import cv2
import time
import logging
import torch
from ultralytics import YOLO
from PIL import Image
import os

# Import custom modules
from precision_grid_mapper import PrecisionGridMapper
from robotic_arm_4dof import RoboticArm4DOF, MockRoboticArm4DOF
from hardware_deployment import HardwareDeployment

# Configure logging
# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for extensive logging
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("logs/system_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
CAMERA_INDEX = 0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
GRID_ROWS = 12
GRID_COLS = 16
MODEL_PATH = 'yolov8n.pt'  # Replace with your model path
ARM_LENGTH_CM = 30
CAPTURE_DELAY_S = 5
OUTPUT_BBOX_IMAGE_PATH = 'output/processed_frame_bboxes.jpg'
OUTPUT_GRID_IMAGE_PATH = 'output/processed_frame_grid.jpg'

def main():
    """Main function to run the final integrated system."""
    logger.info("====================================================")
    logger.info("  STARTING PRECISION WEEDER-ROBOT SYSTEM         ")
    logger.info("====================================================")

    # --- Initialize Components ---\
    logger.debug("Initializing components...")
    grid_mapper = PrecisionGridMapper(IMAGE_WIDTH, IMAGE_HEIGHT, GRID_ROWS, GRID_COLS)
    hardware = HardwareDeployment()
    logger.debug("Grid Mapper initialized.")

    try:
        arm = RoboticArm4DOF(arm_length_cm=ARM_LENGTH_CM)
        if not arm.initialize_hardware():
            logger.warning("Hardware initialization failed. Falling back to mock arm.")
            arm = MockRoboticArm4DOF(arm_length_cm=ARM_LENGTH_CM)
    except Exception as e:
        logger.error(f"Critical error initializing arm: {e}", exc_info=True)
        logger.info("Using MockRoboticArm as a fallback.")
        arm = MockRoboticArm4DOF(arm_length_cm=ARM_LENGTH_CM)
    logger.debug("Robotic Arm initialized.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.critical("Failed to open camera. System cannot continue.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
    logger.debug(f"Camera at index {CAMERA_INDEX} opened successfully.")

    try:
        model = YOLO(MODEL_PATH)
        logger.info(f"YOLO model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.critical(f"Failed to load YOLO model: {e}", exc_info=True)
        cap.release()
        return

    try:
        logger.info(f"System ready. Waiting for {CAPTURE_DELAY_S} seconds before capture.")
        time.sleep(CAPTURE_DELAY_S)

        logger.debug("Attempting to capture frame...")
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame from camera.")
            return
        logger.info("Frame captured successfully.")

        logger.debug("Preprocessing frame...")
        enhanced_frame = hardware.smart_enhance(frame)
        balanced_frame = hardware.auto_white_balance(enhanced_frame)
        logger.debug("Frame preprocessing complete.")

        logger.info("Performing weed detection...")
        results = model(balanced_frame, verbose=False)
        detections = results[0].boxes.data
        logger.info(f"Detection complete. Found {len(detections)} potential targets.")

        if len(detections) > 0:
            # For simplicity, target the first detected weed
            target_detection = detections[0]
            x1, y1, x2, y2, conf, cls = target_detection
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            logger.debug(f"Highest priority target: class {int(cls)} with confidence {conf:.2f} at center ({center_x}, {center_y})")

            grid_x, grid_y = grid_mapper.pixel_to_grid(center_x, center_y)
            logger.info(f"Target mapped to grid cell ({grid_x}, {grid_y}).")

            # Assume a fixed distance for this iteration
            fixed_distance_cm = 20
            logger.debug(f"Using fixed distance of {fixed_distance_cm}cm for arm reach.")

            logger.info("Initiating robotic arm sequence...")
            arm.perform_pick_and_place(target_grid_col=grid_x, grid_cols=GRID_COLS, distance_cm=fixed_distance_cm)

        else:
            logger.info("No weeds detected. No action taken.")

        logger.debug("Preparing to save output images.")
        os.makedirs('output', exist_ok=True)

        # --- Image with Bounding Boxes ---
        image_with_bboxes = balanced_frame.copy()
        for det in detections:
            x1, y1, x2, y2, _, _ = map(int, det)
            cv2.rectangle(image_with_bboxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(OUTPUT_BBOX_IMAGE_PATH, image_with_bboxes)
        logger.info(f"Image with bounding boxes saved to {OUTPUT_BBOX_IMAGE_PATH}")

        # --- Image with Grid and Target Cell ---
        image_with_grid = balanced_frame.copy()
        image_with_grid = grid_mapper.draw_grid(image_with_grid)
        if len(detections) > 0:
            target_detection = detections[0]
            x1, y1, x2, y2, _, _ = map(int, target_detection)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            grid_x, grid_y = grid_mapper.pixel_to_grid(center_x, center_y)
            image_with_grid = grid_mapper.draw_target_cell(image_with_grid, grid_x, grid_y)
        cv2.imwrite(OUTPUT_GRID_IMAGE_PATH, image_with_grid)
        logger.info(f"Image with grid and target cell saved to {OUTPUT_GRID_IMAGE_PATH}")

    except KeyboardInterrupt:
        logger.warning("System operation interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during main loop: {e}", exc_info=True)
    finally:
        logger.info("Initiating system shutdown and resource cleanup.")
        cap.release()
        logger.debug("Camera released.")
        arm.cleanup()
        logger.debug("Arm resources cleaned up.")
        logger.info("====================================================")
        logger.info("  SYSTEM SHUTDOWN COMPLETE                       ")
        logger.info("====================================================")

if __name__ == "__main__":
    main()