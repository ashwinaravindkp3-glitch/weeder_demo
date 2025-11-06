import cv2
import time
import logging
import torch
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np

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
        logger.info("=" * 60)
        logger.info("CAMERA PREVIEW MODE - Press 'c' to capture, 'q' to quit")
        logger.info("=" * 60)

        frame_to_process = None

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera.")
                break

            # Calculate frame statistics for debugging
            mean_brightness = np.mean(frame)
            mean_color = np.mean(frame, axis=(0, 1))

            # Create display frame with debug info
            display_frame = frame.copy()

            # Add debug information overlay
            cv2.putText(display_frame, f"Brightness: {mean_brightness:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"BGR: ({mean_color[0]:.1f}, {mean_color[1]:.1f}, {mean_color[2]:.1f})",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'c' to CAPTURE and process", (10, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, "Press 'q' to QUIT", (10, 470),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show camera preview
            cv2.imshow("Camera Preview - Debug Mode", display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("User quit camera preview.")
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('c'):
                logger.info("User captured frame for processing.")
                frame_to_process = frame.copy()
                cv2.destroyAllWindows()
                break

        if frame_to_process is None:
            logger.error("No frame captured.")
            return

        frame = frame_to_process
        mean_brightness = np.mean(frame)
        logger.info(f"Frame captured successfully. Brightness: {mean_brightness:.1f}")

        logger.debug("Preprocessing frame...")
        enhanced_frame = hardware.smart_enhance(frame)
        balanced_frame = hardware.auto_white_balance(enhanced_frame)
        logger.debug("Frame preprocessing complete.")

        logger.info("Performing weed detection...")
        results = model(balanced_frame, verbose=False)
        detections = results[0].boxes.data
        logger.info(f"Detection complete. Found {len(detections)} total detections.")

        # Separate crops and weeds by class
        crops = []
        weeds = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0:  # Class 0 = Crop
                crops.append(det)
            elif int(cls) == 1:  # Class 1 = Weed
                weeds.append(det)

        logger.info(f"Classified: {len(crops)} crops, {len(weeds)} weeds")

        if len(weeds) > 0:
            logger.info(f"Processing {len(weeds)} detected weed(s)...")

            # Prioritize weed detections using the grid mapper
            weed_list = []
            for i, det in enumerate(weeds):
                x1, y1, x2, y2, conf, cls = det
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                grid_x, grid_y = grid_mapper.pixel_to_grid(center_x, center_y)

                # Calculate distance from center for prioritization
                center_col = GRID_COLS // 2
                center_row = GRID_ROWS // 2
                distance_from_center = abs(grid_x - center_col) + abs(grid_y - center_row)

                weed_list.append({
                    'id': i + 1,
                    'detection': det,
                    'center_x': center_x,
                    'center_y': center_y,
                    'grid_x': grid_x,
                    'grid_y': grid_y,
                    'confidence': float(conf),
                    'class': int(cls),
                    'distance_from_center': distance_from_center
                })

            # Sort by distance from center (closest first) for optimal arm movement
            weed_list.sort(key=lambda w: w['distance_from_center'])

            logger.info("=" * 60)
            logger.info("WEED REMOVAL SEQUENCE - Priority Order:")
            logger.info("=" * 60)
            for weed in weed_list:
                logger.info(f"  Weed {weed['id']}: Grid ({weed['grid_x']}, {weed['grid_y']}), "
                           f"Distance: {weed['distance_from_center']}, "
                           f"Confidence: {weed['confidence']:.2f}")
            logger.info("=" * 60)

            # Track removal statistics
            removal_stats = {
                'total': len(weed_list),
                'successful': 0,
                'failed': 0,
                'removed_weeds': []
            }

            # Process each weed
            fixed_distance_cm = 20
            for idx, weed in enumerate(weed_list, 1):
                logger.info(f"\n>>> Processing Weed {idx}/{len(weed_list)} <<<")
                logger.info(f"Position: Grid ({weed['grid_x']}, {weed['grid_y']}), "
                           f"Pixel ({weed['center_x']}, {weed['center_y']})")
                logger.info(f"Confidence: {weed['confidence']:.2f}, Class: {weed['class']}")

                try:
                    # Initiate robotic arm sequence
                    logger.info("Initiating robotic arm sequence...")
                    arm.perform_pick_and_place(
                        target_grid_col=weed['grid_x'],
                        grid_cols=GRID_COLS,
                        distance_cm=fixed_distance_cm
                    )

                    removal_stats['successful'] += 1
                    removal_stats['removed_weeds'].append(weed['id'])
                    logger.info(f"✓ Weed {idx} removed successfully!")

                except Exception as e:
                    removal_stats['failed'] += 1
                    logger.error(f"✗ Failed to remove weed {idx}: {e}")

                # Small delay between removals to allow arm to stabilize
                if idx < len(weed_list):
                    logger.debug("Waiting 1 second before next weed...")
                    time.sleep(1)

            # Display final statistics
            logger.info("\n" + "=" * 60)
            logger.info("WEED REMOVAL SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total Weeds Detected: {removal_stats['total']}")
            logger.info(f"Successfully Removed: {removal_stats['successful']}")
            logger.info(f"Failed: {removal_stats['failed']}")
            success_rate = (removal_stats['successful'] / removal_stats['total'] * 100) if removal_stats['total'] > 0 else 0
            logger.info(f"Success Rate: {success_rate:.1f}%")
            logger.info(f"Removed Weed IDs: {removal_stats['removed_weeds']}")
            logger.info("=" * 60)

        else:
            logger.info("No weeds detected. No action taken.")

        logger.debug("Preparing to save output images.")
        os.makedirs('output', exist_ok=True)

        # --- Image with Bounding Boxes ---
        image_with_bboxes = balanced_frame.copy()

        # Draw crops with blue boxes
        for i, det in enumerate(crops):
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Draw bounding box in blue for crops
            cv2.rectangle(image_with_bboxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Add crop label
            label = f"Crop {i+1}"
            cv2.putText(image_with_bboxes, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw weeds with green boxes
        for i, det in enumerate(weeds):
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Draw bounding box in green for weeds
            cv2.rectangle(image_with_bboxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add weed label
            label = f"Weed {i+1}"
            cv2.putText(image_with_bboxes, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Add summary text
        summary_text = f"Crops: {len(crops)}, Weeds: {len(weeds)}"
        cv2.putText(image_with_bboxes, summary_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imwrite(OUTPUT_BBOX_IMAGE_PATH, image_with_bboxes)
        logger.info(f"Image with bounding boxes saved to {OUTPUT_BBOX_IMAGE_PATH}")

        # --- Image with Grid and WEED Target Cells ---
        image_with_grid = balanced_frame.copy()
        image_with_grid = grid_mapper.draw_grid(image_with_grid)

        if len(weeds) > 0:
            # Draw weed target cells with different colors based on priority
            colors = [
                (0, 0, 255),    # Red - highest priority (closest)
                (0, 165, 255),  # Orange
                (0, 255, 255),  # Yellow
                (0, 255, 0),    # Green
                (255, 0, 255),  # Magenta
            ]

            # Draw ONLY weed target cells (not crops)
            for i, det in enumerate(weeds):
                x1, y1, x2, y2, _, _ = det[0], det[1], det[2], det[3], det[4], det[5]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                grid_x, grid_y = grid_mapper.pixel_to_grid(center_x, center_y)

                # Use different colors to show priority (cycle through colors if more than 5)
                color = colors[i % len(colors)]
                image_with_grid = grid_mapper.draw_target_cell(image_with_grid, grid_x, grid_y, color=color, alpha=0.3)

                # Add weed ID number in the cell
                cell_center_x = grid_x * (IMAGE_WIDTH // GRID_COLS) + (IMAGE_WIDTH // GRID_COLS) // 2
                cell_center_y = grid_y * (IMAGE_HEIGHT // GRID_ROWS) + (IMAGE_HEIGHT // GRID_ROWS) // 2
                cv2.putText(image_with_grid, str(i+1), (cell_center_x - 10, cell_center_y + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        # Add summary text
        summary_text = f"Crops: {len(crops)}, Weeds: {len(weeds)}"
        cv2.putText(image_with_grid, summary_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imwrite(OUTPUT_GRID_IMAGE_PATH, image_with_grid)
        logger.info(f"Image with grid and WEED target cells saved to {OUTPUT_GRID_IMAGE_PATH}")

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