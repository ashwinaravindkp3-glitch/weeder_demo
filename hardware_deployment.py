import cv2
from ultralytics import YOLO
import time
import os
from datetime import datetime
import numpy as np
import logging

# Configure logging for hardware deployment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- HARDWARE CONFIGURATION ---
OUTPUT_FOLDER = "hardware_detections"
CAMERA_INDEX = 0  # Default camera index for Raspberry Pi
COUNTDOWN_SECONDS = 5  # Reduced for faster testing

# --- OPTIMIZED Preprocessing for Low Light ---
BRIGHTNESS = 25
CONTRAST = 1.4
CLAHE_CLIP_LIMIT = 2.0
CONFIDENCE_THRESHOLD = 0.30  # Lower threshold for better detection in poor conditions

# Setup output directory
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

class HardwareDeployment:
    def __init__(self):
        self.model = None
        self.cap = None
        self.state = "WAITING"
        self.countdown_start_time = 0
        
    def initialize_camera(self):
        """Initialize camera with optimal settings for Raspberry Pi"""
        logger.info("Initializing camera...")
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not self.cap.isOpened():
            logger.error(f"Cannot open camera at index {CAMERA_INDEX}")
            return False
            
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 50)
        
        logger.info("Camera initialized successfully")
        return True
        
    def load_model(self, model_path):
        """Load the YOLO model"""
        logger.info("Loading YOLO model...")
        try:
            self.model = YOLO(model_path)
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def smart_enhance(self, frame):
        """Enhanced preprocessing for low-light conditions"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Apply brightness and contrast adjustments
        enhanced = cv2.convertScaleAbs(enhanced, alpha=CONTRAST, beta=BRIGHTNESS)
        
        return enhanced
    
    def auto_white_balance(self, frame):
        """Automatic white balance correction"""
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result
    
    def process_frame(self, frame):
        """Process frame with enhanced preprocessing"""
        # Step 1: Auto white balance
        balanced = self.auto_white_balance(frame)
        
        # Step 2: Smart enhancement
        preprocessed_frame = self.smart_enhance(balanced)
        
        # Step 3: Run inference
        results = self.model(preprocessed_frame, conf=CONFIDENCE_THRESHOLD)
        
        return results, preprocessed_frame
    
    def save_results(self, original_frame, preprocessed_frame, results):
        """Save detection results with comparison"""
        annotated_frame = results[0].plot()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Save annotated result
        filename = f"detection_{timestamp}.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(output_path, annotated_frame)
        
        # Save comparison (original, preprocessed, annotated)
        comparison = np.hstack([original_frame, preprocessed_frame, annotated_frame])
        comparison_path = os.path.join(OUTPUT_FOLDER, f"comparison_{timestamp}.jpg")
        cv2.imwrite(comparison_path, comparison)
        
        # Print detection stats
        detections = results[0].boxes
        crops = sum(1 for box in detections if int(box.cls[0]) == 0)
        weeds = sum(1 for box in detections if int(box.cls[0]) == 1)
        
        avg_confidence = np.mean([float(box.conf[0]) for box in detections]) if detections else 0
        
        logger.info(f"Detection saved: {output_path}")
        logger.info(f"Detected: {crops} crops, {weeds} weeds")
        logger.info(f"Average confidence: {avg_confidence:.2f}")
        
        return crops, weeds, avg_confidence
    
    def run_hardware_test(self, model_path):
        """Main hardware deployment test"""
        logger.info("Starting hardware deployment test...")
        
        # Initialize camera
        if not self.initialize_camera():
            return False
            
        # Load model
        if not self.load_model(model_path):
            self.cap.release()
            return False
        
        logger.info("Hardware deployment ready!")
        logger.info("Instructions:")
        logger.info("- Press 'c' to capture and detect")
        logger.info("- Press 't' for continuous test mode")
        logger.info("- Press 'q' to quit")
        
        test_count = 0
        
        while True:
            success, frame = self.cap.read()
            if not success:
                logger.error("Cannot read frame from camera")
                break
            
            # Display current state
            display_frame = frame.copy()
            
            if self.state == "WAITING":
                cv2.putText(display_frame, "Press 'c' to capture", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 't' for test mode", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press 'q' to quit", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            elif self.state == "COUNTDOWN":
                elapsed_time = time.time() - self.countdown_start_time
                remaining_time = COUNTDOWN_SECONDS - int(elapsed_time)
                
                if remaining_time <= 0:
                    self.state = "CAPTURING"
                else:
                    cv2.putText(display_frame, f"Capturing in: {remaining_time}", 
                               (50, display_frame.shape[0] // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
            
            elif self.state == "CAPTURING":
                logger.info("Processing capture...")
                
                # Process frame
                results, preprocessed_frame = self.process_frame(frame)
                
                # Save results
                crops, weeds, confidence = self.save_results(frame, preprocessed_frame, results)
                
                # Show annotated result briefly
                annotated_frame = results[0].plot()
                cv2.putText(annotated_frame, f"Crops: {crops}, Weeds: {weeds}, Conf: {confidence:.2f}", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Detection Result", annotated_frame)
                cv2.waitKey(2000)  # Show for 2 seconds
                cv2.destroyWindow("Detection Result")
                
                test_count += 1
                self.state = "WAITING"
            
            # Show live preview
            cv2.imshow("Hardware Deployment - Press 'c' to capture", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and self.state == "WAITING":
                self.state = "COUNTDOWN"
                self.countdown_start_time = time.time()
                logger.info("Countdown started...")
            elif key == ord('t') and self.state == "WAITING":
                # Quick test mode - capture immediately
                logger.info("Quick test mode activated")
                self.state = "CAPTURING"
        
        logger.info(f"Hardware test completed. Total captures: {test_count}")
        self.cleanup()
        return True
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Hardware deployment cleanup complete")

# Quick test function for immediate deployment
def quick_hardware_test(model_path):
    """Run a quick hardware test with minimal setup"""
    deployment = HardwareDeployment()
    return deployment.run_hardware_test(model_path)

if __name__ == "__main__":
    # Model path - update this to your actual model path
    MODEL_PATH = r'C:\Users\ASHWIN K P\Downloads\Yolo_dataset-20251104T114915Z-1-001\Yolo_dataset\runs\yolov8_brinjal_weed\weights\best.pt'
    
    logger.info("Starting hardware deployment for Raspberry Pi")
    
    # Run hardware test
    success = quick_hardware_test(MODEL_PATH)
    
    if success:
        logger.info("Hardware deployment test completed successfully!")
    else:
        logger.error("Hardware deployment test failed!")
    
    logger.info("Ready for robotic arm integration next.")