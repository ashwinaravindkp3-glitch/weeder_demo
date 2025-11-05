import cv2
import numpy as np
import time
import logging
from datetime import datetime
import os
from ultralytics import YOLO
import json

# Import our custom modules
from precision_grid_mapper import PrecisionGridMapper
from constrained_arm_controller import ConstrainedArmController, MockConstrainedArm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hardware_deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HardwareWeedDetectionSystem:
    """
    Complete hardware deployment system for weed detection and removal
    using 2-servo constrained robotic arm and precision grid mapping.
    """
    
    def __init__(self, camera_id=0, model_path='yolov8n.pt'):
        """
        Initialize the hardware weed detection system.
        
        Args:
            camera_id: Camera device ID (default 0)
            model_path: Path to YOLO model file
        """
        self.camera_id = camera_id
        self.model_path = model_path
        
        # Initialize components
        self.camera = None
        self.model = None
        self.grid_mapper = PrecisionGridMapper()
        self.arm_controller = None
        
        # Detection settings
        self.confidence_threshold = 0.5
        self.target_classes = ['plant', 'weed']  # Adjust based on your model
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'detections_found': 0,
            'weeds_removed': 0,
            'failed_removals': 0,
            'processing_time': []
        }
        
        logger.info("Hardware weed detection system initialized")
    
    def initialize_camera(self):
        """Initialize camera with optimal settings for plant detection."""
        logger.info("Initializing camera...")
        
        try:
            self.camera = cv2.VideoCapture(self.camera_id)
            
            # Set camera properties for optimal plant detection
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
            self.camera.set(cv2.CAP_PROP_EXPOSURE, -6)  # Lower exposure for plants
            self.camera.set(cv2.CAP_PROP_AUTO_WB, 0)  # Manual white balance
            
            # Test camera
            ret, frame = self.camera.read()
            if not ret or frame is None:
                logger.error("Failed to read from camera")
                return False
            
            logger.info(f"Camera initialized: {frame.shape[1]}x{frame.shape[0]}")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def initialize_model(self):
        """Initialize YOLO model for weed detection."""
        logger.info("Loading YOLO model...")
        
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    def initialize_arm(self):
        """Initialize robotic arm controller."""
        logger.info("Initializing robotic arm...")
        
        try:
            # Try to use real hardware first
            try:
                import RPi.GPIO as GPIO
                self.arm_controller = ConstrainedArmController()
                logger.info("Using real hardware arm controller")
            except ImportError:
                self.arm_controller = MockConstrainedArm()
                logger.info("Using mock arm controller")
            
            if not self.arm_controller.initialize_hardware():
                logger.error("Failed to initialize arm hardware")
                return False
            
            logger.info("Robotic arm initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Arm initialization failed: {e}")
            return False
    
    def smart_enhance(self, frame):
        """Enhance image for better plant detection in various lighting conditions."""
        # Convert to LAB color space for better color separation
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Apply slight sharpening
        kernel = np.array([[-0.1, -0.1, -0.1],
                          [-0.1, 1.5, -0.1],
                          [-0.1, -0.1, -0.1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def auto_white_balance(self, frame):
        """Apply automatic white balance for consistent color detection."""
        # Simple gray world assumption
        b, g, r = cv2.split(frame)
        
        # Calculate average values
        avg_b = np.mean(b)
        avg_g = np.mean(g)
        avg_r = np.mean(r)
        
        # Calculate scaling factors
        avg_gray = (avg_b + avg_g + avg_r) / 3
        scale_b = avg_gray / avg_b
        scale_g = avg_gray / avg_g
        scale_r = avg_gray / avg_r
        
        # Apply scaling
        b = np.clip(b * scale_b, 0, 255).astype(np.uint8)
        g = np.clip(g * scale_g, 0, 255).astype(np.uint8)
        r = np.clip(r * scale_r, 0, 255).astype(np.uint8)
        
        return cv2.merge([b, g, r])
    
    def process_frame(self, frame):
        """Process a single frame for weed detection."""
        start_time = time.time()
        
        # Enhance frame
        enhanced = self.smart_enhance(frame)
        enhanced = self.auto_white_balance(enhanced)
        
        # Run YOLO detection
        results = self.model(enhanced, conf=self.confidence_threshold)
        
        # Extract detections
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Filter by target classes if specified
                    class_name = self.model.names[cls]
                    if self.target_classes and class_name not in self.target_classes:
                        continue
                    
                    # Calculate center coordinates
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    
                    detections.append({
                        'x': center_x,
                        'y': center_y,
                        'width': width,
                        'height': height,
                        'confidence': float(conf),
                        'class': class_name,
                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                    })
        
        processing_time = time.time() - start_time
        self.stats['processing_time'].append(processing_time)
        
        logger.info(f"Frame processed: {len(detections)} detections in {processing_time:.3f}s")
        
        return detections, enhanced
    
    def execute_weeding_action(self, detection):
        """Execute weeding action based on detection."""
        try:
            logger.info(f"Executing weeding action for detection at ({detection['x']}, {detection['y']})")
            
            # Get movement command from grid mapper
            movement = self.grid_mapper.detection_to_movement_command(
                detection['x'], detection['y'], detection['width'], detection['height']
            )
            
            logger.info(f"Movement command: {movement['direction']}")
            logger.info(f"Movement needed: {movement['movement_needed']}")
            
            # Check if weeding can be done from current position
            if movement['movement_needed'] == "WEED_IMMEDIATELY":
                logger.info("Target at center - executing immediate weeding!")
                self.arm_controller.perform_weeding_sequence()
                self.stats['weeds_removed'] += 1
                return True
            else:
                # Log movement recommendation for future robot mobility
                logger.info(f"Cannot weed from current position - {movement['direction']}")
                self.stats['failed_removals'] += 1
                return False
                
        except Exception as e:
            logger.error(f"Weeding action failed: {e}")
            self.stats['failed_removals'] += 1
            return False
    
    def draw_detections(self, frame, detections, prioritized_detections=None):
        """Draw detection boxes and grid information on frame."""
        display_frame = frame.copy()
        
        # Draw grid
        for i in range(self.grid_mapper.grid_cols + 1):
            x = i * self.grid_mapper.cell_width
            cv2.line(display_frame, (x, 0), (x, self.grid_mapper.image_height), (100, 100, 100), 1)
        
        for i in range(self.grid_mapper.grid_rows + 1):
            y = i * self.grid_mapper.cell_height
            cv2.line(display_frame, (0, y), (self.grid_mapper.image_width, y), (100, 100, 100), 1)
        
        # Mark center
        center_x = self.grid_mapper.center_col * self.grid_mapper.cell_width + self.grid_mapper.cell_width // 2
        center_y = self.grid_mapper.center_row * self.grid_mapper.cell_height + self.grid_mapper.cell_height // 2
        cv2.circle(display_frame, (center_x, center_y), 10, (0, 255, 0), -1)
        cv2.putText(display_frame, "ARM", (center_x - 15, center_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Determine color based on priority
            color = (0, 255, 255)  # Yellow for normal
            if prioritized_detections:
                for prioritized in prioritized_detections:
                    if (prioritized['pixel_position'][0] == detection['x'] and 
                        prioritized['pixel_position'][1] == detection['y']):
                        if prioritized['priority'] == 0:  # Center target
                            color = (0, 255, 0)  # Green
                        else:
                            color = (0, 165, 255)  # Orange
                        break
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{detection['class']} {detection['confidence']:.2f}"
            cv2.putText(display_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add grid position
            col, row = self.grid_mapper.pixel_to_grid(detection['x'], detection['y'])
            grid_label = f"Grid({col},{row})"
            cv2.putText(display_frame, grid_label, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return display_frame
    
    def run_main_loop(self, max_frames=None, auto_weed=True):
        """Run the main detection and weeding loop."""
        logger.info("Starting main detection loop...")
        
        if not self.initialize_camera():
            return False
        
        if not self.initialize_model():
            return False
        
        if not self.initialize_arm():
            return False
        
        logger.info("All systems initialized successfully!")
        logger.info("Press 'q' to quit, 'w' to weed current targets, 's' to save current frame")
        
        frame_count = 0
        
        try:
            while True:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                # Process frame
                detections, enhanced = self.process_frame(frame)
                self.stats['total_frames'] += 1
                self.stats['detections_found'] += len(detections)
                
                # Prioritize detections
                if detections:
                    detection_coords = [(d['x'], d['y']) for d in detections]
                    prioritized = self.grid_mapper.get_weeding_priority(detection_coords)
                    
                    # Log prioritized targets
                    logger.info(f"Found {len(detections)} targets. Priority order:")
                    for item in prioritized:
                        logger.info(f"  Target {item['detection_id']+1}: "
                                   f"Grid {item['grid_position']} - "
                                   f"{item['movement_command']['direction']}")
                
                # Auto-weeding or manual control
                if auto_weed and detections:
                    # Try to weed the highest priority target (closest to center)
                    if prioritized and prioritized[0]['priority'] == 0:  # Target at center
                        success = self.execute_weeding_action(detections[prioritized[0]['detection_id']])
                        if success:
                            time.sleep(2)  # Pause after successful weeding
                
                # Prepare display
                if detections:
                    display_frame = self.draw_detections(frame, detections, prioritized if 'prioritized' in locals() else None)
                else:
                    display_frame = frame.copy()
                
                # Add statistics overlay
                stats_text = (f"Frames: {self.stats['total_frames']} | "
                            f"Detections: {self.stats['detections_found']} | "
                            f"Weeded: {self.stats['weeds_removed']} | "
                            f"Failed: {self.stats['failed_removals']}")
                
                cv2.putText(display_frame, stats_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show frames
                cv2.imshow('Weed Detection', display_frame)
                cv2.imshow('Enhanced', enhanced)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('w') and detections:
                    # Manual weeding of highest priority target
                    if prioritized and prioritized[0]['priority'] == 0:
                        self.execute_weeding_action(detections[prioritized[0]['detection_id']])
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"weed_detection_{timestamp}.jpg", display_frame)
                    logger.info(f"Saved frame as weed_detection_{timestamp}.jpg")
                
                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    break
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.cleanup()
            self.print_final_statistics()
        
        return True
    
    def cleanup(self):
        """Cleanup all resources."""
        logger.info("Cleaning up resources...")
        
        if self.camera:
            self.camera.release()
        
        if self.arm_controller:
            self.arm_controller.cleanup()
        
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")
    
    def print_final_statistics(self):
        """Print final system statistics."""
        logger.info("=== FINAL STATISTICS ===")
        logger.info(f"Total frames processed: {self.stats['total_frames']}")
        logger.info(f"Total detections found: {self.stats['detections_found']}")
        logger.info(f"Weeds successfully removed: {self.stats['weeds_removed']}")
        logger.info(f"Failed removal attempts: {self.stats['failed_removals']}")
        
        if self.stats['processing_time']:
            avg_time = sum(self.stats['processing_time']) / len(self.stats['processing_time'])
            logger.info(f"Average processing time: {avg_time:.3f}s ({1/avg_time:.1f} FPS)")
        
        if self.stats['total_frames'] > 0:
            detection_rate = (self.stats['detections_found'] / self.stats['total_frames']) * 100
            logger.info(f"Detection rate: {detection_rate:.1f}%")
        
        if self.stats['detections_found'] > 0:
            success_rate = (self.stats['weeds_removed'] / self.stats['detections_found']) * 100
            logger.info(f"Weeding success rate: {success_rate:.1f}%")

def main():
    """Main function to run the hardware deployment system."""
    logger.info("Starting Hardware Weed Detection and Removal System")
    logger.info("This system uses a 2-servo constrained arm and precision grid mapping")
    
    # Configuration
    config = {
        'camera_id': 0,
        'model_path': 'yolov8n.pt',  # Update with your trained model
        'auto_weed': True,  # Set to False for manual control only
        'max_frames': None  # Set to number for limited run, None for continuous
    }
    
    # Create and run system
    system = HardwareWeedDetectionSystem(
        camera_id=config['camera_id'],
        model_path=config['model_path']
    )
    
    success = system.run_main_loop(
        auto_weed=config['auto_weed'],
        max_frames=config['max_frames']
    )
    
    if success:
        logger.info("Hardware deployment completed successfully!")
    else:
        logger.error("Hardware deployment failed!")

if __name__ == "__main__":
    main()