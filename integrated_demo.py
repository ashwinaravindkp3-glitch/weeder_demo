#!/usr/bin/env python3
"""
Complete Integrated Demo: Camera Calibration + Weed Detection + Robotic Arm

This demo showcases the complete autonomous weed detection and removal system
with camera calibration, weed detection, and robotic arm coordination.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import time
from typing import List, Tuple, Optional

# Import our modules
from detection.weed_detector import WeedDetectionSystem, create_demo_detection_system
from calibration.camera_calibration import CameraCalibrator, InteractiveCalibrator
from arm_control.robotic_arm import WeedRemovalArm

class IntegratedDemo:
    """Complete integrated demo system"""
    
    def __init__(self):
        import logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing IntegratedDemo")
        print("Initializing Integrated AgriBot Demo System")
        print("=" * 60)
        
        # Initialize components
        self.logger.info("Initializing components")
        self.weed_detector = create_demo_detection_system()
        self.camera_calibrator = CameraCalibrator()
        self.robotic_arm = WeedRemovalArm()
        
        # Demo state
        self.logger.info("Initializing demo state")
        self.calibration_points = []
        self.detected_weeds = []
        self.removal_success_count = 0
        self.total_weeds_removed = 0
        
        self.logger.info("IntegratedDemo initialized successfully")
        print("All components initialized successfully!")
    
    def calibrate_camera_demo(self) -> bool:
        """Demonstrate camera calibration process"""
        self.logger.info("Starting camera calibration demo")
        print("\nCamera Calibration Demo")
        print("-" * 30)
        
        # Simulate calibration points (normally would be from actual camera)
        calibration_data = [
            # (pixel_x, pixel_y, world_x, world_y)
            (100, 100, 0.0, 0.0),
            (300, 100, 10.0, 0.0),
            (500, 100, 20.0, 0.0),
            (100, 300, 0.0, 10.0),
            (300, 300, 10.0, 10.0),
            (500, 300, 20.0, 10.0),
            (100, 400, 0.0, 15.0),
            (300, 400, 10.0, 15.0),
            (500, 400, 20.0, 15.0),
        ]
        
        self.logger.info(f"Adding {len(calibration_data)} calibration points")
        print(f"Adding {len(calibration_data)} calibration points...")
        
        for i, (px, py, wx, wy) in enumerate(calibration_data):
            self.camera_calibrator.add_calibration_point((px, py), (wx, wy, 0.0))
            self.logger.debug(f"Added calibration point {i+1}: Pixel({px}, {py}) -> World({wx}, {wy})")
            print(f"  Point {i+1}: Pixel({px}, {py}) -> World({wx}, {wy})")
        
        # Perform calibration
        self.logger.info("Performing perspective calibration")
        success = self.camera_calibrator.calibrate_perspective()
        
        if success:
            self.logger.info("Camera calibration successful")
            print("Camera calibration successful!")
            
            # Test calibration accuracy
            test_points = [(200, 200), (400, 250)]
            self.logger.info("Testing calibration accuracy")
            print("\nTesting calibration accuracy:")
            
            for px, py in test_points:
                world_coords = self.camera_calibrator.pixel_to_world(px, py)
                self.logger.debug(f"Test point: Pixel({px}, {py}) -> World({world_coords[0]:.2f}, {world_coords[1]:.2f})")
                print(f"  Pixel({px}, {py}) -> World({world_coords[0]:.2f}, {world_coords[1]:.2f})")
            
            # Show calibration statistics
            stats = self.camera_calibrator.get_calibration_accuracy()
            self.logger.info(f"Calibration stats: {stats}")
            print(f"\nCalibration Statistics:")
            if 'avg_error' in stats:
                print(f"  Average Error: {stats['avg_error']:.3f} mm")
            if 'max_error' in stats:
                print(f"  Max Error: {stats['max_error']:.3f} mm")
            print(f"  Total Points: {len(self.camera_calibrator.calibration_points)}")
            
            return True
        else:
            self.logger.error("Camera calibration failed")
            print("Camera calibration failed!")
            return False
    
    def detect_weeds_demo(self) -> List:
        """Demonstrate weed detection process"""
        self.logger.info("Starting weed detection demo")
        print("\nWeed Detection Demo")
        print("-" * 25)
        
        # Create a synthetic test image (normally would be from camera)
        self.logger.info("Creating synthetic field image")
        test_image = self._create_synthetic_field_image()
        
        print("Detecting weeds in synthetic field image...")
        
        # Perform detection
        self.logger.info("Performing weed detection")
        start_time = time.time()
        detections = self.weed_detector.detect_weeds(test_image)
        detection_time = time.time() - start_time
        self.logger.info(f"Detection completed in {detection_time:.3f} seconds. Found {len(detections)} weeds.")
        
        print(f"✅ Detection completed in {detection_time:.3f} seconds")
        print(f"Found {len(detections)} weeds")
        
        # Store detections
        self.detected_weeds = detections
        
        # Display detection results
        if detections:
            print("\nDetected Weeds:")
            for i, weed in enumerate(detections):
                # Convert pixel coordinates to world coordinates
                world_coords = self.camera_calibrator.pixel_to_world(
                    int(weed.center[0]), int(weed.center[1])
                )
                self.logger.debug(f"Weed {i+1}: Confidence={weed.confidence:.2f}, Pixel({int(weed.center[0])}, {int(weed.center[1])}), World({world_coords[0]:.1f}, {world_coords[1]:.1f})")
                print(f"  Weed {i+1}: Confidence={weed.confidence:.2f}, "
                      f"Pixel({int(weed.center[0])}, {int(weed.center[1])}), "
                      f"World({world_coords[0]:.1f}, {world_coords[1]:.1f})")
        
        # Get detection statistics
        stats = self.weed_detector.get_detection_statistics()
        self.logger.info(f"Detection stats: {stats}")
        print(f"\nDetection Statistics:")
        print(f"  Total Detections: {stats['total_detections']}")
        print(f"  Detection Rate: {stats['detection_rate']:.2f}")
        
        return detections
    
    def robotic_arm_demo(self) -> bool:
        """Demonstrate robotic arm weed removal"""
        self.logger.info("Starting robotic arm demo")
        print("\n🤖 Robotic Arm Weed Removal Demo")
        print("-" * 35)
        
        if not self.detected_weeds:
            self.logger.warning("No weeds detected, running detection first")
            print("No weeds detected. Running detection first...")
            self.detect_weeds_demo()
        
        if not self.detected_weeds:
            self.logger.error("No weeds to remove after running detection")
            print("❌ No weeds to remove!")
            return False
        
        self.logger.info(f"Starting removal of {len(self.detected_weeds)} detected weeds")
        print(f"Starting removal of {len(self.detected_weeds)} detected weeds...")
        
        # Move arm to home position
        self.logger.info("Moving arm to home position")
        print("Moving arm to home position...")
        self.robotic_arm.move_to_home()
        time.sleep(1)
        
        # Remove each weed
        for i, weed in enumerate(self.detected_weeds):
            self.logger.info(f"Removing weed {i+1}/{len(self.detected_weeds)}")
            print(f"\n--- Removing Weed {i+1}/{len(self.detected_weeds)} ---")
            
            # Convert pixel coordinates to world coordinates
            world_coords = self.camera_calibrator.pixel_to_world(
                int(weed.center[0]), int(weed.center[1])
            )
            
            self.logger.debug(f"Target: Pixel({int(weed.center[0])}, {int(weed.center[1])}) -> World({world_coords[0]:.1f}, {world_coords[1]:.1f})")
            print(f"Target: Pixel({int(weed.center[0])}, {int(weed.center[1])}) "
                  f"-> World({world_coords[0]:.1f}, {world_coords[1]:.1f})")
            
            # Perform weed removal
            success = self.robotic_arm.remove_weed(
                pixel_x=int(weed.center[0]),
                pixel_y=int(weed.center[1]),
                world_coords=(world_coords[0], world_coords[1], 0.0)
            )
            
            if success:
                self.removal_success_count += 1
                self.total_weeds_removed += 1
                self.logger.info(f"Weed {i+1} removed successfully")
                print(f"✅ Weed {i+1} removed successfully!")
            else:
                self.logger.error(f"Failed to remove weed {i+1}")
                print(f"❌ Failed to remove weed {i+1}")
            
            # Small delay between removals
            time.sleep(0.5)
        
        # Return to home position
        self.logger.info("Returning arm to home position")
        print("\nReturning arm to home position...")
        self.robotic_arm.move_to_home()
        
        # Show final statistics
        removal_rate = 100 * self.removal_success_count / len(self.detected_weeds)
        self.logger.info(f"Weed removal summary: {self.removal_success_count}/{len(self.detected_weeds)} removed ({removal_rate:.1f}% success)")
        print(f"\n🎯 Weed Removal Summary:")
        print(f"  Total Weeds Detected: {len(self.detected_weeds)}")
        print(f"  Successfully Removed: {self.removal_success_count}")
        print(f"  Success Rate: {removal_rate:.1f}%")
        
        return self.removal_success_count > 0
    
    def complete_system_demo(self):
        """Run complete integrated system demonstration"""
        self.logger.info("Starting complete system demo")
        print("🚀 COMPLETE INTEGRATED SYSTEM DEMO")
        print("=" * 50)
        print("This demo showcases the full autonomous weed detection and removal process:")
        print("1. Camera calibration for coordinate mapping")
        print("2. Weed detection using multiple methods")
        print("3. Robotic arm weed removal")
        print("4. Performance monitoring and statistics")
        print()
        
        # Step 1: Camera Calibration
        self.logger.info("Step 1: Camera Calibration")
        print("STEP 1: Camera Calibration")
        calibration_success = self.calibrate_camera_demo()
        
        if not calibration_success:
            self.logger.error("Calibration failed, cannot proceed with demo")
            print("❌ Calibration failed, cannot proceed with demo")
            return False
        
        # Step 2: Weed Detection
        self.logger.info("Step 2: Weed Detection")
        print("\nSTEP 2: Weed Detection")
        detections = self.detect_weeds_demo()
        
        if not detections:
            self.logger.warning("No weeds detected, but continuing with demo")
            print("⚠️  No weeds detected, but continuing with demo")
        
        # Step 3: Robotic Arm Removal
        self.logger.info("Step 3: Robotic Arm Weed Removal")
        print("\nSTEP 3: Robotic Arm Weed Removal")
        removal_success = self.robotic_arm_demo()
        
        # Step 4: Final Statistics
        self.logger.info("Step 4: Final Statistics")
        print("\nSTEP 4: Final System Statistics")
        self.display_final_statistics()
        
        self.logger.info("Complete system demo finished")
        print("\n🎉 COMPLETE SYSTEM DEMO FINISHED!")
        return True
    
    def _create_synthetic_field_image(self) -> np.ndarray:
        """Create a synthetic field image with weeds for testing"""
        # Create base field image (green background)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :] = (34, 139, 34)  # Forest green for field
        
        # Add some random variation to simulate field texture
        noise = np.random.randint(-20, 20, (480, 640, 3))
        image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Add synthetic weeds (bright green spots)
        weed_positions = [
            (150, 120), (320, 200), (450, 150),
            (200, 300), (380, 350), (500, 280)
        ]
        
        for x, y in weed_positions:
            # Draw weed as a bright green circle
            cv2.circle(image, (x, y), 15, (0, 255, 0), -1)
            
            # Add some texture to the weed
            for _ in range(5):
                offset_x = np.random.randint(-10, 10)
                offset_y = np.random.randint(-10, 10)
                cv2.circle(image, (x + offset_x, y + offset_y), 3, (50, 205, 50), -1)
        
        return image
    
    def display_final_statistics(self):
        """Display comprehensive system statistics"""
        self.logger.info("Displaying final statistics")
        print("\n" + "=" * 40)
        print("📊 COMPLETE SYSTEM STATISTICS")
        print("=" * 40)
        
        # Camera calibration stats
        calib_stats = self.camera_calibrator.get_calibration_accuracy()
        self.logger.info(f"Final calibration stats: {calib_stats}")
        print(f"\n🎯 Camera Calibration:")
        if 'avg_error' in calib_stats:
            print(f"  Average Error: {calib_stats['avg_error']:.3f} mm")
        if 'max_error' in calib_stats:
            print(f"  Max Error: {calib_stats['max_error']:.3f} mm")
        print(f"  Calibration Points: {len(self.camera_calibrator.calibration_points)}")
        
        # Detection stats
        detection_stats = self.weed_detector.get_detection_statistics()
        self.logger.info(f"Final detection stats: {detection_stats}")
        print(f"\n🔍 Weed Detection:")
        print(f"  Total Detections: {detection_stats['total_detections']}")
        print(f"  Detection Rate: {detection_stats['detection_rate']:.2f}")
        print(f"  Active Detector: {detection_stats['active_detector']}")
        
        # Per-detector stats
        print(f"\n📋 Per-Detector Statistics:")
        for detector_name, stats in detection_stats['detector_stats'].items():
            print(f"  {detector_name.capitalize()}:")
            print(f"    Total Detections: {stats['total_detections']}")
            print(f"    Avg Confidence: {stats['avg_confidence']:.2f}")
            print(f"    Avg Processing Time: {stats['processing_time']:.3f}s")
        
        # Robotic arm stats
        arm_stats = self.robotic_arm.get_statistics()
        self.logger.info(f"Final arm stats: {arm_stats}")
        print(f"\n🤖 Robotic Arm:")
        print(f"  Total Weeds Removed: {arm_stats['weeds_removed']}")
        print(f"  Success Rate: {arm_stats['success_rate']:.1f}%")
        print(f"  Average Removal Time: {arm_stats['avg_removal_time']:.1f}s")
        print(f"  Total Distance Moved: {arm_stats['total_distance']:.1f}cm")
        
        # Overall system performance
        self.logger.info("Calculating overall system performance")
        print(f"\n🌟 Overall System Performance:")
        if self.detected_weeds:
            success_rate = (self.removal_success_count / len(self.detected_weeds)) * 100
            print(f"  Overall Success Rate: {success_rate:.1f}%")
        else:
            print("  Overall Success Rate: N/A (no weeds detected)")
        
        print(f"  Total Operations: {detection_stats['total_frames']}")
        print("=" * 40)

def main():
    """Main demo function"""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting main demo function")
    
    print("AUTONOMOUS WEED DETECTION & REMOVAL SYSTEM")
    print("Complete Integrated Demo")
    print("=" * 60)
    
    try:
        # Create and run integrated demo
        logging.info("Creating IntegratedDemo instance")
        demo = IntegratedDemo()
        
        # Run complete system demonstration
        logging.info("Running complete_system_demo")
        success = demo.complete_system_demo()
        
        if success:
            logging.info("Demo completed successfully")
            print("\n✅ DEMO COMPLETED SUCCESSFULLY!")
            print("The AgriBot system is ready for real-world deployment.")
        else:
            logging.warning("Demo completed with issues")
            print("\n⚠️  DEMO COMPLETED WITH ISSUES")
            print("Check the output above for details.")
        
        return 0
        
    except KeyboardInterrupt:
        logging.warning("Demo interrupted by user.")
        print("\n\nDemo interrupted by user.")
        return 1
    except Exception as e:
        logging.error(f"Demo failed with error: {e}", exc_info=True)
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())