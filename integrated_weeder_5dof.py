#!/usr/bin/env python3
"""
Integrated Weed Detection and Removal System with 5-DOF Arm
============================================================

This module integrates:
- YOLOv8 weed detection
- Precision grid mapping
- 5-DOF robotic arm control
- Complete autonomous weed removal pipeline

Usage:
    # Real hardware mode
    python integrated_weeder_5dof.py

    # Mock/simulation mode
    python integrated_weeder_5dof.py --mock

Author: Claude
Date: 2025-11-08
"""

import cv2
import numpy as np
import time
import logging
import argparse
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from robotic_arm_5dof import RoboticArm5DOF, JointAngles
from precision_grid_mapper import PrecisionGridMapper

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Detection will be mocked.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WeedDetection:
    """Represents a detected weed"""
    x: int              # Center X in pixels
    y: int              # Center Y in pixels
    width: int          # Bounding box width
    height: int         # Bounding box height
    confidence: float   # Detection confidence (0-1)
    grid_x: int        # Grid column
    grid_y: int        # Grid row
    priority: int      # Removal priority (0=highest)


class CoordinateCalibration:
    """
    Calibrates between camera coordinates (pixels) and robot coordinates (cm)
    """

    def __init__(self):
        """
        Initialize coordinate calibration

        These values should be calibrated for your specific setup
        """
        # Camera field of view in cm at ground level
        self.camera_width_cm = 50.0   # Width of camera view at ground (cm)
        self.camera_height_cm = 37.5  # Height of camera view at ground (cm)

        # Camera position relative to robot base
        self.camera_offset_x = 0.0    # Camera X offset from robot base (cm)
        self.camera_offset_y = 30.0   # Camera Y offset (forward) from robot base (cm)

        # Image dimensions
        self.image_width_px = 640
        self.image_height_px = 480

        # Calculate pixel-to-cm conversion factors
        self.px_to_cm_x = self.camera_width_cm / self.image_width_px
        self.px_to_cm_y = self.camera_height_cm / self.image_height_px

        logger.info(f"Calibration initialized: {self.px_to_cm_x:.3f} cm/px (X), "
                   f"{self.px_to_cm_y:.3f} cm/px (Y)")

    def pixel_to_robot_coords(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to robot coordinates

        Args:
            pixel_x: X coordinate in pixels (0-639)
            pixel_y: Y coordinate in pixels (0-479)

        Returns:
            (robot_x, robot_y) in centimeters relative to robot base
        """
        # Convert pixels to cm, with origin at image center
        cam_x = (pixel_x - self.image_width_px / 2) * self.px_to_cm_x
        cam_y = (pixel_y - self.image_height_px / 2) * self.px_to_cm_y

        # Transform to robot coordinates
        # Camera X = Robot X (lateral)
        # Camera Y = Robot Y (forward/back), accounting for offset
        robot_x = cam_x + self.camera_offset_x
        robot_y = cam_y + self.camera_offset_y

        return (robot_x, robot_y)

    def save_calibration(self, filename: str = "calibration.txt"):
        """Save calibration parameters to file"""
        with open(filename, 'w') as f:
            f.write(f"camera_width_cm={self.camera_width_cm}\n")
            f.write(f"camera_height_cm={self.camera_height_cm}\n")
            f.write(f"camera_offset_x={self.camera_offset_x}\n")
            f.write(f"camera_offset_y={self.camera_offset_y}\n")
        logger.info(f"Calibration saved to {filename}")

    def load_calibration(self, filename: str = "calibration.txt"):
        """Load calibration parameters from file"""
        try:
            with open(filename, 'r') as f:
                for line in f:
                    key, value = line.strip().split('=')
                    setattr(self, key, float(value))

            # Recalculate conversion factors
            self.px_to_cm_x = self.camera_width_cm / self.image_width_px
            self.px_to_cm_y = self.camera_height_cm / self.image_height_px

            logger.info(f"Calibration loaded from {filename}")
        except FileNotFoundError:
            logger.warning(f"Calibration file {filename} not found, using defaults")


class IntegratedWeedRemovalSystem:
    """
    Complete integrated system for autonomous weed detection and removal
    """

    def __init__(self, mock_mode: bool = False, camera_index: int = 0):
        """
        Initialize the integrated system

        Args:
            mock_mode: If True, run in simulation mode
            camera_index: Camera device index
        """
        self.mock_mode = mock_mode

        # Initialize components
        logger.info("Initializing integrated weed removal system...")

        # 1. Initialize robotic arm
        self.arm = RoboticArm5DOF(mock_mode=mock_mode)
        logger.info("✓ Robotic arm initialized")

        # 2. Initialize grid mapper
        self.grid_mapper = PrecisionGridMapper()
        logger.info("✓ Grid mapper initialized")

        # 3. Initialize coordinate calibration
        self.calibration = CoordinateCalibration()
        self.calibration.load_calibration()
        logger.info("✓ Coordinate calibration loaded")

        # 4. Initialize camera
        if not mock_mode:
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open camera {camera_index}")
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            logger.info("✓ Camera initialized")
        else:
            self.camera = None
            logger.info("✓ Camera skipped (mock mode)")

        # 5. Initialize YOLO detector
        if YOLO_AVAILABLE and not mock_mode:
            try:
                self.detector = YOLO('yolov8n.pt')
                logger.info("✓ YOLO detector loaded")
            except Exception as e:
                logger.warning(f"YOLO failed to load: {e}. Using mock detection.")
                self.detector = None
        else:
            self.detector = None
            logger.info("✓ Detection mocked")

        # Statistics
        self.stats = {
            'weeds_detected': 0,
            'weeds_removed': 0,
            'failed_removals': 0,
            'total_cycles': 0
        }

        logger.info("=" * 60)
        logger.info("System ready!")
        logger.info("=" * 60)

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the camera

        Returns:
            Frame as numpy array, or None if failed
        """
        if self.mock_mode:
            # Create mock frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some mock weeds (green circles)
            cv2.circle(frame, (200, 300), 15, (0, 255, 0), -1)
            cv2.circle(frame, (400, 250), 12, (0, 255, 0), -1)
            return frame

        ret, frame = self.camera.read()
        if not ret:
            logger.error("Failed to capture frame")
            return None

        return frame

    def detect_weeds(self, frame: np.ndarray) -> List[WeedDetection]:
        """
        Detect weeds in the frame

        Args:
            frame: Input image

        Returns:
            List of WeedDetection objects
        """
        if self.mock_mode or self.detector is None:
            # Return mock detections
            mock_weeds = [
                WeedDetection(200, 300, 30, 30, 0.95, 0, 0, 0),
                WeedDetection(400, 250, 24, 24, 0.87, 0, 0, 1),
            ]
            # Calculate grid positions
            for weed in mock_weeds:
                weed.grid_x, weed.grid_y = self.grid_mapper.pixel_to_grid(weed.x, weed.y)
            return mock_weeds

        # Run YOLO detection
        results = self.detector(frame, conf=0.5, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class (0=crop, 1=weed) - adjust based on your model
                cls = int(box.cls[0])
                if cls != 1:  # Only process weeds
                    continue

                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                width = int(x2 - x1)
                height = int(y2 - y1)
                confidence = float(box.conf[0])

                # Convert to grid coordinates
                grid_x, grid_y = self.grid_mapper.pixel_to_grid(center_x, center_y)

                weed = WeedDetection(
                    x=center_x,
                    y=center_y,
                    width=width,
                    height=height,
                    confidence=confidence,
                    grid_x=grid_x,
                    grid_y=grid_y,
                    priority=0  # Will be calculated later
                )
                detections.append(weed)

        return detections

    def prioritize_weeds(self, weeds: List[WeedDetection]) -> List[WeedDetection]:
        """
        Sort weeds by priority (closest to center first)

        Args:
            weeds: List of detected weeds

        Returns:
            Sorted list with priority assigned
        """
        center_x = self.grid_mapper.center_col
        center_y = self.grid_mapper.center_row

        # Calculate Manhattan distance from center for each weed
        for weed in weeds:
            distance = abs(weed.grid_x - center_x) + abs(weed.grid_y - center_y)
            weed.priority = distance

        # Sort by priority (lowest = closest = highest priority)
        weeds.sort(key=lambda w: w.priority)

        # Update priority indices
        for i, weed in enumerate(weeds):
            weed.priority = i

        return weeds

    def remove_weed(self, weed: WeedDetection) -> bool:
        """
        Remove a single weed using the robotic arm

        Args:
            weed: WeedDetection to remove

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Removing weed at pixel ({weed.x}, {weed.y}), "
                   f"grid ({weed.grid_x}, {weed.grid_y})")

        try:
            # Convert pixel coordinates to robot coordinates
            robot_x, robot_y = self.calibration.pixel_to_robot_coords(weed.x, weed.y)

            logger.info(f"Robot coordinates: ({robot_x:.1f}, {robot_y:.1f}) cm")

            # Execute weed plucking sequence
            success = self.arm.execute_weed_pluck_sequence(
                weed_x=robot_x,
                weed_y=robot_y,
                weed_z=0.0  # Ground level
            )

            if success:
                self.stats['weeds_removed'] += 1
                logger.info("✓ Weed removed successfully")
            else:
                self.stats['failed_removals'] += 1
                logger.warning("✗ Weed removal failed")

            return success

        except Exception as e:
            logger.error(f"Error during weed removal: {e}")
            self.stats['failed_removals'] += 1
            return False

    def run_single_cycle(self) -> Dict:
        """
        Run a single detection and removal cycle

        Returns:
            Dictionary with cycle statistics
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"CYCLE {self.stats['total_cycles'] + 1}")
        logger.info("=" * 60)

        cycle_stats = {
            'weeds_detected': 0,
            'weeds_removed': 0,
            'weeds_failed': 0
        }

        # 1. Capture frame
        logger.info("Step 1: Capturing frame...")
        frame = self.capture_frame()
        if frame is None:
            logger.error("Failed to capture frame")
            return cycle_stats

        # 2. Detect weeds
        logger.info("Step 2: Detecting weeds...")
        weeds = self.detect_weeds(frame)
        cycle_stats['weeds_detected'] = len(weeds)
        self.stats['weeds_detected'] += len(weeds)

        logger.info(f"Found {len(weeds)} weed(s)")

        if len(weeds) == 0:
            logger.info("No weeds detected, cycle complete")
            self.stats['total_cycles'] += 1
            return cycle_stats

        # 3. Prioritize weeds
        logger.info("Step 3: Prioritizing weeds...")
        weeds = self.prioritize_weeds(weeds)

        for i, weed in enumerate(weeds, 1):
            logger.info(f"  {i}. Priority {weed.priority}: "
                       f"Grid ({weed.grid_x}, {weed.grid_y}), "
                       f"Confidence: {weed.confidence:.2f}")

        # 4. Remove weeds in priority order
        logger.info("Step 4: Removing weeds...")
        for i, weed in enumerate(weeds, 1):
            logger.info(f"\nRemoving weed {i}/{len(weeds)}...")
            success = self.remove_weed(weed)

            if success:
                cycle_stats['weeds_removed'] += 1
            else:
                cycle_stats['weeds_failed'] += 1

            # Small delay between weeds
            time.sleep(0.5)

        # Update stats
        self.stats['total_cycles'] += 1

        logger.info("\n" + "=" * 60)
        logger.info("CYCLE COMPLETE")
        logger.info(f"  Detected: {cycle_stats['weeds_detected']}")
        logger.info(f"  Removed:  {cycle_stats['weeds_removed']}")
        logger.info(f"  Failed:   {cycle_stats['weeds_failed']}")
        logger.info("=" * 60)

        return cycle_stats

    def run_continuous(self, max_cycles: Optional[int] = None):
        """
        Run continuous weed detection and removal

        Args:
            max_cycles: Maximum number of cycles, or None for infinite
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING CONTINUOUS OPERATION")
        logger.info("=" * 60)
        logger.info(f"Max cycles: {max_cycles if max_cycles else 'Infinite'}")
        logger.info("Press Ctrl+C to stop\n")

        cycle = 0
        try:
            while max_cycles is None or cycle < max_cycles:
                self.run_single_cycle()
                cycle += 1

                # Delay between cycles
                time.sleep(2.0)

        except KeyboardInterrupt:
            logger.info("\n\nStopped by user")

        finally:
            self.print_statistics()

    def print_statistics(self):
        """Print overall statistics"""
        logger.info("\n" + "=" * 60)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total cycles:      {self.stats['total_cycles']}")
        logger.info(f"Weeds detected:    {self.stats['weeds_detected']}")
        logger.info(f"Weeds removed:     {self.stats['weeds_removed']}")
        logger.info(f"Failed removals:   {self.stats['failed_removals']}")

        if self.stats['weeds_detected'] > 0:
            success_rate = (self.stats['weeds_removed'] / self.stats['weeds_detected']) * 100
            logger.info(f"Success rate:      {success_rate:.1f}%")

        logger.info("=" * 60)

    def cleanup(self):
        """Cleanup resources"""
        logger.info("\nCleaning up...")

        if self.camera is not None:
            self.camera.release()

        self.arm.cleanup()

        cv2.destroyAllWindows()
        logger.info("✓ Cleanup complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Integrated 5-DOF Weed Removal System')
    parser.add_argument('--mock', action='store_true',
                       help='Run in mock/simulation mode')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--cycles', type=int, default=None,
                       help='Number of cycles to run (default: infinite)')
    parser.add_argument('--single', action='store_true',
                       help='Run single cycle only')

    args = parser.parse_args()

    try:
        # Create system
        system = IntegratedWeedRemovalSystem(
            mock_mode=args.mock,
            camera_index=args.camera
        )

        # Run
        if args.single:
            system.run_single_cycle()
        else:
            system.run_continuous(max_cycles=args.cycles)

    except Exception as e:
        logger.error(f"System error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if 'system' in locals():
            system.cleanup()


if __name__ == "__main__":
    main()
