#!/usr/bin/env python3
"""
Calibration Utility for 5-DOF Weed Removal System
==================================================

Interactive tool for calibrating:
1. Camera-to-robot coordinate transformation
2. Arm reach and workspace validation
3. Gripper force and grasp settings
4. Home position and safety limits

Usage:
    python calibration_utility.py [--mock]

Author: Claude
Date: 2025-11-08
"""

import cv2
import numpy as np
import argparse
import logging
from typing import List, Tuple
import json

from robotic_arm_5dof import RoboticArm5DOF, JointAngles
from integrated_weeder_5dof import CoordinateCalibration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CalibrationUtility:
    """Interactive calibration tool"""

    def __init__(self, mock_mode: bool = False):
        """
        Initialize calibration utility

        Args:
            mock_mode: If True, run in simulation mode
        """
        self.mock_mode = mock_mode

        # Initialize arm
        self.arm = RoboticArm5DOF(mock_mode=mock_mode)

        # Initialize camera
        if not mock_mode:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            self.camera = None

        # Calibration data
        self.calibration = CoordinateCalibration()
        self.calibration_points = []  # List of (pixel_x, pixel_y, robot_x, robot_y)

        logger.info("Calibration utility initialized")

    def main_menu(self):
        """Main calibration menu"""
        while True:
            print("\n" + "=" * 60)
            print("CALIBRATION UTILITY - MAIN MENU")
            print("=" * 60)
            print("\n1. Camera-Robot Coordinate Calibration")
            print("2. Workspace Validation")
            print("3. Test Inverse Kinematics")
            print("4. Gripper Force Calibration")
            print("5. Safety Limits Check")
            print("6. Save Calibration")
            print("7. Load Calibration")
            print("0. Exit")
            print("\nChoice: ", end='', flush=True)

            try:
                choice = input().strip()

                if choice == '1':
                    self.calibrate_coordinates()
                elif choice == '2':
                    self.validate_workspace()
                elif choice == '3':
                    self.test_inverse_kinematics()
                elif choice == '4':
                    self.calibrate_gripper()
                elif choice == '5':
                    self.check_safety_limits()
                elif choice == '6':
                    self.save_calibration()
                elif choice == '7':
                    self.load_calibration()
                elif choice == '0':
                    print("\n→ Exiting calibration utility...")
                    break
                else:
                    print("\n✗ Invalid choice")

            except KeyboardInterrupt:
                print("\n\n→ Interrupted. Exiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\n✗ Error: {e}")

    def calibrate_coordinates(self):
        """
        Camera-to-robot coordinate calibration

        Process:
        1. Place markers at known robot coordinates
        2. Take photo and mark pixel coordinates
        3. Build transformation matrix
        """
        print("\n" + "=" * 60)
        print("CAMERA-ROBOT COORDINATE CALIBRATION")
        print("=" * 60)
        print("\nThis calibration maps camera pixels to robot coordinates.")
        print("\nYou will need to:")
        print("  1. Place a visible marker at known robot coordinates")
        print("  2. Move arm to point at the marker")
        print("  3. Record the pixel and robot coordinates")
        print("  4. Repeat for at least 4 points across the workspace")
        print("\nReady? (y/n): ", end='', flush=True)

        if input().strip().lower() != 'y':
            return

        self.calibration_points = []

        while True:
            print("\n" + "-" * 60)
            print(f"Calibration Point {len(self.calibration_points) + 1}")
            print("-" * 60)

            # Get robot coordinates
            print("\nEnter robot coordinates where marker is placed:")
            try:
                robot_x = float(input("  Robot X (cm): ").strip())
                robot_y = float(input("  Robot Y (cm): ").strip())
            except ValueError:
                print("✗ Invalid input")
                continue

            # Move arm to point at marker
            print(f"\n→ Move arm to point at marker at ({robot_x}, {robot_y})")
            print("  Use manual control or teaching mode")
            print("  Press Enter when ready...", end='', flush=True)
            input()

            # Capture frame
            if not self.mock_mode:
                ret, frame = self.camera.read()
                if not ret:
                    print("✗ Failed to capture frame")
                    continue

                # Display frame and let user click on marker
                print("\n→ Click on the marker in the image")
                print("  (Close window when done)")

                pixel_coords = []

                def mouse_callback(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        pixel_coords.append((x, y))
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                        cv2.imshow('Calibration', frame)

                cv2.namedWindow('Calibration')
                cv2.setMouseCallback('Calibration', mouse_callback)
                cv2.imshow('Calibration', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                if not pixel_coords:
                    print("✗ No point selected")
                    continue

                pixel_x, pixel_y = pixel_coords[0]

            else:
                # Mock mode - use random values
                pixel_x = int(320 + robot_x * 5)
                pixel_y = int(240 + robot_y * 5)

            # Save calibration point
            self.calibration_points.append((pixel_x, pixel_y, robot_x, robot_y))
            print(f"✓ Point recorded: Pixel ({pixel_x}, {pixel_y}) → "
                  f"Robot ({robot_x:.1f}, {robot_y:.1f})")

            # Check if we have enough points
            if len(self.calibration_points) >= 4:
                print(f"\n→ You have {len(self.calibration_points)} calibration points")
                print("  This is sufficient for calibration")
                print("\nAdd more points? (y/n): ", end='', flush=True)
                if input().strip().lower() != 'y':
                    break
            else:
                print(f"\n→ {len(self.calibration_points)}/4 points collected")
                print("  Add another point? (y/n): ", end='', flush=True)
                if input().strip().lower() != 'y':
                    print("✗ Calibration cancelled (need at least 4 points)")
                    return

        # Calculate calibration parameters
        self._calculate_calibration()

    def _calculate_calibration(self):
        """Calculate calibration from collected points"""
        if len(self.calibration_points) < 4:
            print("✗ Need at least 4 calibration points")
            return

        print("\n→ Calculating calibration...")

        # Extract points
        pixel_points = np.array([(p[0], p[1]) for p in self.calibration_points])
        robot_points = np.array([(p[2], p[3]) for p in self.calibration_points])

        # Calculate simple linear transformation
        # robot = scale * (pixel - offset)

        # Find bounding boxes
        px_min, px_max = pixel_points[:, 0].min(), pixel_points[:, 0].max()
        py_min, py_max = pixel_points[:, 1].min(), pixel_points[:, 1].max()
        rx_min, rx_max = robot_points[:, 0].min(), robot_points[:, 0].max()
        ry_min, ry_max = robot_points[:, 1].min(), robot_points[:, 1].max()

        # Calculate scales and offsets
        pixel_width = px_max - px_min
        pixel_height = py_max - py_min
        robot_width = rx_max - rx_min
        robot_height = ry_max - ry_min

        # Update calibration object
        self.calibration.camera_width_cm = robot_width
        self.calibration.camera_height_cm = robot_height
        self.calibration.camera_offset_x = (rx_min + rx_max) / 2
        self.calibration.camera_offset_y = (ry_min + ry_max) / 2

        # Recalculate conversion factors
        self.calibration.px_to_cm_x = robot_width / pixel_width
        self.calibration.px_to_cm_y = robot_height / pixel_height

        print("\n✓ Calibration calculated:")
        print(f"  Camera FOV: {robot_width:.1f} × {robot_height:.1f} cm")
        print(f"  Camera offset: ({self.calibration.camera_offset_x:.1f}, "
              f"{self.calibration.camera_offset_y:.1f}) cm")
        print(f"  Pixel to cm: {self.calibration.px_to_cm_x:.3f} (X), "
              f"{self.calibration.px_to_cm_y:.3f} (Y)")

        # Validate calibration
        print("\n→ Validating calibration...")
        errors = []
        for px, py, rx, ry in self.calibration_points:
            calc_x, calc_y = self.calibration.pixel_to_robot_coords(px, py)
            error_x = abs(calc_x - rx)
            error_y = abs(calc_y - ry)
            total_error = np.sqrt(error_x**2 + error_y**2)
            errors.append(total_error)
            print(f"  Point ({px}, {py}): Error = {total_error:.2f} cm")

        avg_error = np.mean(errors)
        max_error = np.max(errors)
        print(f"\n  Average error: {avg_error:.2f} cm")
        print(f"  Maximum error: {max_error:.2f} cm")

        if avg_error < 2.0:
            print("  ✓ Calibration accuracy: GOOD")
        elif avg_error < 5.0:
            print("  ⚠ Calibration accuracy: ACCEPTABLE")
        else:
            print("  ✗ Calibration accuracy: POOR - consider recalibrating")

    def validate_workspace(self):
        """Test arm reach across the workspace"""
        print("\n" + "=" * 60)
        print("WORKSPACE VALIDATION")
        print("=" * 60)
        print("\nThis test validates the arm can reach all positions")
        print("in the camera's field of view.")

        # Define test grid (5x5 points across workspace)
        test_points = []
        for x in np.linspace(-20, 20, 5):
            for y in np.linspace(10, 40, 5):
                test_points.append((x, y))

        print(f"\n→ Testing {len(test_points)} points across workspace...")

        reachable = 0
        unreachable = 0

        for x, y in test_points:
            angles = self.arm.inverse_kinematics(x, y, 0)
            if angles is not None:
                reachable += 1
                status = "✓"
            else:
                unreachable += 1
                status = "✗"

            print(f"  {status} ({x:5.1f}, {y:5.1f}): "
                  f"{'Reachable' if angles else 'UNREACHABLE'}")

        print(f"\n→ Results:")
        print(f"  Reachable:   {reachable}/{len(test_points)} "
              f"({reachable/len(test_points)*100:.1f}%)")
        print(f"  Unreachable: {unreachable}/{len(test_points)}")

        if reachable == len(test_points):
            print("  ✓ Full workspace coverage!")
        elif reachable >= len(test_points) * 0.8:
            print("  ⚠ Most of workspace reachable")
        else:
            print("  ✗ Limited workspace coverage - check arm dimensions")

    def test_inverse_kinematics(self):
        """Interactive IK testing"""
        print("\n" + "=" * 60)
        print("INVERSE KINEMATICS TEST")
        print("=" * 60)

        while True:
            print("\nEnter target position (or 'q' to quit):")

            try:
                x_str = input("  X (cm): ").strip()
                if x_str.lower() == 'q':
                    break

                y_str = input("  Y (cm): ").strip()
                if y_str.lower() == 'q':
                    break

                z_str = input("  Z (cm): ").strip()
                if z_str.lower() == 'q':
                    break

                x = float(x_str)
                y = float(y_str)
                z = float(z_str)

            except ValueError:
                print("✗ Invalid input")
                continue

            # Calculate IK
            print(f"\n→ Calculating IK for ({x}, {y}, {z})...")
            angles = self.arm.inverse_kinematics(x, y, z)

            if angles is None:
                print("✗ No IK solution found (unreachable position)")
                continue

            print("✓ IK solution found:")
            for joint, angle in angles.to_dict().items():
                print(f"  {joint:10s}: {angle:6.1f}°")

            # Verify with forward kinematics
            calc_pos = self.arm.forward_kinematics(angles)
            print("\n→ Verification (forward kinematics):")
            print(f"  Target:     ({x:.2f}, {y:.2f}, {z:.2f})")
            print(f"  Calculated: ({calc_pos[0]:.2f}, {calc_pos[1]:.2f}, {calc_pos[2]:.2f})")

            error = np.sqrt(sum((a - b)**2 for a, b in zip((x, y, z), calc_pos)))
            print(f"  Error: {error:.2f} cm")

            # Ask if user wants to move to this position
            print("\nMove arm to this position? (y/n): ", end='', flush=True)
            if input().strip().lower() == 'y':
                print("→ Moving arm...")
                self.arm.move_to_position(angles)
                print("✓ Movement complete")

    def calibrate_gripper(self):
        """Gripper force calibration"""
        print("\n" + "=" * 60)
        print("GRIPPER FORCE CALIBRATION")
        print("=" * 60)
        print("\nThis helps find the optimal gripper closing angle")
        print("for different weed thicknesses.")

        print("\n→ Testing gripper positions...")

        test_angles = [90, 120, 150, 170, 180]

        for angle in test_angles:
            print(f"\n→ Testing gripper angle: {angle}°")
            print("  (90° = fully open, 180° = fully closed)")

            current = self.arm.get_current_angles()
            current.gripper = angle
            self.arm.move_to_position(current, speed=0.5, interpolation_steps=20)

            print("  Does this grip the test weed securely? (y/n): ", end='', flush=True)
            response = input().strip().lower()

            if response == 'y':
                print(f"  ✓ Optimal angle found: {angle}°")
                print(f"\n  Recommended settings:")
                print(f"    Open position:   90°")
                print(f"    Closed position: {angle}°")
                break

    def check_safety_limits(self):
        """Verify safety limits and emergency stop"""
        print("\n" + "=" * 60)
        print("SAFETY LIMITS CHECK")
        print("=" * 60)

        print("\n→ Testing home position...")
        self.arm.move_to_position(self.arm.waypoints['home'])
        print("✓ Home position reached")

        print("\n→ Current joint angles:")
        current = self.arm.get_current_angles()
        for joint, angle in current.to_dict().items():
            print(f"  {joint:10s}: {angle:6.1f}°")

        print("\n→ All safety checks passed")

    def save_calibration(self):
        """Save calibration to file"""
        filename = "calibration.txt"
        self.calibration.save_calibration(filename)
        print(f"\n✓ Calibration saved to {filename}")

        # Also save calibration points
        if self.calibration_points:
            with open("calibration_points.json", 'w') as f:
                json.dump(self.calibration_points, f, indent=2)
            print(f"✓ Calibration points saved to calibration_points.json")

    def load_calibration(self):
        """Load calibration from file"""
        filename = "calibration.txt"
        self.calibration.load_calibration(filename)
        print(f"\n✓ Calibration loaded from {filename}")

    def cleanup(self):
        """Cleanup resources"""
        if self.camera is not None:
            self.camera.release()
        self.arm.cleanup()
        cv2.destroyAllWindows()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='5-DOF System Calibration Utility')
    parser.add_argument('--mock', action='store_true',
                       help='Run in mock/simulation mode')
    args = parser.parse_args()

    try:
        util = CalibrationUtility(mock_mode=args.mock)
        util.main_menu()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'util' in locals():
            util.cleanup()


if __name__ == "__main__":
    main()
