#!/usr/bin/env python3
"""
Lookup Table Controller for 5-DOF Arm (No Encoders)
====================================================

This is the MOST PRACTICAL approach for teaching a robotic arm
without encoder feedback in limited time (41 hours).

Method: Manual Lookup Table with Interpolation
- Manually position arm at grid points and record PWM values
- Store in lookup table
- Interpolate for positions between grid points

Author: Claude
Date: 2025-11-08
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.interpolate import griddata

# Try to import GPIO
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MotorPWM:
    """PWM values for all 5 motors (0-100%)"""
    base: float      # Motor 1: Base rotation
    shoulder: float  # Motor 2: Shoulder lift
    elbow: float     # Motor 3: Elbow bend
    wrist: float     # Motor 4: Wrist tilt
    gripper: float   # Motor 5: Gripper open/close

    def to_list(self) -> List[float]:
        return [self.base, self.shoulder, self.elbow, self.wrist, self.gripper]

    def to_dict(self) -> Dict[str, float]:
        return {
            'base': self.base,
            'shoulder': self.shoulder,
            'elbow': self.elbow,
            'wrist': self.wrist,
            'gripper': self.gripper
        }


class SimpleMotorController:
    """
    Simple open-loop motor controller (no encoder feedback)
    Uses PWM or timed pulses to control servos/DC motors
    """

    def __init__(self, pin: int, mock_mode: bool = False):
        """
        Initialize motor controller

        Args:
            pin: GPIO pin for PWM control
            mock_mode: If True, simulate without hardware
        """
        self.pin = pin
        self.mock_mode = mock_mode or not GPIO_AVAILABLE
        self.current_pwm = 50.0  # Neutral position

        if not self.mock_mode:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(pin, GPIO.OUT)
            self.pwm = GPIO.PWM(pin, 50)  # 50Hz for servos
            self.pwm.start(0)

    def set_pwm(self, duty_cycle: float):
        """
        Set motor PWM (0-100%)

        Args:
            duty_cycle: PWM duty cycle percentage (0-100)
        """
        duty_cycle = max(0, min(100, duty_cycle))  # Clamp to 0-100
        self.current_pwm = duty_cycle

        if not self.mock_mode:
            # For servos: typically 2.5-12.5% duty cycle maps to 0-180°
            # Adjust this formula based on your motors
            servo_duty = 2.5 + (duty_cycle / 100.0) * 10.0
            self.pwm.ChangeDutyCycle(servo_duty)

        logger.debug(f"Motor pin {self.pin}: {duty_cycle:.1f}%")

    def get_pwm(self) -> float:
        """Get current PWM value"""
        return self.current_pwm

    def stop(self):
        """Stop motor"""
        if not self.mock_mode:
            self.pwm.ChangeDutyCycle(0)

    def cleanup(self):
        """Cleanup GPIO"""
        if not self.mock_mode:
            self.stop()
            self.pwm.stop()


class LookupTableController:
    """
    Lookup table-based arm controller
    Learns motor PWM values for grid positions through manual teaching
    """

    # GPIO pins for motors
    MOTOR_PINS = {
        'base': 11,
        'shoulder': 13,
        'elbow': 15,
        'wrist': 29,
        'gripper': 18
    }

    def __init__(self, mock_mode: bool = False):
        """
        Initialize lookup table controller

        Args:
            mock_mode: If True, simulate without hardware
        """
        self.mock_mode = mock_mode

        # Initialize motors
        self.motors = {
            name: SimpleMotorController(pin, mock_mode)
            for name, pin in self.MOTOR_PINS.items()
        }

        # Lookup table: (x, y) position -> MotorPWM values
        # We'll store multiple poses: approach, grasp, pull, etc.
        self.lookup_table = {
            'approach': {},  # (x, y) -> MotorPWM for approach pose
            'grasp': {},     # (x, y) -> MotorPWM for grasp pose
            'pull': {},      # (x, y) -> MotorPWM for pull pose
        }

        # Fixed positions (not position-dependent)
        self.fixed_positions = {
            'home': MotorPWM(50, 50, 50, 50, 30),      # Home position
            'dispose': MotorPWM(75, 45, 50, 50, 30),   # Disposal area
        }

        logger.info("Lookup table controller initialized")

    def teach_position(self, pose: str, x: float, y: float):
        """
        Teach a position by manually adjusting motors

        Args:
            pose: Pose name ('approach', 'grasp', 'pull')
            x: X coordinate (cm)
            y: Y coordinate (cm)
        """
        print(f"\n{'=' * 60}")
        print(f"TEACHING: {pose} pose at position ({x}, {y})")
        print(f"{'=' * 60}")
        print("\nManually adjust motors to desired position")
        print("Controls:")
        print("  1-5: Select motor (Base/Shoulder/Elbow/Wrist/Gripper)")
        print("  +/-: Increase/decrease PWM by 5%")
        print("  [/]: Increase/decrease PWM by 1%")
        print("  d: Display current PWM values")
        print("  t: Test current position")
        print("  s: Save position")
        print("  q: Quit without saving")

        selected_motor = 0
        motor_names = list(self.motors.keys())

        while True:
            print(f"\nSelected: {motor_names[selected_motor]}")
            print("Command: ", end='', flush=True)

            cmd = input().strip().lower()

            if cmd == 'q':
                print("→ Cancelled")
                return False

            elif cmd in ['1', '2', '3', '4', '5']:
                selected_motor = int(cmd) - 1
                print(f"→ Selected {motor_names[selected_motor]}")

            elif cmd in ['+', '-', '[', ']']:
                motor_name = motor_names[selected_motor]
                motor = self.motors[motor_name]
                current = motor.get_pwm()

                if cmd == '+':
                    change = 5.0
                elif cmd == '-':
                    change = -5.0
                elif cmd == '[':
                    change = 1.0
                else:
                    change = -1.0

                new_pwm = max(0, min(100, current + change))
                motor.set_pwm(new_pwm)
                print(f"→ {motor_name}: {current:.1f}% → {new_pwm:.1f}%")

            elif cmd == 'd':
                print("\nCurrent PWM values:")
                for name, motor in self.motors.items():
                    print(f"  {name:10s}: {motor.get_pwm():5.1f}%")

            elif cmd == 't':
                print("→ Testing position (moving to current values)...")
                self._execute_current_position()
                print("✓ Position reached")

            elif cmd == 's':
                # Save current position
                pwm = MotorPWM(
                    base=self.motors['base'].get_pwm(),
                    shoulder=self.motors['shoulder'].get_pwm(),
                    elbow=self.motors['elbow'].get_pwm(),
                    wrist=self.motors['wrist'].get_pwm(),
                    gripper=self.motors['gripper'].get_pwm()
                )

                if pose in self.lookup_table:
                    self.lookup_table[pose][(x, y)] = pwm
                    print(f"\n✓ Position saved to lookup table!")
                    print(f"  Pose: {pose}")
                    print(f"  Position: ({x}, {y})")
                    print(f"  PWM: {pwm.to_dict()}")
                    return True
                else:
                    print(f"✗ Invalid pose: {pose}")
                    return False

            else:
                print("✗ Invalid command")

    def _execute_current_position(self):
        """Execute move to current motor PWM values"""
        import time
        for motor in self.motors.values():
            motor.set_pwm(motor.get_pwm())
        time.sleep(1.0)  # Wait for motors to reach position

    def move_to_position(self, pose: str, x: float, y: float):
        """
        Move to a position using lookup table and interpolation

        Args:
            pose: Pose name ('approach', 'grasp', 'pull', 'home', 'dispose')
            x: X coordinate (cm) - ignored for fixed poses
            y: Y coordinate (cm) - ignored for fixed poses
        """
        import time

        # Check if it's a fixed position
        if pose in self.fixed_positions:
            pwm = self.fixed_positions[pose]
            logger.info(f"Moving to fixed position: {pose}")
        else:
            # Interpolate from lookup table
            pwm = self._interpolate_pwm(pose, x, y)
            if pwm is None:
                logger.error(f"Cannot interpolate for {pose} at ({x}, {y})")
                return False
            logger.info(f"Moving to {pose} at ({x:.1f}, {y:.1f})")

        # Set all motor PWMs
        for i, (name, motor) in enumerate(self.motors.items()):
            pwm_value = pwm.to_list()[i]
            motor.set_pwm(pwm_value)

        # Wait for motors to reach position (no feedback, just wait)
        time.sleep(1.5)
        return True

    def _interpolate_pwm(self, pose: str, x: float, y: float) -> Optional[MotorPWM]:
        """
        Interpolate PWM values for a position from lookup table

        Args:
            pose: Pose name
            x: X coordinate
            y: Y coordinate

        Returns:
            Interpolated MotorPWM or None if not possible
        """
        if pose not in self.lookup_table:
            return None

        table = self.lookup_table[pose]
        if len(table) < 3:
            logger.warning(f"Need at least 3 points in lookup table for {pose}, "
                         f"only have {len(table)}")
            return None

        # Extract known points
        positions = np.array([list(pos) for pos in table.keys()])
        pwm_values = np.array([pwm.to_list() for pwm in table.values()])

        # Interpolate each motor PWM separately
        target = np.array([x, y])

        interpolated = []
        for motor_idx in range(5):
            values = pwm_values[:, motor_idx]

            # Use nearest neighbor for points outside convex hull
            interp_value = griddata(
                positions, values, target,
                method='linear', fill_value=np.nan
            )

            if np.isnan(interp_value):
                # Fall back to nearest neighbor
                interp_value = griddata(
                    positions, values, target,
                    method='nearest'
                )

            interpolated.append(float(interp_value))

        return MotorPWM(*interpolated)

    def execute_weed_pluck(self, x: float, y: float) -> bool:
        """
        Execute complete weed plucking sequence

        Args:
            x: Weed X position (cm)
            y: Weed Y position (cm)

        Returns:
            True if successful
        """
        logger.info(f"Executing weed pluck at ({x:.1f}, {y:.1f})")

        try:
            # 1. Home
            print("→ Moving to home")
            self.move_to_position('home', 0, 0)

            # 2. Approach
            print("→ Approaching weed")
            self.move_to_position('approach', x, y)

            # 3. Grasp
            print("→ Grasping weed")
            self.move_to_position('grasp', x, y)

            # Close gripper
            self.motors['gripper'].set_pwm(70)  # Closed
            import time
            time.sleep(1.0)

            # 4. Pull
            print("→ Pulling weed")
            self.move_to_position('pull', x, y)

            # 5. Dispose
            print("→ Moving to disposal")
            self.move_to_position('dispose', 0, 0)

            # Open gripper
            self.motors['gripper'].set_pwm(30)  # Open
            time.sleep(1.0)

            # 6. Home
            print("→ Returning home")
            self.move_to_position('home', 0, 0)

            logger.info("✓ Weed pluck complete!")
            return True

        except Exception as e:
            logger.error(f"Weed pluck failed: {e}")
            return False

    def save_lookup_table(self, filename: str = "lookup_table.json"):
        """Save lookup table to file"""
        data = {
            'fixed_positions': {
                name: pwm.to_dict()
                for name, pwm in self.fixed_positions.items()
            },
            'lookup_table': {}
        }

        for pose, table in self.lookup_table.items():
            data['lookup_table'][pose] = {
                f"{x},{y}": pwm.to_dict()
                for (x, y), pwm in table.items()
            }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Lookup table saved to {filename}")

    def load_lookup_table(self, filename: str = "lookup_table.json"):
        """Load lookup table from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            # Load fixed positions
            for name, pwm_dict in data['fixed_positions'].items():
                self.fixed_positions[name] = MotorPWM(**pwm_dict)

            # Load lookup table
            for pose, table_data in data['lookup_table'].items():
                self.lookup_table[pose] = {}
                for pos_str, pwm_dict in table_data.items():
                    x, y = map(float, pos_str.split(','))
                    self.lookup_table[pose][(x, y)] = MotorPWM(**pwm_dict)

            logger.info(f"Lookup table loaded from {filename}")
            logger.info(f"  Fixed positions: {len(self.fixed_positions)}")
            for pose, table in self.lookup_table.items():
                logger.info(f"  {pose}: {len(table)} points")

        except FileNotFoundError:
            logger.warning(f"File {filename} not found")

    def build_grid_lookup_table(self, grid_size: int = 5):
        """
        Interactive session to build lookup table on a grid

        Args:
            grid_size: Size of grid (5 = 5x5 = 25 positions)
        """
        print("\n" + "=" * 70)
        print("BUILD LOOKUP TABLE - GRID MODE")
        print("=" * 70)
        print(f"\nYou'll teach the arm at {grid_size}×{grid_size} = "
              f"{grid_size * grid_size} grid positions")
        print("\nWorkspace: X: -20 to +20 cm, Y: 10 to 40 cm")
        print("\nFor each position, you'll teach 3 poses:")
        print("  1. Approach (above weed)")
        print("  2. Grasp (at weed)")
        print("  3. Pull (lifted)")
        print(f"\nTotal positions to teach: {grid_size * grid_size * 3}")
        print("\nReady? (y/n): ", end='', flush=True)

        if input().strip().lower() != 'y':
            return

        # Generate grid
        x_values = np.linspace(-20, 20, grid_size)
        y_values = np.linspace(10, 40, grid_size)

        total_positions = len(x_values) * len(y_values) * 3
        current = 0

        for y in y_values:
            for x in x_values:
                print(f"\n{'=' * 70}")
                print(f"Grid Position ({x:.1f}, {y:.1f})")
                print(f"Progress: {current}/{total_positions} positions taught")
                print(f"{'=' * 70}")

                # Teach each pose
                for pose in ['approach', 'grasp', 'pull']:
                    current += 1
                    print(f"\n[{current}/{total_positions}] Teaching '{pose}' "
                          f"at ({x:.1f}, {y:.1f})")

                    success = self.teach_position(pose, x, y)
                    if not success:
                        print("\n⚠ Position not saved. Continue? (y/n): ",
                              end='', flush=True)
                        if input().strip().lower() != 'y':
                            return

        print("\n" + "=" * 70)
        print("✓ LOOKUP TABLE COMPLETE!")
        print(f"  Total positions taught: {current}")
        print("=" * 70)

        # Save
        print("\nSave lookup table? (y/n): ", end='', flush=True)
        if input().strip().lower() == 'y':
            self.save_lookup_table()

    def cleanup(self):
        """Cleanup resources"""
        for motor in self.motors.values():
            motor.cleanup()

        if not self.mock_mode and GPIO_AVAILABLE:
            GPIO.cleanup()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Lookup Table Controller for 5-DOF Arm'
    )
    parser.add_argument('--mock', action='store_true',
                       help='Run in mock mode')
    parser.add_argument('--build', type=int, metavar='GRID_SIZE',
                       help='Build lookup table with NxN grid')
    parser.add_argument('--test', type=float, nargs=2, metavar=('X', 'Y'),
                       help='Test weed pluck at position (X, Y)')

    args = parser.parse_args()

    controller = LookupTableController(mock_mode=args.mock)

    try:
        # Load existing table if available
        controller.load_lookup_table()

        if args.build:
            # Build lookup table
            controller.build_grid_lookup_table(grid_size=args.build)

        elif args.test:
            # Test weed pluck
            x, y = args.test
            print(f"\nTesting weed pluck at ({x}, {y})")
            controller.execute_weed_pluck(x, y)

        else:
            # Interactive menu
            while True:
                print("\n" + "=" * 60)
                print("LOOKUP TABLE CONTROLLER - MENU")
                print("=" * 60)
                print("\n1. Build Lookup Table (Grid)")
                print("2. Teach Single Position")
                print("3. Test Weed Pluck")
                print("4. Save Lookup Table")
                print("5. Load Lookup Table")
                print("0. Exit")
                print("\nChoice: ", end='', flush=True)

                choice = input().strip()

                if choice == '1':
                    grid_size = int(input("Grid size (3-7): ").strip())
                    controller.build_grid_lookup_table(grid_size)

                elif choice == '2':
                    pose = input("Pose (approach/grasp/pull): ").strip()
                    x = float(input("X (cm): ").strip())
                    y = float(input("Y (cm): ").strip())
                    controller.teach_position(pose, x, y)

                elif choice == '3':
                    x = float(input("Weed X (cm): ").strip())
                    y = float(input("Weed Y (cm): ").strip())
                    controller.execute_weed_pluck(x, y)

                elif choice == '4':
                    controller.save_lookup_table()

                elif choice == '5':
                    controller.load_lookup_table()

                elif choice == '0':
                    break

    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()
