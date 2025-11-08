#!/usr/bin/env python3
"""
5-DOF Robotic Arm Controller for Weed Plucking
================================================

This module provides a complete 5-DOF robotic arm controller with:
- DC motor control with encoder feedback
- Inverse kinematics for positioning
- Manual teaching mode for waypoint recording
- Smooth trajectory execution
- Integration with weed detection system

DOF Configuration:
1. Base (Motor 1): Rotation around vertical axis (0-360°)
2. Shoulder (Motor 2): Vertical lift (0-180°)
3. Elbow (Motor 3): Arm bend (0-180°)
4. Wrist (Motor 4): Wrist rotation/tilt (0-180°)
5. Gripper (Motor 5): Open/close gripper (0-180°)

Author: Claude
Date: 2025-11-08
"""

import time
import math
import json
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import GPIO, fall back to mock if not available
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    logger.info("RPi.GPIO available - using real hardware")
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False
    logger.warning("RPi.GPIO not available - using mock mode")


@dataclass
class JointAngles:
    """Represents angles for all 5 joints"""
    base: float        # 0-360 degrees
    shoulder: float    # 0-180 degrees
    elbow: float       # 0-180 degrees
    wrist: float       # 0-180 degrees
    gripper: float     # 0-180 degrees (0=open, 180=closed)

    def to_list(self) -> List[float]:
        """Convert to list for easy iteration"""
        return [self.base, self.shoulder, self.elbow, self.wrist, self.gripper]

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'base': self.base,
            'shoulder': self.shoulder,
            'elbow': self.elbow,
            'wrist': self.wrist,
            'gripper': self.gripper
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'JointAngles':
        """Create from dictionary"""
        return cls(
            base=data['base'],
            shoulder=data['shoulder'],
            elbow=data['elbow'],
            wrist=data['wrist'],
            gripper=data['gripper']
        )


@dataclass
class ArmDimensions:
    """Physical dimensions of the arm in centimeters"""
    base_height: float = 10.0      # Height from ground to shoulder joint
    upper_arm: float = 15.0        # Shoulder to elbow length
    forearm: float = 12.0          # Elbow to wrist length
    gripper: float = 8.0           # Wrist to gripper tip length


class DCMotor:
    """
    DC Motor controller with encoder feedback
    Simulates or controls a real DC motor for robotic arm joints
    """

    def __init__(self, motor_id: int, pwm_pin: int, dir_pin: int,
                 encoder_pin_a: int, encoder_pin_b: int,
                 gear_ratio: float = 1.0, mock_mode: bool = False):
        """
        Initialize DC motor controller

        Args:
            motor_id: Unique motor identifier (1-5)
            pwm_pin: GPIO pin for PWM speed control
            dir_pin: GPIO pin for direction control
            encoder_pin_a: GPIO pin for encoder channel A
            encoder_pin_b: GPIO pin for encoder channel B
            gear_ratio: Gear reduction ratio (motor revs per output rev)
            mock_mode: If True, simulate motor without hardware
        """
        self.motor_id = motor_id
        self.pwm_pin = pwm_pin
        self.dir_pin = dir_pin
        self.encoder_pin_a = encoder_pin_a
        self.encoder_pin_b = encoder_pin_b
        self.gear_ratio = gear_ratio
        self.mock_mode = mock_mode or not GPIO_AVAILABLE

        # Current state
        self.current_angle = 0.0  # Current angle in degrees
        self.target_angle = 0.0   # Target angle in degrees
        self.encoder_count = 0    # Raw encoder count

        # Motor parameters
        self.max_speed = 255      # Max PWM value (0-255)
        self.min_speed = 50       # Minimum speed to overcome friction

        # Initialize hardware
        if not self.mock_mode:
            self._setup_gpio()

        logger.info(f"Motor {motor_id} initialized ({'MOCK' if self.mock_mode else 'REAL'})")

    def _setup_gpio(self):
        """Setup GPIO pins for motor control"""
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pwm_pin, GPIO.OUT)
        GPIO.setup(self.dir_pin, GPIO.OUT)
        GPIO.setup(self.encoder_pin_a, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.encoder_pin_b, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # Setup PWM at 1kHz
        self.pwm = GPIO.PWM(self.pwm_pin, 1000)
        self.pwm.start(0)

        # Setup encoder interrupt
        GPIO.add_event_detect(self.encoder_pin_a, GPIO.BOTH,
                            callback=self._encoder_callback)

    def _encoder_callback(self, channel):
        """Handle encoder interrupts to track position"""
        a_state = GPIO.input(self.encoder_pin_a)
        b_state = GPIO.input(self.encoder_pin_b)

        # Quadrature decoding
        if a_state == b_state:
            self.encoder_count += 1
        else:
            self.encoder_count -= 1

        # Convert encoder count to angle (assuming 1024 PPR encoder)
        self.current_angle = (self.encoder_count / 1024.0) * 360.0 / self.gear_ratio

    def move_to_angle(self, target_angle: float, max_speed: int = 200):
        """
        Move motor to target angle with speed control

        Args:
            target_angle: Desired angle in degrees
            max_speed: Maximum PWM speed (0-255)
        """
        self.target_angle = target_angle

        if self.mock_mode:
            # Simulate smooth movement
            self.current_angle = target_angle
            return

        # Calculate error
        error = target_angle - self.current_angle

        # Determine direction
        direction = 1 if error > 0 else 0
        GPIO.output(self.dir_pin, direction)

        # Calculate speed with simple proportional control
        speed = min(abs(error) * 10, max_speed)
        speed = max(speed, self.min_speed) if abs(error) > 1 else 0

        # Set motor speed
        self.pwm.ChangeDutyCycle((speed / 255.0) * 100)

    def get_angle(self) -> float:
        """Get current motor angle"""
        return self.current_angle

    def set_angle_mock(self, angle: float):
        """Manually set angle (for mock mode)"""
        self.current_angle = angle

    def stop(self):
        """Stop the motor"""
        if not self.mock_mode:
            self.pwm.ChangeDutyCycle(0)

    def cleanup(self):
        """Cleanup GPIO resources"""
        if not self.mock_mode:
            self.stop()
            self.pwm.stop()


class RoboticArm5DOF:
    """
    5-DOF Robotic Arm Controller with Inverse Kinematics
    """

    # GPIO pin assignments for each motor
    # Format: (PWM_PIN, DIR_PIN, ENC_A, ENC_B)
    MOTOR_PINS = {
        1: (11, 12, 7, 8),    # Base
        2: (13, 19, 21, 22),  # Shoulder
        3: (15, 16, 23, 24),  # Elbow
        4: (29, 31, 32, 33),  # Wrist
        5: (18, 22, 35, 36)   # Gripper
    }

    def __init__(self, mock_mode: bool = False):
        """
        Initialize 5-DOF robotic arm

        Args:
            mock_mode: If True, simulate arm without hardware
        """
        self.mock_mode = mock_mode
        self.dimensions = ArmDimensions()

        # Initialize motors
        self.motors = {}
        for motor_id, pins in self.MOTOR_PINS.items():
            pwm_pin, dir_pin, enc_a, enc_b = pins
            self.motors[motor_id] = DCMotor(
                motor_id=motor_id,
                pwm_pin=pwm_pin,
                dir_pin=dir_pin,
                encoder_pin_a=enc_a,
                encoder_pin_b=enc_b,
                gear_ratio=50.0,  # Typical gear ratio for robotic arm
                mock_mode=mock_mode
            )

        # Predefined positions
        self.waypoints = self._initialize_waypoints()

        # Current state
        self.current_position = self.waypoints['home']

        logger.info(f"5-DOF Robotic Arm initialized ({'MOCK' if mock_mode else 'REAL'})")

    def _initialize_waypoints(self) -> Dict[str, JointAngles]:
        """Initialize predefined waypoints for common positions"""
        return {
            'home': JointAngles(
                base=180.0,      # Centered
                shoulder=90.0,   # Horizontal
                elbow=90.0,      # Straight
                wrist=90.0,      # Neutral
                gripper=90.0     # Half open
            ),
            'approach': JointAngles(
                base=180.0,
                shoulder=120.0,  # Lowered
                elbow=110.0,
                wrist=80.0,
                gripper=45.0     # Open
            ),
            'grasp': JointAngles(
                base=180.0,
                shoulder=135.0,  # Lower to ground
                elbow=120.0,
                wrist=70.0,
                gripper=45.0     # Open, ready to grasp
            ),
            'pull': JointAngles(
                base=180.0,
                shoulder=100.0,  # Lift up
                elbow=100.0,
                wrist=80.0,
                gripper=150.0    # Closed
            ),
            'dispose': JointAngles(
                base=270.0,      # Rotate to disposal area
                shoulder=110.0,
                elbow=100.0,
                wrist=90.0,
                gripper=150.0    # Still holding
            )
        }

    def inverse_kinematics(self, x: float, y: float, z: float,
                          wrist_angle: float = 0.0) -> Optional[JointAngles]:
        """
        Calculate joint angles to reach target position

        Uses geometric inverse kinematics for a 5-DOF arm

        Args:
            x: Target X position (cm) - lateral distance
            y: Target Y position (cm) - forward distance
            z: Target Z position (cm) - height from ground
            wrist_angle: Desired wrist orientation (degrees)

        Returns:
            JointAngles if solution exists, None otherwise
        """
        try:
            # 1. Calculate base rotation (simple rotation in XY plane)
            base_angle = math.degrees(math.atan2(x, y))
            base_angle = (base_angle + 360) % 360  # Normalize to 0-360

            # 2. Calculate horizontal reach
            horizontal_reach = math.sqrt(x**2 + y**2)

            # 3. Calculate vertical reach (accounting for base height)
            vertical_reach = z - self.dimensions.base_height

            # 4. Calculate distance from shoulder to target (in 2D plane)
            distance_to_target = math.sqrt(horizontal_reach**2 + vertical_reach**2)

            # Account for gripper length
            effective_target = distance_to_target - self.dimensions.gripper

            # 5. Check if target is reachable
            max_reach = self.dimensions.upper_arm + self.dimensions.forearm
            min_reach = abs(self.dimensions.upper_arm - self.dimensions.forearm)

            if effective_target > max_reach or effective_target < min_reach:
                logger.warning(f"Target unreachable: distance={effective_target:.2f}, "
                             f"min={min_reach:.2f}, max={max_reach:.2f}")
                return None

            # 6. Calculate elbow angle using law of cosines
            # cos(elbow) = (a² + b² - c²) / (2ab)
            a = self.dimensions.upper_arm
            b = self.dimensions.forearm
            c = effective_target

            cos_elbow = (a**2 + b**2 - c**2) / (2 * a * b)
            cos_elbow = max(-1.0, min(1.0, cos_elbow))  # Clamp to valid range
            elbow_angle = math.degrees(math.acos(cos_elbow))

            # 7. Calculate shoulder angle
            # First, find angle to target from horizontal
            angle_to_target = math.degrees(math.atan2(vertical_reach, horizontal_reach))

            # Then, find angle contribution from upper arm
            cos_shoulder_offset = (a**2 + c**2 - b**2) / (2 * a * c)
            cos_shoulder_offset = max(-1.0, min(1.0, cos_shoulder_offset))
            shoulder_offset = math.degrees(math.acos(cos_shoulder_offset))

            shoulder_angle = angle_to_target + shoulder_offset

            # 8. Calculate wrist angle to maintain desired orientation
            # Wrist angle compensates for shoulder and elbow to keep gripper level
            wrist_compensate = 90 - (shoulder_angle + elbow_angle - 90)
            wrist_final = wrist_compensate + wrist_angle

            # 9. Normalize all angles to valid ranges
            shoulder_angle = max(0, min(180, shoulder_angle))
            elbow_angle = max(0, min(180, elbow_angle))
            wrist_final = max(0, min(180, wrist_final))

            result = JointAngles(
                base=base_angle,
                shoulder=shoulder_angle,
                elbow=elbow_angle,
                wrist=wrist_final,
                gripper=self.current_position.gripper  # Keep current gripper state
            )

            logger.debug(f"IK solution: {result.to_dict()}")
            return result

        except Exception as e:
            logger.error(f"IK calculation failed: {e}")
            return None

    def forward_kinematics(self, angles: JointAngles) -> Tuple[float, float, float]:
        """
        Calculate end-effector position from joint angles

        Args:
            angles: Joint angles

        Returns:
            (x, y, z) position in cm
        """
        # Convert to radians
        base_rad = math.radians(angles.base)
        shoulder_rad = math.radians(angles.shoulder)
        elbow_rad = math.radians(angles.elbow)
        wrist_rad = math.radians(angles.wrist)

        # Calculate in 2D plane (shoulder to gripper)
        # Shoulder contribution
        sx = self.dimensions.upper_arm * math.cos(shoulder_rad)
        sz = self.dimensions.upper_arm * math.sin(shoulder_rad)

        # Elbow contribution (relative to shoulder angle)
        ex = self.dimensions.forearm * math.cos(shoulder_rad + elbow_rad - math.pi/2)
        ez = self.dimensions.forearm * math.sin(shoulder_rad + elbow_rad - math.pi/2)

        # Gripper contribution
        total_arm_angle = shoulder_rad + elbow_rad + wrist_rad - math.pi
        gx = self.dimensions.gripper * math.cos(total_arm_angle)
        gz = self.dimensions.gripper * math.sin(total_arm_angle)

        # Total horizontal and vertical reach
        horizontal = sx + ex + gx
        vertical = sz + ez + gz + self.dimensions.base_height

        # Rotate into XY plane using base angle
        x = horizontal * math.sin(base_rad)
        y = horizontal * math.cos(base_rad)
        z = vertical

        return (x, y, z)

    def move_to_position(self, target: JointAngles, speed: float = 1.0,
                        interpolation_steps: int = 50):
        """
        Move arm to target position with smooth interpolation

        Args:
            target: Target joint angles
            speed: Movement speed multiplier (0.1 to 2.0)
            interpolation_steps: Number of intermediate steps
        """
        logger.info(f"Moving to position: {target.to_dict()}")

        # Generate interpolated trajectory
        trajectory = self._interpolate_trajectory(
            self.current_position, target, interpolation_steps
        )

        # Execute trajectory
        for step, angles in enumerate(trajectory):
            # Move each motor
            for i, (motor_id, motor) in enumerate(self.motors.items()):
                angle = angles.to_list()[i]
                motor.move_to_angle(angle)

            # Wait based on speed
            time.sleep(0.02 / speed)  # 20ms per step baseline

            # Update current position
            self.current_position = angles

            if step % 10 == 0:
                logger.debug(f"Step {step}/{interpolation_steps}")

        logger.info("Movement complete")

    def _interpolate_trajectory(self, start: JointAngles, end: JointAngles,
                               steps: int) -> List[JointAngles]:
        """
        Generate smooth trajectory between two positions

        Uses linear interpolation with ease-in-out for smooth motion

        Args:
            start: Starting position
            end: Ending position
            steps: Number of interpolation steps

        Returns:
            List of intermediate JointAngles
        """
        trajectory = []

        for i in range(steps + 1):
            # Ease-in-out interpolation factor
            t = i / steps
            t_smooth = self._ease_in_out(t)

            # Interpolate each joint
            base = start.base + (end.base - start.base) * t_smooth
            shoulder = start.shoulder + (end.shoulder - start.shoulder) * t_smooth
            elbow = start.elbow + (end.elbow - start.elbow) * t_smooth
            wrist = start.wrist + (end.wrist - start.wrist) * t_smooth
            gripper = start.gripper + (end.gripper - start.gripper) * t_smooth

            trajectory.append(JointAngles(base, shoulder, elbow, wrist, gripper))

        return trajectory

    @staticmethod
    def _ease_in_out(t: float) -> float:
        """Ease-in-out function for smooth acceleration/deceleration"""
        return t * t * (3.0 - 2.0 * t)

    def execute_weed_pluck_sequence(self, weed_x: float, weed_y: float,
                                    weed_z: float = 0.0):
        """
        Execute complete weed plucking sequence

        Args:
            weed_x: Weed X position (cm)
            weed_y: Weed Y position (cm)
            weed_z: Weed height (cm), default is ground level
        """
        logger.info(f"Executing weed pluck at ({weed_x}, {weed_y}, {weed_z})")

        try:
            # 1. Move to home position
            logger.info("Step 1: Moving to home position")
            self.move_to_position(self.waypoints['home'])
            time.sleep(0.5)

            # 2. Calculate approach position (10cm above weed)
            logger.info("Step 2: Calculating approach position")
            approach_angles = self.inverse_kinematics(weed_x, weed_y, weed_z + 10)
            if approach_angles is None:
                logger.error("Cannot reach approach position")
                return False

            approach_angles.gripper = 45.0  # Open gripper
            self.move_to_position(approach_angles)
            time.sleep(0.3)

            # 3. Lower to grasp position
            logger.info("Step 3: Lowering to grasp position")
            grasp_angles = self.inverse_kinematics(weed_x, weed_y, weed_z + 2)
            if grasp_angles is None:
                logger.error("Cannot reach grasp position")
                return False

            grasp_angles.gripper = 45.0  # Keep open
            self.move_to_position(grasp_angles, speed=0.5)
            time.sleep(0.2)

            # 4. Close gripper
            logger.info("Step 4: Closing gripper")
            grasp_angles.gripper = 150.0  # Close
            self.move_to_position(grasp_angles, speed=0.3, interpolation_steps=20)
            time.sleep(0.5)

            # 5. Pull weed up
            logger.info("Step 5: Pulling weed")
            pull_angles = self.inverse_kinematics(weed_x, weed_y, weed_z + 20)
            if pull_angles:
                pull_angles.gripper = 150.0  # Keep closed
                self.move_to_position(pull_angles, speed=0.7)
            time.sleep(0.3)

            # 6. Move to disposal position
            logger.info("Step 6: Moving to disposal")
            self.move_to_position(self.waypoints['dispose'])
            time.sleep(0.3)

            # 7. Release gripper
            logger.info("Step 7: Releasing weed")
            dispose_with_open = self.waypoints['dispose']
            dispose_with_open.gripper = 45.0  # Open
            self.move_to_position(dispose_with_open, speed=0.5, interpolation_steps=20)
            time.sleep(0.3)

            # 8. Return home
            logger.info("Step 8: Returning home")
            self.move_to_position(self.waypoints['home'])

            logger.info("Weed pluck sequence complete!")
            return True

        except Exception as e:
            logger.error(f"Weed pluck sequence failed: {e}")
            return False

    def save_waypoint(self, name: str, angles: Optional[JointAngles] = None):
        """
        Save current or specified position as a waypoint

        Args:
            name: Name for the waypoint
            angles: JointAngles to save, or None to use current position
        """
        if angles is None:
            angles = self.current_position

        self.waypoints[name] = angles
        logger.info(f"Waypoint '{name}' saved: {angles.to_dict()}")

    def load_waypoint(self, name: str) -> Optional[JointAngles]:
        """Load a saved waypoint"""
        return self.waypoints.get(name)

    def get_current_angles(self) -> JointAngles:
        """Get current joint angles from motors"""
        return JointAngles(
            base=self.motors[1].get_angle(),
            shoulder=self.motors[2].get_angle(),
            elbow=self.motors[3].get_angle(),
            wrist=self.motors[4].get_angle(),
            gripper=self.motors[5].get_angle()
        )

    def cleanup(self):
        """Cleanup all motors and GPIO"""
        logger.info("Cleaning up robotic arm")
        for motor in self.motors.values():
            motor.cleanup()

        if not self.mock_mode and GPIO_AVAILABLE:
            GPIO.cleanup()


# Convenience functions
def create_arm(mock_mode: bool = True) -> RoboticArm5DOF:
    """Create and return a RoboticArm5DOF instance"""
    return RoboticArm5DOF(mock_mode=mock_mode)


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("5-DOF Robotic Arm - Demo Mode")
    print("=" * 60)

    # Create arm in mock mode
    arm = create_arm(mock_mode=True)

    # Test inverse kinematics
    print("\n1. Testing Inverse Kinematics:")
    target_pos = (10, 25, 15)  # x, y, z in cm
    print(f"   Target position: {target_pos}")

    angles = arm.inverse_kinematics(*target_pos)
    if angles:
        print(f"   Solution found: {angles.to_dict()}")

        # Verify with forward kinematics
        calculated_pos = arm.forward_kinematics(angles)
        print(f"   Verification: {tuple(round(p, 2) for p in calculated_pos)}")
    else:
        print("   No solution found")

    # Test weed plucking sequence
    print("\n2. Testing Weed Plucking Sequence:")
    weed_position = (8, 20, 0)  # Weed at ground level
    print(f"   Weed position: {weed_position}")

    success = arm.execute_weed_pluck_sequence(*weed_position)
    print(f"   Result: {'SUCCESS' if success else 'FAILED'}")

    # Cleanup
    arm.cleanup()
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
