import time
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO
    HARDWARE_AVAILABLE = True
except ImportError:
    logger.warning("RPi.GPIO not found. Running in mock mode.")
    HARDWARE_AVAILABLE = False

class RoboticArm4DOF:
    """Controller for a 4-DOF robotic arm with base, shoulder, elbow, and gripper."""
    
    def __init__(self, arm_length_cm=30):
        # GPIO pins for all four servos
        self.SERVO_BASE = 11      # Rotational base
        self.SERVO_SHOULDER = 13  # Shoulder joint (vertical movement)
        self.SERVO_ELBOW = 15     # Elbow joint
        self.SERVO_GRIPPER = 18   # Gripper/claw
        
        # PWM frequency
        self.PWM_FREQ = 50
        
        # Arm properties
        self.ARM_LENGTH = arm_length_cm
        
        # Servo angle limits and initial positions
        self.servo_positions = {
            'base': 90,       # Center position
            'shoulder': 90,   # Horizontal position
            'elbow': 180,     # Up position
            'gripper': 90     # Open position
        }
        
        self.pwm_controllers = {}
        
    def initialize_hardware(self):
        """Initialize GPIO for all four servos."""
        if not HARDWARE_AVAILABLE:
            logger.info("Mock mode: Simulating 4-DOF arm")
            return True
            
        logger.info("Initializing 4-DOF robotic arm...")
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        
        servo_pins = [self.SERVO_BASE, self.SERVO_SHOULDER, self.SERVO_ELBOW, self.SERVO_GRIPPER]
        
        for pin in servo_pins:
            GPIO.setup(pin, GPIO.OUT)
            self.pwm_controllers[pin] = GPIO.PWM(pin, self.PWM_FREQ)
            self.pwm_controllers[pin].start(0)
            
        # Initialize to safe starting positions
        self.move_to_home()
        
        logger.info("4-DOF arm initialized successfully")
        return True
    
    def _angle_to_duty_cycle(self, angle: int) -> float:
        """Convert angle to PWM duty cycle."""
        return 2.5 + (angle / 180.0) * 10.0
    
    def _move_servo(self, servo_name: str, angle: int, delay: float = 0.5):
        """Move a specific servo to a given angle."""
        pin_map = {
            'base': self.SERVO_BASE,
            'shoulder': self.SERVO_SHOULDER,
            'elbow': self.SERVO_ELBOW,
            'gripper': self.SERVO_GRIPPER
        }
        pin = pin_map[servo_name]
        
        if HARDWARE_AVAILABLE:
            duty_cycle = self._angle_to_duty_cycle(angle)
            self.pwm_controllers[pin].ChangeDutyCycle(duty_cycle)
            time.sleep(delay)
        
        self.servo_positions[servo_name] = angle
        logger.info(f"Moved {servo_name} to {angle}°")
    
    def move_to_home(self):
        """Move the arm to a safe home position."""
        logger.info("Moving arm to home position...")
        self._move_servo('base', 90, 0.5)
        self._move_servo('shoulder', 90, 0.5)
        self._move_servo('elbow', 180, 0.5)
        self._move_servo('gripper', 90, 0.5)
        logger.info("Arm at home position")
    
    def calculate_shoulder_angle(self, distance_cm: float) -> int:
        """Calculate the required shoulder angle to reach a given distance."""
        if distance_cm > self.ARM_LENGTH:
            logger.warning(f"Distance {distance_cm}cm is out of reach. Targeting max range.")
            distance_cm = self.ARM_LENGTH
        
        # Inverse kinematics for a simple 2-link arm (shoulder-elbow)
        # Assuming elbow is fixed, we calculate shoulder angle
        # This is a simplification; a more complex model would be needed for true IK
        angle_rad = math.acos(distance_cm / self.ARM_LENGTH)
        angle_deg = math.degrees(angle_rad)
        
        # Convert to servo angle (0-180 range)
        # Assuming 90 degrees is horizontal forward
        servo_angle = 90 - angle_deg
        
        logger.info(f"Calculated shoulder angle: {servo_angle:.1f}° for distance {distance_cm}cm")
        return int(servo_angle)
    
    def perform_pick_and_place(self, target_grid_col: int, grid_cols: int, distance_cm: float):
        """Execute the full pick-and-place sequence for a given target."""
        logger.info(f"--- Starting Pick-and-Place Sequence ---")
        logger.debug(f"Target column: {target_grid_col}, Total columns: {grid_cols}, Distance: {distance_cm}cm")

        # 1. Move to home position first for a consistent starting state
        self.move_to_home()

        # 2. Calculate and move base
        base_angle_range = 120  # e.g., from 30 to 150 degrees
        min_base_angle = 30
        base_angle = min_base_angle + (target_grid_col / (grid_cols - 1)) * base_angle_range
        logger.info(f"Step 1: Rotating base to angle {int(base_angle)}°")
        self._move_servo('base', int(base_angle), 1.5)

        # 3. Calculate and move shoulder to reach the target
        shoulder_angle = self.calculate_shoulder_angle(distance_cm)
        logger.info(f"Step 2: Adjusting shoulder to angle {shoulder_angle}°")
        self._move_servo('shoulder', shoulder_angle, 1.0)

        # 4. Perform the pick action (elbow down, grip, elbow up)
        logger.info("Step 3: Executing pick action")
        self._move_servo('elbow', 120, 1.0)  # Lower elbow
        self._move_servo('gripper', 180, 0.7) # Close gripper
        self._move_servo('elbow', 180, 1.0)  # Lift elbow

        # 5. Move to the disposal tray (base at 0 degrees)
        logger.info("Step 4: Moving to disposal tray")
        self._move_servo('base', 0, 2.0) # Slower movement for stability
        self._move_servo('shoulder', 90, 1.0) # Level shoulder for drop

        # 6. Release the weed
        logger.info("Step 5: Releasing gripper")
        self._move_servo('gripper', 90, 0.7)

        # 7. Return to home position
        logger.info("Step 6: Returning to home position")
        self.move_to_home()

        logger.info("--- Pick-and-Place Sequence Completed ---")
    
    def cleanup(self):
        """Cleanup GPIO resources."""
        if HARDWARE_AVAILABLE:
            logger.info("Cleaning up 4-DOF arm...")
            for pwm in self.pwm_controllers.values():
                pwm.stop()
            GPIO.cleanup()
        logger.info("Cleanup completed")

# Mock controller for testing without hardware
class MockRoboticArm4DOF:
    """Mock version of the 4-DOF arm for testing."""
    
    def __init__(self, arm_length_cm=30):
        logger.info("Using mock 4-DOF robotic arm")
        self.servo_positions = {'base': 90, 'shoulder': 90, 'elbow': 180, 'gripper': 90}
    
    def initialize_hardware(self):
        logger.info("Mock: Initialized 4-DOF arm")
        return True
    
    def _move_servo(self, servo_name: str, angle: int, delay: float = 0.5):
        self.servo_positions[servo_name] = angle
        logger.info(f"Mock: Moved {servo_name} to {angle}°")
        time.sleep(delay)
    
    def move_to_home(self):
        logger.info("Mock: Moving to home position")
        time.sleep(1)
    
    def calculate_shoulder_angle(self, distance_cm: float) -> int:
        logger.info(f"Mock: Calculated shoulder angle for {distance_cm}cm")
        return 80  # Dummy value
    
    def perform_pick_and_place(self, target_grid_col: int, grid_cols: int, distance_cm: float):
        logger.info(f"Mock: Executing pick-and-place for target at grid {target_grid_col}")
        time.sleep(8)  # Simulate sequence time
        logger.info("Mock: Pick-and-place completed!")
    
    def cleanup(self):
        logger.info("Mock: Cleanup completed")

def main():
    """Main function to test the 4-DOF robotic arm."""
    logger.info("Starting 4-DOF robotic arm test...")
    
    try:
        arm = RoboticArm4DOF()
    except NameError:
        arm = MockRoboticArm4DOF()
    
    if not arm.initialize_hardware():
        logger.error("Failed to initialize hardware")
        return
    
    try:
        # Test a pick-and-place sequence
        logger.info("=== TESTING PICK-AND-PLACE ===")
        # Simulate a target at grid column 12 (of 16) and 20cm away
        arm.perform_pick_and_place(target_grid_col=12, grid_cols=16, distance_cm=20)
        
        logger.info("Test completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        
    finally:
        arm.cleanup()

if __name__ == "__main__":
    main()