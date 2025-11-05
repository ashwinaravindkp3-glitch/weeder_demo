import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO
    HARDWARE_AVAILABLE = True
except ImportError:
    logger.warning("RPi.GPIO not found. Running in mock mode.")
    HARDWARE_AVAILABLE = False

class ConstrainedArmController:
    """Controller for 2-servo robotic arm with hardcoded movements based on testing."""
    
    def __init__(self):
        # GPIO pins for working servos (based on your testing)
        self.SERVO_ELBOW = 15     # Up/Down motion
        self.SERVO_GRIPPER = 18   # Claw control
        
        # PWM frequency
        self.PWM_FREQ = 50
        
        # Hardcoded positions from your testing
        self.ELBOW_START = 180    # Starting position (up)
        self.ELBOW_DOWN = 120     # Down position for weeding
        self.GRIPPER_OPEN = 90    # Open position
        self.GRIPPER_CLOSED = 180 # Closed position for holding
        
        self.pwm_controllers = {}
        
    def initialize_hardware(self):
        """Initialize GPIO for the two working servos."""
        if not HARDWARE_AVAILABLE:
            logger.info("Mock mode: Simulating 2-servo arm")
            return True
            
        logger.info("Initializing constrained 2-servo arm...")
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        
        # Setup only the working servo pins
        working_pins = [self.SERVO_ELBOW, self.SERVO_GRIPPER]
        
        for pin in working_pins:
            GPIO.setup(pin, GPIO.OUT)
            self.pwm_controllers[pin] = GPIO.PWM(pin, self.PWM_FREQ)
            self.pwm_controllers[pin].start(0)
            
        # Initialize to safe starting positions
        self._move_servo_raw(self.SERVO_ELBOW, self.ELBOW_START)
        self._move_servo_raw(self.SERVO_GRIPPER, self.GRIPPER_OPEN)
        
        logger.info("2-servo arm initialized successfully")
        return True
    
    def _angle_to_duty_cycle(self, angle: int) -> float:
        """Convert angle to PWM duty cycle."""
        return 2.5 + (angle / 180.0) * 10.0
    
    def _move_servo_raw(self, pin: int, angle: int, delay: float = 0.5):
        """Move servo to specified angle."""
        if HARDWARE_AVAILABLE:
            duty_cycle = self._angle_to_duty_cycle(angle)
            self.pwm_controllers[pin].ChangeDutyCycle(duty_cycle)
            time.sleep(delay)
        logger.info(f"Moved servo on pin {pin} to {angle}°")
    
    def perform_weeding_sequence(self):
        """Execute the hardcoded weeding sequence from your testing."""
        logger.info("Starting hardcoded weeding sequence...")
        
        # Step 1: Start at top position
        logger.info("Step 1: Moving to start position")
        self._move_servo_raw(self.SERVO_ELBOW, self.ELBOW_START, 1.0)
        
        # Step 2: Open gripper
        logger.info("Step 2: Opening gripper")
        self._move_servo_raw(self.SERVO_GRIPPER, self.GRIPPER_OPEN, 0.5)
        
        # Step 3: Move down slowly to weeding position
        logger.info("Step 3: Moving down to weeding position")
        # Gradual movement from 180° to 120°
        current_angle = self.ELBOW_START
        target_angle = self.ELBOW_DOWN
        step_delay = 0.1
        
        while current_angle > target_angle:
            current_angle -= 5  # Move in 5-degree increments
            if current_angle < target_angle:
                current_angle = target_angle
            self._move_servo_raw(self.SERVO_ELBOW, current_angle, step_delay)
        
        # Step 4: Close gripper to grab weed
        logger.info("Step 4: Closing gripper to grab weed")
        self._move_servo_raw(self.SERVO_GRIPPER, self.GRIPPER_CLOSED, 0.5)
        
        # Step 5: Hold gripper closed and lift up
        logger.info("Step 5: Lifting weed up")
        # Gradual movement back to start position
        current_angle = self.ELBOW_DOWN
        target_angle = self.ELBOW_START
        
        while current_angle < target_angle:
            current_angle += 5  # Move in 5-degree increments
            if current_angle > target_angle:
                current_angle = target_angle
            self._move_servo_raw(self.SERVO_ELBOW, current_angle, step_delay)
        
        # Step 6: Move to disposal position (simulate by moving to side)
        logger.info("Step 6: Moving to disposal position")
        # Brief pause to simulate movement
        time.sleep(0.5)
        
        # Step 7: Open gripper to release weed
        logger.info("Step 7: Opening gripper to release weed")
        self._move_servo_raw(self.SERVO_GRIPPER, self.GRIPPER_OPEN, 0.5)
        
        # Step 8: Return to home position
        logger.info("Step 8: Returning to home position")
        self._move_servo_raw(self.SERVO_ELBOW, self.ELBOW_START, 1.0)
        
        logger.info("Hardcoded weeding sequence completed!")
    
    def test_individual_servos(self):
        """Test individual servo movements for verification."""
        logger.info("Testing individual servo movements...")
        
        # Test elbow movement
        logger.info("Testing elbow servo (up/down)")
        self._move_servo_raw(self.SERVO_ELBOW, self.ELBOW_START, 1.0)
        self._move_servo_raw(self.SERVO_ELBOW, self.ELBOW_DOWN, 1.0)
        self._move_servo_raw(self.SERVO_ELBOW, self.ELBOW_START, 1.0)
        
        # Test gripper movement
        logger.info("Testing gripper servo (open/close)")
        self._move_servo_raw(self.SERVO_GRIPPER, self.GRIPPER_OPEN, 0.5)
        self._move_servo_raw(self.SERVO_GRIPPER, self.GRIPPER_CLOSED, 0.5)
        self._move_servo_raw(self.SERVO_GRIPPER, self.GRIPPER_OPEN, 0.5)
        
        logger.info("Individual servo testing completed!")
    
    def cleanup(self):
        """Cleanup GPIO resources."""
        if HARDWARE_AVAILABLE:
            logger.info("Cleaning up 2-servo arm...")
            for pwm in self.pwm_controllers.values():
                pwm.stop()
            GPIO.cleanup()
        logger.info("Cleanup completed")

# Mock controller for testing without hardware
class MockConstrainedArm:
    """Mock version of the constrained arm for testing."""
    
    def __init__(self):
        logger.info("Using mock constrained 2-servo arm")
        self.elbow_position = 180
        self.gripper_position = 90
    
    def initialize_hardware(self):
        logger.info("Mock: Initialized 2-servo arm")
        return True
    
    def _move_servo_raw(self, pin: int, angle: int, delay: float = 0.5):
        if pin == 15:  # Elbow
            self.elbow_position = angle
            logger.info(f"Mock: Elbow moved to {angle}°")
        elif pin == 18:  # Gripper
            self.gripper_position = angle
            logger.info(f"Mock: Gripper moved to {angle}°")
        time.sleep(delay)
    
    def perform_weeding_sequence(self):
        logger.info("Mock: Executing hardcoded weeding sequence...")
        time.sleep(8)  # Simulate sequence time
        logger.info("Mock: Weeding sequence completed!")
    
    def test_individual_servos(self):
        logger.info("Mock: Testing individual servos...")
        time.sleep(4)  # Simulate test time
        logger.info("Mock: Individual servo testing completed!")
    
    def cleanup(self):
        logger.info("Mock: Cleanup completed")

def main():
    """Main function to test the constrained 2-servo arm."""
    logger.info("Starting constrained 2-servo arm test...")
    
    # Choose appropriate controller
    try:
        import RPi.GPIO as GPIO
        arm = ConstrainedArmController()
        logger.info("Using real hardware controller")
    except ImportError:
        arm = MockConstrainedArm()
        logger.info("Using mock controller")
    
    # Initialize hardware
    if not arm.initialize_hardware():
        logger.error("Failed to initialize hardware")
        return False
    
    try:
        # Test individual servos first
        logger.info("=== INDIVIDUAL SERVO TEST ===")
        arm.test_individual_servos()
        
        # Perform full weeding sequence
        logger.info("=== FULL WEEDING SEQUENCE TEST ===")
        arm.perform_weeding_sequence()
        
        logger.info("All tests completed successfully!")
        return True
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return False
        
    finally:
        arm.cleanup()

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("Constrained 2-servo arm is ready for integration!")
    else:
        logger.error("Constrained 2-servo arm test failed!")