import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ServoTester:
    """Interactive tool for testing and diagnosing individual servos."""

    def __init__(self):
        # GPIO pin assignments (adjust to your wiring)
        self.SERVO_PINS = {
            'shoulder': 13,
            'elbow': 15,
            'gripper': 18,
            'base': 11,
            'wrist': 16,
        }
        
        # PWM frequency for standard servos
        self.PWM_FREQ = 50
        
        self.pwm_controllers = {}

    def initialize_hardware(self):
        """Initialize GPIO and PWM controllers."""
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.HARDWARE_AVAILABLE = True
        except (ImportError, RuntimeError):
            logger.warning("RPi.GPIO not found or not running on a Raspberry Pi. Hardware functions disabled.")
            self.HARDWARE_AVAILABLE = False
            return

        logger.info("Initializing GPIO for servo testing...")
        self.GPIO.setmode(self.GPIO.BOARD)
        self.GPIO.setwarnings(False)

        for name, pin in self.SERVO_PINS.items():
            self.GPIO.setup(pin, self.GPIO.OUT)
            self.pwm_controllers[name] = self.GPIO.PWM(pin, self.PWM_FREQ)
            self.pwm_controllers[name].start(0)
            logger.info(f"Initialized {name} servo on pin {pin}")

    def angle_to_duty_cycle(self, angle: int) -> float:
        """Converts an angle (0-180) to a PWM duty cycle (2.5-12.5)."""
        return 2.5 + (angle / 180.0) * 10.0

    def move_servo(self, name: str, angle: int):
        """Move a servo to a specific angle."""
        if name not in self.SERVO_PINS:
            logger.error(f"Servo '{name}' not found.")
            return

        if not 0 <= angle <= 180:
            logger.warning("Angle must be between 0 and 180.")
            return

        if self.HARDWARE_AVAILABLE:
            duty_cycle = self.angle_to_duty_cycle(angle)
            self.pwm_controllers[name].ChangeDutyCycle(duty_cycle)
        
        logger.info(f"Moved {name} servo to {angle}°")

    def servo_testing_loop(self, servo_name: str):
        """Interactive loop for testing a single servo."""
        logger.info(f"--- Testing {servo_name.upper()} Servo ---")
        print("Enter an angle (0-180) or 'b' to go back.")
        
        current_angle = 90  # Start at a neutral position
        self.move_servo(servo_name, current_angle)

        while True:
            try:
                user_input = input(f"Set {servo_name} angle (current: {current_angle}°): ").strip().lower()

                if user_input == 'b':
                    # Set to a safe position before exiting
                    self.move_servo(servo_name, 90)
                    break
                
                angle = int(user_input)
                self.move_servo(servo_name, angle)
                current_angle = angle

            except ValueError:
                logger.warning("Invalid input. Please enter a number between 0 and 180.")
            except KeyboardInterrupt:
                break

    def main_menu(self):
        """Display the main menu and handle user selection."""
        while True:
            print("\n--- Servo Diagnostic Tool ---")
            print("1. Test Shoulder Servo")
            print("2. Test Elbow Servo")
            print("3. Test Gripper Servo")
            print("4. Test Base Servo")
            print("5. Test Wrist Servo")
            print("q. Quit")
            
            choice = input("Select an option: ").strip().lower()

            if choice == '1':
                self.servo_testing_loop('shoulder')
            elif choice == '2':
                self.servo_testing_loop('elbow')
            elif choice == '3':
                self.servo_testing_loop('gripper')
            elif choice == '4':
                self.servo_testing_loop('base')
            elif choice == '5':
                self.servo_testing_loop('wrist')
            elif choice == 'q':
                break
            else:
                logger.warning("Invalid choice. Please try again.")

    def cleanup(self):
        """Stop PWM and clean up GPIO resources."""
        if self.HARDWARE_AVAILABLE:
            logger.info("Cleaning up GPIO resources...")
            for pwm in self.pwm_controllers.values():
                pwm.stop()
            self.GPIO.cleanup()
        logger.info("Servo diagnostic tool finished.")

def run_diagnostics():
    """Main function to run the servo diagnostic tool."""
    tester = ServoTester()
    try:
        tester.initialize_hardware()
        tester.main_menu()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        tester.cleanup()

if __name__ == "__main__":
    run_diagnostics()