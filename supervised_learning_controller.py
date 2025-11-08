#!/usr/bin/env python3
"""
Supervised Learning Controller for 5-DOF Arm (No Encoders)
============================================================

This approach learns from human demonstrations to control the arm.

Method: Learning from Demonstrations
- Manually demonstrate weed plucking 50-200 times
- Record: weed position (from camera) -> motor PWM values
- Train neural network to predict motor commands
- Deploy trained model for autonomous operation

Time estimate for 41 hours:
- 100 demonstrations × 5 min = 8 hours
- Data processing: 1 hour
- Training: 2-3 hours
- Testing: 3 hours
- Total: ~14 hours ✓

Author: Claude
Date: 2025-11-08
"""

import numpy as np
import json
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time

# Try to import ML libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")

from lookup_table_controller import SimpleMotorController, MotorPWM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Demonstration:
    """Single demonstration of weed plucking"""
    weed_x: float           # Weed X position (cm)
    weed_y: float           # Weed Y position (cm)
    weed_size: float        # Weed size estimate
    approach_pwm: MotorPWM  # PWM for approach pose
    grasp_pwm: MotorPWM     # PWM for grasp pose
    pull_pwm: MotorPWM      # PWM for pull pose
    success: bool           # Whether demonstration was successful


class MotorControlNetwork(nn.Module):
    """
    Neural network to predict motor PWM values from weed position

    Architecture:
    - Input: [weed_x, weed_y, weed_size] (3 features)
    - Hidden layers: 64 -> 128 -> 64 neurons
    - Output: 5 motor PWM values × 3 poses = 15 outputs
    """

    def __init__(self):
        super(MotorControlNetwork, self).__init__()

        self.network = nn.Sequential(
            # Input layer
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            # Hidden layers
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),

            # Output layer: 15 outputs (5 motors × 3 poses)
            nn.Linear(64, 15),
            nn.Sigmoid()  # Output 0-1, will scale to 0-100%
        )

    def forward(self, x):
        """Forward pass"""
        return self.network(x) * 100.0  # Scale to 0-100%


class SupervisedLearningController:
    """
    Controller that learns from human demonstrations
    """

    MOTOR_PINS = {
        'base': 11,
        'shoulder': 13,
        'elbow': 15,
        'wrist': 29,
        'gripper': 18
    }

    def __init__(self, mock_mode: bool = False):
        """
        Initialize supervised learning controller

        Args:
            mock_mode: If True, simulate without hardware
        """
        self.mock_mode = mock_mode

        # Initialize motors
        self.motors = {
            name: SimpleMotorController(pin, mock_mode)
            for name, pin in self.MOTOR_PINS.items()
        }

        # Demonstrations storage
        self.demonstrations = []

        # Neural network model
        self.model = None
        if TORCH_AVAILABLE:
            self.model = MotorControlNetwork()

        # Fixed positions
        self.fixed_positions = {
            'home': MotorPWM(50, 50, 50, 50, 30),
            'dispose': MotorPWM(75, 45, 50, 50, 30),
        }

        logger.info("Supervised learning controller initialized")

    def record_demonstration(self, weed_x: float, weed_y: float,
                           weed_size: float = 10.0) -> bool:
        """
        Record a single demonstration

        User manually controls arm to demonstrate weed plucking,
        system records motor commands at each pose

        Args:
            weed_x: Weed X position (cm)
            weed_y: Weed Y position (cm)
            weed_size: Weed size estimate (cm)

        Returns:
            True if demonstration recorded successfully
        """
        print(f"\n{'=' * 70}")
        print(f"RECORDING DEMONSTRATION #{len(self.demonstrations) + 1}")
        print(f"{'=' * 70}")
        print(f"\nWeed position: ({weed_x:.1f}, {weed_y:.1f}), size: {weed_size:.1f}cm")
        print("\nYou will manually demonstrate 3 poses:")
        print("  1. APPROACH - Position arm above weed")
        print("  2. GRASP - Lower to grasp weed")
        print("  3. PULL - Pull weed upward")

        # Dictionary to store PWM for each pose
        recorded_pwms = {}

        # Record each pose
        for pose in ['approach', 'grasp', 'pull']:
            print(f"\n{'-' * 70}")
            print(f"Pose: {pose.upper()}")
            print(f"{'-' * 70}")

            pwm = self._manually_position_arm(pose)
            if pwm is None:
                print("✗ Demonstration cancelled")
                return False

            recorded_pwms[pose] = pwm

        # Ask if demonstration was successful
        print("\n" + "=" * 70)
        print("Was this demonstration successful? (y/n): ", end='', flush=True)
        success = input().strip().lower() == 'y'

        # Create and save demonstration
        demo = Demonstration(
            weed_x=weed_x,
            weed_y=weed_y,
            weed_size=weed_size,
            approach_pwm=recorded_pwms['approach'],
            grasp_pwm=recorded_pwms['grasp'],
            pull_pwm=recorded_pwms['pull'],
            success=success
        )

        self.demonstrations.append(demo)

        print(f"\n✓ Demonstration #{len(self.demonstrations)} recorded!")
        print(f"  Success: {success}")
        print(f"  Total demonstrations: {len(self.demonstrations)}")

        return True

    def _manually_position_arm(self, pose: str) -> Optional[MotorPWM]:
        """
        Manually position arm and return PWM values

        Args:
            pose: Pose name for display

        Returns:
            MotorPWM with current values, or None if cancelled
        """
        print(f"\nManually position arm for '{pose}':")
        print("Controls:")
        print("  1-5: Select motor")
        print("  +/-: Adjust by 5%")
        print("  [/]: Adjust by 1%")
        print("  d: Display current PWM")
        print("  s: Save and continue")
        print("  q: Cancel")

        selected_motor = 0
        motor_names = list(self.motors.keys())

        while True:
            print(f"\nSelected: {motor_names[selected_motor]} | Command: ",
                  end='', flush=True)

            cmd = input().strip().lower()

            if cmd == 'q':
                return None

            elif cmd in ['1', '2', '3', '4', '5']:
                selected_motor = int(cmd) - 1

            elif cmd in ['+', '-', '[', ']']:
                motor = self.motors[motor_names[selected_motor]]
                current = motor.get_pwm()

                change = {'+': 5, '-': -5, '[': 1, ']': -1}[cmd]
                new_pwm = max(0, min(100, current + change))

                motor.set_pwm(new_pwm)
                print(f"{motor_names[selected_motor]}: {current:.1f}% → {new_pwm:.1f}%")

            elif cmd == 'd':
                print("\nCurrent PWM:")
                for name, motor in self.motors.items():
                    print(f"  {name:10s}: {motor.get_pwm():5.1f}%")

            elif cmd == 's':
                # Return current motor PWMs
                return MotorPWM(
                    base=self.motors['base'].get_pwm(),
                    shoulder=self.motors['shoulder'].get_pwm(),
                    elbow=self.motors['elbow'].get_pwm(),
                    wrist=self.motors['wrist'].get_pwm(),
                    gripper=self.motors['gripper'].get_pwm()
                )

    def collect_demonstrations(self, num_demos: int = 50):
        """
        Interactive session to collect multiple demonstrations

        Args:
            num_demos: Target number of demonstrations
        """
        print("\n" + "=" * 70)
        print("COLLECT DEMONSTRATIONS")
        print("=" * 70)
        print(f"\nTarget: {num_demos} demonstrations")
        print(f"Current: {len(self.demonstrations)} demonstrations")
        print("\nFor each demonstration, you'll manually pluck a weed")
        print("and the system will record your motor commands.")

        while len(self.demonstrations) < num_demos:
            print(f"\n{'=' * 70}")
            print(f"Demonstration {len(self.demonstrations) + 1}/{num_demos}")
            print(f"{'=' * 70}")

            # Get weed position (in real system, from camera)
            print("\nEnter weed position:")
            try:
                x = float(input("  X (cm, -20 to 20): ").strip())
                y = float(input("  Y (cm, 10 to 40): ").strip())
                size = float(input("  Size (cm, optional, default 10): ").strip() or "10")
            except ValueError:
                print("✗ Invalid input")
                continue

            # Record demonstration
            success = self.record_demonstration(x, y, size)

            if not success:
                print("\nContinue collecting? (y/n): ", end='', flush=True)
                if input().strip().lower() != 'y':
                    break

        print(f"\n✓ Collected {len(self.demonstrations)} demonstrations!")

        # Save demonstrations
        print("\nSave demonstrations? (y/n): ", end='', flush=True)
        if input().strip().lower() == 'y':
            self.save_demonstrations()

    def train_model(self, epochs: int = 100, learning_rate: float = 0.001):
        """
        Train neural network from demonstrations

        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        if not TORCH_AVAILABLE:
            print("✗ PyTorch not available. Install with: pip install torch")
            return False

        if len(self.demonstrations) < 10:
            print(f"✗ Need at least 10 demonstrations, have {len(self.demonstrations)}")
            return False

        print("\n" + "=" * 70)
        print("TRAINING NEURAL NETWORK")
        print("=" * 70)

        # Filter successful demonstrations only
        successful = [d for d in self.demonstrations if d.success]
        print(f"\nUsing {len(successful)}/{len(self.demonstrations)} "
              f"successful demonstrations")

        if len(successful) < 10:
            print("✗ Need at least 10 successful demonstrations")
            return False

        # Prepare training data
        X = []  # Inputs: [weed_x, weed_y, weed_size]
        Y = []  # Outputs: 15 PWM values (5 motors × 3 poses)

        for demo in successful:
            X.append([demo.weed_x, demo.weed_y, demo.weed_size])

            # Concatenate PWMs from all 3 poses
            y = (demo.approach_pwm.to_list() +
                 demo.grasp_pwm.to_list() +
                 demo.pull_pwm.to_list())
            Y.append(y)

        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)

        # Normalize inputs
        X_mean = X.mean(dim=0)
        X_std = X.std(dim=0) + 1e-8
        X_normalized = (X - X_mean) / X_std

        # Save normalization parameters
        self.normalization = {'mean': X_mean, 'std': X_std}

        # Split into train/validation
        n_train = int(0.8 * len(X))
        indices = torch.randperm(len(X))

        train_X = X_normalized[indices[:n_train]]
        train_Y = Y[indices[:n_train]]
        val_X = X_normalized[indices[n_train:]]
        val_Y = Y[indices[n_train:]]

        print(f"\nTraining set: {len(train_X)} samples")
        print(f"Validation set: {len(val_X)} samples")

        # Setup training
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        print(f"\nTraining for {epochs} epochs...")
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()

            predictions = self.model(train_X)
            loss = criterion(predictions, train_Y)

            loss.backward()
            optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(val_X)
                val_loss = criterion(val_predictions, val_Y)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1:3d}: "
                      f"Train Loss = {loss.item():.4f}, "
                      f"Val Loss = {val_loss.item():.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        print(f"\n✓ Training complete!")
        print(f"  Best validation loss: {best_val_loss:.4f}")

        # Save model
        print("\nSave model? (y/n): ", end='', flush=True)
        if input().strip().lower() == 'y':
            self.save_model()

        return True

    def predict_motor_commands(self, weed_x: float, weed_y: float,
                              weed_size: float = 10.0) -> Optional[dict]:
        """
        Predict motor PWM values for a weed position

        Args:
            weed_x: Weed X position (cm)
            weed_y: Weed Y position (cm)
            weed_size: Weed size estimate (cm)

        Returns:
            Dictionary with 'approach', 'grasp', 'pull' MotorPWM values
        """
        if self.model is None or not TORCH_AVAILABLE:
            logger.error("Model not available")
            return None

        self.model.eval()

        # Prepare input
        x = torch.FloatTensor([[weed_x, weed_y, weed_size]])

        # Normalize
        if hasattr(self, 'normalization'):
            x = (x - self.normalization['mean']) / self.normalization['std']

        # Predict
        with torch.no_grad():
            prediction = self.model(x)[0].numpy()

        # Split into 3 poses
        approach_pwm = MotorPWM(*prediction[0:5])
        grasp_pwm = MotorPWM(*prediction[5:10])
        pull_pwm = MotorPWM(*prediction[10:15])

        return {
            'approach': approach_pwm,
            'grasp': grasp_pwm,
            'pull': pull_pwm
        }

    def execute_weed_pluck(self, weed_x: float, weed_y: float,
                          weed_size: float = 10.0) -> bool:
        """
        Execute weed plucking using learned model

        Args:
            weed_x: Weed X position (cm)
            weed_y: Weed Y position (cm)
            weed_size: Weed size estimate (cm)

        Returns:
            True if successful
        """
        logger.info(f"Executing learned weed pluck at ({weed_x:.1f}, {weed_y:.1f})")

        # Predict motor commands
        commands = self.predict_motor_commands(weed_x, weed_y, weed_size)
        if commands is None:
            logger.error("Failed to predict motor commands")
            return False

        try:
            # 1. Home
            print("→ Moving to home")
            self._move_to_pwm(self.fixed_positions['home'])

            # 2. Approach
            print("→ Approaching weed")
            self._move_to_pwm(commands['approach'])

            # 3. Grasp
            print("→ Grasping weed")
            grasp = commands['grasp']
            grasp.gripper = 30  # Open first
            self._move_to_pwm(grasp)
            time.sleep(0.5)

            # Close gripper
            grasp.gripper = 70  # Closed
            self._move_to_pwm(grasp)
            time.sleep(1.0)

            # 4. Pull
            print("→ Pulling weed")
            self._move_to_pwm(commands['pull'])

            # 5. Dispose
            print("→ Moving to disposal")
            self._move_to_pwm(self.fixed_positions['dispose'])

            # Open gripper
            dispose = self.fixed_positions['dispose']
            dispose.gripper = 30
            self._move_to_pwm(dispose)
            time.sleep(1.0)

            # 6. Home
            print("→ Returning home")
            self._move_to_pwm(self.fixed_positions['home'])

            logger.info("✓ Weed pluck complete!")
            return True

        except Exception as e:
            logger.error(f"Weed pluck failed: {e}")
            return False

    def _move_to_pwm(self, pwm: MotorPWM):
        """Move all motors to PWM values"""
        for i, (name, motor) in enumerate(self.motors.items()):
            motor.set_pwm(pwm.to_list()[i])
        time.sleep(1.5)  # Wait for movement

    def save_demonstrations(self, filename: str = "demonstrations.json"):
        """Save demonstrations to file"""
        data = []
        for demo in self.demonstrations:
            data.append({
                'weed_x': demo.weed_x,
                'weed_y': demo.weed_y,
                'weed_size': demo.weed_size,
                'approach_pwm': demo.approach_pwm.to_dict(),
                'grasp_pwm': demo.grasp_pwm.to_dict(),
                'pull_pwm': demo.pull_pwm.to_dict(),
                'success': demo.success
            })

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(data)} demonstrations to {filename}")

    def load_demonstrations(self, filename: str = "demonstrations.json"):
        """Load demonstrations from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.demonstrations = []
            for item in data:
                demo = Demonstration(
                    weed_x=item['weed_x'],
                    weed_y=item['weed_y'],
                    weed_size=item['weed_size'],
                    approach_pwm=MotorPWM(**item['approach_pwm']),
                    grasp_pwm=MotorPWM(**item['grasp_pwm']),
                    pull_pwm=MotorPWM(**item['pull_pwm']),
                    success=item['success']
                )
                self.demonstrations.append(demo)

            logger.info(f"Loaded {len(self.demonstrations)} demonstrations from {filename}")

        except FileNotFoundError:
            logger.warning(f"File {filename} not found")

    def save_model(self, filename: str = "motor_control_model.pth"):
        """Save trained model"""
        if self.model is None:
            return

        torch.save({
            'model_state': self.model.state_dict(),
            'normalization': self.normalization
        }, filename)

        logger.info(f"Model saved to {filename}")

    def load_model(self, filename: str = "motor_control_model.pth"):
        """Load trained model"""
        if not TORCH_AVAILABLE or self.model is None:
            return

        try:
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state'])
            self.normalization = checkpoint['normalization']

            logger.info(f"Model loaded from {filename}")

        except FileNotFoundError:
            logger.warning(f"File {filename} not found")

    def cleanup(self):
        """Cleanup resources"""
        for motor in self.motors.values():
            motor.cleanup()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Supervised Learning Controller for 5-DOF Arm'
    )
    parser.add_argument('--mock', action='store_true', help='Run in mock mode')
    parser.add_argument('--collect', type=int, metavar='N',
                       help='Collect N demonstrations')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--test', type=float, nargs=2, metavar=('X', 'Y'),
                       help='Test at position (X, Y)')

    args = parser.parse_args()

    controller = SupervisedLearningController(mock_mode=args.mock)

    try:
        if args.collect:
            controller.collect_demonstrations(num_demos=args.collect)

        elif args.train:
            controller.load_demonstrations()
            controller.train_model(epochs=200)

        elif args.test:
            controller.load_model()
            x, y = args.test
            controller.execute_weed_pluck(x, y)

        else:
            # Interactive menu
            controller.load_demonstrations()
            controller.load_model()

            while True:
                print("\n" + "=" * 60)
                print("SUPERVISED LEARNING CONTROLLER - MENU")
                print("=" * 60)
                print(f"\nDemonstrations: {len(controller.demonstrations)}")
                print("\n1. Collect Demonstrations")
                print("2. Train Model")
                print("3. Test Weed Pluck")
                print("4. Save/Load")
                print("0. Exit")
                print("\nChoice: ", end='', flush=True)

                choice = input().strip()

                if choice == '1':
                    n = int(input("Number of demos to collect: ").strip())
                    controller.collect_demonstrations(n)

                elif choice == '2':
                    controller.train_model()

                elif choice == '3':
                    x = float(input("Weed X (cm): ").strip())
                    y = float(input("Weed Y (cm): ").strip())
                    controller.execute_weed_pluck(x, y)

                elif choice == '4':
                    print("1. Save  2. Load: ", end='', flush=True)
                    sub = input().strip()
                    if sub == '1':
                        controller.save_demonstrations()
                        controller.save_model()
                    else:
                        controller.load_demonstrations()
                        controller.load_model()

                elif choice == '0':
                    break

    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()
