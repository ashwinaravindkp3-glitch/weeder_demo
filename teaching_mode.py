#!/usr/bin/env python3
"""
Teaching Mode Interface for 5-DOF Robotic Arm
==============================================

This module provides an interactive teaching mode interface that allows
operators to manually position the robotic arm and record waypoints.

Features:
- Interactive motor control (manual positioning)
- Waypoint recording and playback
- Sequence creation and editing
- Save/load waypoint configurations

Usage:
    python teaching_mode.py

Author: Claude
Date: 2025-11-08
"""

import sys
import json
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path

from robotic_arm_5dof import RoboticArm5DOF, JointAngles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TeachingMode:
    """
    Interactive teaching mode for recording arm positions
    """

    def __init__(self, arm: RoboticArm5DOF, config_file: str = "waypoints.json"):
        """
        Initialize teaching mode

        Args:
            arm: RoboticArm5DOF instance
            config_file: File to save/load waypoint configurations
        """
        self.arm = arm
        self.config_file = config_file
        self.recorded_waypoints = {}
        self.current_sequence = []

        # Load existing waypoints if available
        self.load_config()

    def load_config(self):
        """Load waypoints from configuration file"""
        config_path = Path(self.config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)

                # Convert dictionaries back to JointAngles
                for name, angles_dict in data.get('waypoints', {}).items():
                    self.recorded_waypoints[name] = JointAngles.from_dict(angles_dict)

                logger.info(f"Loaded {len(self.recorded_waypoints)} waypoints from {self.config_file}")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        else:
            logger.info(f"No existing config file found at {self.config_file}")

    def save_config(self):
        """Save waypoints to configuration file"""
        try:
            # Convert JointAngles to dictionaries for JSON serialization
            data = {
                'waypoints': {
                    name: angles.to_dict()
                    for name, angles in self.recorded_waypoints.items()
                }
            }

            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.recorded_waypoints)} waypoints to {self.config_file}")
            print(f"\n✓ Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            print(f"\n✗ Failed to save configuration: {e}")

    def manual_position_control(self):
        """
        Interactive mode to manually control each joint

        Allows fine-tuning of arm position before recording waypoint
        """
        print("\n" + "=" * 60)
        print("MANUAL POSITION CONTROL")
        print("=" * 60)
        print("\nControls:")
        print("  1-5: Select motor (Base/Shoulder/Elbow/Wrist/Gripper)")
        print("  +/-: Increase/decrease angle by 5°")
        print("  [/]: Increase/decrease angle by 1°")
        print("  h: Move to home position")
        print("  r: Record current position as waypoint")
        print("  d: Display current angles")
        print("  q: Quit manual control")
        print("=" * 60)

        selected_motor = 1
        motor_names = ["Base", "Shoulder", "Elbow", "Wrist", "Gripper"]

        while True:
            print(f"\nSelected: Motor {selected_motor} ({motor_names[selected_motor-1]})")
            print("Command: ", end='', flush=True)

            try:
                cmd = input().strip().lower()

                if cmd == 'q':
                    break
                elif cmd in ['1', '2', '3', '4', '5']:
                    selected_motor = int(cmd)
                    print(f"→ Selected Motor {selected_motor} ({motor_names[selected_motor-1]})")

                elif cmd in ['+', '-', '[', ']']:
                    # Get current angle
                    current_angles = self.arm.get_current_angles()
                    angles_list = current_angles.to_list()
                    current = angles_list[selected_motor - 1]

                    # Calculate change
                    if cmd == '+':
                        change = 5.0
                    elif cmd == '-':
                        change = -5.0
                    elif cmd == '[':
                        change = 1.0
                    else:  # ']'
                        change = -1.0

                    # Apply change with limits
                    new_angle = current + change

                    # Enforce limits based on motor
                    if selected_motor == 1:  # Base: 0-360
                        new_angle = new_angle % 360
                    else:  # All others: 0-180
                        new_angle = max(0, min(180, new_angle))

                    # Update angle
                    angles_list[selected_motor - 1] = new_angle
                    new_position = JointAngles(*angles_list)

                    # Move to new position
                    self.arm.move_to_position(new_position, speed=2.0, interpolation_steps=10)
                    print(f"→ {motor_names[selected_motor-1]}: {current:.1f}° → {new_angle:.1f}°")

                elif cmd == 'h':
                    print("→ Moving to home position...")
                    self.arm.move_to_position(self.arm.waypoints['home'])
                    print("✓ At home position")

                elif cmd == 'r':
                    self.record_waypoint()

                elif cmd == 'd':
                    self.display_current_position()

                else:
                    print("✗ Invalid command")

            except KeyboardInterrupt:
                print("\n\n→ Exiting manual control...")
                break
            except Exception as e:
                print(f"✗ Error: {e}")

    def record_waypoint(self):
        """Record current position as a named waypoint"""
        print("\n" + "-" * 60)
        print("RECORD WAYPOINT")
        print("-" * 60)

        # Display current position
        current = self.arm.get_current_angles()
        print(f"\nCurrent position:")
        for name, angle in current.to_dict().items():
            print(f"  {name:10s}: {angle:6.1f}°")

        # Get waypoint name
        print("\nEnter waypoint name (or 'cancel' to abort): ", end='', flush=True)
        name = input().strip()

        if name.lower() == 'cancel' or not name:
            print("→ Recording cancelled")
            return

        # Save waypoint
        self.recorded_waypoints[name] = current
        self.arm.save_waypoint(name, current)

        print(f"\n✓ Waypoint '{name}' recorded!")
        print("-" * 60)

    def display_current_position(self):
        """Display current arm position"""
        print("\n" + "-" * 60)
        current = self.arm.get_current_angles()
        position = self.arm.forward_kinematics(current)

        print("CURRENT POSITION:")
        print("\nJoint Angles:")
        for name, angle in current.to_dict().items():
            print(f"  {name:10s}: {angle:6.1f}°")

        print("\nEnd-Effector Position:")
        print(f"  X: {position[0]:6.2f} cm")
        print(f"  Y: {position[1]:6.2f} cm")
        print(f"  Z: {position[2]:6.2f} cm")
        print("-" * 60)

    def list_waypoints(self):
        """Display all recorded waypoints"""
        print("\n" + "=" * 60)
        print("RECORDED WAYPOINTS")
        print("=" * 60)

        if not self.recorded_waypoints:
            print("\nNo waypoints recorded yet.")
        else:
            for i, (name, angles) in enumerate(self.recorded_waypoints.items(), 1):
                print(f"\n{i}. {name}")
                for joint, angle in angles.to_dict().items():
                    print(f"     {joint:10s}: {angle:6.1f}°")

        print("=" * 60)

    def playback_waypoint(self):
        """Move to a recorded waypoint"""
        if not self.recorded_waypoints:
            print("\n✗ No waypoints available for playback")
            return

        print("\n" + "=" * 60)
        print("PLAYBACK WAYPOINT")
        print("=" * 60)

        # List available waypoints
        waypoint_list = list(self.recorded_waypoints.keys())
        for i, name in enumerate(waypoint_list, 1):
            print(f"{i}. {name}")

        print("\nEnter waypoint number or name: ", end='', flush=True)
        selection = input().strip()

        # Find waypoint
        waypoint = None
        if selection.isdigit():
            idx = int(selection) - 1
            if 0 <= idx < len(waypoint_list):
                waypoint = self.recorded_waypoints[waypoint_list[idx]]
                name = waypoint_list[idx]
        elif selection in self.recorded_waypoints:
            waypoint = self.recorded_waypoints[selection]
            name = selection

        if waypoint:
            print(f"\n→ Moving to waypoint '{name}'...")
            self.arm.move_to_position(waypoint)
            print("✓ Waypoint reached")
        else:
            print("✗ Waypoint not found")

    def create_sequence(self):
        """Create a sequence of waypoints"""
        print("\n" + "=" * 60)
        print("CREATE WAYPOINT SEQUENCE")
        print("=" * 60)

        if not self.recorded_waypoints:
            print("\n✗ No waypoints available. Record some waypoints first.")
            return

        sequence = []
        waypoint_list = list(self.recorded_waypoints.keys())

        print("\nAvailable waypoints:")
        for i, name in enumerate(waypoint_list, 1):
            print(f"  {i}. {name}")

        print("\nEnter waypoint numbers in sequence (comma-separated):")
        print("Example: 1,3,2,4")
        print("Input: ", end='', flush=True)

        try:
            selection = input().strip()
            indices = [int(x.strip()) - 1 for x in selection.split(',')]

            for idx in indices:
                if 0 <= idx < len(waypoint_list):
                    name = waypoint_list[idx]
                    sequence.append((name, self.recorded_waypoints[name]))
                else:
                    print(f"✗ Invalid waypoint index: {idx + 1}")
                    return

            # Preview sequence
            print("\nSequence preview:")
            for i, (name, _) in enumerate(sequence, 1):
                print(f"  {i}. {name}")

            print("\nExecute this sequence? (y/n): ", end='', flush=True)
            if input().strip().lower() == 'y':
                self.execute_sequence(sequence)
            else:
                print("→ Sequence cancelled")

        except ValueError:
            print("✗ Invalid input format")

    def execute_sequence(self, sequence: List[tuple]):
        """
        Execute a sequence of waypoints

        Args:
            sequence: List of (name, JointAngles) tuples
        """
        print("\n" + "=" * 60)
        print("EXECUTING SEQUENCE")
        print("=" * 60)

        for i, (name, angles) in enumerate(sequence, 1):
            print(f"\n→ Step {i}/{len(sequence)}: Moving to '{name}'...")
            self.arm.move_to_position(angles)
            time.sleep(0.5)  # Pause between waypoints

        print("\n✓ Sequence complete!")
        print("=" * 60)

    def main_menu(self):
        """Main interactive menu"""
        while True:
            print("\n" + "=" * 60)
            print("TEACHING MODE - MAIN MENU")
            print("=" * 60)
            print("\n1. Manual Position Control (record waypoints)")
            print("2. List Recorded Waypoints")
            print("3. Playback Waypoint")
            print("4. Create & Execute Sequence")
            print("5. Save Configuration")
            print("6. Display Current Position")
            print("7. Move to Home")
            print("0. Exit")
            print("\nChoice: ", end='', flush=True)

            try:
                choice = input().strip()

                if choice == '1':
                    self.manual_position_control()
                elif choice == '2':
                    self.list_waypoints()
                elif choice == '3':
                    self.playback_waypoint()
                elif choice == '4':
                    self.create_sequence()
                elif choice == '5':
                    self.save_config()
                elif choice == '6':
                    self.display_current_position()
                elif choice == '7':
                    print("\n→ Moving to home position...")
                    self.arm.move_to_position(self.arm.waypoints['home'])
                    print("✓ At home position")
                elif choice == '0':
                    print("\n→ Exiting teaching mode...")
                    break
                else:
                    print("\n✗ Invalid choice")

            except KeyboardInterrupt:
                print("\n\n→ Interrupted. Exiting...")
                break
            except Exception as e:
                logger.error(f"Error in main menu: {e}")
                print(f"\n✗ Error: {e}")

        # Save before exiting
        print("\nSave changes? (y/n): ", end='', flush=True)
        if input().strip().lower() == 'y':
            self.save_config()


def main():
    """Main entry point for teaching mode"""
    print("\n" + "=" * 60)
    print("5-DOF ROBOTIC ARM - TEACHING MODE")
    print("=" * 60)

    # Ask for mock or real mode
    print("\nRun in MOCK mode (simulation)? (y/n): ", end='', flush=True)
    mock_mode = input().strip().lower() == 'y'

    # Create arm
    try:
        arm = RoboticArm5DOF(mock_mode=mock_mode)
        print(f"\n✓ Arm initialized ({'MOCK' if mock_mode else 'REAL'} mode)")

        # Move to home position
        print("\n→ Moving to home position...")
        arm.move_to_position(arm.waypoints['home'])
        print("✓ Ready!")

        # Start teaching mode
        teaching = TeachingMode(arm, config_file="waypoints.json")
        teaching.main_menu()

        # Cleanup
        print("\n→ Cleaning up...")
        arm.cleanup()
        print("✓ Done!")

    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
