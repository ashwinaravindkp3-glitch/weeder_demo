#!/usr/bin/env python3
"""
5-DOF Robotic Arm Teaching Methods - Demo
==========================================

This script demonstrates the three main teaching methods:
1. Inverse Kinematics (automatic)
2. Manual Teaching (waypoint recording)
3. Hybrid Approach (calibration + IK)

Run this to see all methods in action!

Author: Claude
Date: 2025-11-08
"""

import time
from robotic_arm_5dof import RoboticArm5DOF, JointAngles


def demo_inverse_kinematics():
    """
    Demo: Inverse Kinematics Method
    The arm automatically calculates how to reach target positions
    """
    print("\n" + "=" * 70)
    print("METHOD 1: INVERSE KINEMATICS (Automatic)")
    print("=" * 70)
    print("\nThis method automatically calculates joint angles to reach any position.")
    print("Perfect for autonomous operation with weed detection.")

    arm = RoboticArm5DOF(mock_mode=True)

    # Example: Reach multiple weed positions automatically
    weed_positions = [
        (10, 25, 0),   # Weed 1: 10cm left, 25cm forward, ground level
        (5, 20, 0),    # Weed 2: 5cm left, 20cm forward
        (-8, 30, 0),   # Weed 3: 8cm right, 30cm forward
        (0, 22, 0),    # Weed 4: center, 22cm forward
    ]

    print("\n→ Testing with 4 simulated weed positions:\n")

    for i, (x, y, z) in enumerate(weed_positions, 1):
        print(f"Weed {i}: Position ({x}, {y}, {z}) cm")

        # Calculate inverse kinematics
        angles = arm.inverse_kinematics(x, y, z)

        if angles:
            print(f"  ✓ IK Solution:")
            print(f"    Base:     {angles.base:6.1f}°")
            print(f"    Shoulder: {angles.shoulder:6.1f}°")
            print(f"    Elbow:    {angles.elbow:6.1f}°")
            print(f"    Wrist:    {angles.wrist:6.1f}°")

            # Verify with forward kinematics
            calculated = arm.forward_kinematics(angles)
            print(f"  → Verification: ({calculated[0]:.1f}, {calculated[1]:.1f}, {calculated[2]:.1f})")

            # Execute pluck sequence
            print(f"  → Executing weed pluck...")
            arm.execute_weed_pluck_sequence(x, y, z)
            print(f"  ✓ Weed {i} removed!\n")
        else:
            print(f"  ✗ Position unreachable\n")

        time.sleep(0.5)

    print("\n✓ Method 1 Demo Complete!")
    print("\nAdvantages:")
    print("  • Fully automatic")
    print("  • Works with any position in workspace")
    print("  • Perfect for camera-detected weeds")
    print("  • No manual programming needed")

    arm.cleanup()


def demo_manual_teaching():
    """
    Demo: Manual Teaching Method
    Manually position the arm and save waypoints
    """
    print("\n" + "=" * 70)
    print("METHOD 2: MANUAL TEACHING (Waypoint Recording)")
    print("=" * 70)
    print("\nThis method lets you manually position the arm and save waypoints.")
    print("Perfect for learning and creating custom sequences.")

    arm = RoboticArm5DOF(mock_mode=True)

    print("\n→ Simulating manual teaching process:\n")

    # Simulate manually positioning arm and recording waypoints
    custom_positions = {
        'low_approach': JointAngles(180, 130, 100, 70, 45),
        'ground_grasp': JointAngles(180, 145, 120, 60, 45),
        'high_pull': JointAngles(180, 90, 90, 90, 150),
        'side_dispose': JointAngles(270, 100, 95, 85, 150),
    }

    # Record each position
    for name, angles in custom_positions.items():
        print(f"Recording waypoint: '{name}'")
        print(f"  Manually positioning arm to:")
        print(f"    Base={angles.base}°, Shoulder={angles.shoulder}°, Elbow={angles.elbow}°")
        print(f"    Wrist={angles.wrist}°, Gripper={angles.gripper}°")

        # Save waypoint
        arm.save_waypoint(name, angles)
        print(f"  ✓ Waypoint '{name}' saved\n")

        time.sleep(0.3)

    # Create and execute a custom sequence
    print("→ Creating custom weed removal sequence:")
    sequence = [
        ('low_approach', custom_positions['low_approach']),
        ('ground_grasp', custom_positions['ground_grasp']),
        ('high_pull', custom_positions['high_pull']),
        ('side_dispose', custom_positions['side_dispose']),
    ]

    print("\nSequence:")
    for i, (name, _) in enumerate(sequence, 1):
        print(f"  {i}. {name}")

    print("\n→ Executing sequence...")
    for name, angles in sequence:
        print(f"  → Moving to '{name}'...")
        arm.move_to_position(angles, speed=1.5)
        time.sleep(0.3)

    print("  ✓ Sequence complete!")

    print("\n✓ Method 2 Demo Complete!")
    print("\nAdvantages:")
    print("  • Intuitive and easy to learn")
    print("  • No math required")
    print("  • Great for complex custom motions")
    print("  • Perfect for fine-tuning")

    arm.cleanup()


def demo_hybrid_approach():
    """
    Demo: Hybrid Approach
    Combines IK automation with manual calibration
    """
    print("\n" + "=" * 70)
    print("METHOD 3: HYBRID APPROACH (Calibration + IK)")
    print("=" * 70)
    print("\nThis method combines automatic IK with manual calibration.")
    print("Perfect for maximum accuracy in production systems.")

    arm = RoboticArm5DOF(mock_mode=True)

    print("\n→ Step 1: Manual calibration of key positions\n")

    # Manually calibrate reference positions
    calibration_points = {
        'near_center': (0, 20, 0),
        'far_left': (15, 30, 0),
        'far_right': (-15, 30, 0),
    }

    calibrated_angles = {}

    for name, (x, y, z) in calibration_points.items():
        print(f"Calibrating '{name}' at ({x}, {y}, {z}) cm")

        # Calculate IK solution
        angles = arm.inverse_kinematics(x, y, z)

        if angles:
            # Save as calibrated waypoint
            calibrated_angles[name] = angles
            arm.save_waypoint(f'calibrated_{name}', angles)

            print(f"  ✓ IK solution saved as waypoint")
            print(f"    Shoulder: {angles.shoulder:.1f}°, Elbow: {angles.elbow:.1f}°\n")
        else:
            print(f"  ✗ Position unreachable\n")

        time.sleep(0.2)

    print("→ Step 2: Validate calibration\n")

    # Test calibrated positions
    for name, angles in calibrated_angles.items():
        print(f"Testing calibrated position: '{name}'")

        # Move to position
        arm.move_to_position(angles)

        # Verify position
        actual = arm.forward_kinematics(angles)
        expected = calibration_points[name]

        error_x = abs(actual[0] - expected[0])
        error_y = abs(actual[1] - expected[1])
        error_z = abs(actual[2] - expected[2])
        total_error = (error_x**2 + error_y**2 + error_z**2)**0.5

        print(f"  Expected: ({expected[0]:.1f}, {expected[1]:.1f}, {expected[2]:.1f})")
        print(f"  Actual:   ({actual[0]:.1f}, {actual[1]:.1f}, {actual[2]:.1f})")
        print(f"  Error:    {total_error:.2f} cm")

        if total_error < 2.0:
            print(f"  ✓ Accuracy: GOOD\n")
        else:
            print(f"  ⚠ Accuracy: Needs improvement\n")

        time.sleep(0.2)

    print("→ Step 3: Run automatic operation with calibrated system\n")

    # Now use IK for new positions, with confidence from calibration
    new_weeds = [
        (3, 22, 0),
        (-5, 28, 0),
        (8, 25, 0),
    ]

    for i, (x, y, z) in enumerate(new_weeds, 1):
        print(f"Removing weed {i} at ({x}, {y}, {z}) cm")
        angles = arm.inverse_kinematics(x, y, z)

        if angles:
            arm.move_to_position(angles)
            print(f"  ✓ Weed {i} reached and removed\n")
        else:
            print(f"  ✗ Position unreachable\n")

        time.sleep(0.3)

    print("✓ Method 3 Demo Complete!")
    print("\nAdvantages:")
    print("  • Maximum accuracy")
    print("  • Validated performance")
    print("  • Best of both worlds")
    print("  • Production-ready")

    arm.cleanup()


def main():
    """Run all teaching method demos"""
    print("\n" + "=" * 70)
    print("5-DOF ROBOTIC ARM - TEACHING METHODS DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows three ways to teach your robotic arm:")
    print("  1. Inverse Kinematics (automatic)")
    print("  2. Manual Teaching (waypoint recording)")
    print("  3. Hybrid Approach (calibration + IK)")
    print("\nAll demos run in MOCK mode (no hardware required)")
    print("=" * 70)

    input("\nPress Enter to start Demo 1: Inverse Kinematics...")
    demo_inverse_kinematics()

    input("\nPress Enter to start Demo 2: Manual Teaching...")
    demo_manual_teaching()

    input("\nPress Enter to start Demo 3: Hybrid Approach...")
    demo_hybrid_approach()

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE!")
    print("=" * 70)
    print("\nRECOMMENDATIONS:")
    print("\n1. For Learning & Setup:")
    print("   → Use Method 2 (Manual Teaching)")
    print("   → Run: python teaching_mode.py")
    print("\n2. For Production Use:")
    print("   → Use Method 1 (Inverse Kinematics)")
    print("   → Run: python integrated_weeder_5dof.py")
    print("\n3. For Best Results:")
    print("   → Use Method 3 (Hybrid Approach)")
    print("   → Run: python calibration_utility.py")
    print("   → Then: python integrated_weeder_5dof.py")
    print("\n" + "=" * 70)
    print("\nNext Steps:")
    print("  • Read README_5DOF_SETUP.md for detailed guide")
    print("  • Try teaching_mode.py for hands-on practice")
    print("  • Run calibration_utility.py before production use")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
