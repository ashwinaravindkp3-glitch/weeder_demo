#!/usr/bin/env python3
"""
Autonomous Weed Detection and Removal Demo

This is the main demo script that shows the robotic arm picking weeds.
Run this script to see the complete system in action!
"""

import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm_control.robotic_arm import WeedRemovalArm

def main():
    """Main demo function"""
    print("🌱 AUTONOMOUS WEED DETECTION & REMOVAL DEMO")
    print("=" * 60)
    print("🤖 This demo shows the robotic arm picking weeds!")
    print("📹 The arm will demonstrate realistic weed removal sequences")
    print("⚡ Get ready to see some robotic farming action!")
    print()
    
    # Initialize the weed removal arm
    print("🚀 Initializing robotic arm...")
    arm = WeedRemovalArm()
    
    try:
        # Move to home position first
        print("🏠 Moving to home position...")
        arm.move_to_home()
        time.sleep(2)
        
        print("\n🎯 DEMO 1: Single Weed Removal")
        print("-" * 30)
        
        # Simulate detecting a weed at specific coordinates
        weed_pixel_x, weed_pixel_y = 400, 300  # Camera coordinates
        weed_world_x, weed_world_y, weed_world_z = 80, 60, 10  # Real world coordinates in mm
        
        print(f"📸 Weed detected at camera position ({weed_pixel_x}, {weed_pixel_y})")
        print(f"🗺️  Converted to world coordinates ({weed_world_x}, {weed_world_y}, {weed_world_z}) mm")
        
        # Remove the weed!
        success = arm.remove_weed(weed_pixel_x, weed_pixel_y, 
                                 (weed_world_x, weed_world_y, weed_world_z))
        
        if success:
            print("✅ SUCCESS: Weed removed successfully!")
        else:
            print("❌ FAILED: Could not remove weed")
        
        time.sleep(3)
        
        print("\n🎯 DEMO 2: Multiple Weed Removal Sequence")
        print("-" * 40)
        
        # Run the complete demo sequence
        arm.demo_weed_removal_sequence()
        
        print("\n🎯 DEMO 3: Precision Weed Picking")
        print("-" * 35)
        
        # Test precision movements
        precision_weeds = [
            (30, 30, 5),    # Close, low weed
            (120, -80, 15), # Far, high weed
            (-90, 45, 8),   # Side weed
        ]
        
        for i, (x, y, z) in enumerate(precision_weeds, 1):
            print(f"\n🔍 Precision weed {i}: ({x}, {y}, {z}) mm")
            pixel_x, pixel_y = int(x * 3 + 320), int(y * 2 + 240)
            
            success = arm.remove_weed(pixel_x, pixel_y, (x, y, z))
            
            if success:
                print(f"✅ Precision weed {i} removed!")
            else:
                print(f"❌ Could not reach precision weed {i}")
            
            time.sleep(2)
        
        # Final statistics
        print("\n📊 FINAL DEMO RESULTS")
        print("=" * 25)
        stats = arm.get_removal_stats()
        
        print(f"🌿 Total weeds processed: {stats['total_weeds_removed']}")
        print(f"✅ Successful removals: {stats['successful_removals']}")
        print(f"❌ Failed removals: {stats['failed_removals']}")
        print(f"📈 Success rate: {stats['success_rate']:.1f}%")
        print(f"⏱️  Average removal time: {stats['average_removal_time']:.1f} seconds")
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("🤖 The robotic arm has demonstrated autonomous weed removal!")
        
        # Show final arm position
        final_pos = arm.get_current_position()
        print(f"📍 Final arm position: ({final_pos.x:.1f}, {final_pos.y:.1f}, {final_pos.z:.1f}) mm")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        
    finally:
        print("\n🧹 Cleaning up...")
        arm.move_to_home()
        time.sleep(1)
        arm.cleanup()
        print("✅ Cleanup complete - demo finished!")

if __name__ == '__main__':
    main()