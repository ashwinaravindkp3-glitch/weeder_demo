#!/usr/bin/env python3
"""
Quick Start Script for Autonomous Weed Detection and Removal Demo

Just run: python run_demo.py
"""

import subprocess
import sys

def main():
    print("🌱 Starting Autonomous Weed Detection & Removal Demo...")
    print("=" * 60)
    
    try:
        # Run the main demo
        result = subprocess.run([sys.executable, 'demo.py'], 
                                capture_output=False, 
                                text=True)
        
        if result.returncode == 0:
            print("\n✅ Demo completed successfully!")
        else:
            print(f"\n❌ Demo failed with return code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")

if __name__ == '__main__':
    main()