# 5-DOF Robotic Arm Weed Removal System - Complete Guide

## Overview

This guide explains how to teach a 5 DOF (Degree of Freedom) robotic arm using DC motors to autonomously detect and pluck weeds.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Hardware Setup](#hardware-setup)
3. [Teaching Methods](#teaching-methods)
4. [Quick Start Guide](#quick-start-guide)
5. [Calibration](#calibration)
6. [Operation Modes](#operation-modes)
7. [Troubleshooting](#troubleshooting)

---

## System Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPLETE SYSTEM                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. PERCEPTION                                              │
│     ├─ Camera (640×480)                                     │
│     ├─ YOLOv8 Detection                                     │
│     └─ Grid Mapper (16×12 grid)                            │
│                                                              │
│  2. COORDINATION                                            │
│     ├─ Coordinate Calibration                              │
│     ├─ Weed Prioritization                                 │
│     └─ Motion Planning                                      │
│                                                              │
│  3. ACTUATION                                               │
│     ├─ 5-DOF Arm Controller                                │
│     ├─ Inverse Kinematics                                  │
│     ├─ Trajectory Generation                               │
│     └─ DC Motor Control                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5 Degrees of Freedom

1. **Base (Motor 1)**: Rotates arm left/right (0-360°)
2. **Shoulder (Motor 2)**: Lifts arm up/down (0-180°)
3. **Elbow (Motor 3)**: Bends arm (0-180°)
4. **Wrist (Motor 4)**: Tilts end-effector (0-180°)
5. **Gripper (Motor 5)**: Opens/closes to grasp (0-180°)

---

## Hardware Setup

### Required Hardware

1. **5× DC Motors with Encoders**
   - Recommended: 12V DC motors with 1024 PPR encoders
   - Gear ratio: 50:1 (or similar)

2. **Motor Drivers**
   - 5× H-bridge motor drivers (e.g., L298N)
   - Supports PWM speed control and direction control

3. **Raspberry Pi 4** (or similar)
   - GPIO for motor control
   - USB camera support

4. **USB Camera**
   - Resolution: 640×480 or higher
   - Mounted with clear view of working area

5. **Power Supply**
   - 12V for motors (5-10A depending on motors)
   - 5V for Raspberry Pi

### GPIO Pin Assignments

| Motor    | PWM Pin | DIR Pin | Encoder A | Encoder B |
|----------|---------|---------|-----------|-----------|
| Base     | 11      | 12      | 7         | 8         |
| Shoulder | 13      | 19      | 21        | 22        |
| Elbow    | 15      | 16      | 23        | 24        |
| Wrist    | 29      | 31      | 32        | 33        |
| Gripper  | 18      | 22      | 35        | 36        |

**Note**: Pin numbers use BOARD numbering scheme.

### Wiring Diagram

```
Raspberry Pi                Motor Driver              DC Motor
┌──────────┐               ┌──────────┐             ┌────────┐
│          │               │          │             │        │
│  PWM ────┼──────────────>│ PWM IN   │             │        │
│  DIR ────┼──────────────>│ DIR IN   ├────────────>│ Motor  │
│          │               │          │             │        │
│  ENC_A <─┼───────────────┼──────────┼─────────────│ Enc A  │
│  ENC_B <─┼───────────────┼──────────┼─────────────│ Enc B  │
│          │               │          │             │        │
│  GND ────┼───────────────┤ GND      │             │  GND   │
└──────────┘               └──────────┘             └────────┘
                                 │
                                 │
                           12V Power Supply
```

---

## Teaching Methods

### Method 1: Inverse Kinematics (RECOMMENDED)

**Best for**: Production use, consistent operation

The arm automatically calculates joint angles to reach weed positions detected by the camera.

**How it works**:
1. Camera detects weed at pixel coordinates
2. Coordinates converted to robot workspace (cm)
3. Inverse kinematics calculates required joint angles
4. Arm executes smooth trajectory to target
5. Predefined sequence plucks weed

**Advantages**:
- ✅ Fully automatic
- ✅ Consistent and repeatable
- ✅ Works for any position in workspace
- ✅ No manual programming needed

**Usage**:
```bash
# Run integrated system
python integrated_weeder_5dof.py

# Run in mock mode (no hardware)
python integrated_weeder_5dof.py --mock

# Run single cycle
python integrated_weeder_5dof.py --single
```

---

### Method 2: Manual Teaching (INTERACTIVE)

**Best for**: Initial setup, custom sequences, edge cases

Manually position the arm and record waypoints for later playback.

**How it works**:
1. Enter teaching mode
2. Manually control each motor to desired position
3. Record position as named waypoint
4. Create sequences of waypoints
5. Play back sequences automatically

**Advantages**:
- ✅ Intuitive and easy to learn
- ✅ Good for complex motions
- ✅ No math required
- ✅ Great for fine-tuning

**Usage**:
```bash
# Start teaching mode
python teaching_mode.py

# In mock mode
python teaching_mode.py
```

**Teaching Mode Interface**:
```
Controls:
  1-5: Select motor (Base/Shoulder/Elbow/Wrist/Gripper)
  +/-: Increase/decrease angle by 5°
  [/]: Increase/decrease angle by 1°
  h: Move to home position
  r: Record current position as waypoint
  d: Display current angles
  q: Quit manual control
```

**Example Workflow**:
1. Select motor 2 (shoulder): Press `2`
2. Lower arm: Press `-` multiple times
3. Fine-tune: Press `]` or `[`
4. Record position: Press `r`, name it "grasp_position"
5. Repeat for other positions
6. Create sequence: Link waypoints together
7. Save configuration

---

### Method 3: Calibration-Based (HYBRID)

**Best for**: Maximum accuracy

Combines automatic IK with manual calibration for optimal results.

**How it works**:
1. Perform camera-robot coordinate calibration
2. Validate workspace reachability
3. Test and tune IK solutions
4. Run automatic operation with calibrated parameters

**Usage**:
```bash
# Start calibration utility
python calibration_utility.py

# In mock mode
python calibration_utility.py --mock
```

**Calibration Steps**:

1. **Camera-Robot Coordinate Calibration**
   - Place markers at known robot positions
   - Record pixel and robot coordinates for 4+ points
   - System calculates transformation matrix
   - Validates accuracy

2. **Workspace Validation**
   - Tests 25 points across workspace
   - Identifies unreachable areas
   - Ensures full coverage

3. **IK Testing**
   - Manually test specific positions
   - Verify IK accuracy
   - Fine-tune arm parameters

4. **Gripper Calibration**
   - Test different closing angles
   - Find optimal grip force
   - Configure open/close positions

---

## Quick Start Guide

### Installation

1. **Clone repository**:
```bash
cd ~/weeder_demo
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify hardware** (real hardware only):
```bash
# Test individual motors
python servo_diagnostic.py
```

### First Run (Mock Mode)

Perfect for testing without hardware:

```bash
# 1. Test the arm controller
python robotic_arm_5dof.py

# 2. Try teaching mode
python teaching_mode.py

# 3. Run calibration
python calibration_utility.py --mock

# 4. Run full system
python integrated_weeder_5dof.py --mock --single
```

### First Run (Real Hardware)

1. **Safety First**:
   - Ensure workspace is clear
   - Have emergency stop ready
   - Start with low motor speeds

2. **Calibration**:
```bash
# Run calibration utility
python calibration_utility.py

# Follow prompts for:
# - Camera-robot coordinate calibration
# - Workspace validation
# - Gripper calibration

# Save calibration when complete
```

3. **Teaching Mode** (optional but recommended):
```bash
python teaching_mode.py

# Record key positions:
# - home
# - approach
# - grasp
# - pull
# - dispose
```

4. **Run System**:
```bash
# Single cycle test
python integrated_weeder_5dof.py --single

# Continuous operation
python integrated_weeder_5dof.py

# With cycle limit
python integrated_weeder_5dof.py --cycles 10
```

---

## Calibration

### Why Calibrate?

Calibration ensures accurate mapping between:
- Camera pixels → Robot workspace coordinates
- Robot workspace → Joint angles

### Calibration Files

After calibration, these files are created:

- `calibration.txt` - Camera-robot transformation parameters
- `calibration_points.json` - Calibration point data
- `waypoints.json` - Manually taught waypoints

### Calibration Quality

The calibration utility reports accuracy:

- **Average error < 2 cm**: ✓ GOOD - Ready for operation
- **Average error 2-5 cm**: ⚠ ACCEPTABLE - May need refinement
- **Average error > 5 cm**: ✗ POOR - Recalibrate with more points

### Tips for Good Calibration

1. **Use 6-8 calibration points** (minimum 4)
2. **Spread points across entire workspace**
3. **Include corners and center**
4. **Use precise, visible markers**
5. **Ensure good lighting**
6. **Keep camera stable** (no movement during calibration)

---

## Operation Modes

### 1. Mock Mode (Simulation)

Test without hardware:
```bash
python integrated_weeder_5dof.py --mock
```

**Features**:
- Simulates all motors
- Mock weed detection
- Safe for development
- No GPIO access needed

### 2. Single Cycle

Run one detection and removal cycle:
```bash
python integrated_weeder_5dof.py --single
```

**Process**:
1. Capture frame
2. Detect weeds
3. Prioritize by distance
4. Remove each weed
5. Print statistics
6. Exit

### 3. Continuous Operation

Autonomous weed removal:
```bash
python integrated_weeder_5dof.py --cycles 10
```

**Process**:
- Runs specified number of cycles (or infinite if not specified)
- 2-second delay between cycles
- Press Ctrl+C to stop
- Prints final statistics

### 4. Teaching Mode

Manual waypoint recording:
```bash
python teaching_mode.py
```

**Features**:
- Interactive motor control
- Waypoint recording
- Sequence creation
- Save/load configurations

---

## Weed Plucking Sequence

The arm executes this sequence for each weed:

```
1. HOME POSITION
   └─> Start at safe home position

2. APPROACH (10 cm above weed)
   └─> Move to position above weed
   └─> Open gripper

3. GRASP (at weed base)
   └─> Lower to ground level
   └─> Close gripper around weed

4. PULL (lift 20 cm)
   └─> Pull weed upward with tension
   └─> Maintain grip

5. DISPOSE (rotate to disposal area)
   └─> Move to disposal zone
   └─> Open gripper
   └─> Release weed

6. RETURN HOME
   └─> Move back to home position
   └─> Ready for next weed
```

**Typical timing**:
- Each sequence: 8-12 seconds
- Multiple weeds: Processed in priority order (closest first)

---

## Troubleshooting

### Problem: "No IK solution found"

**Cause**: Target position unreachable

**Solutions**:
1. Check arm dimensions in `robotic_arm_5dof.py`
2. Verify camera calibration
3. Run workspace validation
4. Adjust camera position for better coverage

### Problem: "Motor not moving"

**Cause**: Hardware connection or GPIO issue

**Solutions**:
1. Check motor power supply
2. Verify GPIO pin connections
3. Test with `servo_diagnostic.py`
4. Check motor driver connections
5. Verify encoder wiring

### Problem: "Poor calibration accuracy"

**Cause**: Insufficient or inaccurate calibration points

**Solutions**:
1. Use more calibration points (6-8 recommended)
2. Ensure even distribution across workspace
3. Use precise, visible markers
4. Improve lighting conditions
5. Stabilize camera mounting

### Problem: "Gripper not grasping weeds"

**Cause**: Incorrect gripper angle or force

**Solutions**:
1. Run gripper calibration
2. Test different closing angles
3. Adjust grasp position (lower/higher)
4. Check gripper mechanism

### Problem: "Weed detection not working"

**Cause**: YOLO model or camera issue

**Solutions**:
1. Verify `yolov8n.pt` model file exists
2. Check camera connection and settings
3. Test with mock mode first
4. Adjust detection confidence threshold
5. Ensure good lighting

### Problem: "Arm moves too fast/slow"

**Cause**: Speed parameter needs adjustment

**Solutions**:
1. Adjust `speed` parameter in `move_to_position()`
2. Modify `interpolation_steps` for smoother motion
3. Check motor max_speed settings
4. Adjust PWM frequency if needed

---

## Advanced Configuration

### Arm Dimensions

Edit `robotic_arm_5dof.py` to match your arm:

```python
@dataclass
class ArmDimensions:
    base_height: float = 10.0      # cm - base to shoulder
    upper_arm: float = 15.0        # cm - shoulder to elbow
    forearm: float = 12.0          # cm - elbow to wrist
    gripper: float = 8.0           # cm - wrist to gripper tip
```

### Motor Parameters

Adjust for your motors:

```python
class DCMotor:
    def __init__(self, ...):
        self.max_speed = 255      # Max PWM (0-255)
        self.min_speed = 50       # Min speed to overcome friction
        self.gear_ratio = 50.0    # Gear reduction ratio
```

### Detection Parameters

Tune weed detection:

```python
# In integrated_weeder_5dof.py
results = self.detector(frame,
                       conf=0.5,      # Confidence threshold
                       verbose=False)
```

### Waypoint Customization

Edit predefined waypoints:

```python
# In robotic_arm_5dof.py
'home': JointAngles(
    base=180.0,      # Adjust these values
    shoulder=90.0,
    elbow=90.0,
    wrist=90.0,
    gripper=90.0
)
```

---

## Performance Optimization

### Speed vs Accuracy Trade-offs

1. **Fast Mode** (lower accuracy):
```python
arm.move_to_position(target, speed=2.0, interpolation_steps=20)
```

2. **Accurate Mode** (slower):
```python
arm.move_to_position(target, speed=0.5, interpolation_steps=100)
```

### Parallel Processing

For multiple weeds, prioritization ensures efficient operation:
- Closest weeds processed first
- Minimal base rotation
- Optimized path planning

---

## Safety Features

1. **Software Safety**:
   - Angle limits enforced (0-180° or 0-360°)
   - IK validation before movement
   - Reachability checks
   - Home position on startup

2. **Hardware Safety** (recommended):
   - Emergency stop button
   - Current limiters on motors
   - Mechanical end stops
   - Workspace barriers

3. **Operational Safety**:
   - Always start in mock mode
   - Test sequences slowly first
   - Keep workspace clear
   - Monitor first few cycles

---

## Files Reference

| File | Purpose |
|------|---------|
| `robotic_arm_5dof.py` | Main 5-DOF arm controller |
| `teaching_mode.py` | Interactive waypoint recording |
| `integrated_weeder_5dof.py` | Complete integrated system |
| `calibration_utility.py` | Calibration and testing tools |
| `precision_grid_mapper.py` | Grid-based localization |
| `hardware_deployment.py` | YOLO detection and camera |

---

## Summary

### Best Method for Teaching Your Arm

**For beginners**: Start with **Teaching Mode**
- Learn how the arm moves
- Record safe positions
- Build intuition

**For production**: Use **Inverse Kinematics**
- Fully automatic
- Works with camera detection
- Optimal for continuous operation

**For best results**: Combine both methods
- Use teaching mode for initial setup
- Calibrate coordinates carefully
- Run with automatic IK system

### Next Steps

1. ✅ Install system and dependencies
2. ✅ Test in mock mode
3. ✅ Connect and test hardware
4. ✅ Run calibration
5. ✅ Teach key positions
6. ✅ Test with single weed
7. ✅ Run continuous operation
8. ✅ Monitor and optimize

---

## Support

For issues or questions:
- Check the troubleshooting section
- Review log files (`hardware_deployment.log`)
- Test components individually
- Use mock mode for debugging

Good luck with your robotic weed removal system! 🤖🌱
