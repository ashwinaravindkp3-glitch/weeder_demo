# Preprocessing Settings Guide

## Overview
The system includes image preprocessing to enhance weed detection. These settings can be adjusted based on your lighting conditions.

## Configuration Location
Edit these values in `final_integrated_system.py`:

```python
# --- Preprocessing Configuration ---
ENABLE_PREPROCESSING = True  # Set to False to disable all preprocessing
BRIGHTNESS_ADJUSTMENT = 0    # 0 = none, higher = brighter (0-40)
CONTRAST_MULTIPLIER = 1.0    # 1.0 = no change, >1.0 = more contrast
CLAHE_CLIP_LIMIT = 1.5       # Lower = less aggressive (0-3.0)
```

## Recommended Presets

### ☀️ Bright Environment (CURRENT DEFAULT)
**Use when**: Good lighting, sunny day, well-lit indoor space

```python
ENABLE_PREPROCESSING = True
BRIGHTNESS_ADJUSTMENT = 0     # No brightness boost needed
CONTRAST_MULTIPLIER = 1.0     # Normal contrast
CLAHE_CLIP_LIMIT = 1.5        # Mild enhancement
```

**Result**: Minimal processing, preserves natural colors

---

### 🌤️ Normal Lighting
**Use when**: Moderate lighting, cloudy day, typical indoor lighting

```python
ENABLE_PREPROCESSING = True
BRIGHTNESS_ADJUSTMENT = 10    # Slight brightness boost
CONTRAST_MULTIPLIER = 1.2     # Slight contrast increase
CLAHE_CLIP_LIMIT = 2.0        # Moderate enhancement
```

**Result**: Balanced enhancement for typical conditions

---

### 🌙 Low Light / Dark Environment
**Use when**: Poor lighting, evening, shadowy areas

```python
ENABLE_PREPROCESSING = True
BRIGHTNESS_ADJUSTMENT = 25    # Significant brightness boost
CONTRAST_MULTIPLIER = 1.4     # Strong contrast increase
CLAHE_CLIP_LIMIT = 2.5        # Aggressive enhancement
```

**Result**: Maximum enhancement for dark conditions

---

### 📷 No Preprocessing (Raw Camera)
**Use when**: Perfect lighting, want unmodified feed, testing

```python
ENABLE_PREPROCESSING = False
# Other values ignored when disabled
```

**Result**: Raw camera feed, no modifications

---

## Troubleshooting

### Problem: Image too bright / washed out
**Symptoms**:
- Weeds and crops look too pale
- Colors are washed out
- Overexposed appearance

**Solution**:
1. Reduce `BRIGHTNESS_ADJUSTMENT` (try 0)
2. Reduce `CONTRAST_MULTIPLIER` (try 1.0)
3. Reduce `CLAHE_CLIP_LIMIT` (try 1.0)
4. Or disable preprocessing entirely

### Problem: Image too dark
**Symptoms**:
- Hard to see weeds
- Overall dim appearance
- Poor detection rate

**Solution**:
1. Increase `BRIGHTNESS_ADJUSTMENT` (try 20-30)
2. Increase `CONTRAST_MULTIPLIER` (try 1.3-1.5)
3. Increase `CLAHE_CLIP_LIMIT` (try 2.0-2.5)

### Problem: Colors look unnatural
**Symptoms**:
- Strange color tints
- Unrealistic green/blue hues

**Solution**:
1. Reduce `CLAHE_CLIP_LIMIT` (try 1.0 or 0)
2. Set `BRIGHTNESS_ADJUSTMENT = 0`
3. Set `CONTRAST_MULTIPLIER = 1.0`
4. Or disable white balance by commenting out the line in code

### Problem: Detection not working well
**Symptoms**:
- YOLO missing weeds
- False detections
- Low confidence scores

**Solution**:
1. Use camera preview to check image quality
2. Adjust preprocessing to match training data appearance
3. If model was trained on bright images, use minimal preprocessing
4. If model was trained on enhanced images, use more preprocessing

---

## Parameter Details

### BRIGHTNESS_ADJUSTMENT
- **Range**: 0-40 (can go higher but not recommended)
- **Effect**: Adds constant value to all pixels
- **0**: No change (recommended for bright environments)
- **10-20**: Moderate boost
- **25-40**: Strong boost (for dark environments)

### CONTRAST_MULTIPLIER
- **Range**: 0.5-2.0 (1.0 = normal)
- **Effect**: Multiplies pixel values (spreads histogram)
- **1.0**: No change (recommended for bright environments)
- **1.2-1.3**: Slight increase
- **1.4-1.5**: Strong increase (for low contrast scenes)

### CLAHE_CLIP_LIMIT
- **Range**: 0-5.0 (0 = disabled)
- **Effect**: Adaptive histogram equalization (enhances local contrast)
- **0**: Disabled
- **1.0-1.5**: Mild (recommended for bright environments)
- **2.0-2.5**: Moderate
- **3.0+**: Aggressive (can cause artifacts)

---

## Quick Test Procedure

1. **Start with preprocessing disabled**:
   ```python
   ENABLE_PREPROCESSING = False
   ```

2. **Run and check output**:
   - Look at bounding box image
   - Check detection quality
   - Observe colors and brightness

3. **If too dark, enable minimal preprocessing**:
   ```python
   ENABLE_PREPROCESSING = True
   BRIGHTNESS_ADJUSTMENT = 10
   CONTRAST_MULTIPLIER = 1.2
   CLAHE_CLIP_LIMIT = 1.5
   ```

4. **Adjust incrementally**:
   - Change one parameter at a time
   - Test after each change
   - Find the minimal settings that work well

---

## Current Default (Bright Environment)

The system is now configured for **bright environments** with minimal preprocessing:
- ✅ Brightness: 0 (no boost)
- ✅ Contrast: 1.0 (normal)
- ✅ CLAHE: 1.5 (mild enhancement)

This prevents overexposure in well-lit conditions while still providing slight contrast enhancement for better edge detection.
