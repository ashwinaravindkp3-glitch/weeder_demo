#!/usr/bin/env python3
"""
Quick test script to verify color output is working correctly.
This creates a synthetic field image and saves it to verify colors are not grey.
"""

import cv2
import numpy as np
import os

# Configuration
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
OUTPUT_DIR = 'output'

def create_synthetic_field_image():
    """
    Create a synthetic field image with weeds for testing.

    Returns:
        numpy.ndarray: A synthetic field image with weeds
    """
    # Create base field image (green background)
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    image[:, :] = (34, 139, 34)  # Forest green for field

    # Add some random variation to simulate field texture
    noise = np.random.randint(-20, 20, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)

    # Add synthetic weeds (bright green/yellow spots to simulate weeds)
    weed_positions = [
        (150, 120), (320, 200), (450, 150),
        (200, 300), (380, 350), (500, 280),
        (100, 400), (250, 180)
    ]

    for x, y in weed_positions:
        # Draw weed as a bright yellowish-green circle
        cv2.circle(image, (x, y), 20, (0, 255, 150), -1)

        # Add some texture to the weed
        for _ in range(8):
            offset_x = np.random.randint(-15, 15)
            offset_y = np.random.randint(-15, 15)
            cv2.circle(image, (x + offset_x, y + offset_y), 5, (50, 220, 100), -1)

        # Add weed center
        cv2.circle(image, (x, y), 5, (0, 255, 255), -1)

    # Add some crop plants (darker green circles)
    crop_positions = [
        (80, 80), (180, 80), (280, 80), (380, 80), (480, 80), (580, 80),
        (80, 240), (180, 240), (280, 240), (380, 240), (480, 240), (580, 240),
        (80, 400), (180, 400), (280, 400), (380, 400), (480, 400), (580, 400)
    ]

    for x, y in crop_positions:
        cv2.circle(image, (x, y), 12, (20, 100, 20), -1)
        cv2.circle(image, (x, y), 6, (10, 80, 10), -1)

    print(f"Created synthetic field image: {IMAGE_WIDTH}x{IMAGE_HEIGHT} with {len(weed_positions)} weeds")

    return image

def main():
    """Test color output generation."""
    print("=" * 60)
    print("Testing Color Output Fix")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create synthetic field image
    print("\nCreating synthetic field image...")
    field_image = create_synthetic_field_image()

    # Calculate and display color statistics
    mean_brightness = np.mean(field_image)
    mean_color = np.mean(field_image, axis=(0, 1))

    print(f"\nImage Statistics:")
    print(f"  Mean Brightness: {mean_brightness:.2f}")
    print(f"  Mean Color (BGR): ({mean_color[0]:.1f}, {mean_color[1]:.1f}, {mean_color[2]:.1f})")
    print(f"  Image Shape: {field_image.shape}")

    # Save the image
    output_path = os.path.join(OUTPUT_DIR, 'test_color_output.jpg')
    cv2.imwrite(output_path, field_image)
    print(f"\nImage saved to: {output_path}")

    # Verify the image is not grey
    if mean_brightness < 20:
        print("\n❌ ERROR: Image is too dark (grey/black)!")
        return False
    elif mean_color[1] > 100:  # Green channel should be high
        print("\n✅ SUCCESS: Image has proper colors (not grey)!")
        return True
    else:
        print("\n⚠️  WARNING: Image colors may not be correct")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
