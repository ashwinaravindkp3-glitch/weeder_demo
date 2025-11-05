import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrecisionGridMapper:
    """
    Maps 640x480 camera coordinates to precise grid locations for plant targeting.
    Uses a 16x12 grid (40x40 pixel cells) for optimal precision.
    """
    
    def __init__(self, image_width=640, image_height=480, grid_cols=16, grid_rows=12):
        """
        Initialize the precision grid mapper.
        
        Args:
            image_width: Camera image width (default 640)
            image_height: Camera image height (default 480)
            grid_cols: Number of grid columns (default 16 for 40px cells)
            grid_rows: Number of grid rows (default 12 for 40px cells)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        
        # Calculate cell dimensions
        self.cell_width = image_width // grid_cols
        self.cell_height = image_height // grid_rows
        
        # Center of the grid (arm position)
        self.center_col = grid_cols // 2
        self.center_row = grid_rows // 2
        
        logger.info(f"Grid mapper initialized: {grid_cols}x{grid_rows} grid, "
                   f"cell size {self.cell_width}x{self.cell_height}px")
        logger.info(f"Center position: ({self.center_col}, {self.center_row})")
    
    def pixel_to_grid(self, x_pixel, y_pixel):
        """
        Convert pixel coordinates to grid coordinates.
        
        Args:
            x_pixel: X coordinate in pixels (0-639)
            y_pixel: Y coordinate in pixels (0-479)
            
        Returns:
            tuple: (grid_col, grid_row) coordinates
        """
        # Ensure coordinates are within bounds
        x_pixel = max(0, min(x_pixel, self.image_width - 1))
        y_pixel = max(0, min(y_pixel, self.image_height - 1))
        
        # Convert to grid coordinates
        grid_col = x_pixel // self.cell_width
        grid_row = y_pixel // self.cell_height
        
        return (grid_col, grid_row)
    
    def detection_to_movement_command(self, detection_x, detection_y, detection_width=0, detection_height=0):
        """
        Convert a plant detection to arm movement commands.
        
        Args:
            detection_x: X coordinate of detection center (pixels)
            detection_y: Y coordinate of detection center (pixels)
            detection_width: Width of detection box (optional)
            detection_height: Height of detection box (optional)
            
        Returns:
            dict: Movement command with direction, distance, and precision info
        """
        # Convert to grid coordinates
        grid_col, grid_row = self.pixel_to_grid(detection_x, detection_y)
        
        # Calculate relative position from center
        col_offset = grid_col - self.center_col
        row_offset = grid_row - self.center_row
        
        # Determine movement direction and magnitude
        movement = {
            'grid_position': (grid_col, grid_row),
            'pixel_position': (detection_x, detection_y),
            'offset_from_center': (col_offset, row_offset),
            'direction': self._get_direction_name(col_offset, row_offset),
            'movement_needed': self._calculate_movement_needed(col_offset, row_offset),
            'precision_level': self._calculate_precision_level(detection_width, detection_height)
        }
        
        logger.info(f"Detection at ({detection_x}, {detection_y}) -> Grid ({grid_col}, {grid_row}) "
                   f"-> Offset ({col_offset}, {row_offset}) -> {movement['direction']}")
        
        return movement
    
    def _get_direction_name(self, col_offset, row_offset):
        """Convert offset to human-readable direction."""
        if col_offset == 0 and row_offset == 0:
            return "CENTER (immediate weeding)"
        
        directions = []
        
        # Horizontal direction
        if col_offset < 0:
            directions.append(f"LEFT {abs(col_offset)}")
        elif col_offset > 0:
            directions.append(f"RIGHT {col_offset}")
        
        # Vertical direction
        if row_offset < 0:
            directions.append(f"UP {abs(row_offset)}")
        elif row_offset > 0:
            directions.append(f"DOWN {row_offset}")
        
        return " + ".join(directions)
    
    def _calculate_movement_needed(self, col_offset, row_offset):
        """Calculate if movement is needed and how much."""
        # Calculate Manhattan distance from center
        distance = abs(col_offset) + abs(row_offset)
        
        # Determine if weeding can be done from current position
        if distance == 0:
            return "WEED_IMMEDIATELY"
        elif distance <= 2:  # Within 2 cells of center
            return "SMALL_ADJUSTMENT"
        elif distance <= 4:  # Within 4 cells
            return "MODERATE_MOVEMENT"
        else:
            return "LARGE_MOVEMENT"
    
    def _calculate_precision_level(self, detection_width, detection_height):
        """Calculate precision level based on detection size."""
        if detection_width == 0 or detection_height == 0:
            return "UNKNOWN_SIZE"
        
        # Calculate detection area in pixels
        area = detection_width * detection_height
        
        # Classify by size
        if area < 400:  # < 20x20 pixels
            return "SMALL_PLANT"
        elif area < 1600:  # < 40x40 pixels
            return "MEDIUM_PLANT"
        else:
            return "LARGE_PLANT"
    
    def get_grid_visualization(self, detections=None):
        """
        Create a visual representation of the grid with optional detections.
        
        Args:
            detections: List of (x, y) detection coordinates
            
        Returns:
            str: ASCII representation of the grid
        """
        # Create empty grid
        grid = [['.' for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        
        # Mark center
        grid[self.center_row][self.center_col] = 'C'
        
        # Add detections
        if detections:
            for i, (x, y) in enumerate(detections):
                col, row = self.pixel_to_grid(x, y)
                if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                    if row == self.center_row and col == self.center_col:
                        grid[row][col] = 'X'  # Target at center
                    else:
                        grid[row][col] = str(i + 1)  # Numbered targets
        
        # Build ASCII representation
        result = f"Grid: {self.grid_cols}x{self.grid_rows} (Center: C, Targets: X/Numbers)\n"
        result += "-" * (self.grid_cols * 2 + 1) + "\n"
        
        for row in range(self.grid_rows):
            result += "|"
            for col in range(self.grid_cols):
                result += grid[row][col] + ""
            result += "|\n"
        
        result += "-" * (self.grid_cols * 2 + 1) + "\n"
        result += f"Cell size: {self.cell_width}x{self.cell_height}px\n"
        
        return result

    def draw_grid(self, image):
        """
        Draw the grid on the given image.

        Args:
            image: The image to draw the grid on.

        Returns:
            The image with the grid drawn on it.
        """
        for x in range(0, self.image_width, self.cell_width):
            cv2.line(image, (x, 0), (x, self.image_height), (255, 255, 255), 1)
        for y in range(0, self.image_height, self.cell_height):
            cv2.line(image, (0, y), (self.image_width, y), (255, 255, 255), 1)
        return image
    
    def get_weeding_priority(self, detections):
        """
        Sort detections by weeding priority (closest to center first).
        
        Args:
            detections: List of (x, y) detection coordinates
            
        Returns:
            list: Sorted list of detections with priority info
        """
        prioritized = []
        
        for i, (x, y) in enumerate(detections):
            movement = self.detection_to_movement_command(x, y)
            distance = abs(movement['offset_from_center'][0]) + abs(movement['offset_from_center'][1])
            
            prioritized.append({
                'detection_id': i,
                'pixel_position': (x, y),
                'grid_position': movement['grid_position'],
                'distance_from_center': distance,
                'movement_command': movement,
                'priority': distance  # Lower distance = higher priority
            })
        
        # Sort by distance (priority)
        prioritized.sort(key=lambda x: x['priority'])
        
        return prioritized

# Example usage and testing
if __name__ == "__main__":
    # Create grid mapper
    mapper = PrecisionGridMapper()
    
    # Test detections at various positions
    test_detections = [
        (320, 240),  # Center (should be immediate weeding)
        (160, 120),  # Top-left
        (480, 360),  # Bottom-right
        (100, 400),  # Bottom-left corner
        (550, 50),   # Top-right corner
        (300, 250),  # Near center
    ]
    
    logger.info("=== PRECISION GRID MAPPER TEST ===")
    logger.info(f"Image size: {mapper.image_width}x{mapper.image_height}")
    logger.info(f"Grid size: {mapper.grid_cols}x{mapper.grid_rows}")
    logger.info(f"Cell size: {mapper.cell_width}x{mapper.cell_height} pixels")
    
    # Test each detection
    for i, (x, y) in enumerate(test_detections):
        logger.info(f"\n--- Detection {i+1}: ({x}, {y}) ---")
        movement = mapper.detection_to_movement_command(x, y, 30, 30)  # 30x30 pixel detection
        
        logger.info(f"Grid position: {movement['grid_position']}")
        logger.info(f"Offset from center: {movement['offset_from_center']}")
        logger.info(f"Direction: {movement['direction']}")
        logger.info(f"Movement needed: {movement['movement_needed']}")
        logger.info(f"Precision level: {movement['precision_level']}")
    
    # Show grid visualization
    logger.info(f"\n{mapper.get_grid_visualization(test_detections)}")
    
    # Show priority sorting
    logger.info("=== WEEDING PRIORITY (closest to center first) ===")
    prioritized = mapper.get_weeding_priority(test_detections)
    for item in prioritized:
        logger.info(f"Detection {item['detection_id']+1}: "
                   f"Grid {item['grid_position']} -> "
                   f"Distance {item['distance_from_center']} -> "
                   f"Priority {item['priority']}")