import numpy as np
from typing import Tuple


class StereoGenerator:
    """Generate stereo pairs from depth maps"""
    
    def __init__(self, ipd_mm: float = 12.0):
        """
        Initialize stereo generator
        
        Args:
            ipd_mm: Interpupillary distance in millimeters (default: 12.0)
        """
        self.ipd_mm = ipd_mm
        print(f"StereoGenerator initialized with IPD: {ipd_mm}mm")
    
    def calculate_disparity(self, depth: np.ndarray, image_width: int) -> np.ndarray:
        """
        Convert depth to disparity
        
        Key principle: disparity ∝ depth
        - depth=1.0 (near) → max disparity
        - depth=0.0 (far) → zero disparity
        
        Args:
            depth: Normalized depth (0=far, 1=near)
            image_width: Image width in pixels
            
        Returns:
            Disparity in pixels
        """
        # Max disparity based on IPD
        # 12mm IPD → small max disparity (comfortable)
        # 65mm IPD → larger max disparity (standard)
        max_disparity = (self.ipd_mm / 65.0) * (image_width * 0.05)
        
        # Disparity proportional to depth
        disparity = depth * max_disparity
        
        return disparity.astype(np.float32)
    
    def shift_image(self, image: np.ndarray, disparity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Shift image right based on disparity
        
        Args:
            image: RGB image (H, W, 3)
            disparity: Disparity map (H, W)
            
        Returns:
            (shifted_image, occlusion_mask)
        """
        h, w = image.shape[:2]
        
        shifted = np.zeros_like(image, dtype=np.uint8)
        count_buffer = np.zeros((h, w), dtype=np.int32)
        depth_buffer = np.full((h, w), -np.inf, dtype=np.float32)
        
        # Coordinates
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Destination coordinates
        x_dest = x_coords.astype(np.float32) + disparity
        x_dest_int = np.round(x_dest).astype(np.int32)
        
        # Valid pixels
        valid_mask = (x_dest_int >= 0) & (x_dest_int < w)
        
        valid_y = y_coords[valid_mask]
        valid_x_src = x_coords[valid_mask]
        valid_x_dest = x_dest_int[valid_mask]
        valid_disparity = disparity[valid_mask]
        
        # Sort far to near (near pixels overwrite far)
        sort_indices = np.argsort(valid_disparity)
        
        for idx in sort_indices:
            y = valid_y[idx]
            x_src = valid_x_src[idx]
            x_dst = valid_x_dest[idx]
            disp = valid_disparity[idx]
            
            if disp >= depth_buffer[y, x_dst]:
                shifted[y, x_dst] = image[y, x_src]
                depth_buffer[y, x_dst] = disp
                count_buffer[y, x_dst] += 1
        
        occlusion_mask = (count_buffer == 0).astype(np.uint8) * 255
        
        return shifted, occlusion_mask
    
    def generate_stereo_pair(self, 
                            image: np.ndarray, 
                            depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate stereo pair
        
        Args:
            image: RGB image (H, W, 3) uint8
            depth: Depth map (H, W) float32, [0, 1] where 0=far, 1=near
            
        Returns:
            (left_view, right_view, left_mask, right_mask)
        """
        if image.shape[:2] != depth.shape:
            raise ValueError(f"Shape mismatch: image {image.shape[:2]} vs depth {depth.shape}")
        
        # Ensure correct types
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        depth = np.clip(depth, 0.0, 1.0)
        
        h, w = image.shape[:2]
        
        # Calculate disparity
        disparity = self.calculate_disparity(depth, w)
        
        print(f"Disparity statistics:")
        print(f"  Mean: {np.mean(disparity):.2f} pixels")
        print(f"  Max: {np.max(disparity):.2f} pixels ({np.max(disparity)/w*100:.2f}% of width)")
        
        # Left = original (no shift, no occlusions)
        left_view = image.copy()
        left_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Right = shifted
        right_view, right_mask = self.shift_image(image, disparity)
        
        right_holes = np.sum(right_mask > 0)
        print(f"Occlusions:")
        print(f"  Left: 0 pixels (0.00%) - original image")
        print(f"  Right: {right_holes:,} pixels ({100*right_holes/(h*w):.2f}%)")
        
        return left_view, right_view, left_mask, right_mask
    
    def create_side_by_side(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.hstack([left, right])
    
    def create_anaglyph(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        anaglyph = np.zeros_like(left)
        anaglyph[:, :, 0] = left[:, :, 0]
        anaglyph[:, :, 1] = right[:, :, 1]
        anaglyph[:, :, 2] = right[:, :, 2]
        return anaglyph