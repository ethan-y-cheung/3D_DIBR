import numpy as np
from typing import Tuple


class StereoGenerator:

    def __init__(self, ipd_mm: float = 12.0):
        self.ipd_mm = ipd_mm
        print(f"StereoGenerator initialized with IPD: {ipd_mm}mm")

    def calculate_disparity(self, depth: np.ndarray, image_width: int) -> np.ndarray:
        # Empirical disparity formula:
        # - 65.0mm is average human IPD, used to scale the configured IPD
        # - 0.05 limits max pixel shift to 5% of image width at full depth
        # Together: max_disparity = (user_ipd / avg_ipd) * (5% of width)
        max_disparity = (self.ipd_mm / 65.0) * (image_width * 0.05)

        # Disparity proportional to depth (closer = more shift)
        disparity = depth * max_disparity

        return disparity.astype(np.float32)

    def shift_image(self, image: np.ndarray, disparity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]

        shifted = np.zeros_like(image, dtype=np.uint8)

        # Coordinates
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        # Destination coordinates
        x_dest = x_coords.astype(np.float32) + disparity
        x_dest_int = np.round(x_dest).astype(np.int32)

        # Valid pixels (destination within image bounds)
        valid_mask = (x_dest_int >= 0) & (x_dest_int < w)

        valid_y = y_coords[valid_mask]
        valid_x_src = x_coords[valid_mask]
        valid_x_dest = x_dest_int[valid_mask]
        valid_disparity = disparity[valid_mask]

        # Sort far to near (ascending disparity) so near pixels overwrite far ones
        sort_indices = np.argsort(valid_disparity)

        valid_y = valid_y[sort_indices]
        valid_x_src = valid_x_src[sort_indices]
        valid_x_dest = valid_x_dest[sort_indices]
        valid_disparity = valid_disparity[sort_indices]

        # Vectorized Z-buffer: process in sorted order using advanced indexing.
        # Since arrays are sorted far-to-near, later (nearer) pixels naturally
        # overwrite earlier (farther) ones at the same destination, achieving
        # correct occlusion handling without an explicit per-pixel loop.
        shifted[valid_y, valid_x_dest] = image[valid_y, valid_x_src]

        # Occlusion mask: pixels that were never written to
        occluded = np.full((h, w), True)
        occluded[valid_y, valid_x_dest] = False
        occlusion_mask = occluded.astype(np.uint8) * 255

        return shifted, occlusion_mask

    def generate_stereo_pair(self, image: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        # Left = original (no shift needed, left view is the source image)
        left_view = image.copy()
        left_mask = np.zeros((h, w), dtype=np.uint8)

        # Right = shifted
        right_view, right_mask = self.shift_image(image, disparity)

        return left_view, right_view, left_mask, right_mask

    def create_side_by_side(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.hstack([left, right])

    def create_anaglyph(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        anaglyph = np.zeros_like(left)
        anaglyph[:, :, 0] = left[:, :, 0]
        anaglyph[:, :, 1] = right[:, :, 1]
        anaglyph[:, :, 2] = right[:, :, 2]
        return anaglyph
