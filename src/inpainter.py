import numpy as np
import cv2
from typing import Tuple

class Inpainter:
    def __init__(self, method: str = 'telea', radius: int = 3):

        self.method = method
        self.radius = radius
        
        if method == 'telea':
            self.cv_method = cv2.INPAINT_TELEA
        elif method == 'ns':
            self.cv_method = cv2.INPAINT_NS
        else:
            raise ValueError(f"Unknown method: {method}. Use 'telea' or 'ns'")
        
        print(f"Inpainter initialized:")
        print(f"  Method: {method}")
    
    def inpaint_view(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Ensure correct types
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # Check if there are any pixels to inpaint
        if np.sum(mask > 0) == 0:
            return image.copy()
        
        # OpenCV inpaint
        inpainted = cv2.inpaint(image, mask, self.radius, self.cv_method)
        
        return inpainted
    
    def inpaint_stereo_pair(self, 
                           left: np.ndarray, 
                           right: np.ndarray,
                           left_mask: np.ndarray, 
                           right_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        left_inpainted = self.inpaint_view(left, left_mask)
        right_inpainted = self.inpaint_view(right, right_mask)
        
        return left_inpainted, right_inpainted