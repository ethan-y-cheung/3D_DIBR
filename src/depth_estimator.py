import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from typing import Union

from transformers import pipeline


class DepthEstimator:

    def __init__(self, model_size: str = 'small', device: int = 0):
        """
        Args:
            model_size: 'small', 'base', or 'large'
            device: GPU device ID (0, 1, ...) or -1 for CPU
        """
        if device == -1:
            dev = "cpu"
            pipeline_device = -1
        elif torch.cuda.is_available():
            dev = f"cuda:{device}"
            pipeline_device = device
        else:
            dev = "cpu"
            pipeline_device = -1
        print(f"Using {dev}.")

        model_id = f"depth-anything/Depth-Anything-V2-{model_size}-hf"

        try:
            self.estimator = pipeline(
                task="depth-estimation",
                model=model_id,
                device=pipeline_device
            )
            print(f"Model loaded: {model_id}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def estimate(self, image: Union[np.ndarray, Image.Image, Path, str]) -> np.ndarray:
        # Convert input to PIL image if needed
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            arr = image if image.dtype == np.uint8 else image.astype(np.uint8)
            image = Image.fromarray(arr).convert("RGB")

        result = self.estimator(image)
        depth = result["depth"]
        
        # Apply median blur
        depth_array = cv2.medianBlur(np.array(depth, dtype=np.float32), 3)

        # Normalize to [0, 1] range
        depth_min = depth_array.min()
        depth_max = depth_array.max()

        depth_normalized = (depth_array - depth_min) / (depth_max - depth_min) if depth_max - depth_min > 0 else np.zeros_like(depth_array)
        return depth_normalized.astype(np.float32)
