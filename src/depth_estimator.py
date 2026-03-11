import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from typing import Union

from transformers import pipeline


class DepthEstimator:

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device}.")

        model_id = "depth-anything/Depth-Anything-V2-small-hf"  # changeable if base doesn't work
        
        try:
            self.estimator = pipeline(
                task="depth-estimation",
                model=model_id,
                device=0 if device == "cuda" else -1
                #cache_dir=".models"
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def process_image(self, image_path, output_folder): # Only for testing

        print(f"Processing: {image_path.name}")
        image = Image.open(image_path).convert("RGB")
        
        # run depth estimation
        result = self.estimator(image)
        depth = result["depth"]
        
        # fixes things, but also maybe makes edges chopped?
        depth_array = cv2.medianBlur(np.array(depth, dtype=np.float32), 3)
        depth_normalized = ((depth_array - depth_array.min()) / 
                        (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO) # Color map for visualization
        
        output_name = image_path.stem
        colored_path = output_folder / f"{output_name}_depth_mb.png"
        cv2.imwrite(str(colored_path), depth_colored)

    
    def estimate(self, image: Union[np.ndarray, Image.Image, Path, str]) -> np.ndarray:
        # Convert input to PIL image if needed
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        result = self.estimator(image)
        depth = result["depth"]
        depth_array = cv2.medianBlur(np.array(depth, dtype=np.float32), 3) #apply median blur
        
        # Normalize to [0, 1] range
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        
        depth_normalized = (depth_array-depth_min)/(depth_max-depth_min) if depth_max-depth_min > 0 else np.zeros_like(depth_array)
        return depth_normalized.astype(np.float32)



def main():
    input_folder = Path("input/images")
    output_folder = Path("output/images")
    
    output_folder.mkdir(exist_ok=True)
    
    # Check if input folder exists and has images
    if not input_folder.exists():
        print(f"Error: '{input_folder}' not found.")
        return
    
    # Valid file types
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in input_folder.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found.")
        return
    
    depth_estimator = DepthEstimator()
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}]")
        try:
            DepthEstimator.process_image(depth_estimator, image_path, output_folder)
        except Exception as e:
            print(f"  ✗ Error processing {image_path.name}: {str(e)}")
        print()


if __name__ == "__main__":
    main()
