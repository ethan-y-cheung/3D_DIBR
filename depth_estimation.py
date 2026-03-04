import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2

from transformers import pipeline

def setup():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}.")

    model_id = "depth-anything/Depth-Anything-V2-BASE-hf"  # changeable if base doesn't work
    
    try:
        depth_estimator = pipeline(
            task="depth-estimation",
            model=model_id,
            device=0 if device == "cuda" else -1
        )
        print("Model loaded successfully!")
        return depth_estimator
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def process_image(depth_estimator, image_path, output_folder):

    print(f"Processing: {image_path.name}")
    image = Image.open(image_path).convert("RGB")
    
    # Run depth estimation
    result = depth_estimator(image)
    depth = result["depth"]
    
    depth_array = np.array(depth)
    depth_normalized = ((depth_array - depth_array.min()) / 
                       (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO) # Color map for visualization
    
    output_name = image_path.stem
    grayscale_path = output_folder / f"{output_name}_depth_gray.png"
    Image.fromarray(depth_normalized).save(grayscale_path)
    
    colored_path = output_folder / f"{output_name}_depth_colored.png"
    cv2.imwrite(str(colored_path), depth_colored)
    
    raw_path = output_folder / f"{output_name}_depth_raw.npy"
    np.save(raw_path, depth_array)



def main():
    input_folder = Path("input")
    output_folder = Path("output")
    
    output_folder.mkdir(exist_ok=True)
    
    # Check if input folder exists and has images
    if not input_folder.exists():
        print(f"Error: '{input_folder}' folder not found!")
        return
    
    # Valid file types
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in input_folder.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in '{input_folder}' folder!")
        return
    
    # Initialization
    depth_estimator = setup()

    
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}]")
        try:
            process_image(depth_estimator, image_path, output_folder)
        except Exception as e:
            print(f"  ✗ Error processing {image_path.name}: {str(e)}")
        print()


if __name__ == "__main__":
    main()
