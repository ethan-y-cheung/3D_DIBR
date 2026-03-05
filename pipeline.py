import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import sys

from src.depth_estimator import DepthEstimator
from src.stereo_generator import StereoGenerator
from src.inpainter import Inpainter


def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array"""
    img = Image.open(path).convert('RGB')
    return np.array(img)


def save_image(image: np.ndarray, path: Path):
    """Save numpy array as image"""
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def process_image(input_path: Path, 
                 output_dir: Path,
                 depth_estimator: DepthEstimator,
                 stereo_generator: StereoGenerator,
                 inpainter: Inpainter,
                 skip_inpaint: bool = False):
    """
    Process a single image through the pipeline
    
    Args:
        input_path: Path to input image
        output_dir: Directory for outputs
        depth_estimator: Initialized DepthEstimator instance
        stereo_generator: Initialized StereoGenerator instance
        inpainter: Initialized Inpainter instance
        skip_inpaint: Skip inpainting step if True
    """
    print(f"\n{'='*60}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*60}\n")
    
    # Load image
    print("[1/3] Loading image...")
    image = load_image(input_path)
    print(f"  Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Estimate depth and generate stereo
    print("\n[2/3] Estimating depth and generating stereo pair...")
    depth = depth_estimator.estimate(image)
    left_view, right_view, left_mask, right_mask = stereo_generator.generate_stereo_pair(image, depth)
    
    # Inpaint occlusions
    if not skip_inpaint:
        print("\n[3/3] Inpainting occlusions...")
        left_final, right_final = inpainter.inpaint_stereo_pair(
            left_view, right_view, left_mask, right_mask
        )
    else:
        print("\n[3/3] Skipping inpainting")
        left_final, right_final = left_view, right_view
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = input_path.stem
    
    save_image(left_final, output_dir / "stereo" / f"{base_name}_left.png")
    save_image(right_final, output_dir / "stereo" / f"{base_name}_right.png")
    
    print(f"\n✓ Saved outputs to {output_dir}/")
    print(f"{'='*60}\n")
    
    return left_final, right_final


def parse_args():
    parser = argparse.ArgumentParser(description='2D to 3D Stereo Conversion Pipeline')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--ipd', type=float, default=12.0,
                       help='Interpupillary distance in mm (default: 12.0)')
    parser.add_argument('--depth-model', choices=['small', 'base', 'large'], 
                       default='base',
                       help='DepthAnything model size (default: base)')
    parser.add_argument('--inpaint-method', choices=['telea', 'ns'],
                       default='telea',
                       help='Inpainting method (default: telea)')
    parser.add_argument('--inpaint-radius', type=int, default=3,
                       help='Inpainting radius in pixels (default: 3)')
    parser.add_argument('--skip-inpaint', action='store_true',
                       help='Skip inpainting step')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID, -1 for CPU (default: 0)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("2D to 3D Stereo Pipeline")
    print("="*60)
    print(f"IPD: {args.ipd}mm")
    print(f"Depth Model: {args.depth_model}")
    print(f"Inpainting: {args.inpaint_method if not args.skip_inpaint else 'disabled'}")
    print("="*60)
    
    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    # Initialize components
    print("\nInitializing pipeline components...")
    depth_estimator = DepthEstimator()

    stereo_generator = StereoGenerator(ipd_mm=args.ipd)
    inpainter = Inpainter(
        method=args.inpaint_method,
        radius=args.inpaint_radius
    )
    
    # Get list of images
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        if not image_files:
            print(f"\nError: No images found in {input_path}")
            sys.exit(1)
    else:
        print(f"\nError: {input_path} is not a valid file or directory")
        sys.exit(1)
    
    print(f"\nFound {len(image_files)} image(s) to process")
    
    # Process each image
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[Image {idx}/{len(image_files)}]")
        try:
            process_image(img_path, output_dir, depth_estimator, 
                         stereo_generator, inpainter, args.skip_inpaint)
        except Exception as e:
            print(f"\n✗ Error processing {img_path.name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Results: {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()