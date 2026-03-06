import argparse
from pathlib import Path
import numpy as np
import sys
import cv2
from tqdm import tqdm

from src.depth_estimator import DepthEstimator
from src.stereo_generator import StereoGenerator
from src.inpainter import Inpainter


def smooth_depth(current_depth, previous_depth, alpha=0.8):
    if previous_depth is None:
        return current_depth
    return (alpha * current_depth + (1 - alpha) * previous_depth).astype(np.float32)


def process_video(input_path, output_path, depth_estimator, stereo_generator, inpainter, smooth_alpha=0.8):
    print(f"\n{'='*60}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*60}\n")
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {input_path}")
    
    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Temporal smoothing: alpha={smooth_alpha}\n")
    
    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Process frames with progress bar
    previous_depth = None
    
    with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Depth estimation
            depth = depth_estimator.estimate(frame_rgb)
            depth_smooth = smooth_depth(depth, previous_depth, smooth_alpha)
            previous_depth = depth_smooth
            
            # Stereo generation
            left, right, left_mask, right_mask = stereo_generator.generate_stereo_pair(
                frame_rgb, depth_smooth
            )
            
            # Inpaint
            _, right = inpainter.inpaint_stereo_pair(left, right, left_mask, right_mask)

            right_bgr = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
            out.write(right_bgr)
            
            pbar.update(1)
    
    cap.release()
    out.release()
    
    print(f"\n✓ Complete: {output_path}\n")


def parse_args():
    parser = argparse.ArgumentParser(description='2D to 3D Video Stereo Pipeline')
    parser.add_argument('--input', type=str, default='input/videos',
                       help='Input video file or directory (default: input/videos)')
    parser.add_argument('--output', type=str, default='output/video_pair',
                       help='Output directory (default: output/video_pair)')
    parser.add_argument('--ipd', type=float, default=12.0,
                       help='Interpupillary distance in mm (default: 12.0)')
    parser.add_argument('--depth-model', choices=['small', 'base', 'large'], 
                       default='base',
                       help='DepthAnything model size (default: base)')
    parser.add_argument('--inpaint-method', choices=['telea', 'ns'],
                       default='telea',
                       help='Inpainting method (default: telea)')
    parser.add_argument('--smooth-alpha', type=float, default=0.8,
                       help='Temporal smoothing factor (default: 0.8)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("2D to 3D Video Stereo Pipeline")
    print("="*60)
    print(f"IPD: {args.ipd}mm")
    print(f"Depth Model: {args.depth_model}")
    print(f"Inpainting: {args.inpaint_method}")
    print("="*60)
    
    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    # Initialize components
    print("\nInitializing pipeline components...")
    depth_estimator = DepthEstimator()
    stereo_generator = StereoGenerator(ipd_mm=args.ipd)
    inpainter = Inpainter(method=args.inpaint_method, radius=3)
    
    # Get video files
    if input_path.is_file():
        video_files = [input_path]
    elif input_path.is_dir():
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        video_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in video_extensions]
        if not video_files:
            print(f"\nError: No videos found in {input_path}")
            sys.exit(1)
    else:
        print(f"\nError: {input_path} not found")
        sys.exit(1)
    
    print(f"\nFound {len(video_files)} video(s) to process")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each video
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[Video {idx}/{len(video_files)}]")
        output_path = output_dir / f"{video_path.stem}_right.mp4"
        try:
            process_video(video_path, output_path, 
                        depth_estimator, stereo_generator, inpainter,
                        args.smooth_alpha)
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Results: {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()