import os
import sys
import warnings

# Suppress noisy library output before any heavy imports
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_AUTOGRAPH_VERBOSITY'] = '0'
warnings.filterwarnings('ignore')

import argparse
import contextlib
import csv
import re
import time
from pathlib import Path

import cv2
import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                            SpinnerColumn, TextColumn, TimeElapsedColumn,
                            TimeRemainingColumn)
from rich.table import Table
from rich.text import Text
from skimage.metrics import structural_similarity as _ssim

console = Console()

PAIR_RE = re.compile(r'^(cam[01])_(\d+)$')


@contextlib.contextmanager
def _mute():
    """Silence stdout/stderr from noisy libraries."""
    with open(os.devnull, 'w') as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def discover_pairs(test_dir: Path):
    """Return a list of (timestamp, cam0_path, cam1_path) tuples."""
    cam0, cam1 = {}, {}
    for f in sorted(test_dir.glob('*.mp4')):
        m = PAIR_RE.match(f.stem)
        if not m:
            continue
        (cam0 if m.group(1) == 'cam0' else cam1)[m.group(2)] = f

    pairs = []
    for ts in sorted(cam0.keys()):
        if ts in cam1:
            pairs.append((ts, cam0[ts], cam1[ts]))
        else:
            console.print(f"[yellow]Warning: cam0_{ts} has no matching cam1[/yellow]")
    for ts in sorted(cam1.keys() - cam0.keys()):
        console.print(f"[yellow]Warning: cam1_{ts} has no matching cam0[/yellow]")
    return pairs


def evaluate_pair(cam0_path, cam1_path, de, sg, ip, smooth_alpha, csv_writer, pair_id):
    cap0 = cv2.VideoCapture(str(cam0_path))
    cap1 = cv2.VideoCapture(str(cam1_path))
    if not cap0.isOpened() or not cap1.isOpened():
        cap0.release(); cap1.release()
        raise ValueError(f"Cannot open {cam0_path.name} or {cam1_path.name}")

    n0 = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    n1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(n1, n0)
    if n1 != n0:
        console.print(f"  [yellow]Frame count mismatch: cam0={n0}, cam1={n1}; using {total}[/yellow]")

    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap0.get(cv2.CAP_PROP_FPS)

    psnr_vals, ssim_vals = [], []
    prev_depth = None

    with Progress(
        SpinnerColumn(),
        TextColumn(f"  [dim]{w}x{h} @ {fps:.0f}fps[/dim]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("[dim]eta[/dim]"),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as bar:
        task = bar.add_task('', total=total)

        for frame_idx in range(total):
            ret0, frame0_bgr = cap0.read()
            ret1, frame1_bgr = cap1.read()
            if not ret0 or not ret1:
                break

            frame0_rgb = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2RGB)
            frame1_rgb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB)

            depth = de.estimate(frame0_rgb)
            if prev_depth is not None:
                depth = (smooth_alpha * depth
                         + (1 - smooth_alpha) * prev_depth).astype(np.float32)
            prev_depth = depth

            left, right, lm, rm = sg.generate_stereo_pair(frame0_rgb, depth)
            _, right_gen = ip.inpaint_stereo_pair(left, right, lm, rm)

            # Defensive resolution match (shouldn't happen with paired captures)
            if frame1_rgb.shape != right_gen.shape:
                frame1_rgb = cv2.resize(frame1_rgb,
                                        (right_gen.shape[1], right_gen.shape[0]))

            psnr = cv2.PSNR(right_gen, frame1_rgb)
            if np.isinf(psnr):
                psnr = 100.0
            ssim_val = float(_ssim(right_gen, frame1_rgb,
                                   channel_axis=2, data_range=255))

            psnr_vals.append(psnr)
            ssim_vals.append(ssim_val)
            csv_writer.writerow({
                'pair_id': pair_id,
                'frame_idx': frame_idx,
                'psnr': f'{psnr:.4f}',
                'ssim': f'{ssim_val:.4f}',
            })
            bar.advance(task)

    cap0.release(); cap1.release()

    if not psnr_vals:
        return {'pair_id': pair_id, 'frames': 0,
                'psnr_mean': 0.0, 'psnr_std': 0.0,
                'ssim_mean': 0.0, 'ssim_std': 0.0}

    psnr_arr = np.array(psnr_vals)
    ssim_arr = np.array(ssim_vals)
    return {
        'pair_id': pair_id,
        'frames': len(psnr_vals),
        'psnr_mean': float(psnr_arr.mean()),
        'psnr_std':  float(psnr_arr.std()),
        'ssim_mean': float(ssim_arr.mean()),
        'ssim_std':  float(ssim_arr.std()),
    }


def print_summary(results, csv_path, elapsed):
    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
    table.add_column("Pair",      style="bold dim")
    table.add_column("Frames",    justify="right")
    table.add_column("PSNR mean", justify="right")
    table.add_column("PSNR std",  justify="right", style="dim")
    table.add_column("SSIM mean", justify="right")
    table.add_column("SSIM std",  justify="right", style="dim")

    for r in results:
        table.add_row(
            str(r['pair_id']),
            str(r['frames']),
            f"{r['psnr_mean']:.2f}",
            f"{r['psnr_std']:.2f}",
            f"{r['ssim_mean']:.4f}",
            f"{r['ssim_std']:.4f}",
        )

    if results:
        total_frames = sum(r['frames'] for r in results)
        psnr_means = [r['psnr_mean'] for r in results]
        ssim_means = [r['ssim_mean'] for r in results]
        table.add_section()
        table.add_row(
            "[bold]Overall[/bold]",
            f"[bold]{total_frames}[/bold]",
            f"[bold]{np.mean(psnr_means):.2f}[/bold]",
            f"[bold]{np.std(psnr_means):.2f}[/bold]",
            f"[bold]{np.mean(ssim_means):.4f}[/bold]",
            f"[bold]{np.std(ssim_means):.4f}[/bold]",
        )

    console.print()
    console.print(Panel(table,
                        title="[bold cyan]Evaluation Results[/bold cyan]",
                        border_style="cyan", padding=(0, 1)))

    meta = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    meta.add_column(style="bold dim")
    meta.add_column()
    meta.add_row("Time",     f"{elapsed:.1f}s")
    meta.add_row("CSV",      str(csv_path))
    console.print(Panel(meta, border_style="green", padding=(0, 1)))
    console.print()


def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate DIBR pipeline against ground-truth stereo pairs.')
    p.add_argument('--input', type=str, default='test_videos',
                   help='Directory of cam0_*/cam1_* pairs (default: test_videos)')
    p.add_argument('--output', type=str, default='results',
                   help='Output directory for CSV (default: results)')
    p.add_argument('--model', choices=['small', 'base', 'large'], default='base')
    p.add_argument('--ipd', type=float, default=12.0)
    p.add_argument('--smooth-alpha', type=float, default=0.8)
    p.add_argument('--inpaint-method', choices=['telea', 'ns'], default='telea')
    p.add_argument('--inpaint-radius', type=int, default=3)
    p.add_argument('--gpu', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    console.clear()
    title = Text()
    title.append("3D DIBR", style="bold cyan")
    title.append("  -  Pipeline Evaluation", style="dim")
    console.print(Panel(title, border_style="cyan", padding=(1, 4)))
    console.print()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        console.print(f"[red]Input directory not found: {input_dir}[/red]")
        sys.exit(1)

    pairs = discover_pairs(input_dir)
    if not pairs:
        console.print(f"[red]No cam0/cam1 pairs found in {input_dir}[/red]")
        sys.exit(1)

    # Settings preview
    settings = Table(box=None, show_header=False, padding=(0, 2))
    settings.add_column(style="bold dim", width=16)
    settings.add_column()
    settings.add_row("Input",      str(input_dir))
    settings.add_row("Pairs",      str(len(pairs)))
    settings.add_row("Model",      f"DepthAnything-V2-{args.model}")
    settings.add_row("IPD",        f"{args.ipd} mm")
    settings.add_row("Smoothing",  f"alpha={args.smooth_alpha}")
    settings.add_row("Inpaint",    f"{args.inpaint_method}  radius={args.inpaint_radius}")
    settings.add_row("Device",     "CPU" if args.gpu == -1 else f"cuda:{args.gpu}")
    console.print(Panel(settings, title="[bold]Settings[/bold]",
                        border_style="cyan dim", padding=(0, 1)))
    console.print()

    # Load pipeline with immediate spinner
    with console.status("", spinner="dots") as status:
        status.update("[bold]Importing modules...[/bold]")
        with _mute():
            from src.depth_estimator import DepthEstimator
            from src.inpainter import Inpainter
            from src.stereo_generator import StereoGenerator

        status.update(f"[bold]Loading DepthAnything-V2-{args.model}...[/bold]")
        with _mute():
            de = DepthEstimator(model_size=args.model, device=args.gpu)
            sg = StereoGenerator(ipd_mm=args.ipd)
            ip = Inpainter(method=args.inpaint_method, radius=args.inpaint_radius)

    console.print("[green]Pipeline loaded.[/green]")
    console.print()

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'evaluation.csv'

    results = []
    t0 = time.time()

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=['pair_id', 'frame_idx', 'psnr', 'ssim'])
        writer.writeheader()

        for idx, (ts, cam0_path, cam1_path) in enumerate(pairs, 1):
            console.print(f"[bold]({idx}/{len(pairs)}) pair {ts}[/bold]  "
                          f"[dim]{cam0_path.name} vs {cam1_path.name}[/dim]")
            try:
                r = evaluate_pair(cam0_path, cam1_path, de, sg, ip,
                                  args.smooth_alpha, writer, ts)
                console.print(f"  [green]Done[/green]  "
                              f"PSNR={r['psnr_mean']:.2f}  "
                              f"SSIM={r['ssim_mean']:.4f}")
                results.append(r)
            except Exception as e:
                console.print(f"  [red]x {ts}: {e}[/red]")

    print_summary(results, csv_path, time.time() - t0)


if __name__ == '__main__':
    main()
