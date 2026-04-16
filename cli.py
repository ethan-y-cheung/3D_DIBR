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

import contextlib
import time
from pathlib import Path
from typing import Optional

import questionary
from questionary import Style as QStyle
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                            SpinnerColumn, TextColumn, TimeElapsedColumn,
                            TimeRemainingColumn)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console()

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}

# ── Questionary colour theme to match Rich cyan palette ──────────────────────
Q = QStyle([
    ('qmark',       'fg:#00d7ff bold'),
    ('question',    'bold'),
    ('answer',      'fg:#00d7ff bold'),
    ('pointer',     'fg:#00d7ff bold'),
    ('highlighted', 'fg:#00d7ff bold'),
    ('selected',    'fg:#00ff87'),
    ('instruction', 'fg:#555555'),
    ('text',        ''),
    ('disabled',    'fg:#555555 italic'),
])


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _vtext(s: str) -> bool:
    """Validate float string."""
    try:
        float(s)
        return True
    except ValueError:
        return "Enter a number."


def _vint(s: str) -> bool:
    """Validate integer string (including negative)."""
    try:
        int(s)
        return True
    except ValueError:
        return "Enter an integer."


def _vpath(s: str) -> bool:
    """Validate that path exists."""
    return Path(s.strip()).exists() or "Path does not exist."


# ── Display ───────────────────────────────────────────────────────────────────

def _banner():
    console.clear()
    title = Text()
    title.append("3D DIBR", style="bold cyan")
    title.append("  —  2D to 3D Stereo Conversion", style="dim")
    subtitle = Text("Type  ", style="dim")
    subtitle.append("exit", style="bold red")
    subtitle.append("  at any prompt, or press  ", style="dim")
    subtitle.append("Ctrl+C", style="bold red")
    subtitle.append("  to quit.", style="dim")
    console.print(Panel(
        f"{title}\n{subtitle}",
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()


def _section(title: str):
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan dim"))
    console.print()


def _config_summary(cfg: dict):
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column(style="bold dim", width=20)
    table.add_column(style="white")

    table.add_row("Mode",     cfg['mode'])
    table.add_row("Input",    str(cfg['input']))
    table.add_row("Output",   str(cfg['output']))
    table.add_row("Model",    f"DepthAnything-V2-{cfg['model']}")
    table.add_row("IPD",      f"{cfg['ipd']} mm")
    table.add_row("Format",   cfg['format'])
    if cfg['mode'] == 'Videos':
        table.add_row("Smoothing", f"alpha={cfg['smooth_alpha']}")
    table.add_row("Inpaint",  f"{cfg['inpaint_method']}  radius={cfg['inpaint_radius']}")
    table.add_row("Device",   "CPU" if cfg['gpu'] == -1 else f"cuda:{cfg['gpu']}")

    console.print(Panel(table, title="[bold]Ready to process[/bold]",
                        border_style="cyan dim", padding=(0, 1)))
    console.print()


def _result_summary(ok: int, fail: int, elapsed: float, output_dir: Path):
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column(style="bold dim")
    table.add_column()
    status = f"[green]{ok} completed[/green]"
    if fail:
        status += f"  [red]{fail} failed[/red]"
    table.add_row("Result", status)
    table.add_row("Time",   f"{elapsed:.1f}s")
    table.add_row("Output", str(output_dir))
    console.print(Panel(table,
                        border_style="green" if not fail else "yellow",
                        padding=(0, 1)))
    console.print()


# ── Interactive prompts ───────────────────────────────────────────────────────

def _ask_mode() -> Optional[str]:
    console.print()
    answer = questionary.select(
        "What would you like to do?",
        choices=[
            questionary.Choice("Images  — convert images to stereoscopic 3D", value="Images"),
            questionary.Choice("Videos  — convert videos to stereoscopic 3D", value="Videos"),
            questionary.Separator(),
            questionary.Choice("Exit", value="exit"),
        ],
        style=Q,
    ).ask()
    return answer


def _ask_config(mode: str) -> Optional[dict]:
    is_video = mode == 'Videos'
    console.print()
    _section(f"{mode} — Settings")

    # ── Input path ──
    default_input = 'input/videos/' if is_video else 'input/images/'
    raw = questionary.text(
        "Input path:",
        default=default_input,
        validate=_vpath,
        style=Q,
    ).ask()
    if raw is None or raw.strip().lower() == 'exit':
        return None
    input_path = Path(raw.strip())

    # ── Output directory ──
    default_out = 'output/video_pair' if is_video else 'output'
    raw = questionary.text(
        "Output directory:",
        default=default_out,
        style=Q,
    ).ask()
    if raw is None or raw.strip().lower() == 'exit':
        return None
    output_path = Path(raw.strip())

    # ── Model ──
    model = questionary.select(
        "Model size:",
        choices=[
            questionary.Choice("small  — fastest, lower quality",   value="small"),
            questionary.Choice("base   — balanced  (default)",       value="base"),
            questionary.Choice("large  — slowest, best quality",     value="large"),
        ],
        default="base",
        style=Q,
    ).ask()
    if model is None:
        return None

    # ── Output format ──
    fmt = questionary.select(
        "Output format:",
        choices=[
            questionary.Choice("right     — right-eye view",          value="right"),
            questionary.Choice("anaglyph  — red-cyan (3D glasses)",   value="anaglyph"),
            questionary.Choice("sbs       — side-by-side",            value="sbs"),
        ],
        style=Q,
    ).ask()
    if fmt is None:
        return None

    # ── IPD ──
    raw = questionary.text(
        "Interpupillary distance (mm):",
        default="12.0",
        validate=_vtext,
        style=Q,
    ).ask()
    if raw is None or raw.strip().lower() == 'exit':
        return None
    ipd = float(raw)

    cfg = {
        'mode': mode,
        'input': input_path,
        'output': output_path,
        'model': model,
        'format': fmt,
        'ipd': ipd,
        'smooth_alpha': 0.8,
        'inpaint_method': 'telea',
        'inpaint_radius': 3,
        'gpu': 0,
    }

    # ── Video-only: temporal smoothing ──
    if is_video:
        raw = questionary.text(
            "Temporal smoothing alpha (0-1):",
            default="0.8",
            validate=lambda s: (_vtext(s) if _vtext(s) is True
                                else _vtext(s)) if _vtext(s) is not True
                                else (True if 0.0 <= float(s) <= 1.0 else "Must be between 0 and 1."),
            style=Q,
        ).ask()
        if raw is None or raw.strip().lower() == 'exit':
            return None
        cfg['smooth_alpha'] = float(raw)

    # ── Advanced options ──
    console.print()
    show_adv = questionary.confirm(
        "Configure advanced options?",
        default=False,
        style=Q,
    ).ask()

    if show_adv is None:
        return None

    if show_adv:
        console.print()
        method = questionary.select(
            "Inpainting method:",
            choices=[
                questionary.Choice("telea  — fast, edge-preserving (default)",       value="telea"),
                questionary.Choice("ns     — Navier-Stokes, better for large holes", value="ns"),
            ],
            style=Q,
        ).ask()
        if method is None:
            return None
        cfg['inpaint_method'] = method

        raw = questionary.text(
            "Inpaint radius (px):",
            default="3",
            validate=lambda s: s.isdigit() or "Enter a positive integer.",
            style=Q,
        ).ask()
        if raw is None or raw.strip().lower() == 'exit':
            return None
        cfg['inpaint_radius'] = int(raw)

        raw = questionary.text(
            "GPU device ID (-1 for CPU):",
            default="0",
            validate=_vint,
            style=Q,
        ).ask()
        if raw is None or raw.strip().lower() == 'exit':
            return None
        cfg['gpu'] = int(raw)

    return cfg


# ── Pipeline ──────────────────────────────────────────────────────────────────

_MODEL_CACHE: dict = {}


def _load_pipeline(cfg: dict):
    cache_key = (cfg['model'], cfg['gpu'])

    with console.status("", spinner="dots") as status:
        if cache_key not in _MODEL_CACHE:
            status.update(f"[bold]Importing modules...[/bold]")
            with _mute():
                from src.depth_estimator import DepthEstimator
                from src.inpainter import Inpainter
                from src.stereo_generator import StereoGenerator

            status.update(f"[bold]Loading DepthAnything-V2-{cfg['model']}...[/bold]")
            with _mute():
                de = DepthEstimator(model_size=cfg['model'], device=cfg['gpu'])
            _MODEL_CACHE[cache_key] = de
            status.stop()
            console.print("[green]Model loaded.[/green]")
        else:
            with _mute():
                from src.inpainter import Inpainter
                from src.stereo_generator import StereoGenerator
            status.stop()
            console.print("[dim]Reusing cached model.[/dim]")
            de = _MODEL_CACHE[cache_key]

        with _mute():
            sg = StereoGenerator(ipd_mm=cfg['ipd'])
            ip = Inpainter(method=cfg['inpaint_method'], radius=cfg['inpaint_radius'])

    return de, sg, ip


# ── Processing ────────────────────────────────────────────────────────────────

def _run_images(cfg: dict):
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    de, sg, ip = _load_pipeline(cfg)
    console.print()

    input_path: Path = cfg['input']
    output_dir: Path = cfg['output']
    suffix = {'right': '_right', 'anaglyph': '_anaglyph', 'sbs': '_sbs'}[cfg['format']]

    files = ([input_path] if input_path.is_file()
             else sorted(f for f in input_path.iterdir()
                         if f.suffix.lower() in IMAGE_EXTENSIONS))
    if not files:
        console.print(f"[red]No images found in {input_path}[/red]")
        return

    (output_dir / 'stereo').mkdir(parents=True, exist_ok=True)
    (output_dir / 'depth').mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    t0 = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.fields[fname]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("[dim]eta[/dim]"),
        TimeRemainingColumn(),
        console=console,
    ) as bar:
        task = bar.add_task('', total=len(files), fname='')

        for f in files:
            bar.update(task, fname=f.name)
            try:
                img = np.array(PILImage.open(f).convert('RGB'))
                depth = de.estimate(img)
                left, right, lm, rm = sg.generate_stereo_pair(img, depth)
                left_final, right_final = ip.inpaint_stereo_pair(left, right, lm, rm)

                depth_vis = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                PILImage.fromarray(cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)).save(
                    output_dir / 'depth' / f'{f.stem}_depth.png')

                if cfg['format'] == 'anaglyph':
                    out_img = sg.create_anaglyph(left_final, right_final)
                elif cfg['format'] == 'sbs':
                    out_img = sg.create_side_by_side(left_final, right_final)
                else:
                    out_img = right_final

                PILImage.fromarray(out_img).save(
                    output_dir / 'stereo' / f'{f.stem}{suffix}.png')
                ok += 1
            except Exception as e:
                console.print(f"  [red]x {f.name}: {e}[/red]")
                fail += 1
            finally:
                bar.advance(task)

    _result_summary(ok, fail, time.time() - t0, output_dir)


def _run_videos(cfg: dict):
    import cv2
    import numpy as np

    de, sg, ip = _load_pipeline(cfg)
    console.print()

    input_path: Path = cfg['input']
    output_dir: Path = cfg['output']
    suffix = {'right': '_right', 'anaglyph': '_anaglyph', 'sbs': '_sbs'}[cfg['format']]
    smooth_alpha = cfg['smooth_alpha']

    files = ([input_path] if input_path.is_file()
             else sorted(f for f in input_path.iterdir()
                         if f.suffix.lower() in VIDEO_EXTENSIONS))
    if not files:
        console.print(f"[red]No videos found in {input_path}[/red]")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    t0 = time.time()

    for idx, f in enumerate(files, 1):
        console.print(f"[bold]({idx}/{len(files)}) {f.name}[/bold]")
        try:
            cap = cv2.VideoCapture(str(f))
            if not cap.isOpened():
                raise ValueError(f"Cannot open {f}")

            fps   = cap.get(cv2.CAP_PROP_FPS)
            w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = output_dir / f'{f.stem}{suffix}.mp4'
            out_w    = w * 2 if cfg['format'] == 'sbs' else w
            writer   = cv2.VideoWriter(str(out_path),
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps, (out_w, h))
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

                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    depth = de.estimate(frame_rgb)

                    if prev_depth is not None:
                        depth = (smooth_alpha * depth
                                 + (1 - smooth_alpha) * prev_depth).astype(np.float32)
                    prev_depth = depth

                    left, right, lm, rm = sg.generate_stereo_pair(frame_rgb, depth)
                    _, right = ip.inpaint_stereo_pair(left, right, lm, rm)

                    if cfg['format'] == 'anaglyph':
                        out_frame = sg.create_anaglyph(left, right)
                    elif cfg['format'] == 'sbs':
                        out_frame = sg.create_side_by_side(left, right)
                    else:
                        out_frame = right

                    writer.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))
                    bar.advance(task)

            cap.release()
            writer.release()
            console.print(f"  [green]Done[/green]  {out_path}")
            ok += 1

        except Exception as e:
            console.print(f"  [red]x {f.name}: {e}[/red]")
            fail += 1

    _result_summary(ok, fail, time.time() - t0, output_dir)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    _banner()

    while True:
        try:
            mode = _ask_mode()

            if mode is None or mode == 'exit':
                break

            cfg = _ask_config(mode)
            if cfg is None:
                continue

            console.print()
            _config_summary(cfg)

            go = questionary.confirm("Start processing?", default=True, style=Q).ask()
            if not go:
                console.print("[dim]Skipped.[/dim]")
                continue

            console.print()
            if mode == 'Images':
                _run_images(cfg)
            else:
                _run_videos(cfg)

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")
            break

if __name__ == '__main__':
    main()
