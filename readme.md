# 3D DIBR — 2D to 3D Conversion

Converts 2D images and videos into stereoscopic 3D using monocular depth estimation and Depth Image-Based Rendering (DIBR).

---

## Pipeline

### 1. Depth Estimation
Uses **DepthAnything-V2** (via HuggingFace Transformers) to predict a per-pixel depth map from a single 2D image.

- Three model sizes: `small` (fastest), `base` (default), `large` (best quality)
- Median blur (3×3) applied to smooth noise without blurring edges
- Depth normalized to [0, 1]; note: lower values = farther from camera
- GPU (CUDA) or CPU selectable

### 2. Stereo Generation (DIBR)
Synthesizes a right-eye view by shifting pixels horizontally according to their depth.

- Disparity = `(ipd / 65mm) × (5% of image width) × depth` — scales shift with IPD relative to average human eye separation
- Vectorized forward-warping with a NumPy Z-buffer: pixels sorted far→near so foreground objects correctly occlude background
- Occlusion mask tracks pixels not covered by any warped source pixel

### 3. Inpainting
Fills occlusion holes left by the warping step using OpenCV.

- **Telea** (default) — fast, edge-preserving, suitable for small holes
- **NS** (Navier-Stokes) — slower, better for larger hole regions

### 4. Output Formats
- **right** — right-eye view only
- **anaglyph** — red-cyan composite for standard 3D glasses
- **sbs** — side-by-side layout for VR headsets and 3D monitors

### 5. Temporal Smoothing (video only)
Blends consecutive depth maps with an exponential moving average to reduce flicker and jitter between frames.

---

## Installation

```bash
pip install -r requirements.txt
```

> NumPy must be `< 2.0` due to a Transformers compatibility constraint — the pinned version in `requirements.txt` handles this.

---

## Usage

### Interactive CLI (recommended)

```bash
python cli.py
```

Launches a fully interactive session. Use arrow keys to navigate menus, Enter to confirm, and type `exit` at any prompt or press `Ctrl+C` to quit.

**Flow:**

1. Choose **Images** or **Videos**
2. Enter input path (file or directory)
3. Choose output directory, model size, output format, and IPD
4. *(Videos only)* Set temporal smoothing alpha
5. Optionally configure advanced options (inpaint method, radius, GPU)
6. Review the settings summary and confirm
7. Results are saved; the loop returns to step 1 automatically

**Settings reference:**

| Setting | Default | Description |
|---|---|---|
| Model size | `base` | DepthAnything-V2 size: `small`, `base`, `large` |
| IPD | `12.0 mm` | Interpupillary distance — controls 3D intensity |
| Output format | `right` | `right` (right-eye), `anaglyph` (red-cyan), `sbs` (side-by-side) |
| Smooth alpha | `0.8` | *(videos)* Temporal depth blend — higher = more current frame weight |
| Inpaint method | `telea` | `telea` (fast) or `ns` (Navier-Stokes, better for large holes) |
| Inpaint radius | `3` | Inpainting neighbourhood radius in pixels |
| GPU device | `0` | CUDA device ID; `-1` for CPU |

Loaded models are cached across runs — switching between image and video jobs with the same model/GPU won't reload the weights.

---

### Direct scripts

#### `image_pair.py`

```bash
python image_pair.py --input <file_or_dir> [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--input` | *(required)* | Image file or directory |
| `--output` | `output` | Output directory |
| `--ipd` | `12.0` | Interpupillary distance in mm |
| `--depth-model` | `base` | Model size: `small`, `base`, `large` |
| `--inpaint-method` | `telea` | `telea` or `ns` |
| `--inpaint-radius` | `3` | Inpainting radius in pixels |
| `--gpu` | `0` | GPU device ID, `-1` for CPU |

Outputs: `output/stereo/{name}_right.png` and `output/depth/{name}_depth.png`

#### `video_processor.py`

```bash
python video_processor.py --input <file_or_dir> [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--input` | `input/videos` | Video file or directory |
| `--output` | `output/video_pair` | Output directory |
| `--ipd` | `12.0` | Interpupillary distance in mm |
| `--depth-model` | `base` | Model size: `small`, `base`, `large` |
| `--inpaint-method` | `telea` | `telea` or `ns` |
| `--inpaint-radius` | `3` | Inpainting radius in pixels |
| `--smooth-alpha` | `0.8` | Temporal smoothing factor |
| `--output-format` | `right` | `right`, `anaglyph`, or `sbs` |
| `--gpu` | `0` | GPU device ID, `-1` for CPU |

Output: `output/video_pair/{name}_right.mp4` (or `_anaglyph`, `_sbs`)

---

## Output structure

```
output/
├── stereo/       # Stereo images (right view, anaglyph, or SBS)
├── depth/        # Colored depth map visualizations
└── video_pair/   # Stereo video output
```
