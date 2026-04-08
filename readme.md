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

### CLI (recommended)

```bash
python cli.py [COMMAND] [OPTIONS] INPUT
```

#### Image conversion

```bash
# Single image, default settings
python cli.py image input/images/dog1.jpg

# Directory, large model, anaglyph output
python cli.py image input/images/ --model large --format anaglyph

# Custom IPD, CPU only
python cli.py image input/images/ --ipd 18 --gpu -1
```

#### Video conversion

```bash
# Single video, default settings
python cli.py video input/videos/dog1.mp4

# Directory, side-by-side output, less temporal smoothing
python cli.py video input/videos/ --format sbs --smooth-alpha 0.5

# Anaglyph, large model, specific output directory
python cli.py video input/videos/hike.mp4 --format anaglyph --model large --output output/hike
```

#### CLI options

| Option | Default | Description |
|---|---|---|
| `--model` | `base` | DepthAnything-V2 size: `small`, `base`, `large` |
| `--ipd` | `12.0` | Interpupillary distance in mm |
| `--format` | `right` | Output format: `right`, `anaglyph`, `sbs` |
| `--inpaint-method` | `telea` | Inpainting algorithm: `telea`, `ns` |
| `--inpaint-radius` | `3` | Inpainting radius in pixels |
| `--gpu` | `0` | GPU device ID; `-1` for CPU |
| `--output` | `output` | Output directory |
| `--smooth-alpha` | `0.8` | *(video only)* Temporal smoothing factor (0–1) |

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
