<a id="readme-top"></a>

<h1 align="center">smart-thumbnailer</h1>

<p align="center">
  Extract the best thumbnail frames from a video — no cloud, no ML runtime, just OpenCV and NumPy.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/deps-opencv%20%2B%20numpy-brightgreen?style=flat-square" alt="deps: opencv + numpy">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square" alt="MIT">
</p>

---

Naive keyframe extraction tends to dump six nearly identical frames from the same shot. `smart-thumbnailer` scores every candidate on colorfulness, sharpness, saliency, and face area, then picks one winner per time zone — so the output actually covers the full video instead of one lucky second.

The face detector (~2 MB) downloads itself on the first run. No GPU, no PyTorch, no cloud API.

## Features

- Zone-based diversity: splits the timeline into N equal zones, picks the top scorer from each; diversity check uses Lab Bhattacharyya distance so selected frames are perceptually distinct
- Multi-metric scoring: colorfulness (highest weight), Tenengrad sharpness, RMS contrast, MSCN naturalness, clipping-aware exposure, spectral residual saliency, pixel-level stability, face area and position
- dHash deduplication before full scoring (Hamming distance ≤ 12/64 bits)
- Auto thumbnail count that scales with video duration (2 for short clips, up to 10 for feature-length)
- PySceneDetect integration merges detected cut points with uniform samples when installed
- Rotation correction via `cv2.CAP_PROP_ORIENTATION_META` for phone and action camera footage
- Self-contained HTML report with base64-embedded JPEGs and per-frame metric bars

## Installation

```bash
pip install opencv-python numpy
```

For scene-aware sampling (optional):

```bash
pip install "scenedetect[opencv]"
```

Clone the repo (single file, no package install needed):

```bash
git clone https://github.com/fralapo/smart-thumbnailer.git
cd smart-thumbnailer
```

The res10 face detection model (~2 MB) downloads on first use.

## Quick start

```bash
python thumbnailer.py video.mp4
```

Output lands in `thumbnails/video/`:

```
thumbnails/
└── video/
    ├── thumb_1_04m25s.jpg
    ├── thumb_2_07m15s.jpg
    ├── ...
    ├── report.json
    └── preview.html
```

Open `preview.html` in a browser to compare all candidates side by side with metric bars.

## Usage

```
python thumbnailer.py [inputs...] [options]
```

| Flag | Default | Description |
|---|---|---|
| `-o, --output` | `thumbnails/` | Output root directory |
| `-k, --top-k` | auto | Thumbnail count (overrides auto-scale) |
| `-s, --sample-interval` | `5.0` | Seconds between uniform samples |
| `--skip-pct` | `0.05` | Skip this fraction at the start and end |
| `--jpeg-quality` | `95` | JPEG quality (1–100) |
| `--no-faces` | off | Skip face detection (faster) |
| `--no-scene-detect` | off | Skip PySceneDetect |

### Single file

```bash
# Auto count (13-min video → 6 thumbnails)
python thumbnailer.py talk.mp4

# Force 3 thumbnails into a custom dir
python thumbnailer.py talk.mp4 -o out/ -k 3

# Skip both optional detectors for speed
python thumbnailer.py talk.mp4 --no-faces --no-scene-detect

# Sample more densely (every 2 s instead of 5 s)
python thumbnailer.py talk.mp4 -s 2
```

### Batch processing

```bash
# All videos in the same directory as the script
python thumbnailer.py

# All videos inside a specific folder
python thumbnailer.py /path/to/videos/

# All .mp4 files in the script directory
python thumbnailer.py .mp4

# All .mp4 and .mkv in a specific folder
python thumbnailer.py /path/to/videos/ .mp4 .mkv

# Multiple explicit files
python thumbnailer.py clip1.mp4 clip2.mkv interview.mov
```

Each video gets its own subfolder under `thumbnails/`:

```
thumbnails/
├── clip1/
│   ├── thumb_1_01m05s.jpg
│   ├── report.json
│   └── preview.html
├── clip2/
│   └── ...
└── interview/
    └── ...
```

When processing multiple files, a summary prints at the end:

```
========================================================
  Done: 3/3 succeeded
========================================================
```

## Auto thumbnail count

When `-k` is not set, count scales with duration:

| Duration | Thumbnails |
|---|---|
| < 3 min | 2 |
| 3–8 min | 4 |
| 8–20 min | 6 |
| 20–40 min | 8 |
| 40–70 min | 9 |
| 70–120 min | 10 |
| > 120 min | 10 (cap) |

## How it works

```
video
  └─ uniform samples every Ns  +  PySceneDetect cuts (optional)
       └─ fast filter: drop dark (<22), overexposed (>233), blurry frames
            └─ dHash dedup: remove near-identical frames
                 └─ full scoring per frame:
                 │   colorfulness   (Hasler & Süsstrunk 2003 + HSV hue entropy)
                 │   sharpness      (Tenengrad, center-weighted Sobel gradient)
                 │   contrast       (RMS)
                 │   naturalness    (MSCN kurtosis — catches compression artifacts)
                 │   exposure       (clipping-aware — penalises blown highlights)
                 │   saliency       (spectral residual, Hou & Zhang CVPR 2007)
                 │   stability      (pixel absdiff to temporal neighbors)
                 │   faces          (res10 SSD, conf ≥ 0.70, scored by area + position)
                 └─ zone selection: K zones, top scorer per zone (Lab Bhattacharyya diversity check)
                      └─ output: N × JPG  +  report.json  +  preview.html
```

Faces apply as a score multiplier (×1.30) rather than an additive weight. A frame with a readable face in the upper half of the shot will always beat a comparable faceless frame, regardless of other scores. This follows how YouTube's own patents treat face detection.

Scoring weights (all tunable in `thumbnailer.py`):

| Metric | Weight |
|---|---|
| Colorfulness | 25% |
| Sharpness | 20% |
| Contrast | 15% |
| Exposure | 12% |
| Naturalness | 10% |
| Saliency | 10% |
| Stability | 8% |
| Face | ×1.30 boost |

Colorfulness is the highest-weighted metric — counterintuitive, but that is what the CQE study (Panetta et al., IEEE TCE 2013) found when correlating image quality metrics with human judgments.

## Output files

`report.json` has video metadata, per-thumbnail scores, scoring weights, and whether face detection ran.

`preview.html` is fully self-contained. When face detection is disabled, the Faces row shows N/A rather than a misleading zero.

## Requirements

- Python 3.9+
- `opencv-python >= 4.5`
- `numpy >= 1.20`
- `scenedetect[opencv]` — optional, for scene-aware sampling

## License

MIT

<p align="right">(<a href="#readme-top">back to top</a>)</p>
