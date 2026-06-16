#!/usr/bin/env python3
"""
smart-thumbnailer: Extract 3 optimal thumbnail frames from a video.

Usage:
    python thumbnailer.py video.mp4
    python thumbnailer.py video.mp4 -o out/ -k 3 -s 5
    python thumbnailer.py video.mp4 --no-faces
    python thumbnailer.py video.mp4 -k 5 -s 2
"""

import argparse
import base64
import json
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Face detection model files (OpenCV DNN, ResNet-10 SSD)
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent / "models"
PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
CAFFEMODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/"
    "dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)
PROTOTXT_PATH = MODELS_DIR / "deploy.prototxt"
CAFFEMODEL_PATH = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

# ---------------------------------------------------------------------------
# Scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------

W_SHARPNESS = 0.35
W_CONTRAST = 0.20
W_EXPOSURE = 0.15
W_FACES = 0.20
W_COLORFULNESS = 0.10

# Fast-filter thresholds (applied before full scoring)
MIN_BRIGHTNESS = 22.0
MAX_BRIGHTNESS = 233.0
MIN_SHARPNESS_RAW = 15.0   # Laplacian variance below this -> blurry


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FrameCandidate:
    time_sec: float
    frame: np.ndarray
    sharpness_raw: float = 0.0
    contrast_raw: float = 0.0
    exposure_score: float = 0.0
    face_score: float = 0.0
    colorfulness_raw: float = 0.0
    # filled after normalization
    sharpness: float = 0.0
    contrast: float = 0.0
    colorfulness: float = 0.0
    score: float = 0.0


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def laplacian_sharpness(gray: np.ndarray) -> float:
    """Variance of Laplacian — standard blur/sharpness detector."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def colorfulness_hs(bgr: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colorfulness metric."""
    B, G, R = cv2.split(bgr.astype("float32"))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    mu_rg, sigma_rg = float(np.mean(rg)), float(np.std(rg))
    mu_yb, sigma_yb = float(np.mean(yb)), float(np.std(yb))
    sigma_rgyb = np.sqrt(sigma_rg**2 + sigma_yb**2)
    mu_rgyb = np.sqrt(mu_rg**2 + mu_yb**2)
    return sigma_rgyb + 0.3 * mu_rgyb


def exposure_score(brightness: float) -> float:
    """
    Score 0–100 based on distance from ideal brightness window [90, 170].
    Frames outside [20, 233] are already filtered before this is called.
    """
    lo, hi = 90.0, 170.0
    if lo <= brightness <= hi:
        return 100.0
    elif brightness < lo:
        return max(0.0, 100.0 * (brightness / lo))
    else:
        return max(0.0, 100.0 * (1.0 - (brightness - hi) / (255.0 - hi)))


def face_score_from_count(n: int) -> float:
    """Non-linear mapping: 0->0, 1->70, 2->85, 3+->100."""
    return [0.0, 70.0, 85.0, 100.0][min(n, 3)]


def detect_faces(net: cv2.dnn.Net, bgr: np.ndarray, conf_thresh: float = 0.5) -> int:
    """Return number of high-confidence face detections (capped at 3)."""
    blob = cv2.dnn.blobFromImage(
        cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    count = 0
    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] > conf_thresh:
            count += 1
    return min(count, 3)


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def download_models() -> bool:
    """Download face detection model files if missing. Returns True on success."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if not PROTOTXT_PATH.exists():
            print("  Downloading deploy.prototxt...", end=" ", flush=True)
            urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)
            print("done")

        if not CAFFEMODEL_PATH.exists():
            print("  Downloading res10 caffemodel (~2MB)...", end=" ", flush=True)
            urllib.request.urlretrieve(CAFFEMODEL_URL, CAFFEMODEL_PATH)
            print("done")

        return True
    except Exception as e:
        print(f"failed ({e})")
        return False


def load_face_detector() -> Optional[cv2.dnn.Net]:
    """Load OpenCV DNN face detector. Returns None if unavailable."""
    if not download_models():
        return None
    try:
        net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(CAFFEMODEL_PATH))
        return net
    except Exception as e:
        print(f"Warning: cannot load face detector — {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------

def _normalize_array(values: List[float]) -> List[float]:
    """Normalize list to [0, 100] range."""
    vmax = max(values) if values else 1.0
    if vmax < 1e-6:
        return [0.0] * len(values)
    return [(v / vmax) * 100.0 for v in values]


def normalize_and_score(candidates: List[FrameCandidate]) -> None:
    """
    Normalize sharpness, contrast, colorfulness across the batch,
    then compute final weighted score for each candidate.
    """
    if not candidates:
        return

    sharp_norm = _normalize_array([c.sharpness_raw for c in candidates])
    contrast_norm = _normalize_array([c.contrast_raw for c in candidates])
    color_norm = _normalize_array([c.colorfulness_raw for c in candidates])

    for c, sn, cn, cfn in zip(candidates, sharp_norm, contrast_norm, color_norm):
        c.sharpness = sn
        c.contrast = cn
        c.colorfulness = cfn
        c.score = (
            W_SHARPNESS * c.sharpness
            + W_CONTRAST * c.contrast
            + W_EXPOSURE * c.exposure_score
            + W_FACES * c.face_score
            + W_COLORFULNESS * c.colorfulness
        )


# ---------------------------------------------------------------------------
# Zone-based diversity selection
# ---------------------------------------------------------------------------

def select_zone_best(
    candidates: List[FrameCandidate],
    duration: float,
    top_k: int,
    skip_start: float,
    skip_end: float,
) -> List[FrameCandidate]:
    """
    Divide usable video span into top_k equal zones.
    Pick the highest-scoring candidate from each zone.
    Falls back to global top-k if a zone is empty.
    """
    usable_start = skip_start
    usable_end = skip_end
    zone_width = (usable_end - usable_start) / top_k

    selected: List[FrameCandidate] = []
    used_times = set()

    for z in range(top_k):
        z_start = usable_start + z * zone_width
        z_end = z_start + zone_width
        zone = [c for c in candidates if z_start <= c.time_sec < z_end]

        # widen zone if empty (edge case: sparse sampling, very short zone)
        if not zone:
            margin = zone_width * 0.5
            zone = [c for c in candidates
                    if (z_start - margin) <= c.time_sec < (z_end + margin)]

        if zone:
            best = max(zone, key=lambda c: c.score)
            if best.time_sec not in used_times:
                selected.append(best)
                used_times.add(best.time_sec)

    # fill remaining slots from global top scorers (shouldn't be needed normally)
    if len(selected) < top_k:
        for c in sorted(candidates, key=lambda c: c.score, reverse=True):
            if c.time_sec not in used_times and len(selected) < top_k:
                selected.append(c)
                used_times.add(c.time_sec)

    selected.sort(key=lambda c: c.time_sec)
    return selected


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_html_preview(html_path: str, results: List[dict]) -> None:
    """Self-contained HTML preview with base64-embedded thumbnails."""
    cards = ""
    for r in results:
        img_path = r["_abs_path"]
        with open(img_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()

        m = r["metrics"]
        bar = "".join(
            f'<div class="metric"><span class="label">{k}</span>'
            f'<div class="bar-wrap"><div class="bar" style="width:{v:.0f}%"></div></div>'
            f'<span class="val">{v:.0f}</span></div>'
            for k, v in m.items()
        )
        cards += f"""
<div class="card">
  <img src="data:image/jpeg;base64,{b64}" alt="Thumb {r['rank']}">
  <div class="meta">
    <div class="header">
      <span class="rank">#{r['rank']}</span>
      <span class="ts">{r['timestamp']}</span>
      <span class="total-score">Score {r['score']:.1f}</span>
    </div>
    <div class="metrics">{bar}</div>
  </div>
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Thumbnail Candidates</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #0d0d0d;
         color: #ddd; padding: 24px; }}
  h1 {{ font-size: 1rem; color: #666; letter-spacing: .08em;
        text-transform: uppercase; margin-bottom: 20px; }}
  .grid {{ display: flex; gap: 20px; flex-wrap: wrap; }}
  .card {{ background: #1a1a1a; border-radius: 10px; overflow: hidden;
           width: 360px; box-shadow: 0 4px 20px #0006; }}
  .card img {{ width: 100%; display: block; aspect-ratio: 16/9; object-fit: cover; }}
  .meta {{ padding: 14px 16px; }}
  .header {{ display: flex; align-items: baseline; gap: 10px; margin-bottom: 12px; }}
  .rank {{ font-size: 1.1rem; font-weight: 700; color: #fff; }}
  .ts {{ font-size: 0.85rem; color: #888; }}
  .total-score {{ margin-left: auto; font-size: 0.9rem; color: #5bf; font-weight: 600; }}
  .metrics {{ display: flex; flex-direction: column; gap: 5px; }}
  .metric {{ display: flex; align-items: center; gap: 8px; font-size: 0.78rem; }}
  .label {{ width: 72px; color: #777; flex-shrink: 0; }}
  .bar-wrap {{ flex: 1; height: 4px; background: #2e2e2e; border-radius: 2px; }}
  .bar {{ height: 100%; background: #5bf; border-radius: 2px; }}
  .val {{ width: 28px; text-align: right; color: #aaa; }}
</style>
</head>
<body>
<h1>Thumbnail Candidates</h1>
<div class="grid">{cards}
</div>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_thumbnails(
    video_path: str,
    output_dir: str = "thumbnails",
    top_k: int = 3,
    sample_interval: float = 5.0,
    no_faces: bool = False,
    skip_pct: float = 0.05,
) -> List[dict]:
    """
    Full thumbnail extraction pipeline.

    Args:
        video_path:      Path to input video.
        output_dir:      Directory to write output files.
        top_k:           Number of thumbnails to extract.
        sample_interval: Seconds between sampled frames.
        no_faces:        Skip face detection (faster).
        skip_pct:        Fraction of video to skip at start/end (default 5%).

    Returns:
        List of result dicts (rank, file, timestamp, score, metrics).
    """
    # ── Video metadata ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    if duration < 1.0:
        raise RuntimeError("Video too short or unreadable.")

    skip_start = duration * skip_pct
    skip_end = duration * (1.0 - skip_pct)
    expected_samples = int((skip_end - skip_start) / sample_interval)

    mins_total, secs_total = divmod(int(duration), 60)
    print(f"Video    : {os.path.basename(video_path)}")
    print(f"Duration : {mins_total}m{secs_total:02d}s  |  FPS: {fps:.2f}  |  Frames: {total_frames}")
    print(f"Sampling : every {sample_interval}s ->~{expected_samples} candidates")
    print(f"Skip     : first {skip_start:.0f}s + last {duration - skip_end:.0f}s")

    # ── Phase 1: Load face detector ─────────────────────────────────────────
    face_net: Optional[cv2.dnn.Net] = None
    if not no_faces:
        print("\n[1/4] Loading face detector...")
        face_net = load_face_detector()
        if face_net is None:
            print("      ->face detection disabled (model unavailable)")

    # ── Phase 2: Coarse sampling + fast filter ───────────────────────────────
    print("\n[2/4] Sampling frames...")
    candidates: List[FrameCandidate] = []
    t = skip_start

    while t < skip_end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            t += sample_interval
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())

        # Fast filter — cheap checks only
        if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS:
            t += sample_interval
            continue

        sharp_raw = laplacian_sharpness(gray)
        if sharp_raw < MIN_SHARPNESS_RAW:
            t += sample_interval
            continue

        c = FrameCandidate(time_sec=t, frame=frame.copy())
        c.sharpness_raw = sharp_raw
        c.contrast_raw = float(gray.std())
        c.exposure_score = exposure_score(brightness)
        c.colorfulness_raw = colorfulness_hs(frame)
        candidates.append(c)

        t += sample_interval

    cap.release()
    print(f"      ->{len(candidates)} candidates pass fast filter")

    if not candidates:
        raise RuntimeError(
            "No valid frames found. Video may be entirely dark or unreadable."
        )

    # ── Phase 3: Face detection ──────────────────────────────────────────────
    if face_net is not None:
        print(f"\n[3/4] Face detection on {len(candidates)} frames...")
        for i, c in enumerate(candidates):
            c.face_score = face_score_from_count(detect_faces(face_net, c.frame))
            if (i + 1) % 10 == 0 or i == len(candidates) - 1:
                pct = (i + 1) / len(candidates) * 100
                print(f"\r      {i + 1}/{len(candidates)}  ({pct:.0f}%)", end="", flush=True)
        print()
    else:
        print("\n[3/4] Face detection: skipped")

    # ── Phase 4: Normalize + score ───────────────────────────────────────────
    print("\n[4/4] Scoring and selecting...")
    normalize_and_score(candidates)

    selected = select_zone_best(candidates, duration, top_k, skip_start, skip_end)

    if not selected:
        raise RuntimeError("Zone selection returned no frames.")

    # ── Output ───────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    results: List[dict] = []

    for rank, item in enumerate(selected, 1):
        mins, secs = divmod(int(item.time_sec), 60)
        fname = f"thumb_{rank}_{mins:02d}m{secs:02d}s.jpg"
        abs_path = os.path.join(output_dir, fname)
        cv2.imwrite(abs_path, item.frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        result = {
            "rank": rank,
            "file": fname,
            "_abs_path": abs_path,
            "time_sec": round(item.time_sec, 2),
            "timestamp": f"{mins:02d}:{secs:02d}",
            "score": round(item.score, 2),
            "metrics": {
                "sharpness": round(item.sharpness, 1),
                "contrast": round(item.contrast, 1),
                "exposure": round(item.exposure_score, 1),
                "faces": round(item.face_score, 1),
                "colorfulness": round(item.colorfulness, 1),
            },
        }
        results.append(result)
        print(
            f"  #{rank}  {fname}  "
            f"score={item.score:.1f}  "
            f"sharp={item.sharpness:.0f}  "
            f"faces={item.face_score:.0f}  "
            f"@ {mins:02d}:{secs:02d}"
        )

    # JSON report (strip internal _abs_path key)
    report_path = os.path.join(output_dir, "report.json")
    report_data = {
        "video": os.path.abspath(video_path),
        "duration_sec": round(duration, 2),
        "sample_interval_sec": sample_interval,
        "candidates_evaluated": len(candidates),
        "thumbnails": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report_data, fh, indent=2)

    # HTML preview
    html_path = os.path.join(output_dir, "preview.html")
    write_html_preview(html_path, results)

    print(f"\nSaved to: {os.path.abspath(output_dir)}/")
    print(f"  {len(selected)}x JPG  +  report.json  +  preview.html")

    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract optimal thumbnail frames from a video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python thumbnailer.py talk.mp4
  python thumbnailer.py talk.mp4 -o out/ -k 5
  python thumbnailer.py talk.mp4 -s 3 --no-faces
        """,
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "-o", "--output", default="thumbnails",
        help="Output directory (default: thumbnails/)",
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=3,
        help="Number of thumbnails to extract (default: 3)",
    )
    parser.add_argument(
        "-s", "--sample-interval", type=float, default=5.0,
        help="Seconds between sampled frames (default: 5.0). "
             "Lower = more candidates, slower. Recommended: 2–10.",
    )
    parser.add_argument(
        "--no-faces", action="store_true",
        help="Skip face detection (faster, no model download)",
    )
    parser.add_argument(
        "--skip-pct", type=float, default=0.05,
        help="Fraction of video to skip at start and end (default: 0.05 = 5%%)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.top_k < 1:
        print("Error: --top-k must be >= 1", file=sys.stderr)
        sys.exit(1)

    if args.sample_interval < 0.5:
        print("Error: --sample-interval must be >= 0.5", file=sys.stderr)
        sys.exit(1)

    print("=" * 56)
    print("  smart-thumbnailer")
    print("=" * 56)

    extract_thumbnails(
        video_path=args.video,
        output_dir=args.output,
        top_k=args.top_k,
        sample_interval=args.sample_interval,
        no_faces=args.no_faces,
        skip_pct=args.skip_pct,
    )


if __name__ == "__main__":
    main()
