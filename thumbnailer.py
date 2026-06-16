#!/usr/bin/env python3
"""
smart-thumbnailer: Extract optimal thumbnail frames from a video.

Pipeline:
  metadata -> [scene detect] -> coarse sampling -> fast filter
  -> histogram compute -> dHash dedup -> saliency + stability
  -> [face detect] -> normalize + score -> zone select -> output

Usage:
    python thumbnailer.py video.mp4
    python thumbnailer.py video.mp4 -o out/ -k 3 -s 5
    python thumbnailer.py video.mp4 --no-faces --no-scene-detect
    python thumbnailer.py video.mp4 -k 5 -s 2
"""

import argparse
import base64
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Optional: PySceneDetect for scene-aware sampling
# ---------------------------------------------------------------------------
try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    HAS_SCENEDETECT = True
except ImportError:
    HAS_SCENEDETECT = False

# ---------------------------------------------------------------------------
# Face detection model (OpenCV DNN, ResNet-10 SSD)
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
W_SHARPNESS    = 0.30
W_CONTRAST     = 0.15
W_EXPOSURE     = 0.12
W_FACES        = 0.20
W_COLORFULNESS = 0.08
W_SALIENCY     = 0.10
W_STABILITY    = 0.05

# Fast-filter thresholds
MIN_BRIGHTNESS   = 22.0
MAX_BRIGHTNESS   = 233.0
MIN_SHARPNESS_RAW = 15.0

# dHash near-duplicate threshold (out of 64 bits)
DHASH_MAX_HAMMING = 12


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------
@dataclass
class FrameCandidate:
    time_sec: float
    frame: np.ndarray
    # set during histogram phase
    hist: Optional[np.ndarray] = None
    # raw metrics
    sharpness_raw:    float = 0.0
    contrast_raw:     float = 0.0
    exposure_score:   float = 0.0
    face_score:       float = 0.0
    colorfulness_raw: float = 0.0
    saliency_raw:     float = 0.0
    stability_score:  float = 100.0
    # normalized (0-100)
    sharpness:    float = 0.0
    contrast:     float = 0.0
    colorfulness: float = 0.0
    saliency:     float = 0.0
    score:        float = 0.0


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def colorfulness_hs(bgr: np.ndarray) -> float:
    """Hasler & Susstrunk (2003) colorfulness metric."""
    B, G, R = cv2.split(bgr.astype("float32"))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    return float(
        np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )


def spectral_residual_saliency(gray: np.ndarray) -> float:
    """
    Hou & Zhang (CVPR 2007) spectral residual saliency.
    Pure NumPy/FFT — no contrib modules needed.
    Higher = more visually salient/interesting frame.
    """
    img = cv2.resize(gray, (64, 64)).astype("float32")
    dft = np.fft.fft2(img)
    log_amp = np.log(np.abs(dft) + 1e-6)
    smooth = cv2.boxFilter(log_amp.astype("float32"), -1, (3, 3))
    residual = log_amp - smooth
    sal_map = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(dft)))) ** 2
    sal_map = cv2.GaussianBlur(sal_map.astype("float32"), (9, 9), 2.5)
    return float(sal_map.mean())


def exposure_score(brightness: float) -> float:
    """Score 0-100 based on closeness to ideal brightness [90, 170]."""
    lo, hi = 90.0, 170.0
    if lo <= brightness <= hi:
        return 100.0
    elif brightness < lo:
        return max(0.0, 100.0 * (brightness / lo))
    else:
        return max(0.0, 100.0 * (1.0 - (brightness - hi) / (255.0 - hi)))


def face_score_from_count(n: int) -> float:
    return [0.0, 70.0, 85.0, 100.0][min(n, 3)]


def detect_faces(net: cv2.dnn.Net, bgr: np.ndarray, conf_thresh: float = 0.65) -> int:
    blob = cv2.dnn.blobFromImage(
        cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()
    return min(sum(1 for i in range(detections.shape[2])
                   if detections[0, 0, i, 2] > conf_thresh), 3)


def compute_hist(gray: np.ndarray) -> np.ndarray:
    """64-bin normalized grayscale histogram for stability/diversity checks."""
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def dhash(gray: np.ndarray, size: int = 8) -> int:
    """Difference hash (dHash) for perceptual near-duplicate detection."""
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    result = 0
    for bit in diff.flatten():
        result = (result << 1) | int(bit)
    return result


def hamming_dist(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


# ---------------------------------------------------------------------------
# Model download / load
# ---------------------------------------------------------------------------

def download_models() -> bool:
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
    if not download_models():
        return None
    try:
        return cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(CAFFEMODEL_PATH))
    except Exception as e:
        print(f"Warning: cannot load face detector — {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(candidates: List[FrameCandidate]) -> List[FrameCandidate]:
    """
    Remove perceptually near-duplicate frames using dHash.
    Keeps the first representative from each near-duplicate cluster.
    """
    kept: List[FrameCandidate] = []
    seen_hashes: List[int] = []

    for c in candidates:
        gray = cv2.cvtColor(c.frame, cv2.COLOR_BGR2GRAY)
        h = dhash(gray)
        if all(hamming_dist(h, sh) >= DHASH_MAX_HAMMING for sh in seen_hashes):
            kept.append(c)
            seen_hashes.append(h)

    return kept


# ---------------------------------------------------------------------------
# Temporal stability
# ---------------------------------------------------------------------------

def compute_stability_scores(candidates: List[FrameCandidate]) -> None:
    """
    Assign a stability score based on histogram similarity to neighboring frames.
    Transition frames (scene cuts) differ strongly from both neighbors and score low.
    """
    n = len(candidates)
    if n < 3:
        for c in candidates:
            c.stability_score = 100.0
        return

    for i, c in enumerate(candidates):
        prev = candidates[max(0, i - 1)]
        nxt  = candidates[min(n - 1, i + 1)]
        corr_p = float(cv2.compareHist(c.hist, prev.hist, cv2.HISTCMP_CORREL))
        corr_n = float(cv2.compareHist(c.hist, nxt.hist,  cv2.HISTCMP_CORREL))
        # Penalize frames that look very different from BOTH neighbors
        c.stability_score = max(0.0, min(corr_p, corr_n) * 100.0)


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------

def _normalize(values: List[float]) -> List[float]:
    vmax = max(values) if values else 1.0
    if vmax < 1e-9:
        return [0.0] * len(values)
    return [(v / vmax) * 100.0 for v in values]


def normalize_and_score(candidates: List[FrameCandidate]) -> None:
    if not candidates:
        return

    sharp_n  = _normalize([c.sharpness_raw    for c in candidates])
    cont_n   = _normalize([c.contrast_raw      for c in candidates])
    color_n  = _normalize([c.colorfulness_raw  for c in candidates])
    sal_n    = _normalize([c.saliency_raw      for c in candidates])

    for c, sn, cn, cfn, saln in zip(candidates, sharp_n, cont_n, color_n, sal_n):
        c.sharpness    = sn
        c.contrast     = cn
        c.colorfulness = cfn
        c.saliency     = saln
        c.score = (
            W_SHARPNESS    * c.sharpness
            + W_CONTRAST     * c.contrast
            + W_EXPOSURE     * c.exposure_score
            + W_FACES        * c.face_score
            + W_COLORFULNESS * c.colorfulness
            + W_SALIENCY     * c.saliency
            + W_STABILITY    * c.stability_score
        )


# ---------------------------------------------------------------------------
# Zone-based diversity selection
# ---------------------------------------------------------------------------

def _hist_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


def select_zone_best(
    candidates: List[FrameCandidate],
    top_k: int,
    skip_start: float,
    skip_end: float,
    min_visual_dist: float = 0.20,
) -> List[FrameCandidate]:
    """
    Divide usable video span into top_k equal zones.
    From each zone, pick the highest-scoring frame that is also
    visually distinct from already-selected frames (histogram distance).
    """
    zone_width = (skip_end - skip_start) / top_k
    selected: List[FrameCandidate] = []
    used_times: set = set()

    for z in range(top_k):
        z_start = skip_start + z * zone_width
        z_end   = z_start + zone_width

        zone = [c for c in candidates if z_start <= c.time_sec < z_end]
        if not zone:
            margin = zone_width * 0.5
            zone = [c for c in candidates
                    if (z_start - margin) <= c.time_sec < (z_end + margin)]
        if not zone:
            continue

        zone_sorted = sorted(zone, key=lambda c: c.score, reverse=True)

        chosen = None
        for candidate in zone_sorted:
            if candidate.time_sec in used_times:
                continue
            # Check visual diversity vs already-selected frames
            if not selected or all(
                1.0 - _hist_corr(candidate.hist, s.hist) >= min_visual_dist
                for s in selected
            ):
                chosen = candidate
                break

        # Fallback: accept highest scorer even if visually similar
        if chosen is None:
            for candidate in zone_sorted:
                if candidate.time_sec not in used_times:
                    chosen = candidate
                    break

        if chosen:
            selected.append(chosen)
            used_times.add(chosen.time_sec)

    # Fill any remaining slots from global top scorers
    if len(selected) < top_k:
        for c in sorted(candidates, key=lambda c: c.score, reverse=True):
            if c.time_sec not in used_times and len(selected) < top_k:
                selected.append(c)
                used_times.add(c.time_sec)

    selected.sort(key=lambda c: c.time_sec)
    return selected


# ---------------------------------------------------------------------------
# PySceneDetect integration (optional)
# ---------------------------------------------------------------------------

def get_scene_sample_times(
    video_path: str, skip_start: float, skip_end: float
) -> List[float]:
    """
    Use PySceneDetect to find scene boundaries and return a sample time
    1.5s into each scene (post-cut establishing shot, avoids transition blur).
    Returns [] if PySceneDetect is unavailable or fails.
    """
    if not HAS_SCENEDETECT:
        return []
    try:
        video   = open_video(video_path)
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=27.0))
        manager.detect_scenes(video, show_progress=False)
        scenes  = manager.get_scene_list()

        times = []
        for scene_start, scene_end in scenes:
            t_s = scene_start.seconds
            t_e = scene_end.seconds
            if t_e < skip_start or t_s > skip_end:
                continue
            t = max(t_s + 1.5, skip_start)
            if t < min(t_e - 0.5, skip_end):
                times.append(round(t, 3))

        return times
    except Exception as e:
        print(f"  PySceneDetect: {e}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_html_preview(html_path: str, results: List[dict]) -> None:
    """Self-contained HTML with base64-embedded thumbnails and metric bars."""
    METRIC_LABELS = {
        "sharpness":    "Sharp",
        "contrast":     "Contrast",
        "exposure":     "Exposure",
        "faces":        "Faces",
        "colorfulness": "Color",
        "saliency":     "Saliency",
        "stability":    "Stability",
    }

    cards = ""
    for r in results:
        with open(r["_abs_path"], "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()

        bars = "".join(
            f'<div class="m"><span class="ml">{METRIC_LABELS.get(k, k)}</span>'
            f'<div class="bw"><div class="b" style="width:{v:.0f}%"></div></div>'
            f'<span class="mv">{v:.0f}</span></div>'
            for k, v in r["metrics"].items()
        )
        cards += f"""
<div class="card">
  <img src="data:image/jpeg;base64,{b64}" alt="#{r['rank']}">
  <div class="meta">
    <div class="hdr">
      <span class="rank">#{r['rank']}</span>
      <span class="ts">{r['timestamp']}</span>
      <span class="sc">Score {r['score']:.1f}</span>
    </div>
    <div class="metrics">{bars}</div>
  </div>
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Thumbnail Candidates</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:system-ui,sans-serif;background:#0d0d0d;color:#ddd;padding:24px}}
  h1{{font-size:.85rem;color:#555;letter-spacing:.1em;text-transform:uppercase;margin-bottom:20px}}
  .grid{{display:flex;gap:18px;flex-wrap:wrap}}
  .card{{background:#191919;border-radius:10px;overflow:hidden;width:360px;box-shadow:0 4px 24px #0008}}
  .card img{{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}}
  .meta{{padding:12px 14px}}
  .hdr{{display:flex;align-items:baseline;gap:8px;margin-bottom:10px}}
  .rank{{font-size:1rem;font-weight:700;color:#fff}}
  .ts{{font-size:.8rem;color:#777}}
  .sc{{margin-left:auto;font-size:.85rem;color:#5bf;font-weight:600}}
  .metrics{{display:flex;flex-direction:column;gap:4px}}
  .m{{display:flex;align-items:center;gap:7px;font-size:.75rem}}
  .ml{{width:58px;color:#666;flex-shrink:0}}
  .bw{{flex:1;height:3px;background:#2a2a2a;border-radius:2px}}
  .b{{height:100%;background:linear-gradient(90deg,#36a,#5bf);border-radius:2px}}
  .mv{{width:24px;text-align:right;color:#888}}
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
    no_scene_detect: bool = False,
    skip_pct: float = 0.05,
    jpeg_quality: int = 95,
) -> List[dict]:
    """
    Full thumbnail extraction pipeline.

    Args:
        video_path:       Path to input video.
        output_dir:       Directory for output files.
        top_k:            Number of thumbnails to extract.
        sample_interval:  Seconds between uniformly sampled frames.
        no_faces:         Skip face detection.
        no_scene_detect:  Skip PySceneDetect (if available).
        skip_pct:         Fraction of video to skip at start/end.
        jpeg_quality:     JPEG output quality 1-100.
    """
    # ── Video metadata ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps if fps > 0 else 0.0

    if duration < 1.0:
        raise RuntimeError("Video too short or unreadable.")

    skip_start = duration * skip_pct
    skip_end   = duration * (1.0 - skip_pct)

    mins_t, secs_t = divmod(int(duration), 60)
    n_uniform = int((skip_end - skip_start) / sample_interval)

    print(f"Video    : {os.path.basename(video_path)}")
    print(f"Duration : {mins_t}m{secs_t:02d}s  |  FPS: {fps:.2f}  |  Frames: {total_frames}")
    print(f"Skip     : first {skip_start:.0f}s + last {duration - skip_end:.0f}s")

    # ── Phase 1: Collect candidate times ────────────────────────────────────
    uniform_times = [
        round(skip_start + i * sample_interval, 3)
        for i in range(n_uniform + 1)
        if skip_start + i * sample_interval < skip_end
    ]

    scene_times: List[float] = []
    if not no_scene_detect and HAS_SCENEDETECT:
        print("\n[1/5] Scene detection (PySceneDetect)...")
        scene_times = get_scene_sample_times(video_path, skip_start, skip_end)
        print(f"      {len(scene_times)} scene sample points")
    elif not no_scene_detect and not HAS_SCENEDETECT:
        print("\n[1/5] Scene detection: not available (pip install scenedetect[opencv])")
    else:
        print("\n[1/5] Scene detection: skipped")

    # Merge and deduplicate by time (keep unique times within 1s of each other)
    all_times = sorted(set(uniform_times + scene_times))
    merged_times: List[float] = []
    for t in all_times:
        if not merged_times or t - merged_times[-1] >= 1.0:
            merged_times.append(t)

    print(f"      {len(uniform_times)} uniform + {len(scene_times)} scene = {len(merged_times)} sample points")

    # ── Phase 2: Load face detector ──────────────────────────────────────────
    face_net: Optional[cv2.dnn.Net] = None
    if not no_faces:
        print("\n[2/5] Loading face detector...")
        face_net = load_face_detector()
        if face_net is None:
            print("      -> face detection disabled")

    # ── Phase 3: Sample + fast filter + histogram + saliency ────────────────
    print(f"\n[3/5] Sampling {len(merged_times)} frames...")
    candidates: List[FrameCandidate] = []

    for idx, t in enumerate(merged_times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue

        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())

        if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS:
            continue
        sharpness_raw = laplacian_sharpness(gray)
        if sharpness_raw < MIN_SHARPNESS_RAW:
            continue

        c = FrameCandidate(time_sec=t, frame=frame.copy())
        c.sharpness_raw    = sharpness_raw
        c.contrast_raw     = float(gray.std())
        c.exposure_score   = exposure_score(brightness)
        c.colorfulness_raw = colorfulness_hs(frame)
        c.saliency_raw     = spectral_residual_saliency(gray)
        c.hist             = compute_hist(gray)
        candidates.append(c)

    cap.release()
    print(f"      {len(candidates)} candidates pass fast filter")

    if not candidates:
        raise RuntimeError(
            "No valid frames found — video may be entirely dark or unreadable."
        )

    # ── Phase 3b: dHash deduplication ────────────────────────────────────────
    before_dedup = len(candidates)
    candidates = deduplicate(candidates)
    removed = before_dedup - len(candidates)
    if removed:
        print(f"      dHash dedup: removed {removed} near-duplicates -> {len(candidates)} remain")

    if not candidates:
        raise RuntimeError("All frames are near-duplicates. Try a smaller --sample-interval.")

    # ── Phase 3c: Temporal stability scores ───────────────────────────────────
    compute_stability_scores(candidates)

    # ── Phase 4: Face detection ───────────────────────────────────────────────
    if face_net is not None:
        print(f"\n[4/5] Face detection on {len(candidates)} frames...")
        for i, c in enumerate(candidates):
            c.face_score = face_score_from_count(detect_faces(face_net, c.frame))
            if (i + 1) % 10 == 0 or i == len(candidates) - 1:
                pct = (i + 1) / len(candidates) * 100
                print(f"\r      {i + 1}/{len(candidates)}  ({pct:.0f}%)", end="", flush=True)
        print()
    else:
        print("\n[4/5] Face detection: skipped")

    # ── Phase 5: Normalize + score + select ──────────────────────────────────
    print(f"\n[5/5] Scoring and selecting {top_k} thumbnails...")
    normalize_and_score(candidates)
    selected = select_zone_best(candidates, top_k, skip_start, skip_end)

    if not selected:
        raise RuntimeError("Zone selection returned no frames.")

    # ── Output ────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    results: List[dict] = []

    for rank, item in enumerate(selected, 1):
        mins, secs = divmod(int(item.time_sec), 60)
        fname    = f"thumb_{rank}_{mins:02d}m{secs:02d}s.jpg"
        abs_path = os.path.join(output_dir, fname)
        cv2.imwrite(abs_path, item.frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        result = {
            "rank":     rank,
            "file":     fname,
            "_abs_path": abs_path,
            "time_sec": round(item.time_sec, 2),
            "timestamp": f"{mins:02d}:{secs:02d}",
            "score":    round(item.score, 2),
            "metrics": {
                "sharpness":    round(item.sharpness,    1),
                "contrast":     round(item.contrast,     1),
                "exposure":     round(item.exposure_score, 1),
                "faces":        round(item.face_score,   1),
                "colorfulness": round(item.colorfulness, 1),
                "saliency":     round(item.saliency,     1),
                "stability":    round(item.stability_score, 1),
            },
        }
        results.append(result)

        print(
            f"  #{rank}  {fname}  score={item.score:.1f}  "
            f"sharp={item.sharpness:.0f}  sal={item.saliency:.0f}  "
            f"stab={item.stability_score:.0f}  faces={item.face_score:.0f}  "
            f"@ {mins:02d}:{secs:02d}"
        )

    # JSON report
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "video":                os.path.abspath(video_path),
                "duration_sec":         round(duration, 2),
                "sample_interval_sec":  sample_interval,
                "candidates_evaluated": len(candidates),
                "weights": {
                    "sharpness":    W_SHARPNESS,
                    "contrast":     W_CONTRAST,
                    "exposure":     W_EXPOSURE,
                    "faces":        W_FACES,
                    "colorfulness": W_COLORFULNESS,
                    "saliency":     W_SALIENCY,
                    "stability":    W_STABILITY,
                },
                "thumbnails": [
                    {k: v for k, v in r.items() if not k.startswith("_")}
                    for r in results
                ],
            },
            fh,
            indent=2,
        )

    # HTML preview
    html_path = os.path.join(output_dir, "preview.html")
    write_html_preview(html_path, results)

    print(f"\nSaved: {os.path.abspath(output_dir)}/")
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
  python thumbnailer.py talk.mp4 --no-scene-detect

Optional dependencies:
  pip install scenedetect[opencv]   # enables scene-aware sampling
        """,
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-o", "--output",          default="thumbnails",
                        help="Output directory (default: thumbnails/)")
    parser.add_argument("-k", "--top-k",           type=int,   default=3,
                        help="Number of thumbnails to extract (default: 3)")
    parser.add_argument("-s", "--sample-interval", type=float, default=5.0,
                        help="Seconds between sampled frames (default: 5.0)")
    parser.add_argument("--skip-pct",              type=float, default=0.05,
                        help="Fraction to skip at start/end (default: 0.05)")
    parser.add_argument("--jpeg-quality",          type=int,   default=95,
                        help="JPEG output quality 1-100 (default: 95)")
    parser.add_argument("--no-faces",              action="store_true",
                        help="Skip face detection")
    parser.add_argument("--no-scene-detect",       action="store_true",
                        help="Skip PySceneDetect scene-aware sampling")

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
    print("  smart-thumbnailer  v2")
    print("=" * 56)

    extract_thumbnails(
        video_path       = args.video,
        output_dir       = args.output,
        top_k            = args.top_k,
        sample_interval  = args.sample_interval,
        no_faces         = args.no_faces,
        no_scene_detect  = args.no_scene_detect,
        skip_pct         = args.skip_pct,
        jpeg_quality     = args.jpeg_quality,
    )


if __name__ == "__main__":
    main()
