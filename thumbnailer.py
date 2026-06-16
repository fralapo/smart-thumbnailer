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
# Weights based on CQE (Panetta et al. IEEE TCE 2013) + video keyframe research.
# Face is applied as a multiplier (FACE_BOOST) rather than an additive term —
# this follows the YouTube patent approach and Song et al. 2016 (Yahoo Research).
W_COLORFULNESS = 0.25  # CQE: highest perceptual impact
W_SHARPNESS    = 0.20
W_CONTRAST     = 0.15  # CQE: second
W_NATURALNESS  = 0.10  # MSCN kurtosis — catches artifacts/compression
W_EXPOSURE     = 0.12
W_SALIENCY     = 0.10
W_STABILITY    = 0.08
FACE_BOOST     = 1.30  # multiply score by this when face detected

# Fast-filter thresholds
MIN_BRIGHTNESS   = 22.0
MAX_BRIGHTNESS   = 233.0
MIN_SHARPNESS_RAW = 15.0

# dHash near-duplicate threshold (out of 64 bits)
DHASH_MAX_HAMMING = 12

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
    ".webm", ".m4v", ".mpg", ".mpeg", ".ts", ".mts", ".m2ts",
}


# ---------------------------------------------------------------------------
# Auto top-k: number of thumbnails based on video duration
# ---------------------------------------------------------------------------

def auto_top_k(duration_sec: float) -> int:
    """
    Choose how many thumbnails to extract based on video length.

    Scale:
      < 3 min  -> 2
      3-8 min  -> 4
      8-20 min -> 6
      20-40    -> 8
      40-70    -> 9
      70-120   -> 10
      > 120    -> 10 (capped)
    """
    m = duration_sec / 60.0
    if m < 3:   return 2
    if m < 8:   return 4
    if m < 20:  return 6
    if m < 40:  return 8
    if m < 70:  return 9
    if m < 120: return 10
    return 10


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------
@dataclass
class FrameCandidate:
    time_sec: float
    frame: np.ndarray
    # set during histogram phase
    hist: Optional[np.ndarray] = None
    lab_hist: object = None          # list[np.ndarray] — Lab 3-channel, for zone diversity
    # raw metrics
    sharpness_raw:    float = 0.0
    contrast_raw:     float = 0.0
    exposure_score:   float = 0.0
    face_score:       float = 0.0
    face_count:       int   = 0
    face_detected:    bool  = False
    colorfulness_raw: float = 0.0
    naturalness_raw:  float = 0.0
    saliency_raw:     float = 0.0
    stability_score:  float = 100.0
    # normalized (0-100)
    sharpness:    float = 0.0
    contrast:     float = 0.0
    colorfulness: float = 0.0
    naturalness:  float = 0.0
    saliency:     float = 0.0
    score:        float = 0.0


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def laplacian_sharpness(gray: np.ndarray) -> float:
    """Fast blur detector for the pre-filter stage."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad_sharpness(gray: np.ndarray) -> float:
    """
    Tenengrad sharpness (Sobel gradient energy) with Gaussian center weighting.
    Inspired by sharp-frame-extractor (cansik/github).

    Center-weighting improves selection for portrait-framed subjects:
    - Edge energy in the frame center counts more
    - Avoids picking frames sharp only at the edges/background
    - sigma_fraction = 0.22 matches the sharp-frame-extractor default
    """
    g = gray.astype("float32")
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    g2 = gx * gx + gy * gy

    h, w = g2.shape
    sigma_y = max(1.0, h * 0.22)
    sigma_x = max(1.0, w * 0.22)
    ky = cv2.getGaussianKernel(h, sigma_y, ktype=cv2.CV_32F)
    kx = cv2.getGaussianKernel(w, sigma_x, ktype=cv2.CV_32F)
    weights = (ky @ kx.T).astype("float32")
    weights /= float(weights.sum())

    return float((g2 * weights).sum())


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


def exposure_score(gray: np.ndarray) -> float:
    """
    Exposure score 0-100. Penalizes mean brightness far from ideal AND
    frames with clipped highlights/shadows (blown whites, crushed blacks).
    """
    flat = gray.ravel().astype(np.float32)
    brightness = float(flat.mean())

    lo, hi = 90.0, 170.0
    if lo <= brightness <= hi:
        mean_score = 1.0
    elif brightness < lo:
        mean_score = max(0.0, brightness / lo)
    else:
        mean_score = max(0.0, 1.0 - (brightness - hi) / (255.0 - hi))

    pct_highlight = float(np.mean(flat > 245)) * 100.0
    pct_shadow    = float(np.mean(flat < 10))  * 100.0
    clip_penalty  = 1.0 - min(0.8, (pct_highlight + pct_shadow) / 15.0)

    return float(mean_score * clip_penalty * 100.0)


def mscn_naturalness(gray: np.ndarray) -> float:
    """
    MSCN (Mean Subtracted Contrast Normalized) kurtosis naturalness score.
    Natural scenes have MSCN kurtosis ~3-5. Very low (flat/uniform) or very
    high (extreme noise/compression artifacts) both score poorly.
    Returns [0, 1].
    """
    img = gray.astype(np.float64)
    C = 1.0 / 255.0
    mu = cv2.GaussianBlur(img, (7, 7), 7.0 / 6.0)
    sigma = np.sqrt(np.abs(
        cv2.GaussianBlur(img * img, (7, 7), 7.0 / 6.0) - mu * mu
    ))
    mscn = (img - mu) / (sigma + C)
    flat = mscn.ravel()
    std  = float(flat.std()) + 1e-8
    kurt = float(np.mean(((flat - float(flat.mean())) / std) ** 4))
    # Target sweet spot: 3 < kurtosis < 6
    score = np.exp(-0.5 * ((np.log(max(kurt, 0.1)) - np.log(4.0)) / 0.8) ** 2)
    return float(np.clip(score, 0.0, 1.0))


def colorfulness_combined(bgr: np.ndarray) -> float:
    """
    Combined colorfulness: Hasler & Susstrunk (2003) + HSV hue entropy +
    saturation score. Catches grey/washed-out and oversaturated frames that
    H&S alone misclassifies as high-quality.
    """
    B, G, R = cv2.split(bgr.astype("float32"))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    hs = float(
        np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    mask = sat > 30
    if mask.sum() >= 100:
        hue_vals = hue[mask]
        hist_h, _ = np.histogram(hue_vals, bins=18, range=(0, 180))
        hist_h = hist_h[hist_h > 0].astype(np.float32)
        prob   = hist_h / hist_h.sum()
        hue_entropy = float(-np.sum(prob * np.log2(prob + 1e-8))) / 4.17
    else:
        hue_entropy = 0.0

    mean_sat  = float(sat.astype(np.float32).mean() / 255.0)
    sat_score = float(np.exp(-0.5 * ((mean_sat - 0.45) / 0.20) ** 2))

    return hs * (0.60 + 0.25 * hue_entropy + 0.15 * sat_score)


def detect_faces(
    net: cv2.dnn.Net, bgr: np.ndarray, conf_thresh: float = 0.70
) -> List[tuple]:
    """
    Run res10 SSD face detector. Returns list of (x1,y1,x2,y2,conf) tuples.
    conf_thresh=0.70 is recommended for video (0.5 gives too many false positives).
    Mean (104,177,123) is the correct res10 training mean — not (104,117,123).
    """
    h, w = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    dets = net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2, conf))
    return boxes


def face_area_position_score(detections: List[tuple], frame_h: int, frame_w: int) -> float:
    """
    Score face presence as [0..100] based on:
    - Face area ratio (sweet spot 5-25% of frame — readable at thumbnail size)
    - Vertical position (upper 60% preferred)
    - Multiple faces penalised (crowded faces harder to read at thumb size)
    Grounded in Google's actor-centric thumbnail patents (US9892324, US10242265).
    """
    if not detections:
        return 0.0
    frame_area = max(frame_h * frame_w, 1)
    scores = []
    for (x1, y1, x2, y2, conf) in detections:
        area_ratio = (x2 - x1) * (y2 - y1) / frame_area
        area_score = float(np.exp(
            -0.5 * ((np.log(max(area_ratio, 1e-4)) - np.log(0.12)) / 0.9) ** 2
        ))
        face_cy = (y1 + y2) / 2.0
        pos_score = 1.0 - 0.4 * (face_cy / frame_h)
        scores.append(area_score * pos_score * min(conf, 1.0))
    if not scores:
        return 0.0
    primary = max(scores)
    multi_penalty = 1.0 / (1.0 + 0.15 * max(0, len(scores) - 1))
    return float(primary * multi_penalty * 100.0)


def compute_hist(gray: np.ndarray) -> np.ndarray:
    """64-bin normalized grayscale histogram for temporal stability checks."""
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def compute_lab_hist(bgr: np.ndarray) -> List[np.ndarray]:
    """
    3-channel normalized histograms in Lab color space.
    Used for zone-diversity check with Bhattacharyya distance.
    Lab is perceptually calibrated — frames that look similar have low distance.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hists = []
    for ch, bins in [(0, 32), (1, 16), (2, 16)]:
        h = cv2.calcHist([lab], [ch], None, [bins], [0, 256])
        cv2.normalize(h, h)
        hists.append(h)
    return hists


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
# Rotation correction (phone videos, action cameras, etc.)
# ---------------------------------------------------------------------------

def get_video_rotation(cap: cv2.VideoCapture) -> int:
    """
    Read rotation metadata from video stream (OpenCV 4.5+).
    Phone and action-camera videos often embed 90/270-deg rotation tags.
    Returns 0, 90, 180, or 270.
    """
    try:
        angle = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        return angle if angle in (90, 180, 270) else 0
    except Exception:
        return 0


def apply_rotation(frame: np.ndarray, angle: int) -> np.ndarray:
    """Counterrotate frame to correct for video orientation metadata."""
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


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
    Pixel-level inter-frame stillness (Song et al. 2016, Yahoo Research).
    Absdiff to neighbors detects motion/transition frames far better than
    histogram correlation, which is a scene-change detector not a motion detector.
    """
    n = len(candidates)
    if n < 3:
        for c in candidates:
            c.stability_score = 100.0
        return

    for i, c in enumerate(candidates):
        prev_gray = cv2.cvtColor(candidates[max(0, i - 1)].frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(c.frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(candidates[min(n - 1, i + 1)].frame, cv2.COLOR_BGR2GRAY)
        diff_p = float(cv2.absdiff(curr_gray, prev_gray).mean())
        diff_n = float(cv2.absdiff(curr_gray, next_gray).mean())
        motion = (diff_p + diff_n) / 2.0
        # stillness in [0, 1]: low motion = 1.0 (best), high motion = ~0
        stillness = 1.0 / (1.0 + motion / 8.0)
        c.stability_score = stillness * 100.0


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------

def _normalize(values: List[float]) -> List[float]:
    vmax = max(values) if values else 1.0
    if vmax < 1e-9:
        return [0.0] * len(values)
    return [(v / vmax) * 100.0 for v in values]


def normalize_and_score(candidates: List[FrameCandidate]) -> None:
    """
    Normalize raw metrics to [0-100], compute composite score.
    Face is applied as a multiplicative boost (FACE_BOOST) not additive weight —
    following Song et al. 2016 and YouTube actor-centric thumbnail patents.
    """
    if not candidates:
        return

    sharp_n  = _normalize([c.sharpness_raw    for c in candidates])
    cont_n   = _normalize([c.contrast_raw      for c in candidates])
    color_n  = _normalize([c.colorfulness_raw  for c in candidates])
    nat_n    = _normalize([c.naturalness_raw   for c in candidates])
    sal_n    = _normalize([c.saliency_raw      for c in candidates])

    for c, sn, cn, cfn, natn, saln in zip(
        candidates, sharp_n, cont_n, color_n, nat_n, sal_n
    ):
        c.sharpness    = sn
        c.contrast     = cn
        c.colorfulness = cfn
        c.naturalness  = natn
        c.saliency     = saln
        base = (
            W_COLORFULNESS * c.colorfulness
            + W_SHARPNESS    * c.sharpness
            + W_CONTRAST     * c.contrast
            + W_NATURALNESS  * c.naturalness
            + W_EXPOSURE     * c.exposure_score
            + W_SALIENCY     * c.saliency
            + W_STABILITY    * c.stability_score
        )
        c.score = base * (FACE_BOOST if c.face_detected else 1.0)


# ---------------------------------------------------------------------------
# Zone-based diversity selection
# ---------------------------------------------------------------------------

def _lab_diversity(a_hists: List[np.ndarray], b_hists: List[np.ndarray]) -> float:
    """
    Mean Bhattacharyya distance across Lab channels.
    Range [0, 1]: 0=identical, 1=maximally different.
    Perceptually calibrated — visually similar frames cluster correctly.
    """
    return float(np.mean([
        cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)
        for a, b in zip(a_hists, b_hists)
    ]))


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
            # Check visual diversity vs already-selected frames (Lab Bhattacharyya)
            if not selected or all(
                _lab_diversity(candidate.lab_hist, s.lab_hist) >= min_visual_dist
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

def write_html_preview(html_path: str, results: List[dict], faces_disabled: bool = False) -> None:
    """Self-contained HTML with base64-embedded thumbnails and metric bars."""
    METRIC_LABELS = {
        "colorfulness": "Color",
        "sharpness":    "Sharp",
        "contrast":     "Contrast",
        "naturalness":  "Natural",
        "exposure":     "Exposure",
        "saliency":     "Saliency",
        "stability":    "Stability",
        "faces":        "Faces",
    }
    SKIP_IN_HTML = {"face_count", "face_boost"}

    cards = ""
    for r in results:
        with open(r["_abs_path"], "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()

        bars_html = []
        for k, v in r["metrics"].items():
            if k in SKIP_IN_HTML:
                continue  # rendered via "faces" row or omitted
            label = METRIC_LABELS.get(k, k)
            if k == "faces":
                if faces_disabled:
                    bars_html.append(
                        f'<div class="m">'
                        f'<span class="ml">{label}</span>'
                        f'<div class="bw"><div class="b" style="width:0%;background:#333"></div></div>'
                        f'<span class="mv" style="color:#444">N/A</span></div>'
                    )
                else:
                    fc = r["metrics"].get("face_count", 0)
                    suffix = f" ({fc})" if fc else ""
                    bars_html.append(
                        f'<div class="m">'
                        f'<span class="ml">{label}{suffix}</span>'
                        f'<div class="bw"><div class="b" style="width:{v:.0f}%"></div></div>'
                        f'<span class="mv">{v:.0f}</span></div>'
                    )
            else:
                bars_html.append(
                    f'<div class="m"><span class="ml">{label}</span>'
                    f'<div class="bw"><div class="b" style="width:{v:.0f}%"></div></div>'
                    f'<span class="mv">{v:.0f}</span></div>'
                )
        bars = "".join(bars_html)
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
    top_k: Optional[int] = None,
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
        top_k:            Number of thumbnails (None = auto from duration).
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
    rotation     = get_video_rotation(cap)

    if duration < 1.0:
        raise RuntimeError("Video too short or unreadable.")

    skip_start = duration * skip_pct
    skip_end   = duration * (1.0 - skip_pct)

    mins_t, secs_t = divmod(int(duration), 60)
    n_uniform = int((skip_end - skip_start) / sample_interval)

    # Resolve top_k: auto if not specified
    top_k_auto = top_k is None
    if top_k is None:
        top_k = auto_top_k(duration)

    print(f"Video    : {os.path.basename(video_path)}")
    print(f"Duration : {mins_t}m{secs_t:02d}s  |  FPS: {fps:.2f}  |  Frames: {total_frames}")
    print(f"Skip     : first {skip_start:.0f}s + last {duration - skip_end:.0f}s")
    if rotation:
        print(f"Rotation : {rotation} deg (auto-correcting frames)")
    if top_k_auto:
        print(f"Thumbs   : {top_k}  (auto, {mins_t}m video)")
    else:
        print(f"Thumbs   : {top_k}  (manual)")

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

        frame      = apply_rotation(frame, rotation)
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())

        if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS:
            continue
        if laplacian_sharpness(gray) < MIN_SHARPNESS_RAW:
            continue

        c = FrameCandidate(time_sec=t, frame=frame.copy())
        c.sharpness_raw    = tenengrad_sharpness(gray)
        c.contrast_raw     = float(gray.std())
        c.exposure_score   = exposure_score(gray)
        c.colorfulness_raw = colorfulness_combined(frame)
        c.naturalness_raw  = mscn_naturalness(gray)
        c.saliency_raw     = spectral_residual_saliency(gray)
        c.hist             = compute_hist(gray)
        c.lab_hist         = compute_lab_hist(frame)
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
            dets = detect_faces(face_net, c.frame)
            c.face_count    = len(dets)
            c.face_detected = len(dets) > 0
            c.face_score    = face_area_position_score(dets, c.frame.shape[0], c.frame.shape[1])
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
                "colorfulness": round(item.colorfulness, 1),
                "sharpness":    round(item.sharpness,    1),
                "contrast":     round(item.contrast,     1),
                "naturalness":  round(item.naturalness,  1),
                "exposure":     round(item.exposure_score, 1),
                "saliency":     round(item.saliency,     1),
                "stability":    round(item.stability_score, 1),
                "faces":        round(item.face_score,   1),
                "face_count":   item.face_count,
                "face_boost":   FACE_BOOST if item.face_detected else 1.0,
            },
        }
        results.append(result)

        face_info = f"faces={item.face_count}" if not no_faces else "faces=N/A"
        print(
            f"  #{rank}  {fname}  score={item.score:.1f}  "
            f"sharp={item.sharpness:.0f}  sal={item.saliency:.0f}  "
            f"stab={item.stability_score:.0f}  {face_info}  "
            f"@ {mins:02d}:{secs:02d}"
        )

    # JSON report
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "video":                    os.path.abspath(video_path),
                "duration_sec":             round(duration, 2),
                "thumbnails_count":         top_k,
                "thumbnails_count_auto":    top_k_auto,
                "sample_interval_sec":      sample_interval,
                "candidates_evaluated":     len(candidates),
                "face_detection_enabled":   not no_faces,
                "weights": {
                    "colorfulness": W_COLORFULNESS,
                    "sharpness":    W_SHARPNESS,
                    "contrast":     W_CONTRAST,
                    "naturalness":  W_NATURALNESS,
                    "exposure":     W_EXPOSURE,
                    "saliency":     W_SALIENCY,
                    "stability":    W_STABILITY,
                    "face_boost":   FACE_BOOST,
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
    write_html_preview(html_path, results, faces_disabled=no_faces)

    print(f"\nSaved: {os.path.abspath(output_dir)}/")
    print(f"  {len(selected)}x JPG  +  report.json  +  preview.html")

    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_input_videos(inputs: List[str]) -> List[str]:
    """
    Resolve positional CLI args to an ordered list of video file paths.

    Supported forms (can be mixed):
      (none)            -> all videos in the script's directory
      video.mp4         -> single explicit file
      /path/to/dir/     -> all videos inside that directory
      .mp4              -> all .mp4 in script's directory
      /path/dir .mp4    -> all .mp4 in specified directory
    """
    script_dir = Path(__file__).parent

    ext_filters: set = set()
    scan_dirs: List[Path] = []
    video_files: List[str] = []
    seen: set = set()

    for token in inputs:
        # Extension shorthand: starts with "." and has no path separator
        if (token.startswith(".") and len(token) > 1
                and "/" not in token and "\\" not in token):
            ext_filters.add(token.lower())
            continue

        p = Path(token)
        if p.is_dir():
            scan_dirs.append(p)
        elif p.is_file():
            resolved = str(p.resolve())
            if p.suffix.lower() in VIDEO_EXTENSIONS:
                if resolved not in seen:
                    video_files.append(resolved)
                    seen.add(resolved)
            else:
                print(f"Warning: {token!r} skipped -- not a known video format",
                      file=sys.stderr)
        else:
            print(f"Warning: {token!r} not found", file=sys.stderr)

    # Scan dirs when: no input at all, or extension filter given without explicit dir
    need_scan = (not inputs) or bool(ext_filters) or bool(scan_dirs)
    if need_scan and not scan_dirs:
        scan_dirs = [script_dir]

    exts = ext_filters if ext_filters else VIDEO_EXTENSIONS
    for d in scan_dirs:
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                resolved = str(f.resolve())
                if resolved not in seen:
                    video_files.append(resolved)
                    seen.add(resolved)

    return video_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract optimal thumbnail frames from video files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input modes:
  python thumbnailer.py                          all videos in script directory
  python thumbnailer.py video.mp4                single file
  python thumbnailer.py video1.mp4 video2.mkv    multiple explicit files
  python thumbnailer.py /path/to/dir/            all videos in directory
  python thumbnailer.py .mp4                     all .mp4 in script directory
  python thumbnailer.py /path/dir .mp4 .mkv      filtered by extension

Output:
  thumbnails/<video_name>/thumb_N_MMmSSs.jpg
  thumbnails/<video_name>/report.json
  thumbnails/<video_name>/preview.html

Optional dependencies:
  pip install scenedetect[opencv]   # enables scene-aware sampling
        """,
    )
    parser.add_argument("inputs", nargs="*",
                        help="Video files, directories, or extension filters (.mp4)")
    parser.add_argument("-o", "--output",          default=None,
                        help="Output root directory (default: thumbnails/)")
    parser.add_argument("-k", "--top-k",           type=int,   default=None,
                        help="Number of thumbnails to extract (default: auto from duration)")
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

    if args.top_k is not None and args.top_k < 1:
        print("Error: --top-k must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.sample_interval < 0.5:
        print("Error: --sample-interval must be >= 0.5", file=sys.stderr)
        sys.exit(1)

    videos = resolve_input_videos(args.inputs)

    if not videos:
        print("Error: no video files found.", file=sys.stderr)
        sys.exit(1)

    output_root = args.output or "thumbnails"
    batch = len(videos) > 1

    print("=" * 56)
    print("  smart-thumbnailer  v4")
    print("=" * 56)
    if batch:
        print(f"Batch mode: {len(videos)} video(s) queued\n")

    ok = 0
    failed: List[str] = []

    for i, video_path in enumerate(videos, 1):
        stem = Path(video_path).stem
        safe_stem = "".join(c if c not in '<>:"/\\|?*' else "_" for c in stem)

        if batch:
            out_dir = os.path.join(output_root, safe_stem)
            print(f"[{i}/{len(videos)}] {os.path.basename(video_path)}")
        else:
            # Single file: honour -o as direct path for backward compat
            out_dir = args.output if args.output else os.path.join(output_root, safe_stem)

        try:
            extract_thumbnails(
                video_path      = video_path,
                output_dir      = out_dir,
                top_k           = args.top_k,
                sample_interval = args.sample_interval,
                no_faces        = args.no_faces,
                no_scene_detect = args.no_scene_detect,
                skip_pct        = args.skip_pct,
                jpeg_quality    = args.jpeg_quality,
            )
            ok += 1
        except Exception as exc:
            print(f"  Error: {exc}", file=sys.stderr)
            failed.append(os.path.basename(video_path))

    if batch:
        print(f"\n{'=' * 56}")
        print(f"  Done: {ok}/{len(videos)} succeeded")
        if failed:
            print(f"  Failed ({len(failed)}): {', '.join(failed)}")
        print("=" * 56)


if __name__ == "__main__":
    main()
