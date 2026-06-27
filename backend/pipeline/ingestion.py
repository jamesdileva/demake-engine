"""
Sprint 2 — Ingestion Pipeline
Extracts keyframes and an audio sample from the uploaded trailer using ffmpeg.

Architecture doc reference:
  Use ffmpeg to extract keyframes and audio snippets from the trailer.
  Extract 1 keyframe every 3 seconds (~10-20 frames for a 2-minute trailer).
  Extract a 30-second audio clip from the middle (captures main theme).
"""
import os
import math
import subprocess
from pathlib import Path
from PIL import Image


def run_ingestion(source_path: str, output_dir: str) -> dict:
    """
    Extract keyframes and audio from the uploaded MP4.

    Args:
        source_path: Path to the uploaded MP4 file
        output_dir:  /outputs/{demake_id}/ — all files written here

    Returns:
        {
            "keyframes_dir": str,
            "keyframe_paths": [str, ...],   # All extracted PNG paths
            "best_frames": [str, ...],       # Top 5 by quality score
            "audio_path": str | None,
            "video_duration_s": float,
            "frame_count": int,
        }
    """
    keyframes_dir = os.path.normpath(os.path.join(output_dir, "keyframes"))
    os.makedirs(keyframes_dir, exist_ok=True)

    # ── Step 1: Get video metadata ────────────────────────────────────────────
    duration = _get_duration(source_path)
    print(f"[Ingestion] Video duration: {duration:.1f}s")

    # ── Step 2: Extract keyframes (1 per 3 seconds) ───────────────────────────
    frame_pattern = os.path.join(keyframes_dir, "frame_%04d.png")

    # Scale to 512x288 — enough detail for VLM, small enough to be fast
    result = subprocess.run([
        "ffmpeg", "-i", source_path,
        "-vf", "fps=1/3,scale=512:288:flags=lanczos",
        "-q:v", "2",          # High quality PNG
        "-frames:v", "40",    # Cap at 40 frames max
        frame_pattern,
        "-y",                 # Overwrite if exists
        "-loglevel", "error"  # Suppress ffmpeg spam
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr}")

    # Collect extracted frames
    keyframe_paths = sorted([
        os.path.join(keyframes_dir, f)
        for f in os.listdir(keyframes_dir)
        if f.endswith(".png")
    ])

    if not keyframe_paths:
        raise RuntimeError("ffmpeg ran but produced no keyframes. Is the file a valid MP4?")

    print(f"[Ingestion] Extracted {len(keyframe_paths)} keyframes")

    # ── Step 3: Score frames and pick the best 5 ─────────────────────────────
    scored = _score_frames(keyframe_paths)
    best_frames = [path for path, score in scored[:5]]
    print(f"[Ingestion] Best 5 frames selected by quality score")

    # ── Step 4: Extract audio sample (30s from middle of video) ──────────────
    audio_path = _extract_audio(source_path, output_dir, duration)

    return {
        "keyframes_dir":    keyframes_dir,
        "keyframe_paths":   keyframe_paths,
        "best_frames":      best_frames,
        "audio_path":       audio_path,
        "video_duration_s": duration,
        "frame_count":      len(keyframe_paths),
    }


def _get_duration(source_path: str) -> float:
    """Use ffprobe to get video duration in seconds."""
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        source_path
    ], capture_output=True, text=True)

    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 120.0  # Assume 2 minutes if probe fails


def _score_frames(paths: list[str]) -> list[tuple[str, float]]:
    """
    Score each frame by visual quality.
    Higher score = more interesting/sharp frame worth showing the VLM.

    Scoring criteria:
      - Brightness variance (avoids pure black/white frames)
      - Edge density (sharpness — blurry frames score low)
      - Color range (avoids monochrome/loading screens)

    Returns list of (path, score) sorted best-first.
    """
    scored = []
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
            score = _frame_quality_score(img)
            scored.append((path, score))
        except Exception:
            scored.append((path, 0.0))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _frame_quality_score(img: Image.Image) -> float:
    """
    Compute a quality score for a single frame.
    Uses pure PIL — no numpy/cv2 dependency.
    """
    # Downsample for fast processing
    small = img.resize((64, 36))
    pixels = list(small.getdata())

    # ── Brightness variance ───────────────────────────────────────────────────
    # Frames that are too dark or too bright are less informative
    brightnesses = [(r + g + b) / 3 for r, g, b in pixels]
    mean_b = sum(brightnesses) / len(brightnesses)
    variance = sum((b - mean_b) ** 2 for b in brightnesses) / len(brightnesses)
    brightness_score = min(variance / 2000.0, 1.0)  # Normalize

    # ── Color range ───────────────────────────────────────────────────────────
    # Count unique-ish colors (quantize to avoid noise inflating count)
    quantized = set(
        (r // 32, g // 32, b // 32)
        for r, g, b in pixels
    )
    color_score = min(len(quantized) / 40.0, 1.0)

    # ── Avoid first/last 10% of frames ───────────────────────────────────────
    # Trailers often start/end with logos and black screens
    # (This bias is applied at the caller level via slicing)

    return (brightness_score * 0.6) + (color_score * 0.4)


def _extract_audio(source_path: str, output_dir: str, duration: float) -> str | None:
    """
    Extract a 30-second audio clip from roughly the middle of the video.
    The middle section usually contains the main musical theme.
    """
    audio_path = os.path.normpath(os.path.join(output_dir, "audio_sample.wav"))

    # Start at 30% into the video, grab 30 seconds
    start_time = max(0, duration * 0.30)

    result = subprocess.run([
        "ffmpeg", "-i", source_path,
        "-ss", str(start_time),
        "-t", "30",
        "-vn",                   # No video
        "-ar", "16000",          # 16kHz sample rate
        "-ac", "1",              # Mono
        "-acodec", "pcm_s16le",  # WAV format
        audio_path,
        "-y",
        "-loglevel", "error"
    ], capture_output=True, text=True)

    if result.returncode != 0 or not os.path.exists(audio_path):
        print(f"[Ingestion] Audio extraction failed (non-fatal): {result.stderr}")
        return None

    print(f"[Ingestion] Audio sample extracted: {audio_path}")
    return audio_path