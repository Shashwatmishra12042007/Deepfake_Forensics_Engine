"""Video frame extraction and metadata for temporal forensic analysis."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import VIDEO_SAMPLE_FRAMES
from src.utils.errors import MediaLoadError, ProcessingError


def load_video_metadata(path: str | Path) -> dict:
    """Return fps, frame count, duration, resolution."""
    try:
        path = Path(path)
        if not path.is_file():
            raise MediaLoadError("Video file not found", str(path))
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise MediaLoadError("OpenCV could not open video", str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frames / fps if fps > 0 else 0.0
            return {
                "fps": round(fps, 3),
                "frame_count": frames,
                "duration_sec": round(duration, 2),
                "width": w,
                "height": h,
            }
        finally:
            cap.release()
    except MediaLoadError:
        raise
    except (OSError, ValueError) as exc:
        raise MediaLoadError("Failed to read video metadata", str(exc)) from exc


def extract_frames_one_per_second(path: str | Path) -> list[np.ndarray]:
    """Extract one BGR frame per second of video using OpenCV."""
    try:
        path = Path(path)
        if path.suffix.lower() != ".mp4":
            raise MediaLoadError("Only .mp4 videos are supported", str(path))
        if not path.is_file():
            raise MediaLoadError("Video file not found", str(path))

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise MediaLoadError("OpenCV could not open video", str(path))

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0:
                raise ProcessingError("Video has invalid FPS", str(path))
            if total_frames <= 0:
                raise ProcessingError("Video has no readable frames", str(path))

            duration_sec = total_frames / fps
            frames: list[np.ndarray] = []
            second = 0

            while second <= duration_sec:
                frame_idx = min(int(round(second * fps)), total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(frame)
                second += 1

            if not frames:
                raise ProcessingError("No frames extracted at 1 fps", str(path))
            return frames
        finally:
            cap.release()
    except (MediaLoadError, ProcessingError):
        raise
    except (cv2.error, ValueError) as exc:
        raise ProcessingError("1 fps frame extraction failed", str(exc)) from exc


def analyze_video(video_path: str | Path) -> float:
    """
    Sample .mp4 at 1 frame/sec, score each frame with analyze_media,
    and return the mean authenticity confidence for the full duration [0, 100].
    """
    try:
        from src.models.detector import analyze_media

        path = Path(video_path)
        frames = extract_frames_one_per_second(path)
        scores: list[float] = []
        errors: list[str] = []

        for i, frame in enumerate(frames):
            try:
                scores.append(analyze_media(frame))
            except ProcessingError as exc:
                errors.append(f"second {i}: {exc}")

        if not scores:
            detail = "; ".join(errors) if errors else "no frames scored"
            raise ProcessingError("Video analysis produced no scores", detail)

        return round(float(np.mean(scores)), 2)
    except (MediaLoadError, ProcessingError):
        raise
    except (ValueError, TypeError) as exc:
        raise ProcessingError("analyze_video failed", str(exc)) from exc


def extract_frames(
    path: str | Path,
    max_frames: int = VIDEO_SAMPLE_FRAMES,
) -> list[np.ndarray]:
    """Uniformly sample frames as BGR arrays."""
    try:
        path = Path(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise MediaLoadError("OpenCV could not open video", str(path))

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise ProcessingError("Video has no readable frames", str(path))

        indices = np.linspace(0, max(total - 1, 0), num=min(max_frames, total), dtype=int)
        frames: list[np.ndarray] = []
        try:
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(frame)
        finally:
            cap.release()

        if not frames:
            raise ProcessingError("No frames extracted from video", str(path))
        return frames
    except (MediaLoadError, ProcessingError):
        raise
    except (cv2.error, ValueError) as exc:
        raise ProcessingError("Frame extraction failed", str(exc)) from exc
