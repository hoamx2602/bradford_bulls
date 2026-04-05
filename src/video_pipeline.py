"""
Video Download & Metadata Extraction Pipeline
Downloads video from YouTube (or local file) and extracts metadata.
"""

import json
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from .config import VIDEOS_DIR, YT_DLP_FORMAT


@dataclass
class VideoMetadata:
    file_path: Path
    title: str
    duration_sec: float
    fps: float
    width: int
    height: int
    total_frames: int
    source_url: str = ""

    def __str__(self):
        dur_min = self.duration_sec / 60
        return (
            f"Video: {self.title}\n"
            f"  Path:       {self.file_path}\n"
            f"  Duration:   {dur_min:.1f} min ({self.duration_sec:.0f}s)\n"
            f"  Resolution: {self.width}x{self.height}\n"
            f"  FPS:        {self.fps}\n"
            f"  Frames:     {self.total_frames:,}"
        )


def download_youtube(url: str, output_dir: Path = VIDEOS_DIR) -> Path:
    """Download video from YouTube using yt-dlp."""
    if not shutil.which("yt-dlp"):
        raise RuntimeError(
            "yt-dlp not found. Install it:\n"
            "  brew install yt-dlp   (macOS)\n"
            "  pip install yt-dlp    (pip)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", YT_DLP_FORMAT,
        "-o", output_template,
        "--print", "after_move:filepath",
        "--no-playlist",
        url,
    ]

    print(f"[Download] Downloading from: {url}")
    print(f"[Download] Format: {YT_DLP_FORMAT}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    # The last non-empty line of stdout is the filepath
    filepath = result.stdout.strip().split("\n")[-1].strip()
    output_path = Path(filepath)

    if not output_path.exists():
        # Fallback: find the most recent file in output_dir
        files = sorted(output_dir.glob("*.*"), key=lambda f: f.stat().st_mtime)
        if files:
            output_path = files[-1]
        else:
            raise FileNotFoundError(f"Downloaded file not found in {output_dir}")

    print(f"[Download] Saved to: {output_path}")
    return output_path


def get_video_metadata(video_path: Path, source_url: str = "") -> VideoMetadata:
    """Extract metadata from a video file using OpenCV."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    cap.release()

    metadata = VideoMetadata(
        file_path=video_path,
        title=video_path.stem,
        duration_sec=duration_sec,
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
        source_url=source_url,
    )

    print(f"[Metadata]\n{metadata}")
    return metadata


def load_video(source: str) -> VideoMetadata:
    """
    Load video from YouTube URL or local file path.
    Returns VideoMetadata with all info needed for processing.
    """
    source_url = ""

    if source.startswith("http"):
        source_url = source
        video_path = download_youtube(source)
    else:
        video_path = Path(source)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {source}")

    return get_video_metadata(video_path, source_url=source_url)
