"""
Video Download & Metadata Extraction Pipeline
Downloads video from YouTube (or local file) and extracts metadata.
"""

import os
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


_VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".webm", ".mov", ".avi", ".mpeg", ".mpg"})
_AUDIO_ONLY_EXTENSIONS = frozenset({".m4a", ".opus", ".ogg", ".mp3", ".wav", ".aac", ".flac"})


def _find_ffmpeg_exe() -> Path | None:
    """
    Resolve ffmpeg executable. Jupyter/IDE often inherits an old PATH after winget install;
    on Windows we also look under WinGet's Packages folder.
    """
    for key in ("FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"):
        raw = (os.environ.get(key) or "").strip().strip('"')
        if not raw:
            continue
        p = Path(raw)
        if p.is_dir():
            p = p / "ffmpeg.exe" if os.name == "nt" else p / "ffmpeg"
        if p.is_file():
            return p.resolve()

    for name in ("ffmpeg", "ffmpeg.exe"):
        w = shutil.which(name)
        if w:
            return Path(w).resolve()

    if os.name != "nt":
        return None

    local = os.environ.get("LOCALAPPDATA", "")
    winget = Path(local) / "Microsoft" / "WinGet" / "Packages"
    if winget.is_dir():
        for exe in winget.rglob("ffmpeg.exe"):
            if exe.parent.name.lower() == "bin":
                return exe.resolve()
        for exe in winget.rglob("ffmpeg.exe"):
            return exe.resolve()

    scoop = Path(os.environ.get("USERPROFILE", "")) / "scoop" / "shims" / "ffmpeg.exe"
    if scoop.is_file():
        return scoop.resolve()

    return None


def download_youtube(url: str, output_dir: Path = VIDEOS_DIR) -> Path:
    """Download video from YouTube using yt-dlp."""
    if not shutil.which("yt-dlp"):
        raise RuntimeError(
            "yt-dlp not found. Install it:\n"
            "  brew install yt-dlp   (macOS)\n"
            "  pip install yt-dlp    (pip)"
        )
    ffmpeg_exe = _find_ffmpeg_exe()
    if not ffmpeg_exe:
        raise RuntimeError(
            "ffmpeg not found. It is required to merge video+audio from YouTube.\n"
            "Install: https://ffmpeg.org/download.html\n"
            "  winget install FFmpeg   (Windows)\n"
            "  brew install ffmpeg       (macOS)\n"
            "If you just installed ffmpeg, restart the Jupyter kernel (or Cursor) so PATH updates.\n"
            "Or set FFMPEG_BINARY to the full path of ffmpeg.exe."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", YT_DLP_FORMAT,
        "--merge-output-format", "mp4",
        "--ffmpeg-location",
        str(ffmpeg_exe.parent),
        "-o", output_template,
        "--print", "after_move:filepath",
        "--no-playlist",
        url,
    ]

    print(f"[Download] Downloading from: {url}")
    print(f"[Download] Format: {YT_DLP_FORMAT}")
    print(f"[Download] ffmpeg: {ffmpeg_exe}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    # yt-dlp may print one path per moved file; without merge the last line can be audio-only.
    lines = [ln.strip() for ln in result.stdout.strip().split("\n") if ln.strip()]
    candidates = [Path(ln) for ln in lines if Path(ln).exists()]

    def _pick_video(paths: list[Path]) -> Path | None:
        vids = [p for p in paths if p.suffix.lower() in _VIDEO_EXTENSIONS]
        if not vids:
            return None
        return max(vids, key=lambda p: p.stat().st_size)

    output_path = _pick_video(candidates)
    if output_path is None and candidates:
        # Only audio (or unknown) files reported — try newest video-like file in folder
        recent = sorted(output_dir.glob("*.*"), key=lambda f: f.stat().st_mtime, reverse=True)
        output_path = _pick_video(recent)
    if output_path is None:
        if candidates and all(p.suffix.lower() in _AUDIO_ONLY_EXTENSIONS for p in candidates):
            raise RuntimeError(
                "Download produced audio-only file(s). OpenCV needs a video container.\n"
                "Install ffmpeg, delete partial downloads in output/videos, and retry."
            )
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
