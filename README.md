# Bradford Bulls — sponsorship exposure pipeline

Python tooling and notebooks for ingesting match video, sampling frames, and downstream sponsor visibility analysis.

## Python environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` already includes **yt-dlp**; you still need **ffmpeg** on the machine (see below). **OpenCV** comes in via `opencv-python`.

---

## ffmpeg and yt-dlp by operating system

Downloads from YouTube use **yt-dlp** plus **ffmpeg** to merge separate video and audio streams. Without ffmpeg you may only get audio, or low-quality single files.

| OS | Install ffmpeg | Install yt-dlp (if not using pip) | Notes |
|----|----------------|-------------------------------------|--------|
| **macOS** (typical dev machine) | `brew install ffmpeg` | `brew install yt-dlp` *or* use the pip package from `requirements.txt` | Homebrew paths are picked up by Terminal and most Jupyter setups. |
| **Windows** | `winget install FFmpeg` *(Gyan FFmpeg)* or [ffmpeg.org](https://ffmpeg.org/download.html) | `pip install yt-dlp` (from requirements) | After **winget**, Windows updates **PATH** for *new* processes. **Restart the Jupyter kernel**, or restart **Cursor / VS Code**, so Python sees `ffmpeg`. If it still fails, set `FFMPEG_BINARY` to the full path of `ffmpeg.exe` (see below). |
| **Linux** (Debian/Ubuntu) | `sudo apt update && sudo apt install -y ffmpeg` | `pip install yt-dlp` or your distro package | Ensure `ffmpeg` is on `PATH` (`which ffmpeg`). |

### Jupyter / IDE and PATH (especially Windows)

If you install ffmpeg **after** opening the editor or starting a notebook kernel, the running Python process may still have the **old** `PATH`. **Restart the kernel** (or the whole app). The project also resolves WinGet-installed ffmpeg under `%LOCALAPPDATA%\Microsoft\WinGet\Packages` when `PATH` is stale.

### Optional: point Python at ffmpeg explicitly

Set one of:

- **`FFMPEG_BINARY`** — full path to the `ffmpeg` executable (Windows: `...\ffmpeg.exe`).
- **`IMAGEIO_FFMPEG_EXE`** — same idea (some other libraries use this).

---

## Download quality (720p vs 1080p)

YouTube often serves **720p as the best pure MP4** stream, while **1080p** is a **separate DASH stream** (sometimes VP9/WebM). Forcing `bestvideo[ext=mp4]` can cap you at 720p.

Default quality is configured in **`src/config.py`** as `YT_DLP_FORMAT`: it prefers **video up to 1080p** merged with best audio, then falls back to sensible defaults. Merged output is still requested as **MP4** in the download pipeline (`--merge-output-format mp4`).

To change quality, edit `YT_DLP_FORMAT` in `src/config.py`, for example:

- **Allow up to 4K:** use `height<=2160` instead of `height<=1080` in the same string (files will be larger).
- **Stricter / smaller:** lower the height cap or use yt-dlp’s [format selection](https://github.com/yt-dlp/yt-dlp#format-selection) syntax.

---

## Notebooks

From the project root (with `src` importable), e.g.:

```bash
jupyter notebook notebooks/01_video_to_frames.ipynb
```

If imports fail, run Jupyter from the repository root or adjust `PYTHONPATH` so `src` is on the path.

**GPU / CUDA:** notebooks such as `02b_auto_annotate.ipynb` may run entirely on CPU if PyTorch was installed without CUDA, or on Mac if Grounding DINO does not use MPS. See **`docs/GPU_SETUP.md`**.

---

## Further planning

See `PLAN.md` for product and pipeline phases.
