# Architecture

## Mục tiêu

Module `track_annotation` giải quyết bài toán annotation logo trên video rugby Bradford Bulls bằng cách chuyển paradigm từ frame-level sang track-level.

## Vị trí trong pipeline lớn

Pipeline 6 tầng của master plan:

```
[1] Shot Segmentation
    ↓
[2] Player & Team Detection         ← module này (v0)
    ↓
[3] Logo Detection (2-stage)
     - Stage A: class-agnostic       ← train từ output của module này
     - Stage B: brand recognition
    ↓
[4] Multi-Object Tracking
    ↓
[5] Visibility Filtering
    ↓
[6] Exposure Metrics & Valuation
```

Module này phục vụ **Tầng 2 + Tầng 4** trong giai đoạn vòng 1 (player tracking) và **Tầng 3 + Tầng 4** trong vòng 2+ (logo tracking).

## Flow data

```
                       config.yaml
                            |
                            v
   video.mp4  ──>  [pipeline.detect_track]  ──>  list[Track]
                            |
                            v
                  [pipeline.keyframe]   ──>   list[Keyframe] per track
                            |
                            v
                    [pipeline.clip]     ──>   clip.mp4 per track
                            |
                            v
              [pipeline.package_builder] ──>  annotation_package/
                                              ├── manifest.json
                                              ├── tracks/
                                              │   └── track_NNNNN/
                                              └── annotations.jsonl
                            |
                            v
                  [reviewer.app]   ──>   updated annotations.jsonl
                            |
                            v
                  [exporters.yolo / cvat / roboflow]
                            |
                            v
                  YOLO dataset (training-ready)
```

## Tách lớp module

### `config.py`
- Pydantic schema cho mọi knob của pipeline.
- Load từ YAML, override bằng env var với prefix `TRACK_ANN__`.
- Validate type và value range tại load time → fail fast.

### `utils/`
- `logging.py`: loguru-based, console + file rotation.
- `video_io.py`: VideoReader context manager, FPS subsampling, seek-by-frame.
- `geometry.py`: bbox math (IoU, area), sharpness (Laplacian variance), crop with padding.

### `pipeline/`
- `detect_track.py`: ultralytics YOLO + BoT-SORT. Streams frames qua model. Output `list[Track]` đã filter min duration & area.
- `keyframe.py`: chọn N best-evidence frames per track theo strategy có thể config.
- `clip.py`: extract video clip 2s quanh midpoint của track. Backend cv2 hoặc ffmpeg.
- `pose_align.py`: ý tưởng 1 — pose-aligned multi-frame fusion. Optional (chỉ cho ambiguous tracks).
- `package_builder.py`: orchestrator — gọi tuần tự detect_track → keyframe → clip → write package.

### `exporters/`
- `yolo.py`: package → YOLO format (data.yaml + images/labels split). Hỗ trợ single-class (cho Stage A) và multi-class.
- `cvat.py`: package → CVAT 1.1 XML. Cho phép annotator refine bboxes trong CVAT trước khi gán brand.
- `roboflow.py`: upload YOLO dataset lên Roboflow project.

### `reviewer/app.py`
- Streamlit UI. Một track per page với 3 keyframe + clip + brand grid.
- Append annotation vào `annotations.jsonl` (idempotent — re-runnable).
- State: `st.session_state.track_idx` track vị trí hiện tại; resume từ track chưa annotate đầu tiên.

### `cli.py`
- Click-based CLI: `build-package`, `export`, `inspect`.
- Là entrypoint chính khi chạy production.

## Design decisions

### Vì sao Pydantic + YAML thay vì Hydra?

Hydra mạnh nhưng overhead cao cho project nhỏ (1 dev, 2 trận/tháng). Pydantic + YAML đơn giản, validate tốt, đủ cho nhu cầu hiện tại. Có thể migrate sang Hydra sau nếu cần multi-run sweep.

### Vì sao streaming detect+track thay vì batch?

Một trận 2h ở 1080p = ~12 GB raw data nếu load hết vào RAM. Streaming cho phép chạy trên máy 16 GB RAM mà không cần swap.

### Vì sao tách `select_keyframes` và `write_keyframes`?

`select_keyframes` là pure function (không touch disk), dễ test. `write_keyframes` mới thực sự đọc video và lưu file. Tách ra giúp:
- Test logic chọn keyframe không cần video thật (xem `tests/test_keyframe.py`).
- Tái dùng `select_keyframes` khi cần các pipeline khác (ví dụ: export thumbnail cho dashboard).

### Vì sao label propagation per track thay vì per frame?

Cốt lõi của ý tưởng 2: 1 brand decision = N frame label. Annotator chỉ touch 1 lần per track; mọi frame trong track tự động kế thừa label. Giảm thời gian 5–10x đồng thời tăng consistency vì mọi frame trong cùng track có cùng brand_id (không có annotator drift trong track).

### Vì sao annotations.jsonl là append-only thay vì overwrite?

- Audit trail: mọi quyết định annotator được log lại, có timestamp implicit qua thứ tự dòng.
- Crash-safe: process bị kill giữa chừng không mất annotation đã làm.
- Last-write-wins khi đọc: cho phép annotator sửa quyết định cũ bằng cách thêm dòng mới.

## Mở rộng

### Thêm strategy chọn keyframe

Edit `pipeline/keyframe.py:_pick_by_strategy()` thêm branch mới, ví dụ `"least_occluded"`. Update `KeyframeConfig.strategies` Literal. Test trong `tests/test_keyframe.py`.

### Thêm exporter format

Tạo file mới trong `exporters/`. Pattern:
```python
def export_to_<format>(package_dir, output_path, **kwargs) -> Path:
    manifest = json.loads((Path(package_dir) / "manifest.json").read_text())
    annotations = _load_annotations_jsonl(...)
    # ... write format-specific output
    return output_path
```
Add CLI option in `cli.py:cmd_export()`.

### Swap detector

Update YAML config — đổi `detection.weights` và `detection.target_classes`. Không đụng code. Đây là điểm mạnh chính của kiến trúc này.

### Multi-video batch processing

Wrapper script:
```python
for video in videos_dir.glob("*.mp4"):
    build_package(video, output_dir / video.stem, cfg)
```
Hoặc dùng GNU parallel / Snakemake.

## Performance budget (RTX 4500 Ada, 1080p, 2h match)

| Stage | Time | Bottleneck |
|---|---|---|
| detect+track | ~25 min | YOLO inference |
| keyframe write | ~3 min | Disk I/O |
| clip extract  | ~7 min | cv2 VideoWriter (5 min if ffmpeg) |
| package total | ~35 min | — |
| reviewer (per track) | 8–15 sec | Human |

Tổng: ~35 min auto-processing + ~2-4 hour annotation cho 1 trận (giả sử 200-300 tracks).
