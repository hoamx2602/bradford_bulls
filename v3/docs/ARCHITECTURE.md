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
- **Brand model**: `Brand` (master) chứa `list[Variant]`. Mỗi `Variant` có `kit_contexts: list[home|away|special|any]` quy định khi nào active. Loaded từ `data/logo_templates/brands.yaml` riêng (không inline trong pipeline config).
- **MatchContext**: per-match metadata (kit_context, opponent, date, …) — passed via CLI `--kit-context` hoặc sidecar `match.meta.yaml`.

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
- `yolo.py`: package → YOLO format (data.yaml + images/labels split).
  - `class_mode=single`: 1 class "logo" (cho Stage A class-agnostic).
  - `class_mode=brand` (mặc định): N class = các brand_id unique trong annotations (~16 nếu cover hết).
  - `class_mode=variant`: N class = các variant_id unique (~21 nếu cover hết).
- `cvat.py`: package → CVAT 1.1 XML.
- `roboflow.py`: upload YOLO dataset lên Roboflow project.

### `reviewer/app.py`
- Streamlit UI. Một track per page với 3 keyframe + clip + brand grid.
- **Brand-aware**: load `kit_context` từ manifest, chỉ show brand active cho kit này. Variant tự động suy ra (annotator không phải chọn) trừ khi brand có > 1 active variant cho cùng kit_context.
- Append annotation vào `annotations.jsonl` (idempotent — re-runnable).
- State: `st.session_state.track_idx` track vị trí hiện tại; resume từ track chưa annotate đầu tiên.

### `cli.py`
- Click-based CLI: `build-package`, `export`, `inspect`.
- Là entrypoint chính khi chạy production.

## Brand / Variant / Kit-context — chi tiết

### Vì sao tách layer Brand vs Variant

Mỗi sponsor master entity (vd "Aon") có thể có nhiều phiên bản hình ảnh (vd `aon_red` cho home, `aon_white` cho away). Trước refactor v0.2, code treat 21 variant như 21 brand riêng, gây 3 vấn đề:

1. **Reporting**: sponsor quan tâm "Aon được bao nhiêu exposure", không phải `aon_red` riêng — phải aggregate thủ công.
2. **Annotation**: annotator phải tự nhớ trận này home → chọn `aon_red`, không phải `aon_white`. Dễ sai.
3. **Training**: model phải học phân biệt `aon_red` vs `aon_white` mặc dù same brand → phí samples + risk confused.

Refactor v0.2 fix triệt để: brand là master entity, variant là appearance, kit_context là hoàn cảnh — 3 thứ độc lập.

### Cách kit_context hoạt động trong runtime

```
build_package(video, registry, MatchContext(kit_context="home"))
    └─> manifest.logo_templates.active_brands  # snapshot of brands active for "home"
    └─> reviewer/app.py reads this snapshot, shows only home brands
    └─> exporter aggregates by brand_id (or variant_id) per --class-mode

annotation row:
{"track_id": 42, "brand_id": "aon", "variant_id": "aon_red", "kit_context": "home", ...}
```

`active_brands` được snapshot tại build time vào manifest → reviewer/exporter không bị ảnh hưởng nếu registry sau đó thay đổi.

### Variant của brand cùng active cho 1 kit_context

Hiếm khi xảy ra (thường mỗi kit chỉ có 1 variant per brand). Nếu có (vd brand có cả `_v1` và `_v2` cho cùng home kit), reviewer sẽ hiển thị secondary dropdown để annotator chọn variant cụ thể.

### Brand không gắn với kit (single variant brands)

Dùng `kit_contexts: [any]` — luôn active cho mọi kit_context. Hầu hết các sponsor (KLG, ChadLaw, EM Workwear, …) thuộc loại này.

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
