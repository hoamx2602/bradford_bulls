# Bradford Bulls — Track-Level Annotation Pipeline (v3)

Module phụ trợ cho việc annotation logo sponsorship trên video rugby Bradford Bulls.

Triển khai **ý tưởng 2 (track-level annotation)** + **ý tưởng 3 (video clip review)** từ master plan, giải quyết vấn đề frame mờ trên video thể thao 720p/1080p bằng cách chuyển paradigm từ frame-level sang track-level annotation.

Cộng với mô hình **Brand / Variant / Kit-Context** để xử lý chính xác việc cùng một sponsor có nhiều phiên bản logo (light/dark/red/white) tuỳ kit của trận đấu.

---

## Vì sao cần module này

Trên video rugby 720p/1080p, motion blur khiến từng frame riêng lẻ khó nhận diện logo. Cách annotate frame-by-frame (chọn frame nét → vẽ box → chọn brand) tốn thời gian và inconsistent.

Module này thay đổi paradigm:

1. **Detect + track** đối tượng (player hoặc logo, tuỳ giai đoạn)
2. Mỗi track sinh ra một "annotation unit" gồm: 3 keyframe best-evidence + video clip 2 giây
3. Annotator quyết định **brand** (không phải variant) **một lần cho cả track** — variant tự động suy ra từ kit_context của trận
4. Label tự động propagate cho mọi frame trong track

→ Giảm 5–10x thời gian annotation, đồng thời tăng chất lượng vì mỗi quyết định brand dựa trên evidence tốt nhất trong track.

---

## Mô hình Brand / Variant / Kit-Context

| Khái niệm | Định nghĩa | Ví dụ |
|---|---|---|
| **Brand** | Sponsor master entity, dùng cho định giá & reporting | `aon`, `cch`, `mcp` |
| **Variant** | Một phiên bản hình ảnh cụ thể của brand | `aon_red`, `aon_white` |
| **Kit context** | Kit cầu thủ mặc trong trận | `home`, `away`, `special` |
| **Active variant** | Variant nào dùng cho trận này | `aon_red` khi `kit_context=home` |

Mỗi variant trong `data/logo_templates/brands.yaml` khai báo `kit_contexts: [home]` hoặc `[any]`. Tại runtime:

- Annotator chỉ thấy các brand active cho kit_context của trận → giảm dropdown từ 21 → ~16 entry
- Variant tự động suy ra từ context (không cần annotator nhớ "trận này home → chọn aon_red")
- YOLO export mặc định dùng **brand-level class** (16 class) thay vì variant-level (21 class) → consolidate samples, model train tốt hơn
- Stage B brand recognition (sau này) chỉ match với template active cho kit_context → giảm confusion

---

## Cấu trúc thư mục

```
v3/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/                            # YAML configs (Pydantic-validated)
│   ├── default.yaml
│   ├── person_tracking.yaml            # v0: track players
│   └── logo_tracking.yaml              # v1: track logos
│
├── data/                               # Local data (gitignored)
│   ├── videos/                         # Input videos
│   │   └── match.meta.example.yaml     # Example match metadata sidecar
│   ├── logo_templates/
│   │   ├── brands.yaml                 # **Brand/variant/kit_context registry**
│   │   ├── aon/{red,white}.png
│   │   ├── cch/{black,white}.png
│   │   ├── mcp/{home,away}.png
│   │   ├── chadlaw.png
│   │   └── ...
│   └── annotation_packages/            # Output: one folder per video
│
├── weights/
│   └── README.md
│
├── src/
│   └── track_annotation/
│       ├── __init__.py
│       ├── config.py                   # BrandRegistry, MatchContext, etc.
│       ├── cli.py                      # build-package, export, brands, inspect
│       │
│       ├── pipeline/
│       │   ├── detect_track.py
│       │   ├── keyframe.py
│       │   ├── clip.py
│       │   ├── pose_align.py
│       │   └── package_builder.py
│       │
│       ├── exporters/
│       │   ├── cvat.py
│       │   ├── yolo.py                 # class_mode = single | brand | variant
│       │   └── roboflow.py
│       │
│       ├── reviewer/
│       │   └── app.py                  # Brand-aware Streamlit UI
│       │
│       └── utils/
│           ├── video_io.py
│           ├── geometry.py
│           └── logging.py
│
├── scripts/
│   ├── run_pipeline.py
│   ├── run_reviewer.sh
│   ├── validate_setup.py
│   └── normalize_logos.py              # Copy Kit Sponsors → data/logo_templates
│
├── notebooks/
│   └── 01_demo_pipeline.ipynb
│
├── tests/
│   ├── test_geometry.py
│   └── test_keyframe.py
│
└── docs/
    ├── ARCHITECTURE.md
    └── ANNOTATION_GUIDE.md
```

---

## Setup

### 1. Tạo môi trường

```bash
cd v3
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2. Chuẩn bị weights

```bash
cp ../yolo11l.pt weights/yolo11l.pt
```

### 3. Chuẩn bị logo templates

Copy từ `Kit Sponsors/` cũ vào structure mới (1 lệnh):

```bash
# Dry-run để xem mapping
python scripts/normalize_logos.py --source "../Kit Sponsors/Kit Sponsors" --dry-run

# Thực thi
python scripts/normalize_logos.py --source "../Kit Sponsors/Kit Sponsors"
```

Script tự match filename → variant_id qua heuristic (vd `"3 - CCH - Master Logo Black [A3 Digital].png"` → `cch_black`). Các file không match được sẽ liệt kê để bạn map thủ công.

### 4. Verify setup

```bash
python scripts/validate_setup.py
```

Kết quả mong đợi: GPU detected, weights loaded, brands.yaml parse OK, 21 variants với template files đầy đủ.

---

## Sử dụng

### Workflow 1 — Build package với explicit kit context

```bash
python scripts/run_pipeline.py \
    --video data/videos/match_2026_04_15.mp4 \
    --config configs/person_tracking.yaml \
    --output data/annotation_packages/match_2026_04_15 \
    --kit-context home
```

### Workflow 2 — Build package với match metadata sidecar (recommended)

⚠️ **Quan trọng**: nếu chỉ truyền `--kit-context` mà không có sidecar metadata với `target_team` + `ignore_regions`, pipeline sẽ track **mọi người trong frame** (cả đội đối thủ, trọng tài, staff, khán giả) và có thể track cả vùng UI overlay (scoreboard, channel logo). Hậu quả: package có hàng nghìn track sai.

Tạo `data/videos/match_2026_04_15.meta.yaml`:

```yaml
kit_context: home
match_date: "2026-04-15"
opponent: "Wakefield Trinity"

# Filter: chỉ giữ player có màu áo Bradford
target_team:
  primary_colors:
    - {name: red,    h: [0, 10],    s: [120, 255], v: [70, 255]}
    - {name: amber,  h: [15, 30],   s: [100, 255], v: [120, 255]}
  min_team_score: 0.10

# Filter: bỏ vùng overlay UI khỏi detection
ignore_regions:
  - [0.00, 0.85, 0.50, 1.00]    # bottom-left scoreboard
  - [0.85, 0.00, 1.00, 0.15]    # top-right BullsTV logo
```

Xem `data/videos/match.meta.example.yaml` (home, red+amber) và `match.meta.away.example.yaml` (away, white) để có template đầy đủ.

Rồi chạy:

```bash
python scripts/run_pipeline.py \
    --video data/videos/match_2026_04_15.mp4 \
    --config configs/person_tracking.yaml \
    --output data/annotation_packages/match_2026_04_15 \
    --match-meta data/videos/match_2026_04_15.meta.yaml
```

Sidecar approach có ưu điểm: metadata gắn liền với video file, không bị lạc, dễ batch nhiều trận.

### Workflow 3 — Annotate qua Streamlit reviewer

```bash
bash scripts/run_reviewer.sh data/annotation_packages/match_2026_04_15
```

Reviewer tự load `kit_context` từ manifest → chỉ show brand active cho kit này.

### Workflow 4 — Export YOLO format

```bash
# Mặc định: brand-level multi-class (recommended for most training)
python -m track_annotation.cli export \
    --package data/annotation_packages/match_2026_04_15 \
    --format yolo \
    --output data/yolo_dataset

# Single-class (cho Stage A class-agnostic detector)
python -m track_annotation.cli export \
    --package data/annotation_packages/match_2026_04_15 \
    --format yolo \
    --output data/yolo_dataset_stage_a \
    --class-mode single

# Variant-level (cho fine-grained model nâng cao)
python -m track_annotation.cli export \
    --package data/annotation_packages/match_2026_04_15 \
    --format yolo \
    --output data/yolo_dataset_variant \
    --class-mode variant
```

### Workflow 5 — Liệt kê brand cho kit context

```bash
python -m track_annotation.cli brands --kit-context home
python -m track_annotation.cli brands --kit-context away
```

### Workflow 6 — Inspect package

```bash
python -m track_annotation.cli inspect --package data/annotation_packages/match_2026_04_15
```

---

## Format output (annotation package)

```
data/annotation_packages/match_2026_04_15/
├── manifest.json                       # match metadata + kit_context + active brand pool
├── tracks/
│   ├── track_00001/
│   │   ├── keyframe_*_full.jpg
│   │   ├── keyframe_*_crop.jpg
│   │   ├── clip.mp4
│   │   └── meta.json
│   └── ...
└── annotations.jsonl                   # 1 dòng = 1 track-brand assignment
```

**Mỗi dòng `annotations.jsonl`:**

```json
{
  "track_id": 42,
  "brand_id": "aon",
  "variant_id": "aon_red",
  "position": "chest_front",
  "visibility_quality": "clear",
  "is_target_team": true,
  "skip": false,
  "kit_context": "home"
}
```

---

## Hardware / Compute

| Hạng mục | Cấu hình tối thiểu | Khuyến nghị |
|---|---|---|
| GPU | RTX 3060 (8 GB) | RTX 4500 Ada (24 GB) |
| RAM | 16 GB | 32 GB |
| Storage | 100 GB | 500 GB SSD |
| Python | 3.10+ | 3.10 |
| CUDA | 11.8+ | 12.1 |

Throughput trên RTX 4500 Ada (1 trận 2h, 1080p):
- detect + track: ~25 phút
- keyframe + clip extraction: ~10 phút
- **Total: ~35 phút / trận**

---

## Tích hợp với pipeline lớn

Module này là bước **annotation** trong pipeline 6 tầng của master plan. Output (YOLO format) được dùng để train Stage A class-agnostic detector. Sau đó pipeline chuyển sang chế độ logo tracking và lặp lại để annotate vòng 2 (hard example mining).

Xem `docs/ARCHITECTURE.md` để biết chi tiết tích hợp.

---

## License & Authorship

Bradford Bulls Internal Project · 2026
