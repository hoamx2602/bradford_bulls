# Bradford Bulls — Track-Level Annotation Pipeline (v3)

Module phụ trợ cho việc annotation logo sponsorship trên video rugby Bradford Bulls.

Triển khai **ý tưởng 2 (track-level annotation)** + **ý tưởng 3 (video clip review)** từ master plan, giải quyết vấn đề frame mờ trên video thể thao 720p/1080p bằng cách chuyển paradigm từ frame-level sang track-level annotation.

---

## Vì sao cần module này

Trên video rugby 720p/1080p, motion blur khiến từng frame riêng lẻ khó nhận diện logo. Cách annotate frame-by-frame (chọn frame nét → vẽ box → chọn brand) tốn thời gian và inconsistent.

Module này thay đổi paradigm:

1. **Detect + track** đối tượng (player hoặc logo, tuỳ giai đoạn)
2. Mỗi track sinh ra một "annotation unit" gồm: 3 keyframe best-evidence + video clip 2 giây
3. Annotator quyết định brand **một lần cho cả track** thay vì mỗi frame
4. Label tự động propagate cho mọi frame trong track

→ Giảm 5–10x thời gian annotation, đồng thời tăng chất lượng vì mỗi quyết định brand dựa trên evidence tốt nhất trong track.

---

## Cấu trúc thư mục

```
v3/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/                            # YAML configs (Hydra-style)
│   ├── default.yaml                    # Default settings
│   ├── person_tracking.yaml            # v0: track players (before custom logo detector)
│   └── logo_tracking.yaml              # v1: track logos (after Stage A is trained)
│
├── data/                               # Local data (gitignored)
│   ├── videos/                         # Input videos
│   ├── logo_templates/                 # 21 brand templates
│   └── annotation_packages/            # Output: one folder per video
│
├── weights/                            # Model weights (gitignored)
│   └── README.md                       # How to obtain weights
│
├── src/
│   └── track_annotation/
│       ├── __init__.py
│       ├── config.py                   # Pydantic config schema
│       ├── cli.py                      # Command-line interface
│       │
│       ├── pipeline/                   # Core processing pipeline
│       │   ├── detect_track.py         # Detection + BoT-SORT tracking
│       │   ├── keyframe.py             # Best-evidence keyframe selection
│       │   ├── clip.py                 # Video clip extraction
│       │   ├── pose_align.py           # Pose-aligned multi-frame fusion (idea 1)
│       │   └── package_builder.py      # Assemble annotation package
│       │
│       ├── exporters/                  # Format converters
│       │   ├── cvat.py                 # CVAT XML
│       │   ├── yolo.py                 # YOLO format (for training)
│       │   └── roboflow.py             # Roboflow upload
│       │
│       ├── reviewer/                   # Streamlit annotation UI
│       │   └── app.py
│       │
│       └── utils/                      # Shared utilities
│           ├── video_io.py             # Video reading/writing
│           ├── geometry.py             # Bbox math, IoU, sharpness
│           └── logging.py              # Loguru-based logging
│
├── scripts/                            # Standalone runnable scripts
│   ├── run_pipeline.py                 # End-to-end pipeline
│   ├── run_reviewer.sh                 # Launch Streamlit reviewer
│   └── validate_setup.py               # Verify environment + weights
│
├── notebooks/                          # Demo notebooks
│   └── 01_demo_pipeline.ipynb
│
├── tests/                              # Pytest unit tests
│   ├── test_keyframe.py
│   └── test_geometry.py
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

Đặt file YOLO weights vào `weights/`:

```bash
cp ../yolo11l.pt weights/yolo11l.pt
# Sau khi train Stage A custom detector:
# cp path/to/stage_a.pt weights/stage_a_logo_detector.pt
```

### 3. Chuẩn bị logo templates

Copy 21 logo template vào `data/logo_templates/` (ưu tiên PNG transparent, 512×512).
Đã có script chuẩn hoá ở `scripts/normalize_logos.py` (TBD).

### 4. Verify setup

```bash
python scripts/validate_setup.py
```

Kết quả mong đợi: GPU detected, weights loaded, dependencies OK.

---

## Sử dụng

### Workflow đầy đủ — từ video tới annotation package

```bash
# 1. Chạy detect + track + keyframe selection + clip extraction
python -m track_annotation.cli build-package \
    --video data/videos/match_2026_04_15.mp4 \
    --config configs/person_tracking.yaml \
    --output data/annotation_packages/match_2026_04_15

# 2. Mở Streamlit reviewer để annotate
bash scripts/run_reviewer.sh data/annotation_packages/match_2026_04_15

# 3. Sau khi annotate xong, export sang format YOLO để train
python -m track_annotation.cli export \
    --package data/annotation_packages/match_2026_04_15 \
    --format yolo \
    --output data/yolo_dataset
```

### Hai chế độ vận hành

**v0 — Person tracking (giai đoạn vòng 1 annotation):**

Dùng YOLO11-L pretrained COCO để detect & track player. Mỗi player track là một annotation unit; annotator label các logo nhìn thấy trên jersey trong track đó.

```bash
python -m track_annotation.cli build-package \
    --video VIDEO --config configs/person_tracking.yaml --output OUT
```

**v1 — Logo tracking (sau khi có Stage A custom detector):**

Swap detector sang custom Stage A class-agnostic logo detector. Track logo trực tiếp; annotator chỉ cần confirm brand.

```bash
python -m track_annotation.cli build-package \
    --video VIDEO --config configs/logo_tracking.yaml --output OUT
```

---

## Format output (annotation package)

```
data/annotation_packages/match_2026_04_15/
├── manifest.json                       # Metadata: video info, config used, track count
├── tracks/
│   ├── track_00001/
│   │   ├── keyframe_sharpest.jpg       # Frame có sharpness cao nhất
│   │   ├── keyframe_largest.jpg        # Frame có bbox to nhất
│   │   ├── keyframe_midpoint.jpg       # Frame ở giữa track
│   │   ├── clip.mp4                    # Video clip 2 giây
│   │   └── meta.json                   # bbox sequence, timestamps, sharpness scores
│   ├── track_00002/
│   └── ...
└── annotations.jsonl                   # Output từ reviewer (1 dòng = 1 brand assignment)
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

Throughput tham khảo trên RTX 4500 Ada (1 trận 2h, 1080p, 5 fps processing):

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
