# README-v3 — Single-Command Workflow (Host)

End-to-end hướng dẫn chạy v3 pipeline trên máy lab GPU. Tránh chạy rời rạc nhiều script — chỉ một lệnh `run_match.py` cho cả luồng từ video tới annotation package.

> Tài liệu này thay thế phần "Workflow" của `README.md` để tập trung vào host workflow tự động hóa.

---

## Mục tiêu

```
data/videos/match.mp4
       ↓
   ONE COMMAND
       ↓
data/annotation_packages/match/  ← sẵn sàng để annotate trong Streamlit
```

---

## 1. One-time setup (chạy 1 lần khi clone repo)

```bash
cd v3

# Virtualenv
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip

# Dependencies + editable install
pip install -r requirements.txt
pip install -e . --no-deps

# YOLO weights (được auto-download nếu thiếu, nhưng pre-cache cho ổn định)
mkdir -p weights
[ -f weights/yolo11l.pt ] || \
  wget -O weights/yolo11l.pt \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt

# Verify everything
python scripts/validate_setup.py
```

Output mong đợi: tất cả check `[OK]`, GPU detected, `brands.yaml` parse được 16 brands × 21 variants.

---

## 2. Chuẩn bị 1 video

Đặt video vào `data/videos/`:

```bash
cp /path/to/your/match.mp4 data/videos/
```

Mục tiêu: có file ở `data/videos/match.mp4`.

---

## 3. Chạy 1 trong 3 mode

### Mode A — Smoke test (no team filter)

Khi bạn chưa biết HSV ranges của kit, chỉ muốn xem pipeline có track được player không. Bỏ team filter → giữ mọi người trong frame (cả Bradford lẫn đối thủ lẫn staff).

```bash
python scripts/run_match.py \
    --video data/videos/match.mp4 \
    --no-filter \
    --kit-context home \
    --max-duration 60       # bỏ flag này khi chạy thật
```

**Kết quả**: vài trăm tracks bao gồm cả 2 đội. Mở reviewer xem tracks có đúng object không.

### Mode B — Auto-calibrate kit color (recommended cho lần đầu mỗi kit)

Tự động cluster torso colors → user pick cluster đúng → write meta.yaml.

**Step 1**: cluster

```bash
python scripts/run_match.py \
    --video data/videos/match.mp4 \
    --auto-calibrate
```

Script sẽ:
- Sample 40 frames đa dạng từ video
- YOLO detect persons → extract torso crops
- K-Means cluster (K=3) → 3 nhóm torso color
- Save 3 ảnh `cluster_*.jpg` trong `.cache/calib/match/`

**Step 2**: bạn mở `.cache/calib/match/cluster_*.jpg`, chọn cluster nào là Bradford.

**Step 3**: re-run với `--cluster-id N`:

```bash
python scripts/run_match.py \
    --video data/videos/match.mp4 \
    --auto-calibrate \
    --cluster-id 1 \
    --kit-context home \
    --opponent "Hull FC"
```

Auto-generate `data/videos/match.meta.yaml` với HSV ranges từ cluster đó, rồi chạy luôn pipeline.

### Mode C — Manual meta.yaml (recommended cho production)

Bạn đã có meta.yaml chuẩn (tự tay tune hoặc lưu từ lần auto-calibrate trước):

```bash
python scripts/run_match.py --video data/videos/match.mp4
```

Script tự tìm `data/videos/match.meta.yaml` (sidecar cùng tên video). Không cần truyền extra args.

---

## 4. Workflow mặc định (không suy nghĩ — paste vào terminal)

Trận đầu tiên của một kit mới (vd kit black):

```bash
# 1. Cluster torsos
python scripts/run_match.py --video data/videos/M06_black_1080p.mp4 --auto-calibrate

# 2. Inspect cluster_0.jpg / cluster_1.jpg / cluster_2.jpg
#    Pick the cluster that's Bradford (let's say cluster 1)
open .cache/calib/M06_black_1080p/

# 3. Generate meta + run pipeline
python scripts/run_match.py --video data/videos/M06_black_1080p.mp4 \
    --auto-calibrate --cluster-id 1 \
    --kit-context special --opponent "Hull FC"

# 4. Annotate
bash scripts/run_reviewer.sh data/annotation_packages/M06_black_1080p
```

Hoặc đầu-cuối với `--launch-reviewer`:

```bash
python scripts/run_match.py --video data/videos/M06_black_1080p.mp4 \
    --auto-calibrate --cluster-id 1 \
    --kit-context special \
    --launch-reviewer
```

Trận tiếp theo cùng kit: bỏ qua step 1-3, chỉ cần 1 lệnh:

```bash
python scripts/run_match.py --video data/videos/M07_black_1080p.mp4 --launch-reviewer
```

(Reuse meta.yaml từ trận trước bằng cách `cp data/videos/M06_black_1080p.meta.yaml data/videos/M07_black_1080p.meta.yaml`).

---

## 5. Sau annotation: export training data

```bash
# Brand-level multi-class (recommended cho training)
python -m track_annotation.cli export \
    --package data/annotation_packages/M06_black_1080p \
    --format yolo --output data/yolo_dataset_M06 --class-mode brand

# Single-class (cho Stage A class-agnostic detector)
python -m track_annotation.cli export \
    --package data/annotation_packages/M06_black_1080p \
    --format yolo --output data/yolo_dataset_M06_stage_a --class-mode single
```

---

## 6. Tổng quan các script

| Script | Vai trò | Khi nào dùng |
|---|---|---|
| `scripts/run_match.py` | **Wrapper end-to-end** | Lệnh chính, gọi từ ngoài |
| `scripts/run_pipeline.py` | Chỉ detect+track+package | Khi muốn skip validate/inspect/reviewer |
| `scripts/auto_calibrate_kit.py` | Cluster torso → suggest HSV | Khi muốn standalone calibration |
| `scripts/calibrate_meta.py` | HSV/region từ pixel bbox thủ công | Khi muốn fine-tune sau auto-calibrate |
| `scripts/validate_setup.py` | Check môi trường | Sau khi clone hoặc đổi env |
| `scripts/run_reviewer.sh` | Launch Streamlit | Sau khi pipeline xong |
| `scripts/normalize_logos.py` | Copy Kit Sponsors → templates dir | One-time setup logo templates |

---

## 7. Hiểu sâu auto-calibrate

Auto-calibration dựa trên hai assumption:
1. **Diversity sample**: 40 frame trải đều video bắt được mọi tình huống (gần/xa, sáng/tối)
2. **Color separability**: 2-3 đội + staff thường tách biệt rõ trong HSV space sau khi loại grass và skin

Hạn chế đã biết (đã gặp ở v2):
- **Lighting variation**: kit white dưới ánh đèn vàng có HSV khác kit white dưới sky → 1 cluster bị tách thành 2
- **Mud / kit dirt**: cuối hiệp 2 màu áo bị tối đi → drift khỏi cluster center
- **Skin tone overlap**: kit nâu/be dễ confuse với da (đã filter skin-heavy crops nhưng không hoàn hảo)

Khi auto-calibrate cho ra cluster không thuần (vd `cluster_1.jpg` có cả Bradford lẫn Hull):
- **Tăng K**: `--k-clusters 5` để phân tích chi tiết hơn
- **Tăng sample**: `--n-calib-frames 80`
- **Fall back manual**: dùng `scripts/calibrate_meta.py hsv` với pixel bbox thủ công

Auto-calibrate **không phải silver bullet** — nó là 1 helper để có HSV ranges baseline trong 30 giây thay vì 10 phút thủ công. Sau lần đầu, bạn nên fine-tune YAML bằng tay nếu thấy kết quả track chưa chuẩn.

---

## 8. Troubleshooting

| Triệu chứng | Cause khả dĩ | Fix |
|---|---|---|
| `0 tracks generated` | HSV ranges không match kit | Re-run với `--no-filter` để verify pipeline OK, rồi `--auto-calibrate` |
| `5000+ tracks` | Filter quá lỏng + không có ignore_regions | Update meta.yaml với ignore_regions cho UI overlay |
| Track ở vùng scoreboard | `ignore_regions` chưa cover đúng | Dùng `scripts/calibrate_meta.py region` để đo bbox chính xác |
| Track của đội đối thủ chiếm đa số | Color filter chọn cluster sai | Re-run `--auto-calibrate`, pick cluster khác |
| `ModuleNotFoundError: track_annotation` | Quên `pip install -e .` | Chạy lại `pip install -e . --no-deps` trong venv |
| `FileNotFoundError: brands.yaml` | Repo chưa pull file | Verify `data/logo_templates/brands.yaml` tồn tại; check `.gitignore` không nuốt |
| GPU OOM | imgsz=1280 quá lớn cho GPU nhỏ | Edit `configs/person_tracking.yaml`, hạ `detection.imgsz` xuống 960 |

---

## 9. Khi nào dùng v3 vs Roboflow / CVAT

- **v3 reviewer**: nhanh, brand assignment 1-click per track, propagate tự động sang frames trong track. Đủ cho 90% case.
- **CVAT export**: khi cần refine bbox (tracker miss/wrong) trước khi gán brand. `python -m track_annotation.cli export --format cvat` xuất XML.
- **Roboflow**: khi team annotation gồm nhiều người và cần cloud workflow. Export YOLO format rồi upload.

---

## 10. Các bước tiếp theo (sau khi annotate đủ frames)

```
data/yolo_dataset_*/
   ↓
Train Stage A (class-agnostic logo detector)
   ↓
Đặt weights mới vào weights/stage_a_logo_detector.pt
   ↓
Đổi config: configs/logo_tracking.yaml
   ↓
Lặp lại: python scripts/run_match.py --video NEW_MATCH.mp4 (now logo-tracking instead of person-tracking)
```

Khi sang v1 logo tracking, mỗi "track" là một logo instance thật chứ không còn là player → annotator chỉ confirm brand thay vì điền vị trí.

---

## License

Bradford Bulls Internal · 2026
