# Bradford Bulls — AI Sponsorship Exposure Valuation
# Tài liệu đặc tả dự án toàn diện

> Cập nhật: 2026-04-07
> Trạng thái: Phase 1 đang thực hiện

---

## MỤC LỤC

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Bài toán kinh doanh](#2-bài-toán-kinh-doanh)
3. [Kiến trúc hệ thống](#3-kiến-trúc-hệ-thống)
4. [Phase 1: Thu thập & chuẩn bị dữ liệu](#4-phase-1-thu-thập--chuẩn-bị-dữ-liệu)
5. [Phase 2: Huấn luyện model](#5-phase-2-huấn-luyện-model)
6. [Phase 3: Đo lường Exposure](#6-phase-3-đo-lường-exposure)
7. [Phase 4: Mô hình định giá](#7-phase-4-mô-hình-định-giá)
8. [Phase 5: Báo cáo & mở rộng](#8-phase-5-báo-cáo--mở-rộng)
9. [Phân tích rủi ro & giải pháp](#9-phân-tích-rủi-ro--giải-pháp)
10. [Yêu cầu kỹ thuật](#10-yêu-cầu-kỹ-thuật)
11. [Cấu trúc dự án](#11-cấu-trúc-dự-án)
12. [Bài học đã rút ra](#12-bài-học-đã-rút-ra)

---

## 1. Tổng quan dự án

### 1.1. Ý tưởng

Hiện tại Bradford Bulls (và hầu hết các CLB thể thao) định giá vị trí logo sponsor trên áo thi đấu bằng **phỏng đoán cảm tính** — Main Sponsor trả 26%, Sleeve trả 4%, v.v. Không ai biết thực sự logo nào được nhìn thấy nhiều nhất trên TV/video.

Dự án này xây dựng hệ thống AI để **đo lường khách quan** mức độ hiện diện (exposure) của từng logo sponsor trong video trận đấu, từ đó:
- Chứng minh giá trị thực của từng vị trí sponsor bằng dữ liệu
- Đề xuất mức giá hợp lý hơn cho từng vị trí
- Tạo báo cáo cho sponsor: "Logo bạn xuất hiện X giây, chiếm Y% màn hình, tương đương Z giá trị quảng cáo"

### 1.2. Tầm nhìn dài hạn

Xây dựng thành **SaaS product** cho các CLB thể thao khác:
- CLB upload video trận đấu → hệ thống tự động phân tích → báo cáo exposure
- Mô hình kinh doanh: subscription per-season hoặc per-match

### 1.3. Scope hiện tại (MVP)

- 1 đội: Bradford Bulls
- 1 bộ áo: Home kit 2025-26
- Vài trận đấu từ YouTube (1080p broadcast)
- 21 logo classes
- Output: Exposure report per match

---

## 2. Bài toán kinh doanh

### 2.1. Bảng giá hiện tại

Từ file `Bradford Bulls (Curent Pricing).csv`:

| Vị trí | % tổng giá | Ghi chú |
|--------|-----------|---------|
| Main Sponsor | 26% | AON — logo lớn nhất, trước ngực |
| Collar Back | 8% | Phía sau cổ áo |
| Collar Bone | 8% | ATM Hospitality — xương quai xanh |
| Chest (opp Badge) | 7% | CCH — ngực đối diện huy hiệu |
| Sleeve 1 | 4% | ChadLaw |
| Sleeve 2 | 11% | Fairway Flooring / MCP |
| Sleeve 3 | 4% | EM Workwear |
| Top Back | 5% | KLG — lưng trên |
| Nape Neck | 3% | Gáy |
| Bottom Back | 3% | Bartercard — lưng dưới |
| Top Back Shorts | 5% | Quần phía sau trên |
| Shorts Front | 3% | MNA Cladding — quần phía trước |
| Shorts Back 1 | 3% | MNA Support — quần phía sau 1 |
| Shorts Back 2 | 3% | Romantica — quần phía sau 2 |
| Socks | 1% | Top Notch — tất |
| **TOTAL** | **100%** | |

### 2.2. Câu hỏi cần trả lời

1. **Main Sponsor (26%)** thực sự được nhìn thấy nhiều nhất? Hay logo sau lưng (khi cầu thủ chạy xa camera) lại xuất hiện nhiều hơn?
2. **Socks (1%)** có thực sự chỉ 1%? Hay trong close-up cầu thủ đang chạy, tất chiếm diện tích đáng kể?
3. **Sleeve 2 (11%)** đắt gấp gần 3x Sleeve 1 (4%) — có hợp lý không?
4. Sponsor nào đang **overpay** vs **underpay** so với exposure thực?

### 2.3. Giá trị sản phẩm mang lại

**Cho CLB:**
- Dữ liệu đàm phán với sponsor khi ký hợp đồng mới
- Justify tăng/giảm giá vị trí
- Báo cáo cho sponsor hiện tại: "Đây là giá trị bạn nhận được"

**Cho Sponsor:**
- Biết chính xác ROI của vị trí sponsor
- So sánh exposure giữa các vị trí để chọn vị trí tốt nhất
- Dữ liệu cho internal marketing budget allocation

---

## 3. Kiến trúc hệ thống

### 3.1. Pipeline tổng thể

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                     │
│  Video trận đấu (YouTube 1080p, ~90 phút, 30 FPS)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                 ┌───────────▼───────────┐
                 │    PHASE 1            │
                 │    Training Data      │
                 │                       │
                 │  Sample 1 FPS         │
                 │  Filter quality       │
                 │  De-duplicate         │
                 │  → 300-400 frames     │
                 │  → Upload Roboflow    │
                 │  → Manual annotate    │
                 └───────────┬───────────┘
                             │
                 ┌───────────▼───────────┐
                 │    PHASE 2            │
                 │    Train Model        │
                 │                       │
                 │  YOLOv8 custom        │
                 │  21 logo classes      │
                 │  Train on Roboflow    │
                 │  or Colab GPU         │
                 └───────────┬───────────┘
                             │
                 ┌───────────▼───────────┐
                 │    PHASE 3            │
                 │    Measure Exposure   │
                 │                       │
                 │  Run model on FULL    │
                 │  video (30 FPS)       │
                 │  → 162,000 frames     │
                 │  Track per-logo       │
                 │  exposure metrics     │
                 └───────────┬───────────┘
                             │
                 ┌───────────▼───────────┐
                 │    PHASE 4            │
                 │    Pricing Model      │
                 │                       │
                 │  Compare exposure     │
                 │  vs current pricing   │
                 │  → Recommendations    │
                 └───────────┬───────────┘
                             │
                 ┌───────────▼───────────┐
                 │    PHASE 5            │
                 │    Reporting          │
                 │                       │
                 │  Per-match report     │
                 │  Per-sponsor card     │
                 │  Season trends        │
                 │  → SaaS dashboard     │
                 └───────────────────────┘
```

### 3.2. Tech Stack

| Layer | Tool | Lý do chọn |
|-------|------|-----------|
| Video download | yt-dlp + ffmpeg | Tốt nhất cho YouTube, hỗ trợ merge format |
| Frame processing | OpenCV (cv2) | Industry standard, nhanh, GPU support |
| Person detection | YOLOv8m (ultralytics) | Pre-trained COCO, batch inference GPU |
| Logo detection | YOLOv8 custom trained | Nhẹ, realtime, dễ train trên Roboflow |
| Annotation platform | Roboflow | UI tốt, train trực tiếp, export YOLO format |
| De-duplication | imagehash (pHash) | Perceptual hash, robust với minor changes |
| Sharpness scoring | Laplacian + Tenengrad + NVar | Multi-metric, đáng tin cậy hơn single metric |
| Training compute | Google Colab (T4/A100 GPU) | Free/cheap, đủ mạnh |
| Storage | Google Drive | Persistent, accessible từ Colab |
| Local development | MacBook Apple Silicon (MPS) | Code + test nhỏ, không training nặng |
| Reporting | Python (matplotlib, pandas) | Flexible, custom charts |

---

## 4. Phase 1: Thu thập & chuẩn bị dữ liệu

### 4.1. Mục tiêu

Lấy 300-400 full frames chất lượng cao từ video trận đấu, upload lên Roboflow để annotate thủ công.

### 4.2. Pipeline chi tiết

```
Video (1080p, 90 phút, 30 FPS = ~162,000 frames)
    │
    ├─ [1] Sample 1 FPS
    │       162,000 → ~5,400 frames
    │       Lý do: 30 frame liên tiếp gần giống nhau, 1 FPS đủ đại diện
    │
    ├─ [2] YOLOv8 person detection
    │       Filter: chỉ giữ frame có ≥1 cầu thủ
    │       Batch inference 16 frames/batch trên GPU
    │       ~5,400 → ~4,000 frames (bỏ frame không có người: replay đồ họa, sân trống)
    │
    ├─ [3] Sharpness scoring
    │       Multi-metric: Laplacian (40%) + Tenengrad (35%) + NVar (25%)
    │       Filter: score ≥ 0.30
    │       ~4,000 → ~2,500 frames (bỏ frame motion blur nặng)
    │
    ├─ [4] Frame scoring tổng hợp
    │       score = sharpness × 0.40
    │             + player_count_norm × 0.30   (nhiều cầu thủ = nhiều logo)
    │             + player_coverage × 0.30     (cầu thủ lớn = logo rõ hơn)
    │       Sort theo score giảm dần
    │
    ├─ [5] De-duplicate (pHash)
    │       So sánh perceptual hash giữa các frame
    │       Hamming distance < 6 → coi là trùng → bỏ
    │       ~2,500 → ~1,000 frames
    │
    ├─ [6] Time diversity selection
    │       Chia video thành windows 10 giây
    │       Max N frames per window (tránh lấy quá nhiều từ 1 pha)
    │       Lấy top 300-400 theo score
    │
    └─ Output: 300-400 full frames (.jpg, quality 95%)
               + metadata CSV (frame_num, timestamp, score, n_players)
```

### 4.3. Notebook

**File**: `notebooks/colab_03_frame_extraction.ipynb`

| Cell | Nội dung | Ghi chú |
|------|---------|---------|
| 1 | Install packages | Restart runtime sau cell này |
| 2 | Config (TEST_MODE) | TEST: 1000 frames/20 select. PROD: full/400 |
| 3 | Setup (imports, Drive, paths) | |
| 4 | Download/load video | Skip nếu đã có trên Drive |
| 5 | Helper functions | Sharpness, timestamp format |
| 6 | Sample + detect + score | Main loop, GPU batch |
| 7 | De-dup + select top N | pHash + time diversity |
| 8 | Save frames to Drive | JPG 95% + CSV metadata |
| 9 | Preview top 12 | Visual verification |
| 10 | Upload to Roboflow | Raw frames, no annotations |
| 11 | Next steps | Class table, annotation guide |

### 4.4. Config parameters

| Parameter | Test | Production | Ý nghĩa |
|-----------|------|-----------|---------|
| `MAX_FRAMES` | 1000 | None (all) | Số frame quét tối đa |
| `TARGET_FRAMES` | 20 | 400 | Số frame chọn cuối cùng |
| `TARGET_FPS` | 1 | 1 | Sample rate |
| `MIN_PLAYERS` | 1 | 1 | Min cầu thủ trong frame |
| `MIN_SHARPNESS` | 0.30 | 0.30 | Ngưỡng nét tối thiểu (0-1) |
| `DEDUP_HASH_THRESH` | 6 | 6 | pHash distance tối thiểu |
| `TIME_BUCKET_SEC` | 10 | 10 | Window đa dạng thời gian |

### 4.5. Sharpness scoring — chi tiết kỹ thuật

Dùng 3 metrics kết hợp thay vì 1:

```python
score = (
    min(laplacian_variance / 300, 1.0) × 0.40    # Edge detection
  + min(tenengrad / 5000, 1.0)        × 0.35    # Sobel gradient magnitude
  + min(normalized_variance / 50, 1.0) × 0.25    # Contrast/detail
)
```

| Metric | Đo gì | Ưu điểm | Nhược điểm |
|--------|-------|---------|-----------|
| Laplacian variance | Cạnh (edges) | Nhanh, trực quan | Nhạy cảm với noise |
| Tenengrad (Sobel) | Gradient magnitude | Robust với motion blur | Chậm hơn Laplacian |
| Normalized variance | Contrast/chi tiết | Bổ trợ cho 2 cái trên | Không đo blur trực tiếp |

**Tại sao 3 metrics?**: Laplacian đôi khi cho điểm cao với frame có texture nhiễu nhưng thực tế mờ. Kết hợp 3 metrics giảm false positive.

### 4.6. De-duplication — chi tiết kỹ thuật

**Perceptual Hash (pHash)**:
- Resize frame → 32×32 → DCT → lấy top-left 8×8 → binary hash 64-bit
- So sánh bằng Hamming distance (số bit khác nhau)
- Distance < 6 → coi là giống nhau

**Tại sao pHash thay vì exact hash?**: Video broadcast thường có overlay thay đổi (đồng hồ, tỷ số) nhưng nội dung giống nhau. pHash robust với minor changes.

**Tại sao threshold 6?**: 
- 0-4: gần như giống hệt (cùng frame, khác compression)
- 5-8: rất giống (cùng pha, khác vài frame)
- 9-15: tương tự (cùng cảnh, khác góc)
- >15: khác nhau

Threshold 6 loại bỏ frame gần giống nhưng vẫn giữ frame cùng cảnh nhưng khác đáng kể.

### 4.7. Annotation Guidelines

**PHẢI ĐỊNH NGHĨA RÕ TRƯỚC KHI BẮT ĐẦU ANNOTATE:**

#### 4.7.1. Quy tắc vẽ bounding box

| Quy tắc | Chi tiết |
|---------|---------|
| Tight box | Vẽ sát viền logo, không thừa padding |
| Logo bị che < 50% | Annotate — vẽ bbox cho phần nhìn thấy + phần đoán được |
| Logo bị che ≥ 50% | **KHÔNG annotate** |
| Logo < 15px width | **KHÔNG annotate** — quá nhỏ để model học |
| Logo mờ nhưng đọc được | Annotate — model cần học cả trường hợp này |
| Logo mờ không đọc được | **KHÔNG annotate** |
| 2 logo cùng sponsor, khác vị trí | 2 bbox riêng biệt |
| Logo trong replay | Annotate bình thường |
| Logo trên bảng LED sân | **KHÔNG annotate** — chỉ annotate logo trên áo/quần/tất |
| Logo đội đối thủ | **KHÔNG annotate** — chỉ Bradford Bulls |
| Logo trên áo HLV/staff | **KHÔNG annotate** — chỉ áo thi đấu |

#### 4.7.2. Phân biệt variants

| Trường hợp | Cách gán class |
|------------|---------------|
| AON màu đỏ trên nền trắng | `aon_red` (class 0) |
| AON màu trắng trên nền đỏ/đen | `aon_white` (class 1) |
| CCH logo đen | `cch_black` (class 3) |
| CCH logo trắng | `cch_white` (class 4) |
| MCP trên áo đội nhà | `mcp_home` (class 10) |
| MCP trên áo đội khách | `mcp_away` (class 9) |
| Romantica chữ trắng | `romantica_white` (class 18) |
| Romantica chữ đen | `romantica_black` (class 19) |
| Paints & Lacquers nền vàng | `paints_lacquers_yellow` (class 13) |
| Paints & Lacquers nền đỏ | `paints_lacquers_red` (class 17) |
| Không chắc variant nào | Chọn cái gần nhất, **nhất quán** quan trọng hơn chính xác |

#### 4.7.3. Checklist trước khi annotate

- [ ] Tạo tất cả 21 classes trên Roboflow project
- [ ] Annotate thử 10 frame đầu → review lại → điều chỉnh quy tắc nếu cần
- [ ] Đảm bảo mỗi class có ≥15 annotations (tối thiểu để model học)
- [ ] Ghi chú class nào khó annotate → cần thêm frame cho class đó

### 4.8. 21 Logo Classes — đầy đủ

| ID | Code Name | Logo Name | Vị trí trên áo | File tham khảo (Kit Sponsors/) |
|----|-----------|-----------|----------------|-------------------------------|
| 0 | `aon_red` | Aon (Red) | Main Sponsor — ngực | `1 - aon_logo_signature_red_rgb (2).png` |
| 1 | `aon_white` | Aon (White) | Main Sponsor — ngực | `1 - aon_logo_white_rgb (3).png` |
| 2 | `atm_hospitality` | ATM Hospitality | Collar Bone | `2 - ATM-Hospitality-Logo-New-Font.png` |
| 3 | `cch_black` | CCH (Black) | Chest (opp Badge) | `3 - CCH - Master Logo Black [A3 Digital].png` |
| 4 | `cch_white` | CCH (White) | Chest (opp Badge) | `3 - CCH - Master Logo White [A3 Digital].png` |
| 5 | `chadlaw` | ChadLaw | Sleeve 1 | `4 - ChadLaw1.png` |
| 6 | `em_workwear` | EM Workwear | Sleeve 3 | `5 - EM workwear logo.png` |
| 7 | `fairway_flooring` | Fairway Flooring | Sleeve 2 | `6 - Fairway Flooring Ltd Logo nO NUMBER.jpg` |
| 8 | `klg` | KLG | Top Back | `7 - KLG Transparent Final.png` |
| 9 | `mcp_away` | MCP (Away) | Sleeve 2 (away kit) | `8 - MCP Away.png` |
| 10 | `mcp_home` | MCP (Home) | Sleeve 2 (home kit) | `9 - MCP.png` |
| 11 | `mna_cladding` | MNA Cladding | Shorts Front | `10 - MNA Cladding.png` |
| 12 | `mna_support` | MNA Support | Shorts Back 1 | `11 - MNA Support Services.png` |
| 13 | `paints_lacquers_yellow` | Paints & Lacquers (yellow) | — | `12 - yellow.jpg` |
| 14 | `top_notch` | Top Notch | Socks | `13 - Top Notch Logo.png` |
| 15 | `bartercard` | Bartercard | Bottom Back | `Bartercard.eps` |
| 16 | `floor_tonic` | Floor Tonic | — | `Floor tonic Logo.pdf` |
| 17 | `paints_lacquers_red` | Paints & Lacquers (red) | — | `Paints & Laquers Logo FINAL.pdf` |
| 18 | `romantica_white` | Romantica (White) | Shorts Back 2 | `Romantica Beds - Logo FINAL WHITE.pdf` |
| 19 | `romantica_black` | Romantica (Black) | Shorts Back 2 | `romantica black.jpg` |
| 20 | `acs_group` | ACS Group | — | `acs_group_full_colour.svg` |

### 4.9. Số lượng training data cần thiết

| Mục tiêu | Frames | Annotations ước tính | Kết quả kỳ vọng |
|----------|--------|---------------------|-----------------|
| MVP / test | 100-200 | ~500-1000 | mAP 30-50% — proof of concept |
| Good | 300-400 | ~1500-2500 | mAP 50-70% — usable |
| Very good | 500-800 | ~3000-5000 | mAP 70-85% — production ready |
| Excellent | 1000+ | ~5000+ | mAP 85%+ — nhưng diminishing returns |

**Ước tính thời gian annotate:**
- Trung bình ~5-8 logos per frame (không phải frame nào cũng thấy tất cả)
- ~30-60 giây per frame (vẽ bbox + gán class)
- 400 frames × 45s = ~5 giờ annotate

**Class imbalance dự kiến:**
- `aon_red` (Main Sponsor, to nhất) → xuất hiện nhiều nhất → nhiều annotation nhất
- `top_notch` (Socks) → rất ít close-up → ít annotation
- `acs_group` → có thể không xuất hiện trong video → 0 annotation
- Cần thêm frame cho class thiếu (re-run pipeline với filter cụ thể)

---

## 5. Phase 2: Huấn luyện model

### 5.1. Mục tiêu

Train YOLOv8 model detect 21 logo classes trên frame trận đấu.

### 5.2. Option A: Train trên Roboflow

**Ưu điểm**: Không cần setup, UI đẹp, auto-deploy API
**Nhược điểm**: Ít control hyperparameters, giới hạn free tier

```
Roboflow → Versions → Generate (add augmentation) → Train → Deploy
```

Augmentation đề xuất:
- Flip horizontal: **KHÔNG** (text bị ngược → model confused)
- Rotation: ±15° (cầu thủ nghiêng nhẹ)
- Brightness: ±25% (ánh sáng sân khác nhau)
- Blur: 0-2px (simulate motion blur nhẹ)
- Crop: 0-10% (simulate zoom khác nhau)
- Mosaic: có (tăng diversity)

### 5.3. Option B: Train trên Google Colab

**Ưu điểm**: Full control, custom augmentation, larger models
**Nhược điểm**: Cần viết training script

```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # medium model, balance speed/accuracy
results = model.train(
    data="dataset.yaml",    # exported from Roboflow
    epochs=100,
    imgsz=1280,             # larger → better for small logos
    batch=8,                # T4: 8, A100: 16-32
    patience=20,            # early stopping
    device="cuda",
)
```

### 5.4. Chọn model size

| Model | Params | Speed (T4) | Accuracy | Khi nào dùng |
|-------|--------|-----------|----------|-------------|
| YOLOv8n | 3.2M | ~5ms | Thấp nhất | Không khuyến nghị — logo nhỏ |
| YOLOv8s | 11.2M | ~8ms | Trung bình | Nếu cần realtime |
| **YOLOv8m** | **25.9M** | **~15ms** | **Tốt** | **Khuyến nghị — balance** |
| YOLOv8l | 43.7M | ~25ms | Rất tốt | Nếu accuracy quan trọng hơn speed |
| YOLOv8x | 68.2M | ~40ms | Cao nhất | Overkill cho bài này |

**Khuyến nghị**: YOLOv8m với `imgsz=1280` (thay vì default 640). Logo nhỏ cần resolution cao hơn.

### 5.5. Đánh giá model

#### Metrics

| Metric | Ý nghĩa | Ngưỡng chấp nhận |
|--------|---------|------------------|
| mAP@50 | % detection đúng (IoU ≥ 50%) | ≥ 60% |
| mAP@50:95 | Strict hơn, trung bình IoU 50-95% | ≥ 40% |
| Per-class AP | Accuracy từng class riêng | Mỗi class ≥ 30% |
| Precision | Trong các detection, bao nhiêu đúng | ≥ 70% (ít false positive) |
| Recall | Trong các logo thực, bao nhiêu detect được | ≥ 50% |

#### Validation strategy

1. **Roboflow auto-split**: 70% train / 20% valid / 10% test
2. **Manual benchmark**: Chọn 5-10 phút video (300-600 frames), đếm logo bằng tay → so với model
3. **Cross-match test**: Train trận A, test trận B → nếu drop > 20% → cần thêm data

#### Nếu model kém

| Vấn đề | Dấu hiệu | Giải pháp |
|--------|----------|----------|
| Class thiếu data | AP < 20% cho class đó | Thêm frame có class đó, annotate thêm |
| Logo quá nhỏ | Detect kém ở xa camera | Tăng `imgsz` lên 1280-1920 |
| False positive nhiều | Precision thấp | Tăng confidence threshold, thêm negative examples |
| Confuse giữa variants | aon_red vs aon_white lẫn nhau | Merge thành 1 class nếu không cần phân biệt |
| Overfitting | Train loss thấp, val loss cao | Thêm augmentation, thêm data, giảm epochs |

### 5.6. Khó khăn dự kiến

1. **Logo rất nhỏ** (tất, quần) — YOLOv8 struggle với object < 20px. Giải pháp: `imgsz=1280`, hoặc tiled inference (chia frame thành 4 phần, detect từng phần)
2. **Logo bị biến dạng** do cơ thể cầu thủ cong — training data cần cover nhiều pose
3. **Class imbalance** — AON xuất hiện 100x nhiều hơn ACS Group. Giải pháp: class weight trong loss function, hoặc oversample minority class
4. **Áo ướt, bẩn** — thay đổi appearance của logo. Cần data trong nhiều điều kiện

---

## 6. Phase 3: Đo lường Exposure

### 6.1. Mục tiêu

Chạy model trên toàn bộ video, đo lường exposure của từng logo theo thời gian.

### 6.2. Vấn đề cốt lõi: Frame-based vs Human perception

**Người xem** nhìn video liên tục (30 FPS), não ghép các frame thành hình ảnh chuyển động. Logo hơi mờ do motion blur vẫn được nhận biết vì có context từ frame trước/sau.

**Model** phân tích từng frame riêng lẻ. Frame mờ → confidence thấp hoặc miss detection. Nếu chỉ đếm frame nét → **đánh giá thấp 40-60%** exposure thực.

### 6.3. Giải pháp: Multi-level Exposure Metrics

#### Level 1: Raw Presence (đơn giản nhất)

```
exposure_seconds[logo] = count(frames có detection) / FPS
```

Ưu: Đơn giản, dễ hiểu
Nhược: Không phân biệt logo to/nhỏ, rõ/mờ

#### Level 2: Weighted Exposure (khuyến nghị)

```
weighted_exposure[logo] = Σ (confidence_i × 1/FPS)
```

- Frame nét, logo rõ: confidence 0.85 → đóng góp 0.85/30 = 0.028s
- Frame mờ, logo mờ: confidence 0.35 → đóng góp 0.35/30 = 0.012s
- Frame không detect: đóng góp 0

Ưu: Phản ánh mức độ rõ ràng, frame mờ vẫn tính nhưng weight thấp
Nhược: Phụ thuộc vào calibration của confidence score

#### Level 3: Continuous Segments (phản ánh perception)

```
Nhóm detections liên tục thành exposure events:
    Logo A: frame 100-180 (confidence > 0.3) = 1 event, 2.67s
    Logo A: frame 200-210 = 1 event, 0.33s (< 0.5s → não chưa kịp nhận biết)
    Logo A: frame 300-500 = 1 event, 6.67s

Chỉ tính events ≥ 0.5s (minimum perception threshold)
```

Ưu: Gần nhất với human perception
Nhược: Threshold 0.5s cần validate bằng thực nghiệm

#### Level 4: Quality Index (QI) — toàn diện nhất

```
Cho mỗi detection:
    QI_i = size_score × clarity × position_weight × (1 / clutter)

Trong đó:
    size_score = bbox_area / frame_area        (logo to = giá trị cao)
    clarity = confidence                       (rõ = giá trị cao)
    position_weight = f(bbox_center_position)  (giữa màn hình > rìa)
    clutter = 1 + count(other_logos_same_frame) (nhiều logo cạnh tranh = giá trị mỗi logo giảm)

Brand Exposure Value = Σ QI_i × (1/FPS)
```

### 6.4. Xử lý các trường hợp đặc biệt

#### Replay detection

Replay phát lại cùng 1 pha → logo xuất hiện 2-3 lần. Tùy business rule:
- **Option A**: Tính replay = exposure thật (người xem vẫn thấy)
- **Option B**: Tách riêng live vs replay exposure
- **Kỹ thuật detect replay**: Broadcast overlay thay đổi (icon "REPLAY" xuất hiện), hoặc scene similarity detection

#### Overlay/graphics detection

Scoreboard, tên cầu thủ, VAR display che logo:
- Nếu model không detect logo → tự động không tính (OK)
- Nếu model detect logo dưới overlay → confidence thấp (OK với weighted exposure)

#### Camera transition

Chuyển camera (wide → close-up) tạo gián đoạn:
- Continuous segment bị ngắt → 2 events thay vì 1
- Giải pháp: cho phép gap ≤ 0.5s giữa detections vẫn tính là cùng 1 event

### 6.5. Output dữ liệu

```
Per-frame CSV:
    frame_num, timestamp, logo_class, confidence, bbox_x, bbox_y, bbox_w, bbox_h, qi_score

Per-logo summary:
    logo_class, total_frames, exposure_seconds, weighted_exposure,
    n_events, avg_event_duration, max_event_duration,
    avg_size, avg_confidence, qi_total

Per-match report:
    match_info, per_logo_summary, timeline_data, comparison_with_pricing
```

### 6.6. Yêu cầu compute

| Video | Frames | Thời gian ước tính (T4 GPU) | Storage |
|-------|--------|---------------------------|---------|
| 90 phút @ 30 FPS | 162,000 | ~45 phút (YOLOv8m) | ~500MB detection results |
| 90 phút @ 15 FPS | 81,000 | ~22 phút | ~250MB |
| 90 phút @ 5 FPS | 27,000 | ~8 phút | ~80MB |

**Khuyến nghị**: Chạy ở 30 FPS cho accuracy tốt nhất. 45 phút inference trên T4 là chấp nhận được.

---

## 7. Phase 4: Mô hình định giá

### 7.1. Mục tiêu

So sánh exposure thực tế với bảng giá hiện tại, đưa ra khuyến nghị.

### 7.2. Cách trình bày kết quả

**KHÔNG nên**: Đưa absolute value ("Logo AON trị giá $50,000 per match")
→ Dễ bị challenge: "Tại sao $50k? Dựa vào đâu? Nielsen nói khác"

**NÊN**: Đưa relative comparison
→ "AON exposure gấp 8x Bartercard, nhưng chỉ trả gấp 8.7x (26%/3%). Tương đối hợp lý."
→ "Top Notch (Socks) trả 1%, nhưng exposure chỉ bằng 0.2% tổng. Đang overpay 5x."
→ "KLG (Top Back) trả 5%, nhưng exposure = 12% tổng. Đang underpay 2.4x."

### 7.3. Công thức định giá đề xuất

```
Exposure Share[logo] = weighted_exposure[logo] / Σ weighted_exposure[all logos]

Fair Price %[logo] = Exposure Share[logo] × 100

Price Efficiency = Current % / Exposure Share %
    > 1.0 = đang overpay
    = 1.0 = fair
    < 1.0 = đang underpay
```

### 7.4. Factors bổ sung (ngoài exposure time)

| Factor | Ý nghĩa | Cách đo |
|--------|---------|--------|
| Brand prominence | Logo lớn + vị trí trung tâm | Avg QI score |
| Prime time exposure | Xuất hiện trong moments quan trọng (bàn thắng, phạt) | Detect highlight events |
| Exclusive visibility | Lúc chỉ có 1 logo trên màn hình | 1/clutter |
| Close-up frequency | Bao nhiêu lần logo được zoom gần | Count frames với size > threshold |

### 7.5. Khó khăn

1. **Không có media value benchmark**: Giá quảng cáo TV cho Rugby League ở UK không public → khó convert sang money value
2. **Sponsor negotiation**: Kết quả có thể bất lợi cho CLB (Main Sponsor đang trả "đúng" nhưng phân tích cho thấy "thừa") → cần present tactfully
3. **Sample size**: 1-2 trận không đại diện cho cả mùa giải. Cần ≥5 trận để trend đáng tin cậy.

---

## 8. Phase 5: Báo cáo & mở rộng

### 8.1. Per-match Report

```
Bradford Bulls vs [Đối thủ] — Sponsor Exposure Report
Date: [ngày]

EXECUTIVE SUMMARY
- Total match exposure: X minutes of combined logo visibility
- Most visible sponsor: [logo] (Y seconds)
- Best value sponsor: [logo] (exposure/price ratio)

PER-SPONSOR BREAKDOWN
[Bảng: mỗi sponsor → exposure time, exposure share %, current price %,
        efficiency ratio, trend vs previous matches]

TIMELINE
[Chart: timeline showing when each logo was visible throughout the match]

HEATMAP
[Chart: aggregate position of each logo on screen]

RECOMMENDATIONS
- [Logo X] is significantly underpaying relative to exposure
- [Logo Y] has low exposure — consider better placement or price adjustment
```

### 8.2. Season Dashboard (SaaS vision)

- Aggregate across matches
- Trend charts: exposure changing over season
- Comparison: home vs away matches
- Opponent impact: exposure changes based on opponent style of play

### 8.3. Scale cho CLB khác

**Mỗi CLB mới cần:**
1. Logo files của sponsors → tạo class mapping
2. Annotate 300-400 frames từ 1 trận → train model riêng
3. Run pipeline trên các trận → generate reports

**Estimation mỗi CLB mới**: ~1-2 ngày setup + annotate, sau đó tự động.

---

## 9. Phân tích rủi ro & giải pháp

### 9.1. Rủi ro kỹ thuật

| # | Rủi ro | Xác suất | Impact | Giải pháp |
|---|--------|---------|--------|----------|
| T1 | Model accuracy thấp (<50% mAP) | Trung bình | Cao | Thêm training data, tăng image size, thử model lớn hơn |
| T2 | Logo quá nhỏ không detect được (tất, quần) | Cao | Trung bình | Tiled inference, hoặc chấp nhận và ghi chú trong report |
| T3 | Model không generalize sang trận khác | Trung bình | Cao | Train thêm data từ nhiều trận, augmentation mạnh hơn |
| T4 | False positive: detect logo trên LED/sân | Trung bình | Trung bình | Filter theo position (LED thường ở rìa), hoặc train negative class |
| T5 | Video quality khác nhau (YouTube vs broadcast) | Cao | Trung bình | Test trên nhiều nguồn video, adjust confidence threshold |
| T6 | Class imbalance nặng | Cao | Trung bình | Class weights, oversample minority, merge rare variants |
| T7 | Replay bị đếm 2 lần | Trung bình | Thấp | Replay detection, hoặc tách riêng trong report |

### 9.2. Rủi ro methodology

| # | Rủi ro | Xác suất | Impact | Giải pháp |
|---|--------|---------|--------|----------|
| M1 | "Presence" ≠ "Recognition" — overestimate | Cao | Cao | Weighted exposure + min duration + confidence interval |
| M2 | Camera bias — exposure phụ thuộc đạo diễn hình | Cao | Cao | Đo nhiều trận trung bình, note trong report |
| M3 | Sampling bias khi lấy training data | Trung bình | Trung bình | Time diversity, đủ frames |
| M4 | Annotation inconsistency | Trung bình | Cao | Rõ ràng guidelines, annotate thử → review → refine |
| M5 | Không benchmark được với industry standard | Cao | Trung bình | Manual benchmark 5-10 phút, transparent methodology |

### 9.3. Rủi ro kinh doanh

| # | Rủi ro | Xác suất | Impact | Giải pháp |
|---|--------|---------|--------|----------|
| B1 | Kết quả bất lợi cho sponsor đang trả nhiều | Cao | Cao | Trình bày relative, kèm confidence interval |
| B2 | CLB không muốn share data bất lợi với sponsor | Trung bình | Cao | Focus vào value-add cho cả 2 bên |
| B3 | 1-2 trận không đại diện cả mùa giải | Cao | Trung bình | Nêu rõ limitations, cần ≥5 trận |
| B4 | Sponsor challenge methodology | Trung bình | Cao | Transparent, open-source, manual verification option |

---

## 10. Yêu cầu kỹ thuật

### 10.1. Development environment

```
Local (MacBook Apple Silicon):
    - Python 3.10+
    - PyTorch với MPS backend
    - VS Code / Cursor
    - Git
    - Dùng cho: viết code, test nhỏ, review

Google Colab:
    - GPU: T4 (free) hoặc A100 (Pro)
    - Python 3.10
    - Dùng cho: training, inference nặng, frame extraction
    - Storage: Google Drive mount
```

### 10.2. Python packages

```
# Core
ultralytics        # YOLOv8
opencv-python      # Frame processing
torch              # Deep learning backend
torchvision        # Image transforms

# Frame selection
imagehash          # Perceptual hashing
scikit-image       # Image quality metrics
scikit-learn       # KMeans clustering

# Video
yt-dlp             # YouTube download
ffmpeg             # Video processing (system package)

# Annotation & training
roboflow           # Upload, annotate, train, deploy

# Data & visualization
pandas             # DataFrames
numpy              # Numerical
matplotlib         # Charts
Pillow             # Image I/O
tqdm               # Progress bars
```

### 10.3. Hardware requirements

| Task | Min GPU | Recommended | VRAM | Time |
|------|---------|------------|------|------|
| Frame extraction (YOLOv8 inference) | T4 (16GB) | T4 | 4GB | ~20 phút/trận |
| Training YOLOv8m (300 frames) | T4 (16GB) | A100 (40GB) | 8GB+ | 1-3 giờ |
| Full video inference (162K frames) | T4 (16GB) | T4 | 4GB | ~45 phút/trận |
| Annotation | CPU only | — | — | ~5 giờ manual |

### 10.4. Storage requirements

| Data | Size ước tính |
|------|--------------|
| 1 video (1080p, 90 phút) | 1-3 GB |
| 400 selected frames (JPG 95%) | ~200 MB |
| Annotated dataset (YOLO format) | ~300 MB |
| Trained model weights | ~50 MB |
| Detection results (1 match, 30 FPS) | ~500 MB CSV |
| **Total per match** | **~4-5 GB** |

### 10.5. API keys & accounts

| Service | Cần gì | Free tier |
|---------|--------|----------|
| Roboflow | API key, workspace | 10,000 images, 3 model trains |
| Google Colab | Google account | GPU runtime limited (T4) |
| YouTube | Không cần API | Public videos only |

---

## 11. Cấu trúc dự án

```
BRADFORD_BULLS_PROJECT/
│
├── PLAN.md                              # Project plan (tổng quan)
├── docs/
│   ├── PROJECT_SPECIFICATION.md         # ← FILE NÀY (chi tiết đầy đủ)
│   └── GPU_SETUP.md                     # Hướng dẫn setup GPU
│
├── src/                                 # Python modules
│   ├── config.py                        # Config chung (classes, pricing, device)
│   ├── video_pipeline.py                # Download video
│   ├── frame_sampler.py                 # Frame sampling logic
│   ├── player_filter.py                 # Player visibility filter
│   └── ...
│
├── notebooks/                           # Jupyter notebooks
│   ├── colab_03_frame_extraction.ipynb  # ← DÙNG CÁI NÀY — lấy frame + upload Roboflow
│   ├── colab_02_smart_crop_pipeline.ipynb  # (ĐÃ BỎ) crop + auto-annotate
│   ├── colab_01_smart_frame_selection.ipynb # (CŨ) frame selection v1
│   ├── 01_video_to_frames.ipynb         # (CŨ) basic frame extraction
│   └── ...
│
├── Kit Sponsors/                        # Logo files gốc từ Bradford Bulls
│   ├── 1 - aon_logo_signature_red_rgb (2).png
│   ├── 2 - ATM-Hospitality-Logo-New-Font.png
│   └── ... (21 logo files)
│
├── Bradford Bulls (Curent Pricing).csv  # Bảng giá vị trí sponsor hiện tại
│
├── output/                              # Output local (dev)
│   ├── videos/
│   ├── frames/
│   └── metadata/
│
├── logs/                                # Interaction logs (auto-generated)
│   └── YYYY-MM-DD/chat-XXX.md
│
├── requirements.txt                     # Python dependencies
├── README.md                            # Quick start guide
└── .claude/                             # Claude Code config
    └── skills/auto-log/SKILL.md
```

### Google Drive structure (Colab):
```
/content/drive/MyDrive/Bradford_Bulls/
├── videos/                  # Downloaded match videos
├── selected_frames/         # Output from frame extraction pipeline
├── metadata/                # CSV with frame metadata
├── best_crops/              # (legacy) Player crops
└── auto_annotated_crops/    # (legacy) Auto-annotated crops
```

---

## 12. Bài học đã rút ra

### 12.1. Grounding DINO không phù hợp cho logo detection

**Đã thử**: Dùng Grounding DINO (zero-shot) để auto-annotate logo trên player crops.

**Kết quả**: 
- Prompt chung ("logo on jersey") → detect cả người, box quá lớn
- Prompt cụ thể (21 brand names) → hầu hết match vào class đầu tiên (phrase matching quá loose)
- Per-brand prompts → chậm (16 passes) và vẫn không chính xác

**Bài học**: Zero-shot detection phù hợp cho object phổ biến (person, car). Logo sponsor là domain-specific, cần supervised training.

### 12.2. Full frame > Player crop cho annotation

**Đã thử**: Crop từng cầu thủ → annotate logo trên crop.

**Kết quả**: Logo lớn hơn trên crop (10-20% vs 1-3%), nhưng:
- Mất context (không biết cầu thủ đang ở đâu)
- Team filtering phức tạp (K-means jersey color)
- Pipeline quá nhiều bước → nhiều chỗ fail

**Bài học**: Full frame đơn giản hơn, annotator thấy toàn cảnh, cả 2 đội, dễ annotate hơn.

### 12.3. Test trước, production sau

**Đã mắc lỗi**: Chạy full pipeline rồi mới phát hiện lỗi ở cuối.

**Bài học**: `TEST_MODE = True` với data nhỏ trước, verify từng bước, rồi mới chạy production.

### 12.4. Package version conflicts trên Colab

**Đã mắc lỗi**: `autodistill` → `transformers` version conflict → `supervision.BaseDataset` error.

**Bài học**: 
- Cài tất cả packages ở cell đầu tiên
- Restart runtime sau install
- `pip uninstall` packages conflict trước khi install
- Test import ngay sau install (bắt lỗi sớm)

### 12.5. Auto-annotate tốn thời gian hơn manual

**Nghịch lý**: Auto-annotate lý thuyết nhanh hơn, nhưng thực tế:
- Sửa annotation sai mất nhiều thời gian hơn vẽ từ đầu
- Debug pipeline auto-annotate mất nhiều ngày
- Manual annotate 400 frames chỉ mất ~5 giờ

**Bài học**: Với dataset nhỏ (<1000 images), manual annotation thường hiệu quả hơn semi-auto.
