# BRADFORD BULLS - AI Sponsorship Exposure Valuation System

## Tầm nhìn sản phẩm

Xây dựng **SaaS platform** cho phép bất kỳ CLB thể thao nào upload video trận đấu + logo sponsors → AI tự động phân tích và đưa ra **báo cáo giá trị hiện diện thương hiệu** (Brand Exposure Valuation Report).

---

## PHASE 1: VIDEO INGESTION & FRAME SAMPLING

### 1.1 Video Input Pipeline
- Hỗ trợ input: **YouTube URL**, upload file trực tiếp, hoặc cloud storage link
- Dùng **yt-dlp** để download video từ YouTube ở chất lượng cao nhất
- Extract metadata: resolution, FPS, duration, match info

### 1.2 Smart Frame Sampling (giải quyết vấn đề 60FPS)

Đây là điểm quan trọng. Video 4K 60FPS, 2 giờ = **432,000 frames**. Không cần xử lý hết.

**Chiến lược 3 lớp:**

| Lớp | Phương pháp | Mục đích |
|-----|-------------|----------|
| **L1 - Temporal Sampling** | Lấy 2-3 FPS thay vì 60 FPS (giảm ~95% frames) | Loại bỏ frames gần như giống nhau |
| **L2 - Scene Change Detection** | Dùng perceptual hashing (pHash) + structural similarity (SSIM) để so sánh frames liên tiếp, chỉ giữ frames có thay đổi đáng kể | Loại bỏ frames tĩnh (replay chậm, góc máy đứng yên) |
| **L3 - Player Presence Filter** | Dùng person detection (YOLOv8/RT-DETR) để chỉ giữ frames có cầu thủ | Loại bỏ frames chỉ có sân cỏ, khán đài, đồ họa TV |

**Kết quả ước tính:** Từ 432K frames → còn khoảng **8,000-15,000 frames** cần xử lý sâu.

### 1.3 Frame Storage với Timestamp Mapping

Mỗi frame được lưu kèm metadata để **truy ngược chính xác vị trí trong video gốc**:

```
output/
├── frames/
│   ├── frame_000001_00m01s.jpg    # Frame 1 = giây thứ 1
│   ├── frame_000002_00m02s.jpg    # Frame 2 = giây thứ 2
│   ├── frame_000147_02m27s.jpg
│   └── ...
├── metadata/
│   └── frames_index.csv
│       # frame_id, timestamp_sec, timestamp_hms, source_video, fps_original, is_player_visible
│       # 1, 1.0, 00:00:01, match_video.mp4, 60, true
│       # 2, 2.0, 00:00:02, match_video.mp4, 60, true
```

- Mỗi frame filename chứa **frame number + timestamp** để dễ trace
- File `frames_index.csv` lưu full mapping: frame ↔ thời điểm ↔ video gốc
- Có cờ `is_player_visible` để đánh dấu frame nào cầu thủ rõ ràng (dùng cho annotation)

### 1.4 Player Visibility Filtering

Từ tập frames đã lưu, lọc ra các frames mà **cầu thủ hiện rõ ràng bằng mắt thường**:

- **Kích thước tối thiểu**: Player bounding box >= 5% diện tích frame (loại bỏ cầu thủ quá xa)
- **Độ nét tối thiểu**: Laplacian variance > ngưỡng (loại bỏ frames bị motion blur)
- **Không bị che**: Player visibility >= 70% (không bị cầu thủ khác che quá nhiều)
- Output: Thư mục `frames_clear/` chỉ chứa frames đạt chuẩn, sẵn sàng cho annotation

**Về broadcast vs highlights:**
- Phân tích **cả hai** nhưng áp **weight khác nhau**
- Highlights: weight cao hơn (vì viewer attention cao hơn, replayed nhiều lần)
- Broadcast full match: weight chuẩn
- Có thể tính thêm **view count** từ YouTube API để nhân hệ số

---

### 1.5: ANNOTATION WORKFLOW (Roboflow + Jupyter Notebook)

### 1.5.1 Annotation Pipeline

Toàn bộ Phase 1 + Annotation chạy trong **Jupyter Notebook** để dễ visualize và iterate:

```
Notebook 1: 01_video_to_frames.ipynb
  → Download video, extract frames, lưu với timestamps
  → Filter frames có cầu thủ rõ ràng
  → Export frames_clear/ sẵn sàng annotation

Notebook 2: 02_annotation_roboflow.ipynb
  → Upload frames_clear/ lên Roboflow
  → Hướng dẫn annotate trên Roboflow UI
  → Download annotated dataset (YOLO format)

Notebook 3: 03_train_logo_detector.ipynb
  → Load annotated dataset từ Roboflow
  → Fine-tune YOLOv8 trên logo dataset
  → Evaluate model, export weights

Notebook 4: 04_run_detection.ipynb
  → Chạy model trên toàn bộ frames
  → Tính metrics, xuất báo cáo
```

### 1.5.2 Roboflow Annotation Process

1. **Upload**: Push frames đã lọc lên Roboflow project
2. **Label setup**: Tạo label classes tương ứng với mỗi sponsor:
   - `aon` (Main Sponsor)
   - `atm_hospitality` (Collar Bone)
   - `cch_cedar_court` (Chest)
   - `chadlaw` (Sleeve)
   - `em_workwear` (Sleeve)
   - `fairway_flooring` (Sleeve)
   - `klg` (Top Back)
   - `mcp` (Sleeve 2)
   - `mna_cladding` (Shorts)
   - `mna_support` (Shorts)
   - `yellow_bartercard` (Shorts)
   - `top_notch` (Socks)
   - ... (thêm theo danh sách sponsor)
3. **Annotate**: Vẽ bounding box quanh mỗi logo trên áo cầu thủ Bradford Bulls
4. **Augmentation**: Roboflow tự động augment (flip, rotate, brightness, blur...)
5. **Export**: Download dataset ở format YOLOv8 PyTorch

### 1.5.3 Roboflow API Integration (trong Notebook)

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("bradford-bulls").project("kit-sponsor-logos")

# Upload frames để annotate
project.upload(image_path="output/frames_clear/")

# Sau khi annotate xong, download dataset
dataset = project.version(1).download("yolov8")
```

---

### 1.6: GPU COMPATIBILITY & ENVIRONMENT

### Chiến lược chạy đa nền tảng

| Môi trường | GPU | Cách chạy | Use case |
|------------|-----|-----------|----------|
| **MacBook (Apple Silicon)** | MPS (Metal) | `device="mps"` | Dev, prototype, annotate, xử lý vài frames |
| **MacBook (Intel)** | Không | `device="cpu"` | Dev, annotate only |
| **AWS SageMaker** | NVIDIA T4/A10G | `device="cuda"` | Training model, xử lý full video |
| **AWS EC2 G5** | NVIDIA A10G | `device="cuda"` | Batch processing nhiều video |
| **Google Colab Pro** | NVIDIA T4/A100 | `device="cuda"` | Thay thế rẻ hơn cho prototype |

### Auto-detect device trong code

```python
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"          # NVIDIA GPU (cloud/server)
    elif torch.backends.mps.is_available():
        return "mps"           # Apple Silicon GPU
    else:
        return "cpu"           # Fallback

DEVICE = get_device()
print(f"Using device: {DEVICE}")
```

### Workflow đề xuất

```
[MacBook - Local]                    [Cloud - GPU]
     │                                     │
     ├─ Download video                     │
     ├─ Extract frames                     │
     ├─ Filter frames (CPU OK)             │
     ├─ Upload lên Roboflow               │
     ├─ Annotate trên Roboflow UI         │
     │                                     │
     │──── Push code + data lên cloud ────→│
     │                                     ├─ Train YOLOv8 (cần GPU)
     │                                     ├─ Run detection full video
     │                                     ├─ Tính metrics
     │←──── Pull results về ──────────────│
     │                                     │
     ├─ Review kết quả                    │
     ├─ Xuất báo cáo                      │
```

---

## PHASE 2: PLAYER & JERSEY DETECTION

### 2.1 Player Detection & Tracking
- **Model:** YOLOv8x hoặc RT-DETR (real-time detection transformer - SOTA 2025)
- Detect tất cả người trong frame (cầu thủ, trọng tài, ball boy...)
- **Player tracking** qua các frames bằng **ByteTrack** hoặc **BoT-SORT** để biết cầu thủ nào xuất hiện bao lâu

### 2.2 Team Classification
- Phân biệt cầu thủ Bradford Bulls vs đối thủ dựa trên **jersey color clustering** (K-means trên vùng áo)
- Chỉ phân tích logo trên cầu thủ Bradford Bulls (hoặc cả 2 đội nếu client muốn)

### 2.3 Jersey Region Segmentation
- **Keypoint-based body pose estimation** (ViTPose hoặc RTMPose) để xác định vị trí các phần cơ thể
- Map body keypoints → jersey zones theo file pricing:

```
Nape Neck (3%)      ←  cổ sau
Collar Back (8%)    ←  sau cổ áo
Collar Bone (8%)    ←  xương đòn
Main Sponsor (26%)  ←  ngực giữa
Chest opp Badge (7%) ← ngực đối diện huy hiệu
Sleeve 1/2/3 (4%/11%/4%) ← tay áo
Top Back (5%)       ←  lưng trên
Bottom Back (3%)    ←  lưng dưới
Shorts Front/Back (3%/3%/3%) ← quần
Top Back Shorts (5%) ← quần trên sau
Socks (1%)          ←  tất
```

---

## PHASE 3: LOGO DETECTION & RECOGNITION

Đây là **core AI** của hệ thống. Có 3 approach bổ sung nhau:

### 3.1 Approach A - Logo Detection (Primary)
- **Fine-tune YOLOv8** hoặc **Grounding DINO** trên dataset logo của sponsors
- Training data: Dùng logo gốc từ thư mục Kit Sponsors + augmentation (xoay, co giãn, thay đổi ánh sáng, thêm wrinkle/fold effect mô phỏng vải áo)
- Mỗi CLB mới chỉ cần upload logo → system tự generate training data → fine-tune

### 3.2 Approach B - Feature Matching (Backup/Verification)
- Dùng **DINO v2** hoặc **CLIP** embeddings để so sánh vùng áo đã crop với logo gốc
- Tính cosine similarity giữa feature vectors
- Ưu điểm: **Zero-shot** - không cần training khi thêm logo mới
- Dùng làm lớp verify cho Approach A

### 3.3 Approach C - Vision Language Model (cho trường hợp khó)
- Dùng **GPT-4o / Claude Vision** để phân tích các frames mà A+B không confident
- Prompt: "Identify all visible sponsor logos on this rugby player's jersey. For each logo, specify: brand name, position on jersey, visibility level"
- Đắt hơn nhưng accurate cho edge cases

**Chiến lược kết hợp:**
```
Frame → YOLOv8 detect (fast, cheap)
  ├── Confidence > 0.8 → Accept
  ├── 0.4 < Confidence < 0.8 → Verify bằng CLIP matching
  └── Confidence < 0.4 → Gửi qua VLM (GPT-4o/Claude) để phân tích
```

---

## PHASE 4: EXPOSURE METRICS ENGINE

### 4.1 Raw Metrics (đo cho mỗi logo, mỗi frame)

| Metric | Cách đo | Đơn vị |
|--------|---------|--------|
| **Screen Time** | Tổng số frames logo xuất hiện × thời gian mỗi frame | giây |
| **Screen Size** | Diện tích bounding box logo / diện tích frame | % |
| **Clarity Score** | Laplacian variance của vùng logo (đo độ nét) | 0-1 |
| **Visibility Score** | % diện tích logo không bị che (occlusion detection) | 0-1 |
| **Position Score** | Vị trí logo trong frame (trung tâm = cao hơn, rìa = thấp hơn, dựa trên eye-tracking heatmap research) | 0-1 |
| **Angle Score** | Góc nghiêng của logo so với camera (chính diện = 1, nghiêng 90° = 0) | 0-1 |
| **Frequency** | Số lần xuất hiện riêng biệt (các chuỗi liên tục) | count |

### 4.2 Composite Brand Exposure Value (BEV)

```python
# Cho mỗi frame i mà logo j xuất hiện:
frame_exposure[i][j] = (
    screen_size[i][j]      * W_size        # 0.25
  * clarity[i][j]          * W_clarity     # 0.20
  * visibility[i][j]       * W_visibility  # 0.20
  * position_score[i][j]   * W_position    # 0.15
  * angle_score[i][j]      * W_angle       # 0.10
  * video_type_weight      * W_type        # 0.10 (highlight=1.5, broadcast=1.0)
)

# Tổng hợp cho cả video:
total_BEV[j] = sum(frame_exposure[i][j] * frame_duration * view_count_multiplier)
```

### 4.3 Pricing Model

```python
# Từ BEV → Pricing
total_sponsorship_value = contract_total_value  # tổng giá trị hợp đồng tài trợ

# Phân bổ theo BEV thực tế thay vì % cố định
brand_fee[j] = (BEV[j] / sum(BEV[all])) * total_sponsorship_value

# So sánh với pricing hiện tại
delta[j] = brand_fee[j] - current_pricing[j]
# → Brand nào đang được/thiệt so với giá trị thực tế
```

---

## PHASE 5: OUTPUT & REPORTING

### 5.1 MVP Output (CSV/JSON)

```
Brand | Position | Screen Time | Avg Size | Clarity | BEV Score | Suggested % | Current % | Delta
AON   | Main     | 847s        | 3.2%     | 0.82    | 4521      | 28.1%       | 26%       | +2.1%
MCP   | Sleeve2  | 612s        | 1.8%     | 0.71    | 2103      | 13.1%       | 11%       | +2.1%
...
```

### 5.2 Visual Evidence
- Export sample frames với bounding boxes cho mỗi brand
- Heatmap hiển thị vùng xuất hiện nhiều nhất trên màn hình
- Timeline chart: khi nào mỗi brand xuất hiện trong video

---

## TECH STACK

```
┌─────────────────────────────────────────────┐
│                  FRONTEND                    │
│         (Phase sau - Web Dashboard)          │
│         Next.js + React + D3.js             │
├─────────────────────────────────────────────┤
│                  BACKEND                     │
│         Python FastAPI + Celery              │
│         (task queue cho video processing)    │
├─────────────────────────────────────────────┤
│              AI/ML PIPELINE                  │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ YOLOv8x │ │ ViTPose  │ │ CLIP/DINOv2  │ │
│  │(detect) │ │(pose)    │ │(matching)    │ │
│  └─────────┘ └──────────┘ └──────────────┘ │
│  ┌─────────────────┐ ┌──────────────────┐  │
│  │ ByteTrack       │ │ GPT-4o/Claude    │  │
│  │ (tracking)      │ │ (fallback VLM)   │  │
│  └─────────────────┘ └──────────────────┘  │
├─────────────────────────────────────────────┤
│              INFRASTRUCTURE                  │
│  AWS: S3 (storage) + EC2 G5 (GPU)           │
│  hoặc GCP: GCS + Vertex AI                  │
│  Docker + Docker Compose                     │
└─────────────────────────────────────────────┘
```

---

## LỘ TRÌNH TRIỂN KHAI

### Sprint 1 (2 tuần) - Foundation & Frame Extraction
- [ ] Setup project structure, Jupyter Notebook environment
- [ ] `01_video_to_frames.ipynb`: Video download (yt-dlp) + frame extraction
- [ ] Smart frame sampling (L1 + L2 + L3) với timestamp mapping
- [ ] Player visibility filtering → export frames_clear/
- [ ] Auto-detect device (MPS/CUDA/CPU)

### Sprint 2 (2 tuần) - Annotation & Training
- [ ] `02_annotation_roboflow.ipynb`: Upload frames lên Roboflow, setup labels
- [ ] Annotate logos trên Roboflow UI (manual labeling)
- [ ] `03_train_logo_detector.ipynb`: Fine-tune YOLOv8 trên annotated dataset
- [ ] Evaluate model accuracy, iterate annotation nếu cần
- [ ] Team classification (jersey color clustering)

### Sprint 3 (2 tuần) - Detection & Metrics
- [ ] `04_run_detection.ipynb`: Chạy detection trên toàn bộ frames
- [ ] Jersey region segmentation (pose estimation)
- [ ] CLIP/DINOv2 feature matching pipeline (verification layer)
- [ ] Implement tất cả raw metrics (screen time, size, clarity, visibility, position, angle)

### Sprint 3 (2 tuần) - Metrics & Pricing
- [ ] Implement tất cả raw metrics
- [ ] Composite BEV scoring formula
- [ ] Pricing model với so sánh current vs suggested
- [ ] CSV/JSON report generation

### Sprint 4 (1 tuần) - Polish & Scale
- [ ] VLM fallback integration
- [ ] Multi-club support (config-driven, mỗi club upload logo + pricing)
- [ ] Performance optimization
- [ ] Documentation + API design cho SaaS

**Tổng: ~7 tuần cho MVP có thể chạy được**

---

## ĐIỂM KHÁC BIỆT CHO SẢN PHẨM SAAS

Để mở rộng cho các CLB khác:

1. **Onboarding flow**: CLB upload kit design + logo files + current pricing → system tự chuẩn bị model
2. **Zero-shot capability**: Dùng CLIP/VLM nên không cần train lại cho mỗi CLB mới
3. **Multi-sport**: Kiến trúc abstract đủ để apply cho football, rugby, basketball...
4. **Benchmark database**: Tích lũy data từ nhiều CLB → đưa ra industry benchmark
