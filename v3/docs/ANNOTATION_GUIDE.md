# Annotation Guide cho team annotator

Tài liệu hướng dẫn quy trình annotate trên Streamlit reviewer của module `track_annotation`.

## Nguyên tắc cốt lõi

> **Quy tắc 3 giây**: Nếu sau khi xem 3 keyframe + video clip 2 giây, bạn không thể tự tin xác định brand trong vòng 3 giây, hãy chọn **Skip (not visible)**. KHÔNG đoán.

Lý do: ground truth chỉ phản ánh "logo nào con người nhận diện được". Annotation sai làm hỏng cả model.

## Workflow

### 1. Khởi động

```bash
bash scripts/run_reviewer.sh data/annotation_packages/<TÊN_TRẬN>
```

Mở browser tại http://localhost:8501.

### 2. Cho mỗi track

1. **Đọc metadata header**: số frame, duration, mean confidence, mean area.
   - Track ngắn (< 10 frame) thường ít context → cần xem video clip kỹ.
   - Mean area thấp (< 0.5%) = logo nhỏ → khó nhận diện.

2. **Xem 3 keyframes**:
   - `sharpest`: frame nét nhất → ưu tiên đọc brand từ đây.
   - `largest`: frame có bbox to nhất → tốt cho logo nhỏ.
   - `midpoint`: frame ở giữa track → context tổng thể.
   - Mỗi keyframe có cả full frame và crop. Crop để xem chi tiết, full để xem context.

3. **Xem video clip 2 giây** (loop tự động): brain integration giúp đọc logo mờ tốt hơn ảnh tĩnh. Đây là lợi thế lớn nhất so với annotation frame-by-frame.

4. **Tham khảo brand templates**: mở expander "Reference: 21 brand templates" để so sánh visual.

5. **Ra quyết định**:
   - **Brand confident** → chọn brand từ dropdown → Save & next.
   - **Có logo nhưng không xác định brand** → chọn `unknown` → Save & next.
   - **Không nhìn thấy logo / không phải Bradford** → Skip (not visible).

   ⚠️ **Lưu ý quan trọng — brand vs variant**: bạn chỉ chọn **brand** (ví dụ "Aon", "CCH"), không chọn variant (ví dụ aon_red vs aon_white). Hệ thống tự suy ra variant đúng từ kit_context của trận (hiển thị ở sidebar trên cùng "Kit context: home/away/...").

   Trong trường hợp hiếm có 1 brand có nhiều variant cùng active cho 1 kit (vd brand đổi logo giữa mùa giải), reviewer sẽ hiện secondary dropdown — chọn variant cụ thể bạn thấy.

### 3. Position labeling

Chọn vị trí nơi logo xuất hiện trên cơ thể cầu thủ:

| Position | Khi nào dùng |
|---|---|
| `chest_front` | Logo ở ngực, camera nhìn vào mặt cầu thủ |
| `chest_back` | Logo ở lưng, camera nhìn từ phía sau |
| `sleeve_left` / `sleeve_right` | Logo ở tay áo |
| `short_left` / `short_right` | Logo ở quần đùi |
| `collar` | Logo ở cổ áo / vai |
| `other` | Không khớp các trường hợp trên |

### 4. Visibility quality

| Quality | Tiêu chí |
|---|---|
| `clear` | Logo rất rõ, không chần chừ khi nhận diện |
| `partial` | Logo bị che một phần (< 30%), vẫn nhận diện được |
| `blurry` | Mờ nhưng vẫn nhận ra brand từ shape/color |
| `occluded` | Bị che 30–50%, phải dùng video clip để xác nhận |

Nếu bị che > 50% → Skip thay vì label.

## Trường hợp đặc biệt

### Track của đối thủ hoặc trọng tài

Bỏ chọn checkbox `Is target team (Bradford Bulls)` rồi Skip. Hệ thống bỏ qua khi export.

### Cùng cầu thủ nhưng nhiều logo trên người

Hiện tại pipeline track ở mức người (player). Một track = một player = nhiều logo. Trong reviewer hiện tại chỉ chọn được 1 brand per track. Workaround:
- Chọn brand chính rõ nhất cho track này.
- Sau khi pipeline switch sang logo tracking (v1), mỗi logo sẽ là 1 track riêng → vấn đề tự động giải quyết.

### Replay hoặc slow motion

Vẫn label bình thường. Filter replay/live sẽ áp dụng ở tầng metric, không ở annotation.

### Logo mới chưa có trong danh sách 21

Chọn `unknown`. Báo lại với senior để bổ sung vào `brand_ids` config nếu cần.

### Bị stuck (track không thể quyết định)

Skip. Đừng dành quá 30 giây cho 1 track. Mục tiêu: throughput 80–120 track/giờ.

## Quality control

### Self-check cuối mỗi session

- Sample 10 track random đã làm trong ngày → xem lại sau 2h → có thay đổi quyết định không?
- Nếu > 20% thay đổi → review lại guideline với senior.

### Inter-annotator agreement (mỗi vòng)

200 track sẽ được double-labeled bởi 2 annotator độc lập. Đo Cohen's kappa:
- Kappa ≥ 0.80: PASS, dùng để train.
- 0.60 ≤ Kappa < 0.80: review disagreement, retrain annotator.
- Kappa < 0.60: FAIL, restart annotation guideline session.

## Tips về tốc độ

- Dùng phím tắt browser: Ctrl+L để focus address bar (jump to track), F5 reload nếu UI lag.
- Chia ca 90 phút làm + 15 phút nghỉ. Sau 2 ca chất lượng giảm rõ rệt.
- Sidebar "Filter: only un-annotated" để skip qua các track đã làm khi quay lại.

## Phản hồi

Nếu gặp track ambiguous lặp lại nhiều lần (cùng brand, cùng tình huống), ghi lại track_id và báo team. Có thể cần:
- Thêm pose-aligned fusion view (ý tưởng 1) cho loại track này.
- Cập nhật guideline với ví dụ.
- Bổ sung brand template variant.
