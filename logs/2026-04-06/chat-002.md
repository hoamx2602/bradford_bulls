## Timestamp: 2026-04-06 10:14:54 +0100
### User:
lúc chạy file này có cái cell Auto Annotate: ... FileNotFoundError ... AttributeError: 'BertModel' object has no attribute 'get_head_mask'

### Claude:
Mình đã sửa ngay trong `notebooks/colab_01_smart_frame_selection.ipynb` phần **Auto-Annotate** để hết 2 lỗi chính bạn gặp:

- **`FileNotFoundError: ... GroundingDINO_SwinT_OGC.py does not exist`**: mình thêm bước tạo cache và tải **config** + **checkpoint** vào đúng path mà `autodistill` tìm.
- **`AttributeError: 'BertModel' object has no attribute 'get_head_mask'`**: đây là do **lệch phiên bản `transformers`** so với `groundingdino/autodistill-grounding-dino`, nên mình **pin `transformers==4.41.2`** (một phiên bản tương thích ổn định hơn).

Bạn chạy lại theo thứ tự:
1) Chạy cell cài đặt ở mục **8** (cell đó giờ sẽ cài package + download config/ckpt).
2) Nếu Colab báo import lỗi do đã load phiên bản cũ, làm **Runtime → Restart session** rồi chạy lại cell đó.
3) Chạy cell kế tiếp để `GroundingDINO(...)` và `base_model.label(...)`.

Ghi chú: cảnh báo **`HF_TOKEN`** chỉ là khuyến nghị để tăng rate limit; với model public thì vẫn chạy được (chỉ chậm hơn / dễ bị rate-limit).

