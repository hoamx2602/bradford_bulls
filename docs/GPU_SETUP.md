# GPU vs CPU — tại sao `02b_auto_annotate.ipynb` nặng CPU?

Notebook **`notebooks/02b_auto_annotate.ipynb`** có thể tốn CPU rất nhiều trong khi GPU gần như không làm việc vì **hai nhóm lý do** khác nhau.

---

## 1. PyTorch của bạn đang chạy **CPU** (không có CUDA)

Trong output notebook bạn sẽ thấy:

- `[Device] Using: cpu (CPU)` — từ `src/config.py` (`torch.cuda.is_available()` là `False` trên máy Windows này).
- `WARNING: CUDA not available. GroundingDINO will run very slowly.` — thư viện **Grounding DINO** (qua `autodistill-grounding-dino`) dùng PyTorch; **không có CUDA thì suy luận chạy trên CPU**.

Cài `torch` bằng `pip install -r requirements.txt` **mặc định thường là bản CPU-only** (đặc biệt trên Windows), **dù máy có card NVIDIA**. Driver GPU vẫn cài được nhưng PyTorch không “bám” vào GPU.

### Cách xử lý trên Windows (NVIDIA)

1. Gỡ hoặc ghi đè PyTorch bằng bản **có CUDA**, đúng với driver của bạn — chỉ dẫn chính thức: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)  
   - Chọn: *Windows*, *Pip*, *Python*, *CUDA* (ví dụ 12.4 hoặc 12.1 tùy trang hướng dẫn).
2. Cài đúng **môi trường** mà Jupyter/Cursor đang dùng (cùng một `python` / `.venv`).
3. **Khởi động lại kernel** notebook, chạy:

```python
import torch
print(torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
```

Khi `cuda_available: True`, Grounding DINO mới có thể chạy nhanh trên GPU (tùy phiên bản thư viện vẫn có thể có giới hạn; nếu vẫn chậm, xem log có còn cảnh báo CUDA không).

### macOS (MacBook — môi trường bạn thường dùng)

- `src/config.py` sẽ chọn **`mps`** khi Apple Silicon + PyTorch hỗ trợ MPS.
- **Grounding DINO / autodistill** thường được viết và test quanh **CUDA**; trên Mac **có thể vẫn rơi về CPU** nếu mã hoặc phụ thuộc không gửi tensor lên MPS. Khi đó dù Mac có GPU tích hợp, notebook vẫn nặng CPU.
- Cách thực tế để có GPU mạnh cho bước này: **Google Colab / máy có NVIDIA + PyTorch CUDA** — trong repo đã có gợi ý notebook Colab: `notebooks/02b_colab_auto_annotate.ipynb`.

---

## 2. Một phần pipeline **vốn là CPU** (bình thường)

Ở mục chọn frame đa dạng (perceptual hash với `imagehash` + PIL), toàn bộ là **xử lý ảnh trên CPU**. Bước đó có thể đẩy CPU cao và **không cần GPU**.

---

## 3. Task Manager hiển thị “Cursor” ~100% CPU

Kernel Jupyter thường chạy trong tiến trình của Cursor/VS Code, nên **tải CPU của mô hình + vòng lặp xử lý ảnh** có thể gộp vào mục **Cursor** trong Task Manager — điều đó **không có nghĩa** là GPU đang được dùng cho PyTorch.

---

## Tóm tắt hành động

| Mục tiêu | Việc nên làm |
|----------|----------------|
| Dùng GPU NVIDIA trên Windows với notebook hiện tại | Cài **PyTorch + CUDA** đúng hướng dẫn [pytorch.org](https://pytorch.org/get-started/locally/), cùng env với Jupyter, restart kernel, kiểm tra `torch.cuda.is_available()`. |
| MacBook, muốn tránh CPU kéo dài cho Grounding DINO | Thử **Colab** (`02b_colab_auto_annotate.ipynb`) hoặc máy có NVIDIA + CUDA. |
| Giảm tải ở bước chọn frame | Giảm số frame đầu vào hoặc tăng `DIVERSITY_HASH_THRESHOLD` / giảm `TARGET_ANNOTATION_COUNT` trong notebook (khi notebook không còn chạy). |

File này chỉ là tài liệu; **không thay đổi** `02b_auto_annotate.ipynb` để tránh xung đột khi bạn đang chạy notebook.
