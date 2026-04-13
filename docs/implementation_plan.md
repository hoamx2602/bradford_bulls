# Kế hoạch Triển khai Đo lường Phơi sáng Logo Tự động

Tài liệu này tổng hợp thiết kế kiến trúc tổng thể, quy trình xử lý (pipeline), và giải quyết triệt để các rủi ro về sai số dữ liệu (Bias) đã được thảo luận.

## 1. Mục tiêu Cốt lõi
Xây dựng một hệ thống hoàn chỉnh từ đầu đến cuối nhằm tự động theo dõi, tính toán thời gian xuất hiện (Exposure Time) và đánh giá độ chói/rõ nét (Confidence/Quality) của các logo hiển thị trên áo thi đấu của đội Bradford Bulls từ một video trận đấu.

## 2. Sơ đồ Xử lý (Pipeline Architecture)

```mermaid
flowchart TD
    %% Phase 1: Tạo dữ liệu
    subgraph P1 [Phase 1: Xây dựng tập dữ liệu - Hiện tại]
        A(Video gốc 2 tiếng) --> B[Frame Extraction\nTrích xuất khung hình cận cảnh, sắc nét]
        B --> C[Tập 300-400 frames chất lượng cao]
        C --> D{Upload lên nền tảng Roboflow}
    end

    %% Phase 2: Dán nhãn & Training (Bí kíp chống Bias)
    subgraph P2 [Phase 2: Dán nhãn & Huấn luyện (Training)]
        D --> E(Gán nhãn Logo trực tiếp trên các frame)
        D --> F(Thêm 15% Background Frames\nKhán đài, sân cỏ không dán nhãn)
        E --> G(Train mô hình YOLOv8\ntại Roboflow hoặc Google Colab)
        F --> G
        G --> H((Model Weights/Bộ Não AI))
    end

    %% Phase 3: Quét dữ liệu trên Video Thực (Inference)
    subgraph P3 [Phase 3: Quét Video Thực (Inference)]
        H --> I
        A --> I[AI Quét toàn bộ 2 tiếng Video]
        I --> J[Log Tracker: Xuất Tọa độ, Confidence,\nTên Logo từng frame vào file CSV]
        J --> K[Replay Detection\nThuật toán phát hiện & Cắt bỏ đoạn phát lại]
    end

    %% Phase 4: Tính toán kết quả & Xuất Video
    subgraph P4 [Phase 4: Tính toán & Kết xuất báo cáo]
        K --> L[Tính Exposure Time = Số frame / FPS]
        K --> M[Tính Weighted Exposure\nDựa trên tổng Confidence score]
        L --> N[Tạo báo cáo BI / Bảng giá tài trợ cho CLB]
        M --> N
        I --> O[Vẽ khung & text hiển thị đè lên Video gốc]
        O --> P[Xuất Demo Video MP4 Highlight cho Nhà tài trợ]
    end
    
    P1 --> P2
    P2 --> P3
    P3 --> P4
```

## 3. Tổng hợp Thắc mắc, Rủi ro (Bias) và Giải pháp

Trong quá trình thiết kế, một số lo ngại về Data Bias (Thiên lệch dữ liệu) và tính thực tiễn đã được đưa ra. Dưới đây là cách chúng ta sẽ giải quyết triệt để từng vấn đề:

> [!WARNING]
> Sự chênh lệch giữa lúc AI học (chỉ học hình cận nét) và lúc thi trực tiếp trên video (có khán giả, có cảnh rộng) sẽ dẫn đến kết quả định giá sai lệch nếu không được điều chỉnh ở Phase 2 và Phase 3.

### A. Vấn đề cảnh khán đài, sân trống (Nhận diện nhầm áo khán giả)
- **Thắc mắc:** Nếu hệ thống chỉ học từ các frame có logo nét, liệu ra video thực tế nó có nhìn nhầm khán giả mặc áo đỏ/trắng, chi tiết bảng quảng cáo trên sân thành logo của cầu thủ (False Positive) không?
- **Giải pháp:** Bắt buộc áp dụng **Negative Examples (Background / Null Frames)**. Trong lúc gán nhãn, chúng ta lồng vào khoảng 15-20% ảnh cảnh nền trống (không thấy mặt cầu thủ/không có logo rõ rệt) và tuyệt đối để trống không vẽ hộp nào (no bounding boxes ghi nhận). YOLO sẽ được dạy cách "im lặng" khi gặp các cảnh mang nhiều nhiễu như vậy.

### B. Vấn đề cảnh quay góc siêu rộng (Wide Shots)
- **Thắc mắc:** Tại bước trích xuất (Extraction), đa số chỉ lọc ra ảnh cận cảnh. Liệu điều này sẽ khiến mô hình bị "mù" khi camera quay lia góc rộng không?
- **Giải pháp:** 
  1. *Về kỹ thuật luyện AI:* Pipeline `selection.py` hiện tại không cắt bỏ hoàn toàn cảnh rộng mà vẫn duy trì quota khoảng 25% cho cảnh medium và 5% cho wide, cung cấp lượng dữ liệu vừa đủ để AI quan sát từ xa.
  2. *Về quy chuẩn định giá (Valuation Standard):* Nếu logo ở quá xa đến mức AI không nhận dạng được (trả về giá trị rỗng hoặc confidence cực thấp), điều này đồng nghĩa với việc mắt thường của khán giả qua tivi cũng không thể nhận ra thương hiệu đó. Do đó, việc AI tự động đánh giá thấp hoặc lọc bỏ các frame quá xa là một kết quả hoàn toàn đúng với tinh thần định giá thực tiễn của báo cáo.

### C. Vấn đề cảnh chiếu lại (Replay) của nhà đài
- **Thắc mắc:** Cảnh Replay chiếu lại bàn thắng luôn chiếu rất chậm, cận cảnh và rất nét. Logo xuất hiện nhiều, rõ, và điểm báo cáo sẽ nhân đôi vô lý.
- **Giải pháp:** Giải quyết bằng Code Python ở Phase 3. Xây dựng một mô-đun phát hiện cảnh chuyển tiếp (Video Transistion Detection). Các đài thường lướt hiệu ứng (như logo đài giật qua màn hình) để báo hiệu Replay. Khi có log inference từ AI báo về, hệ thống sẽ tự động quét, khoanh vùng khoảng thời gian ở giữa các đoạn chuyển tiếp TV đó để khấu trừ khỏi tổng số giây Exposure.

### D. Vấn đề Đưa toàn bộ Frame Bị Loại vào Huấn luyện (Hard Negative Mining)
- **Thắc mắc:** Thay vì chỉ học 400 frames tốt, liệu ta có nên đưa *tất cả* các frame bị loại (không dán nhãn) vào mô hình để giúp mô hình hiểu đâu là những cảnh "không cần quan tâm" và tăng độ chính xác lên không?
- **Giải pháp:** Việc đưa thêm hình ảnh trống (không dán nhãn) vào là đúng theo nguyên lý Hard Negative Mining giúp giảm ảo giác (False Positive). Tuy nhiên, **tuyệt đối không được đưa toàn bộ**. Có 2 lý do:
  1. *Nguy cơ mất cân bằng (Data Imbalance):* Nếu trộn 400 ảnh tốt với 5.000 ảnh rác, AI sẽ trở nên "lười biếng" và ưu tiên dự đoán "không có logo" để ăn điểm an toàn, dẫn đến bỏ sót toàn bộ logo thực.
  2. *Bẫy Mâu Thuẫn Dữ Liệu:* Nhiều frame bị loại vì mờ, nhưng rốt cuộc *vẫn chứa logo*. Nếu đưa chúng vào mà bỏ trống (không khoanh), AI sẽ bị tẩu hỏa nhập ma (lúc thì bảo khoanh logo đỏ, lúc lại cấm khoanh).
  **Hành động chốt:** Chỉ lấy đúng tỉ lệ vàng **10% (khoảng 40 frames)** là các bối cảnh chắc chắn 100% không chứa cầu thủ (cỏ, khán đài). Pipeline `selection.py` hiện tại đã được cấu trúc tự động bắt 10% `"background"` frames này (xem lại phần Sơ đồ Xử lý).


## 4. Kế hoạch Hành động Ngắn hạn (Next Steps)

1. **Gán nhãn Phase 1 trên Roboflow (Người dùng chủ trì):**
   - Thiết lập chuẩn các Class Logo (Ví dụ khoảng 21 nhãn nhà tài trợ).
   - Vẽ khung chữ nhật (bounding boxes), tránh dán nhãn các logo dưới bị che khuất hơn 50%.
   - Add thêm ảnh Negative (Null Background Frames).

2. **Huấn luyện mô hình YOLOv8 Phase 2 (Hệ thống hướng dẫn):**
   - Hoàn tất Train mô hình; thu về file weights đuôi `.pt`. Điểm mAP tối thiểu cần đạt > 0.8.

3. **Cấu trúc lại Phase 3 & 4 (Hệ thống AI Assistant hỗ trợ lập trình):**
   - Viết kịch bản Python (`tracker.py`) để chạy bộ weights `.pt` qua suốt thời lượng của video clip.
   - Thêm hàm `sv.BoxAnnotator` bằng thư viện `supervision` để vẽ khung trực quan lồng ghép lên Video đính kèm bộ đếm (Ví dụ Demo Video Output).
   - Thiết lập module Python Pandas lấy dữ liệu từ Log file CSV để tổng kết ra số giây xuất hiện riêng biệt cho tùng nhãn hàng.

---

## 5. Tầm nhìn Dài hạn: Kiến trúc SaaS Toàn Diện (Scale out cho Nhiều CLB)

Khi bạn muốn biến công nghệ này thành một Sản phẩm B2B (SaaS - Software as a Service) bán cho hàng loạt các Câu lạc bộ Thể thao, chúng ta không thể "train tay" hay "chạy code script" cho từng khách hàng được. Dưới đây là Bản vẽ Hệ thống toàn diện để tự động hóa mọi khâu.

### Sơ đồ Hệ thống SaaS Định giá Phơi sáng (Exposure Valuation SaaS)

```mermaid
flowchart TD
    %% MULTI-TENANT ONBOARDING
    subgraph Onboarding [1. Multi-Tenant Onboarding (Web App)]
        A1(Người dùng/CLB) --> A2[Nhập Brand Identity\nMàu áo Home/Away, Logo Reference]
        A2 --> A3[(Tenant Config DB)]
    end

    %% MLOPS PIPELINE (TỰ ĐỘNG SINH DATA)
    subgraph MLOps [2. Automated MLOps Pipeline]
        A3 -.-> B1
        B1(Tải Video Trận đấu Ngẫu Nhiên) --> B2[Smart Frame Extraction\nGiống thuật toán bạn đang làm]
        B2 --> B3[Auto-Annotation Engine\nDùng Zero-shot Model: Grounding DINO + SAM]
        B3 --> B4[Human-in-the-Loop Review\nNhân viên QA chỉ cần click Duyệt/Lỗi]
        B4 --> B5[Auto-Train Farm\nTự động train YOLO cho đội bóng đó]
        B5 --> B6[(Model Registry\nLưu trữ Models theo CLB)]
    end

    %% CLOUD INFERENCE ENGINE
    subgraph Inference [3. Distributed Cloud Inference Engine]
        C1(Upload Video Trận Đấu Mới) --> C2{Load Balancer / API Gateway}
        C2 --> C3[Video Splitter\nCắt video thành các đoạn 10 phút]
        
        B6 -.-> C4
        C3 --> C4[GPU Worker Nodes 1..N\nQuét Inference Song Song]
        C4 --> C5[Object Tracking Module\nBoT-SORT: Theo dõi khung hình liên tục]
        C5 --> C6[(Detection Datalake\nHàng triệu logs tọa độ)]
    end

    %% DATA ALCHEMY & REPORTING
    subgraph Analytics [4. Phân tích Dữ liệu & Xử lý Bias]
        C6 --> D1[Replay/Transition Filter\nKhử đoạn chiếu lại]
        D1 --> D2[Quality Indexing Engine\nChấm điểm: Clarity, Size, Clutter]
        D2 --> D3[Media Value Equivalence\nQuy đổi thời gian ra số Tiền tương đương]
    end

    %% END CONSUMER DASHBOARD
    subgraph Dashboard [5. Dashboard Báo Cáo Tài Trợ]
        D3 --> E1[Web Dashboard React/Next.js]
        E1 --> E2(Biểu đồ Exposure)
        E1 --> E3(Cảnh báo lệch góc Camera)
        E1 --> E4[Tự động Render Highlight Video\nAuto Video Editor]
    end

    Onboarding --> MLOps
    MLOps --> Inference
    Inference --> Analytics
    Analytics --> Dashboard
```

### Các Luồng Công Nghệ Nâng Cấp Thiết Yếu Cho SaaS

Để hệ thống hoạt động mượt mà cho 100 đội khác nhau mà không sụp đổ, bạn sẽ cần nâng cấp các khối sau:

**1. Khối Tự Động Gán Nhãn (Zero-shot Auto-Annotation)**
Thay vì bạn phải lên Roboflow vẽ 400 hộp cho mỗi đội bóng mới.Hệ thống sẽ dùng **Grounding DINO** (mà bạn đã chuẩn bị trong requirements) kết hợp Segment Anything (SAM).
Hệ thống sẽ hỏi CLB đưa cho tấm ảnh logo gốc. Grounding DINO sẽ tự tìm logo đó trên áo cầu thủ và vẽ hộp. Con lúc này chỉ cần nhân sự mở ra xem lướt qua xem máy vẽ đúng không, thao tác 400 frames lúc này rút ngắn từ **4 tiếng xuống còn 15 phút**.

**2. Khối Nhận Diện Tốc Độ Cao (Distributed Inference)**
Video 2 tiếng chạy bằng 1 máy tính có thể mất 1 tiếng. Trên nền tảng SaaS, hệ thống sẽ cắt video 2 tiếng thành 12 phần nhỏ (Mỗi phần 10 phút). Giao cho 12 con Server GPU (như AWS EC2 T4) chạy song song. Tốc độ thu thập kết quả cho 1 trận đấu khổng lồ giảm xuống chỉ còn **khoảng 5 phút**.

**3. Tích hợp Object Tracking (Theo dấu đối tượng)**
Bạn không đếm từng frame rời rạc. Bạn sẽ cần gắn thêm thuật toán tracker (ví dụ BoT-SORT/ByteTrack).
Khi cầu thủ di chuyển, AI hiểu được "Logo A ở frame 1" và "Logo A ở frame 2" là **cùng một sự kiện**, từ đó tính ra được con số: *Sự kiện số 4 kéo dài 5.2 giây*. Nếu không có Tracker, báo cáo sẽ bị nhiễu loạn thông tin.

**4. Module Tính Tiền Cắm Rời (Valuation Engine)**
Khách hàng không chỉ muốn biết logo xuất hiện 450 giây. Họ muốn biết 450 giây đó đáng giá bao nhiêu Đô la. Báo cáo của bạn phải quy chiếu với Đơn giá truyền hình (TV Advertising Equivalent Value) kết hợp tham số Size, Clutter để kết xuất ra **số tiền**. Đây sẽ là tính năng "bán lấy tiền" đắt giá nhất của hệ thống!

## User Review Required
Bạn hãy rà soát lại toàn bộ **Kiến trúc vĩ mô** này. Kiến trúc này có đi đúng cái Tầm Nhìn Hệ Sinh Thái (Ecosystem Vision) mà bạn đang nhắm tới không? Hãy xác nhận (Approve) và chúng ta sẽ lưu tài liệu tối thượng này lại làm Bản Thiết Kế Định Hướng của toàn dự án.
