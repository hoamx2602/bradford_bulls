# Phương Pháp Luận & Logic Định Giá Tài Trợ Thể Thao (Sponsorship Valuation Methodology)

Tài liệu này trình bày chi tiết cách hệ thống chuyển đổi dữ liệu thô từ AI (Machine Learning Bounding Boxes) thành các chỉ số kinh doanh có giá trị thực tiễn (Sponsorship Media Value) dành cho các Nhà tài trợ và Câu lạc bộ bóng đá/bóng bầu dục.

Khác với việc chỉ "đếm khung hình" đơn thuần, hệ thống phân tích truyền thông cấp độ doanh nghiệp (như Nielsen Sports hay GumGum) đều áp dụng một cơ chế lọc 4 Tầng (4-Layer Filtering Pipeline) nhằm đảm bảo số liệu minh bạch, chống lặp lại và phản ánh đúng chất lượng hiển thị.

---

## Tầng 1: Lọc Trùng Lặp (Simultaneous Exposure Deduplication)

**Vấn đề:** 
Một khung hình (frame) quay cận một pha ăn mừng có thể lên tới 4 cầu thủ cùng mặc áo đấu mang logo AON. Thuật toán YOLO sẽ trả về 4 tọa độ Bounding Boxes độc lập. Nếu tính tổng thời gian hiển thị của 4 hộp này, hệ thống sẽ gây ra lỗi "đếm kép" (Double Counting), dẫn đến thời gian báo cáo lớn hơn cả thời lượng video thực tế.

**Cách Xử Lý Kỹ Thuật:**
- **Aggregation (Gộp sự kiện):** Thuật toán gộp tất cả các nhận diện trùng lặp trong cùng một mili-giây (cùng một frame) thành **1 Sự Kiện Hiển Thị Duy Nhất (Single Exposure Event)**. Thời gian ghi nhận cho thương hiệu lúc này vẫn là độ dài của số frame đó.
- **Biến số phụ:** Dù gom lại làm 1, hệ thống vẫn lưu thông tin `Logo_Count`. Giá trị này sẽ được dùng làm trọng số thưởng ở các Tầng sau (Sự xuất hiện chùm mang lại sự chú ý cao hơn).

---

## Tầng 2: Mịn hóa và Chống Nhiễu Thời Gian (Temporal Smoothing & Thresholds)

**Vấn đề:** 
Chuyển động của cầu thủ (chạy, xoay người, bị đối phương che khuất) khiến AI không thể nhận diện logo một cách liền mạch 100%. Tín hiệu trả về bị đứt đoạn dạng nhấp nháy: `[Thấy] - [Thấy] - [Mất 3 frames] - [Thấy]`. Khách hàng không thể nhận một báo cáo với hàng nghìn "Sự kiện 0.1 giây" rời rạc.

**Cách Xử Lý Kỹ Thuật:**
1. **Gap Bridging (Thuật toán lấp lỗ hổng):** 
   Triển khai bộ đếm Object Tracking (như thuật toán Hysteresis). Nếu logo bị che lấp và xuất hiện lại trong một khoảng thời gian cực ngắn (VD: Dưới 15 frames / 0.5 giây), hệ thống sẽ ngầm hiểu đây chỉ là nội suy do vật cản, và tự động "nối" chúng thành một dải xuất hiện liền mạch.
2. **Minimum Viewability Threshold (Bộ Lọc Ghi Nhận Não Bộ):**
   Tiêu chuẩn truyền hình quốc tế chỉ ra rằng não người cần ít nhất **1.5 đến 2 giây** để xử lý và ghi nhớ một thương hiệu lướt qua màn hình. Bất kỳ sự kiện xuất hiện nào tổng thời gian dưới tỷ lệ này sẽ bị xóa khỏi báo cáo tài chính cuối cùng, nhằm bảo vệ uy tín định giá cho Câu lạc bộ.

---

## Tầng 3: Đo Lường Điểm Chất Lượng Thực Tế (Quality Indexing)

**Vấn đề:** 
Độ Tự Tin (Confidence Score) từ model AI như YOLO chỉ thể hiện việc "Máy chắc chắn tới đâu đây là cái Logo", nhưng hoàn toàn không thể hiện được "Tầm nhìn rành mạch" (Visibility) mà mắt người xem tivi cảm nhận. Đây là thước đo sống còn để phân cấp giá trị.

**Cách Xử Lý Kỹ Thuật (Tính điểm QI - Quality Index):**
Mỗi giây xuất hiện hợp lệ ở Tầng 2 sẽ được chấm điểm từ 0.0 - 1.0 dựa trên Trọng số Chất lượng cấu thành từ 4 yếu tố không gian:

1. **Size (Kích thước / Share of Voice):**
   + *Công thức:* Tỷ lệ diện tích Bounding Box chia cho Tổng diện tích của Khung hình video (Screen Area).
   + Những pha quay cận cảnh (Close-up) sẽ mang lại chất lượng và giá trị cao gấp hàng chục lần so với viễn cảnh (Wide shot).
2. **Position & Prominence (Vị trí Trung tâm):**
   + Logo nằm ngay trọng tâm màn hình sẽ có hệ số nhân `1.0`. Càng trôi về 4 rìa màn hình (những vùng khán giả hiếm khi liếc nhìn), hệ số sẽ giảm dần xuống `0.5` hoặc thấp hơn.
3. **Clarity (Độ rõ nét & Ánh sáng):**
   + Đây là lúc `Confidence Score` của thuật toán AI kết hợp với kỹ thuật Laplacian (chống Motion Blur) để trừ điểm nếu hình ảnh quá mờ nhòe, tối tăm.
4. **Clutter (Mức độ Cạnh tranh Thương hiệu):**
   + Tính năng nâng cao: Nếu trên màn hình lúc đó có 5 hãng tài trợ khác nhau cùng xuất hiện (trên áo, bảng biển, khán đài), sự mất tập trung của người xem sẽ xuất hiện. Điểm Quality của mỗi logo trên màn hình sẽ bị giảm trừ.

---

## Tầng 4: Kết Xuất Báo Cáo & Quy Đổi Tài Chính (The Valuation Engine)

Tầng cuối cùng chính là "cỗ máy in tiền". Dữ liệu đi qua các tầng trên sẽ xuất ra một Bảng điều khiển (Dashboard) với các thông số ngôn ngữ Marketing thay cho thuật toán:

1. **Total On-Screen Time (Tổng Thời Gian Hiện Diện):** 
   Tổng số giây nhãn hàng xuất hiện thực tế không tính lặp lại. VD: Lên sóng 15 Phút.
2. **Total Impact Events (Số Lần Gây Ảnh Hưởng):** 
   Số bối cảnh xuất hiện kéo dài trên 2 giây. VD: 60 Lần.
3. **100% Equivalent Time (Thời Gian Phơi Sáng Quy Đổi 100%):** 
   Chỉ số đỉnh cao nhất của hệ thống. Bằng cách lấy (Thời Gian Mỗi Giây) X (Chỉ số QI từng Giây đó). 
   VD: 15 Phút hiển thị rải rác nhưng nhiều đoạn góc hẹp, mờ, nhỏ... sẽ được máy xén và nén lại thành **"Tương Đương 5 Phút 20 Giây chiếm lĩnh toàn bộ Màn Hình"**.
4. **Media Advertising Value Equivalence (Giá Trị Quảng Cáo Tương Đương):**
   Lấy thời gian quy đổi `[5 Phút 20 Giây]` nhân cho `[Đơn Giá 1 Giây Quảng Cáo TV]`. Khởi tạo nên bức tranh tài chính chính xác nhất cho nhà đầu tư để đưa ra quyết định tái ký kết tài trợ.
