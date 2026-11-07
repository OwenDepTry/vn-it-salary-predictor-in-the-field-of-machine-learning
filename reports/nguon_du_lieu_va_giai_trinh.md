# Nhật ký nguồn và phương pháp kiểm chứng — Thành viên 1

## 1. Nguồn
**VietJobs:** https://huggingface.co/datasets/dinhieufam/VietJobs — người công bố dinhieufam; dữ liệu CSV, 18 cột theo trình xem dữ liệu. Đây là dữ liệu tuyển dụng nhiều ngành, **không phải chỉ IT**. Tác giả bộ dữ liệu không đồng nghĩa với chủ sở hữu từng tin tuyển dụng. Không khẳng định đã xác thực từng nhà tuyển dụng.

**IT-Job-Posting:** https://github.com/SonPhatTranDeveloper/IT-Job-Posting — tác giả SonPhatTranDeveloper; README công bố 3.101 tin từ LinkedIn, ITviec, TopCV. Danh sách 11 trường trong README không chứa trường lương. Nguồn đối chiếu, không ghép nhãn lương.

## 2. Tính tái lập
Lưu đường dẫn tải, SHA-256, kích thước byte, thời điểm tải UTC vào `data/raw/download_log.json`. Trình bày số dòng thực đếm, không chép số dòng từ tài liệu nhóm khi có sai lệch. Phiên bản nguồn có thể đổi: nên lưu commit GitHub và revision Hugging Face khi hoàn thiện báo cáo.

## 3. Quy tắc kiểm toán
- `salary = 0.0 triệu` hoặc `Thoả thuận` được xếp nhóm cần loại khỏi hồi quy có nhãn số.
- Tin không rõ đơn vị tiền tệ hoặc kỳ trả lương phải kiểm tra thủ công; không tự quy đổi.
- Chức danh IT xác định bằng quy tắc có thể kiểm tra; xuất toàn bộ ứng viên ra CSV để rà soát sai dương/sai âm.
- Phễu sơ bộ khác phễu sạch cuối; số 449 trong phân công là mốc tham chiếu, **không phải kết quả đã đo**.
- Không đưa `salary_min`, `salary_max`, `salary_avg` vào đặc trưng dự đoán nếu chúng tạo nên nhãn, để tránh rò rỉ.

## 4. Điều kiện GO/NO-GO
GO chỉ khi số tin IT có nhãn số hợp lệ đủ cho huấn luyện/đánh giá, luật làm sạch được kiểm thử, phân bố theo nhóm đủ dùng và không rò rỉ nhãn. NO-GO hoặc thu hẹp phạm vi nếu thiếu nhãn, mẫu quá ít hay thiên lệch nghiêm trọng. Quyết định cuối phụ thuộc thống kê thực nghiệm.

## 5. Giới hạn
Tin đăng tuyển không đại diện hoàn toàn cho thị trường; có tin trùng, lương thỏa thuận, mức lương có thể không phản ánh lương thực nhận. Các trường kỹ năng có thể do tác giả trích xuất tự động và có sai số.
