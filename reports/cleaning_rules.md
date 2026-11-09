# Quy tắc làm sạch dữ liệu (GĐ2)

Nguồn: VietJobs (`data/raw/vietjobs/VietJobs.csv`, sha256 trong `data/raw/download_log.json`).
Mã nguồn: `src/data_cleaning.py`. File gốc không bị sửa; cột gốc được giữ với tên `*_raw` / `*_src`.

## 1. Phạm vi "IT"
IT được xác định **theo chức danh**, không theo category của dataset. Lý do: category
`công_nghệ_thông_tin_kỹ_thuật_số` của VietJobs gộp cả thiết kế đồ họa và kỹ thuật số (GĐ1b).

Thứ tự áp dụng:
1. **Loại trừ** nếu chức danh thuộc một trong các nhóm: kinh doanh / bán hàng / marketing / SEO / CSKH;
   tư vấn (trừ "tư vấn triển khai"); thiết kế / đồ họa / UI-UX / 3D / nội thất / video / nội dung;
   lập trình máy công cụ (CNC, PLC, phay, tiện); an ninh / bảo vệ (trừ an ninh mạng,
   an ninh thông tin, an toàn thông tin, cyber); kế toán / thủ kho / thu ngân /
   hành chính / văn phòng (trừ khi có chữ "IT"); giảng dạy; bất động sản; presales / phát triển khách hàng
   / thị trường; gán nhãn dữ liệu / huấn luyện viên AI; smarthome / KNX; giám sát camera.
2. **Vai trò IT** (khớp đầu tiên):
   | job_category | Ví dụ | Cần category CNTT? |
   |---|---|---|
   | Quản lý & phân tích (PM/BA/PO) | BrSE, Scrum Master, IT PM/BA | Có, trừ BrSE/Scrum/có chữ "IT" |
   | Triển khai & ERP | ERP, SAP, Odoo, triển khai phần mềm | "triển khai" chung chung: có |
   | Kiểm thử & QA | Tester, kiểm thử, QA/QC engineer | "QA"/"QC" đơn lẻ: có |
   | Phát triển phần mềm (nhúng) | "nhúng" (so có dấu), embedded, firmware; xét trước các dòng dưới để "Kỹ sư Linux nhúng", "Kỹ Sư IT- Nhúng" vào đúng nhóm | Không |
   | Dữ liệu & AI | Data engineer/scientist, ML, AI engineer | Data analyst / BI: có |
   | Hạ tầng, DevOps & bảo mật | DevOps, cloud, sysadmin, network, security, an ninh mạng, an toàn thông tin, redteam | Không |
   | Hỗ trợ kỹ thuật & IT chung | Helpdesk, IT support, "Nhân viên IT" | "hỗ trợ kỹ thuật" chung chung: có |
   | Phát triển phần mềm | Developer, lập trình, kỹ sư phần mềm, nhà phát triển | "engineer"/"kỹ sư" chung chung: có |
3. Từ dễ va chạm khi bỏ dấu được so khớp có dấu hoặc phân biệt hoa thường:
   "kiểm thử" (khác "kiêm thủ", "kiêm thu ngân"), "SAP" (khác "sắp"), "AI" chỉ khi là vai trò
   (AI Engineer/Developer…), không tính "biết sử dụng AI".
4. Chức danh dịch máy được hỗ trợ: "nhà phát triển", "phụ trợ" (backend), "thống nhất" (Unity),
   "lãnh đạo kỹ thuật" / "kỹ thuật dẫn đầu" (tech lead), "chủ sở hữu sản phẩm" (PO),
   "phân tích kinh doanh" (BA), "kiến trúc sư giải pháp" (solution architect),
   "người thử nghiệm" / "người kiểm tra" (tester — chỉ khi thuộc category CNTT).
5. Bộ quy tắc được tự kiểm tra trên 38 chức danh mẫu mỗi lần chạy (`SELF_CHECK` trong mã).

## 2. Lương
- Đơn vị: triệu VND/tháng. GĐ1a không thấy USD, theo năm/giờ, "từ"/"tới" trong cột `salary`.
- `salary_min`, `salary_max`: parse từ chuỗi `salary` (giữ phần thập phân). Cột int gốc của dataset
  (`salary_min_src`, `salary_max_src`) là floor của giá trị này (GĐ1b: khớp 100%).
- "0.0 triệu" = thỏa thuận (khớp 1:1 với `salary_avg` = "Thoả thuận") → không dùng làm nhãn.
- "x triệu" (điểm đơn) → loại khỏi mô hình chính: không phân biệt được lương cố định với khoảng mở bị cắt.
- "0 - b" và "a - b" với a > b → loại.
- **Nhãn** `salary_label = (salary_min + salary_max) / 2` — giả định xấp xỉ, không phải lương thực nhận.
- Chỉ giữ 3 ≤ salary_min và salary_max ≤ 200 triệu (ngoài phạm vi: nghi sai đơn vị/lỗi nhập).
- Loại khoảng quá rộng: salary_max / salary_min > 5 (ví dụ "5 - 200 triệu", "5 - 100 triệu" phát hiện ở EDA
  GĐ2): trung điểm của khoảng như vậy không phải ước lượng lương có nghĩa. Ngưỡng đặt SAU khi xem EDA,
  ảnh hưởng tối đa 8 tin (số tin có max/min > 3) — ghi nhận là quyết định hậu kiểm.

## 3. Các trường khác
- **Hợp đồng**: chỉ giữ "Toàn thời gian" (thực tập / bán thời gian có thang lương khác).
- **Quốc gia**: chỉ giữ "Việt Nam".
- **Kinh nghiệm**: "Không yêu cầu" = 0; "N tháng" = N/12 năm. Dataset chỉ có MỘT giá trị nên
  `experience_years` thay cho cặp min/max. Nhóm: 0 / <1 / 1 / 2 / 3 / 4+ năm.
- **Cấp bậc** (`seniority`): suy ra từ từ khóa chức danh (manager > lead > senior > middle > junior >
  fresher > intern), còn lại "unknown". "Project/Product/Account Manager" không tính là manager.
- **Thành phố** (`city`): Hà Nội (gồm các quận/huyện), TP.HCM (gồm Thủ Đức, quận số…), Đà Nẵng, Khác,
  Nhiều thành phố. Theo địa danh ghi trong tin, không quy đổi theo địa giới sau sáp nhập 7/2025.
- **Kỹ năng**: parse danh sách, lowercase, giữ dấu chấm đầu (".net"), gộp đồng nghĩa (reactjs→react, node.js→nodejs,
  sql server/mssql, các biến thể "tin học văn phòng"…), lưu dạng `a|b|c`. Chọn top-k kỹ năng làm
  đặc trưng sẽ fit trên train ở GĐ3.
- **Ngày đăng**: không có trong VietJobs → `date` để trống.

## 4. Trùng lặp
- Gộp tin trùng theo (chức danh chuẩn hóa, địa điểm gốc, chuỗi lương gốc, mô tả chuẩn hóa), giữ bản đầu.
- `desc_group` = hash của mô tả chuẩn hóa. GĐ3 chia train/test theo nhóm này để các tin đăng lại
  cùng mô tả không nằm ở cả hai phía.
- Không có cột công ty nên không gộp theo công ty được.
