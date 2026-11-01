# Quy tắc làm sạch dữ liệu – TV2

## 1. Phạm vi dữ liệu

Dữ liệu đầu vào được lấy từ bộ dữ liệu VietJobs (`data/raw/vietjobs/VietJobs.csv`).

Quá trình xử lý tập trung vào các tin tuyển dụng thuộc lĩnh vực Công nghệ thông tin (IT) tại Việt Nam, có thông tin lương phù hợp với phạm vi nghiên cứu.

## 2. Quy tắc làm sạch dữ liệu

### 2.1. Lọc tin tuyển dụng IT

- Sử dụng chức danh công việc để nhận diện các vị trí thuộc lĩnh vực IT.
- Phân loại tin tuyển dụng thành 7 nhóm: Phát triển phần mềm; Kiểm thử & QA; Hạ tầng, DevOps & bảo mật; Dữ liệu & AI; Quản lý & phân tích (PM/BA/PO); Triển khai & ERP; Hỗ trợ kỹ thuật & IT chung.
- Loại bỏ các vị trí không thuộc phạm vi nghiên cứu như bán hàng, marketing, kế toán, thiết kế đồ họa và các công việc phi kỹ thuật.
- Kết quả phân loại được lưu trong cột `job_category`.

### 2.2. Xử lý mức lương

- Chuyển thông tin lương ban đầu (`salary_raw`) thành hai giá trị số `salary_min` và `salary_max`.
- Chuẩn hóa đơn vị lương về triệu VND/tháng.
- Loại bỏ các bản ghi có mức lương thỏa thuận hoặc không xác định được mức lương phù hợp.
- Với bản ghi chỉ có một mức lương, sử dụng giá trị đó cho cả mức tối thiểu và tối đa.
- Xem mức lương bằng 0 là dữ liệu thỏa thuận, không phải mức lương thực tế.
- Chỉ giữ các bản ghi có mức lương trong khoảng 3–200 triệu VND/tháng.
- Kiểm tra điều kiện `salary_min <= salary_max`.
- Loại bỏ các khoảng lương có tỷ lệ giữa mức tối đa và tối thiểu lớn hơn 5.

### 2.3. Chuẩn hóa kinh nghiệm

- Chuyển thông tin kinh nghiệm thành số năm (`experience_years`).
- Quy đổi số tháng kinh nghiệm sang năm.
- Phân nhóm kinh nghiệm bằng `experience_group`.
- Suy luận cấp bậc công việc từ chức danh thông qua `seniority_from_title`.
- Lưu cấp bậc vào cột `seniority`.

### 2.4. Chuẩn hóa địa điểm

- Chuẩn hóa thông tin thành phố bằng `map_city`.
- Ánh xạ các cách viết khác nhau của cùng một địa điểm về tên thành phố thống nhất.
- Lưu kết quả trong cột `city`.

### 2.5. Chuẩn hóa kỹ năng

- Chuẩn hóa tên kỹ năng bằng `normalize_skill`.
- Tách danh sách kỹ năng bằng `parse_skills`.
- Loại bỏ các giá trị rỗng và chuẩn hóa cách biểu diễn kỹ năng.
- Lưu danh sách kỹ năng trong cột `skills`, sử dụng dấu `|` để phân tách.

### 2.6. Xử lý dữ liệu trùng lặp

- Chuẩn hóa thông tin chức danh, địa điểm, mức lương và mô tả công việc trước khi xác định bản ghi trùng lặp.
- Sử dụng thông tin công việc để nhận diện các bản ghi có nội dung tương ứng.
- Tạo mã nhóm mô tả `desc_group` để hỗ trợ việc nhận diện các bản ghi liên quan.
- Việc tạo nhóm dữ liệu ở bước Feature Engineering sử dụng Union-Find để xử lý cả quan hệ trùng lặp bắc cầu.

## 3. Feature Engineering và chia dữ liệu

- Sử dụng Union-Find để tạo nhóm dữ liệu (`group_id`), bao gồm cả các quan hệ trùng lặp bắc cầu.
- Chia dữ liệu theo nhóm bằng `GroupShuffleSplit`.
- Sử dụng `random_state = 42` để đảm bảo khả năng tái lập kết quả.
- Đảm bảo một nhóm không xuất hiện đồng thời trong tập train và test.
- Lưu thông tin chia tập vào `data/processed/split.csv`.
- Kiểm tra danh sách `row_id` để tránh sử dụng nhầm file chia tập khi dữ liệu thay đổi.
- Bộ mã hóa kỹ năng chỉ học vocabulary từ dữ liệu train khi gọi `fit`, tránh rò rỉ thông tin từ tập test.
- Sử dụng `ColumnTransformer` để xử lý các nhóm đặc trưng số, phân loại và kỹ năng.

## 4. Kiểm tra chất lượng dữ liệu

Sau khi làm sạch, dữ liệu được kiểm tra các tiêu chí:

- Số dòng trùng lặp.
- Chức danh bị thiếu.
- Mức lương tối thiểu hoặc tối đa bị thiếu.
- Khoảng lương không hợp lệ.
- Tính nhất quán giữa các cột đặc trưng và nhãn mục tiêu.

### 4.1. Kết quả lọc dữ liệu

Theo lần chạy hiện tại:

| Công đoạn | Số bản ghi |
|---|---:|
| Dữ liệu đầu vào | 48.092 |
| Sau lọc Việt Nam | 47.957 |
| Sau lọc chức danh IT | 1.221 |
| Sau loại lương thỏa thuận | 663 |
| Sau xử lý lương đơn | 555 |
| Sau loại lương không hợp lệ | 555 |
| Sau lọc full-time | 545 |
| Sau kiểm tra giới hạn lương | 544 |
| Sau kiểm tra tỷ lệ lương | 540 |
| Sau loại trùng lặp | 456 |

### 4.2. Kết quả kiểm tra chất lượng

- Số bản ghi sau làm sạch: 456.
- Số dòng trùng lặp: 0.
- Số chức danh bị thiếu: 0.
- Số mức lương tối thiểu bị thiếu: 0.
- Số mức lương tối đa bị thiếu: 0.
- Số khoảng lương không hợp lệ: 0.

## 5. Các tệp đầu ra

- `data/interim/vietjobs_it_candidates.csv`
- `data/processed/it_salary_clean.csv`
- `data/processed/split.csv`
- `reports/gd2_cleaning_log.txt`
- `reports/cleaning_rules.md`
