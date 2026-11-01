# Cleaning Rules – TV2

## 1. Phạm vi dữ liệu

Dữ liệu đầu vào được lấy từ nguồn VietJobs. Quá trình xử lý tập trung vào các tin tuyển dụng thuộc lĩnh vực Công nghệ thông tin (IT) tại Việt Nam có thông tin lương phù hợp với phạm vi nghiên cứu.

## 2. Quy tắc làm sạch

### 2.1. Lọc tin tuyển dụng IT

- Sử dụng chức danh công việc để nhận diện các vị trí IT.
- Loại bỏ các vị trí không thuộc phạm vi IT như bán hàng, marketing, kế toán và các công việc phi kỹ thuật.
- Giữ lại các nhóm vị trí kỹ thuật phù hợp với phạm vi nghiên cứu.

### 2.2. Xử lý mức lương

- Chuyển thông tin lương về dạng số.
- Tách mức lương thành hai trường `salary_min` và `salary_max`.
- Loại bỏ các bản ghi không xác định được mức lương hợp lệ.
- Kiểm tra điều kiện `salary_min <= salary_max`.
- Không xem mức lương bằng 0 là mức lương thực tế.

### 2.3. Chuẩn hóa kinh nghiệm

- Chuyển thông tin kinh nghiệm thành số năm (`experience_years`).
- Phân nhóm kinh nghiệm bằng `experience_group`.
- Suy luận cấp bậc công việc từ chức danh thông qua `seniority_from_title`.

### 2.4. Chuẩn hóa địa điểm

- Chuẩn hóa thông tin thành phố bằng `map_city`.
- Ánh xạ các cách viết khác nhau về cùng một nhóm địa điểm.

### 2.5. Chuẩn hóa kỹ năng

- Chuẩn hóa tên kỹ năng bằng `normalize_skill`.
- Tách danh sách kỹ năng bằng `parse_skills`.
- Lưu danh sách kỹ năng ở dạng có cấu trúc để phục vụ Feature Engineering.

### 2.6. Xử lý dữ liệu trùng lặp

- Kiểm tra và loại bỏ các bản ghi trùng lặp.
- Sử dụng thông tin công việc để nhận diện các nhóm bản ghi liên quan.

## 3. Feature Engineering và chia dữ liệu

- Sử dụng Union-Find để tạo nhóm dữ liệu (`group_id`), bao gồm cả các quan hệ trùng lặp bắc cầu.
- Chia dữ liệu theo nhóm bằng `GroupShuffleSplit`.
- Sử dụng `random_state = 42` để đảm bảo khả năng tái lập.
- Đảm bảo một nhóm không xuất hiện đồng thời trong tập train và test.
- Bộ mã hóa kỹ năng chỉ học từ dữ liệu train khi fit.

## 4. Kiểm tra chất lượng

Sau khi làm sạch, dữ liệu được kiểm tra:
- Số dòng trùng lặp.
- Chức danh bị thiếu.
- Mức lương tối thiểu hoặc tối đa bị thiếu.
- Khoảng lương không hợp lệ.

Kết quả chạy hiện tại:
- Số bản ghi sau làm sạch: 371.
- Số dòng trùng lặp: 0.
- Số chức danh bị thiếu: 0.
- Số mức lương tối thiểu bị thiếu: 0.
- Số mức lương tối đa bị thiếu: 0.
- Số khoảng lương không hợp lệ: 0.

## 5. Đầu ra

- `data/interim/vietjobs_it_candidates.csv`
- `data/processed/it_salary_clean.csv`
- `data/processed/split.csv`
- `reports/gd2_cleaning_log.txt`
- `reports/cleaning_rules.md`