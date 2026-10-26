# Thành viên 1 — VietJobs: thu thập, khảo sát, kiểm toán, EDA

## Nguồn công khai
- VietJobs: https://huggingface.co/datasets/dinhieufam/VietJobs
- IT-Job-Posting: https://github.com/SonPhatTranDeveloper/IT-Job-Posting

Dữ liệu tải trực tiếp từ nhà xuất bản; mã băm SHA-256 được tạo trên **file thực tế tải xuống**. File gốc không được sửa. Thời điểm tải và SHA-256 nằm trong `data/raw/download_log.json`.

## Chạy trên Mac / Windows
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
python notebooks/01a_schema_discovery.py
python notebooks/01b_data_audit.py
# Khi TV2 bàn giao data/processed/it_salary_clean.csv:
python notebooks/02_eda.py
```
Windows: `.venv\\Scripts\\activate` thay cho `source`.

**Lưu ý:** bộ lọc chức danh là bước rà soát ứng viên sơ bộ, không phải bộ lọc cuối. Phễu 449 tin sạch chỉ có thể khẳng định sau khi TV2 hoàn tất và kiểm tra luật lọc. EDA phụ thuộc dữ liệu TV2; không tạo dữ liệu giả. Mọi số trong báo cáo phải lấy từ file output của lần chạy thực tế.

## Câu hỏi bảo vệ
- Dữ liệu từ đâu? URL nguồn, bản gốc và `download_log.json`.
- 48.092 dòng có đúng không? Đếm bằng `pd.read_csv(...).shape[0]` và lưu `gd1a_schema_output.txt`.
- Vì sao 0.0 triệu không phải lương 0? Vì nguồn hiển thị salary_avg là 'Thoả thuận' cho ví dụ có salary=0.0 triệu.
- IT-Job-Posting dùng để làm gì? Đối chiếu danh mục việc IT; không dùng làm nhãn lương do không có cột salary.
- Có phải dự đoán lương thực nhận? Không, chỉ dự đoán mức lương **đăng tuyển** của tin có thông tin lương.
