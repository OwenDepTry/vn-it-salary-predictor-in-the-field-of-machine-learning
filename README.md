# TV2 Data Cleaning

## Thay file trong project
Giải nén ZIP vào thư mục gốc `do-an-ml`, chọn ghi đè 4 file:
- `src/data_cleaning.py`
- `src/features.py`
- `tests/test_data_cleaning.py`
- `tests/test_features.py`

## Chạy kiểm thử
```powershell
python -m pytest -q
```

## Chạy pipeline
```powershell
python src/data_cleaning.py
python src/features.py
```

## Lưu ý về kết quả dữ liệu
Bản code này đã qua 9/9 unit tests và kiểm tra tích hợp được việc tạo group, chia train/test, fit preprocessor.
Tuy nhiên, với `VietJobs.csv` đi kèm trong phiên làm việc, pipeline hiện tạo 313 dòng, chưa đạt mục tiêu 449 dòng trong báo cáo leader. Không dùng kết quả 313 dòng để nộp. Cần hoàn tất hiệu chỉnh tập nhận diện IT theo bảng rà soát của leader trước khi xác nhận hoàn thành.
