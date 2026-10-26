# Dự đoán mức lương đăng tuyển cho các vị trí IT tại Việt Nam

Đồ án môn Máy học: xây dựng mô hình hồi quy dự đoán mức lương (triệu VNĐ/tháng) của tin tuyển dụng IT dựa trên nhóm nghề, thành phố, kinh nghiệm, cấp bậc và kỹ năng, kèm ứng dụng web Streamlit và giao diện dòng lệnh.

- **Bộ dữ liệu chính:** [VietJobs](https://huggingface.co/datasets/dinhieufam/VietJobs) (48.092 bản ghi, 18 cột)
- **Bộ dữ liệu phụ (chỉ để khảo sát):** IT-Job-Posting (3.101 bản ghi, không có cột lương)
- **Tập sạch sau lọc IT và tạo nhãn:** 449 tin; tập test: 86 tin
- **Kết quả trên tập test:** Linear Regression, MAE ≈ 4,64 triệu, R² ≈ 0,53

> ⚠️ Kết quả dự đoán chỉ mang tính tham khảo, phản ánh mức lương *đăng tuyển* trong dữ liệu, không phải mức lương thực nhận.

---

## Thành viên nhóm

| Thành viên | Họ và tên | MSSV | Phụ trách |
|---|---|---|---|
| TV1 | Nguyễn Tiến Đạt | 3124410060 | Thu thập, khảo sát dữ liệu, EDA (GĐ1, GĐ2) |
| TV2 | Lê Ngọc Hiếu | 3124410095 | Làm sạch dữ liệu, xây dựng đặc trưng (GĐ2, GĐ3) |
| TV3 | Phan Thành Đạt | 3124410065 | Huấn luyện, chọn và đánh giá mô hình (GĐ3–GĐ5) |
| TV4 | Phan Tiến Đạt | 3124410066 | Dự đoán, ứng dụng Streamlit, kiểm thử, tích hợp (GĐ5–GĐ7) |

---

## Cấu trúc thư mục

```
.
├── app/
│   └── streamlit_app.py          # Ứng dụng web (3 trang)
├── data/
│   ├── raw/                      # Dữ liệu gốc + download_log.json (không sửa tay)
│   ├── interim/                  # vietjobs_it_candidates.csv
│   └── processed/                # it_salary_clean.csv, split.csv
├── models/
│   ├── salary_model.joblib       # Pipeline đã huấn luyện
│   └── model_card.json           # Metadata, giá trị hợp lệ, metric, câu miễn trừ
├── notebooks/
│   ├── 01a_schema_discovery.py   # Khảo sát schema, tỷ lệ thiếu
│   ├── 01b_data_audit.py         # Phân loại chuỗi lương, phễu dữ liệu, GO/NO-GO
│   └── 02_eda.ipynb              # Khám phá dữ liệu sạch
├── reports/                      # Toàn bộ output, bảng, hình dùng trong báo cáo
│   └── figures/
├── src/
│   ├── download_data.py          # Tải dữ liệu, kiểm sha256
│   ├── data_cleaning.py          # Parse lương, kinh nghiệm, lọc IT, chuẩn hóa kỹ năng
│   ├── features.py               # Gộp nhóm chống rò rỉ, chia train/test, tiền xử lý
│   ├── train.py                  # Baseline, LR (GĐ3); DT, RF + grid search (GĐ4)
│   ├── evaluate.py               # Metric, CV theo nhóm, bootstrap, đánh giá cuối
│   └── predict.py                # SalaryPredictor + CLI
├── tests/                        # 60 test (pytest)
├── requirements.txt
└── README.md
```

---

## Cài đặt

Yêu cầu Python 3 (phiên bản thư viện đã được khóa trong `requirements.txt`).

```bash
git clone <đường-dẫn-repo>
cd <tên-repo>

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Chạy lại toàn bộ dự án (7 bước)

Chạy lần lượt từ thư mục gốc của dự án. Mỗi bước đọc đầu ra của bước trước.

### Bước 1 – Tải dữ liệu (GĐ1)

```bash
python src/download_data.py
```

Tải VietJobs và IT-Job-Posting vào `data/raw/`, tính sha256 và ghi vào `data/raw/download_log.json`.

### Bước 2 – Khảo sát dữ liệu (GĐ1)

```bash
python notebooks/01a_schema_discovery.py
python notebooks/01b_data_audit.py
```

Đầu ra: `reports/gd1a_schema_output.txt`, `reports/gd1b_audit_output.txt`, `reports/gd1b_funnel.csv`, `reports/gd1b_it_title_review.csv`.

### Bước 3 – Làm sạch dữ liệu (GĐ2)

```bash
python src/data_cleaning.py
```

Lọc vị trí IT, parse lương ("0.0 triệu" được coi là thỏa thuận), kinh nghiệm, thành phố, cấp bậc, kỹ năng. Đầu ra: `data/interim/vietjobs_it_candidates.csv`, `data/processed/it_salary_clean.csv` (449 tin), `reports/gd2_cleaning_log.txt`. Các luật làm sạch được ghi trong `reports/cleaning_rules.md`.

### Bước 4 – Khám phá dữ liệu (EDA, GĐ2)

```bash
jupyter notebook notebooks/02_eda.ipynb
```

Chạy toàn bộ các ô. Đầu ra: `reports/gd2_eda_output.txt` và 7 hình `reports/figures/gd2_*.png`.

### Bước 5 – Huấn luyện baseline và Linear Regression (GĐ3)

```bash
python src/train.py
```

Tạo `data/processed/split.csv` (chỉ tạo lần đầu, chia theo nhóm tin đăng lại để chống rò rỉ), huấn luyện DummyRegressor và Linear Regression bằng GroupKFold 5 fold. Đầu ra: `reports/model_results.csv`, `reports/gd3_train_output.txt`, `reports/gd3_lr_coefficients.csv`.

### Bước 6 – Tinh chỉnh và chọn mô hình (GĐ4)

```bash
python src/train.py --stage gd4
```

Grid search cho Decision Tree và Random Forest, chọn mô hình theo quy tắc 1 độ lệch chuẩn. Đầu ra: `reports/gd4_grid_results.csv`, `reports/gd4_selected_model.json`, `reports/gd4_train_output.txt`.

### Bước 7 – Đánh giá cuối trên tập test (GĐ5)

```bash
python src/evaluate.py
```

Đánh giá **một lần duy nhất** trên tập test, tính khoảng tin cậy 95% bằng bootstrap theo nhóm, MAE theo nhóm, permutation importance. Đầu ra:

- `reports/gd5_final_output.txt`, `gd5_group_mae.csv`, `gd5_permutation_importance.csv`, `gd5_test_predictions.csv`, 2 hình `gd5_*.png`
- `models/salary_model.joblib`, `models/model_card.json`

---

## Sử dụng mô hình

### Ứng dụng web

```bash
streamlit run app/streamlit_app.py
```

Ứng dụng gồm 3 trang:

1. **Dự đoán** – nhập nhóm nghề, thành phố, số năm kinh nghiệm, cấp bậc, kỹ năng để nhận mức lương dự đoán kèm khoảng sai số.
2. **Dữ liệu & Giới hạn** – mô tả dữ liệu, luật làm sạch và phạm vi áp dụng của mô hình.
3. **Phân tích mô hình** – so sánh kết quả CV và test, MAE theo nhóm, mức độ quan trọng của đặc trưng.

### Dòng lệnh

```bash
# Xem các giá trị hợp lệ (nhóm nghề, thành phố, cấp bậc, kỹ năng)
python src/predict.py --list

# Dự đoán
python src/predict.py --job "<nhóm nghề>" --city "<thành phố>" --exp 2 --skills "python,sql"
```

Giá trị đầu vào phải nằm trong danh sách của `--list`; kinh nghiệm hợp lệ từ 0 đến 6 năm. Đầu vào sai sẽ bị từ chối với lỗi `InvalidInput`. Kết quả CLI khớp với ứng dụng web.

---

## Kiểm thử

```bash
python -m pytest tests -q
```

Kết quả mong đợi: **60 passed**. Bộ test bao gồm:

- `test_data_cleaning.py` – parse lương, kinh nghiệm, thành phố, chuẩn hóa kỹ năng, 38 chức danh mẫu của bộ lọc IT
- `test_features.py` – gộp nhóm tin đăng lại, không nhóm nào nằm ở cả train và test, từ vựng kỹ năng chỉ học từ dữ liệu fit
- `test_predict_invalid_input.py` – 15 kiểu đầu vào sai bị từ chối; đầu vào hợp lệ trả về đơn vị và câu miễn trừ

`tests/conftest.py` tạo một mô hình nhỏ trên dữ liệu giả, nên có thể chạy test cho phần dự đoán và ứng dụng khi chưa huấn luyện mô hình thật.

---

## Nguyên tắc tái lập và chống rò rỉ

- `random_state=42` ở mọi bước.
- `split.csv` chỉ tạo một lần, **không tạo lại**; các tin đăng lại (kể cả bắc cầu) được gộp cùng nhóm và luôn nằm cùng phía train hoặc test.
- Tập test chỉ được dùng **một lần**, sau khi đã chọn xong mô hình.
- Từ vựng kỹ năng và tiền xử lý chỉ học từ dữ liệu train.
- Mọi con số trong báo cáo đều lấy từ file trong `reports/`.

---

## Quy ước làm việc nhóm

- Mỗi thành viên làm trên nhánh riêng (ví dụ `tv1-data`), tạo pull request vào `main`, cần một người khác review.
- Chạy `python -m pytest tests -q` trước mỗi lần merge; không merge khi có test lỗi.
- Muốn đổi tên cột, khóa JSON trong `model_card.json` hoặc chữ ký hàm dùng chung (`experience_group()`, `normalize_skill()`, `FEATURE_COLUMNS`, `build_preprocessor()`, `get_split()`…) phải báo người dùng trước khi merge.
