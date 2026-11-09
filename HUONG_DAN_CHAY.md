# Code TV3 (Phan Thành Đạt): train.py và evaluate.py

## Các file
| File | Vai trò |
|---|---|
| `src/train.py` | GĐ3 (Dummy + Linear Regression) và GĐ4 (grid search LR/DT/RF có chỉnh số kỹ năng + quy tắc 1 độ lệch chuẩn) |
| `src/evaluate.py` | Hàm dùng chung (metric, GroupKFold, bootstrap theo nhóm, MAE theo nhóm) và GĐ5 `final_evaluation()` |
| `src/features.py` | Của TV2 (đã merge từ `tv2-datacleaning`). TV3 dùng `load_clean`, `make_groups`, `get_split`, `build_preprocessor`, `FEATURE_COLUMNS`, `SENIORITY_COARSE` |
| `tests/make_fake_data.py` | Tạo dữ liệu GIẢ đúng schema `it_salary_clean.csv` của TV2 để chạy thử |
| `tests/test_train_evaluate.py` | Test cho phần TV3 |

## Chạy với dữ liệu thật
```bash
pip install -r requirements.txt     # cần thêm scikit-learn, joblib, pytest (xem dưới)
python src/download_data.py         # tải data/raw/ (không commit)
python src/data_cleaning.py         # GĐ2 -> data/processed/it_salary_clean.csv
python src/train.py                 # bước 5 – GĐ3, lần đầu tạo data/processed/split.csv
python src/train.py --stage gd4     # bước 6 – GĐ4 (~1 phút)
python src/evaluate.py              # bước 7 – GĐ5, CHỈ chạy 1 lần (dùng tập test)
python -m pytest -q
```
Trên Windows nếu lỗi `UnicodeEncodeError` khi in tiếng Việt: đặt `PYTHONIOENCODING=utf-8`.

## Chạy thử với dữ liệu giả
Làm trong một bản sao repo (hoặc xóa kết quả sau đó), vì script ghi vào `data/processed/`, `reports/`, `models/`:
```bash
python tests/make_fake_data.py
python src/train.py && python src/train.py --stage gd4 && python src/evaluate.py --force
```
Trước khi chạy dữ liệu thật: xóa `data/processed/`, `reports/gd3*`, `gd4*`, `gd5*`, `model_results.csv`, `models/`.

## Hợp đồng với TV4 (`src/predict.py`, `app/streamlit_app.py`)
- `models/salary_model.joblib`: Pipeline sklearn chứa `SkillsMultiHot` của `features.py`, nên khi `joblib.load` phải có `src/` trong `sys.path`.
- `models/model_card.json`: `model` (tên), `params`, `n_train`, `created_utc`, `allowed` {`job_category`, `city`, `seniority` (giá trị thô)}, `train_counts` {`job_category`, `city`, `seniority_coarse`}, `experience_years_range`, `skills_vocabulary`, `label_range_train`, `residual_interval_oof` {`p10`, `p90`, `test_coverage`}, `cv_metrics`, `test_metrics` {`mae_mean`, `n_test`, …}, `data_sha256`, `disclaimer`.
- `reports/model_results.csv`: các stage `gd3`, `gd4`, `cv_train`, `test`; baseline tên `dummy_median`; dòng `test` có `mae_ci_low`, `mae_ci_high`.
- `reports/gd5_group_mae.csv`: `column, group, n, mae, mae_dummy, median_label, median_pred, kết luận?`.
- `reports/gd5_permutation_importance.csv`: `feature, mae_increase, std` (chỉ các cột mô hình dùng).
- Đổi tên khóa/cột nào ở trên phải báo TV4 trước.
- `requirements.txt` hiện ở nhánh TV2 chưa có `scikit-learn`, `joblib`, `pytest`; bản của TV4 đã có, sẽ có khi merge `tv4-app`.
