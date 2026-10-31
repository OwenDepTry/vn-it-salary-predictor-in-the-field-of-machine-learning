# Code TV3 (Phan Thành Đạt): train.py và evaluate.py

## Các file
| File | Vai trò |
|---|---|
| `src/train.py` | GĐ3 (Dummy + Linear Regression) và GĐ4 (grid search DT/RF + quy tắc 1-SE) |
| `src/evaluate.py` | Hàm dùng chung (metric, GroupKFold, bootstrap theo nhóm, MAE theo nhóm) và GĐ5 `final_evaluation()` |
| `src/features.py` | **BẢN TẠM** để chạy thử. Khi TV2 có file thật thì dùng file của TV2, **đừng commit đè** |
| `tests/make_fake_data.py` | Tạo dữ liệu giả đúng schema `it_salary_clean.csv` + `split.csv` |
| `tests/test_train_evaluate.py` | 6 test cho phần TV3 |

## Chạy thử với dữ liệu giả
```bash
git checkout main && git pull && git checkout -b tv3-model
# chép thư mục src/ và tests/ ở đây vào repo
pip install pandas numpy matplotlib scikit-learn joblib pytest

python tests/make_fake_data.py      # tạo data/processed/*.csv (giả)
python src/train.py                 # bước 5 – GĐ3
python src/train.py --stage gd4     # bước 6 – GĐ4 (~1 phút)
python src/evaluate.py              # bước 7 – GĐ5, chỉ chạy được 1 lần
python -m pytest tests -q
```
Muốn chạy lại bước 7 với dữ liệu giả: `python src/evaluate.py --force`.
Khi chuyển sang dữ liệu thật, xóa `reports/gd3*`, `gd4*`, `gd5*`, `model_results.csv`, `models/` và `data/processed/` giả trước.

## Khi TV2 bàn giao dữ liệu thật
1. Thay `src/features.py` bằng file của TV2, lấy `it_salary_clean.csv` và `split.csv` thật.
2. Code của mình chỉ cần từ TV2: `FEATURE_COLUMNS`, `TARGET`, `RANDOM_STATE`, `load_clean()`, `get_split(df)`, `build_preprocessor()`; tập sạch có cột `row_id`; split.csv có `row_id, group_id, split`. Nếu TV2 đặt tên khác, sửa trong hàm `load_train_test()` của `evaluate.py`.
3. Chạy lại bước 5, 6, 7. Kết quả ghi theo file trong reports/; không chỉnh mô hình cho khớp số trong README.

## Cần báo cho thành viên khác
- **TV4**: `salary_model.joblib` là Pipeline sklearn có lớp `SkillsMultiHot` của `features.py`, nên khi `joblib.load` phải có `src/` trong `sys.path`. File `model_card.json` có đủ các khóa đã thống nhất, cộng thêm `feature_columns`, `unit`, `baseline_test_metrics`, `created_at`. `residual_interval_oof` là `{"q10": ..., "q90": ...}`. File `gd5_group_mae.csv` có các cột `group_by, value, n, mae, mean_error`.
- **TV4** thêm vào `requirements.txt`: `scikit-learn`, `joblib`, `pytest`.
