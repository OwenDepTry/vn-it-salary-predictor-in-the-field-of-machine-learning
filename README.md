# Dự đoán mức lương đăng tuyển cho các vị trí IT tại Việt Nam

Đồ án môn Máy học: xây dựng mô hình hồi quy dự đoán mức lương (triệu VNĐ/tháng) của tin tuyển dụng IT dựa trên nhóm nghề, thành phố, kinh nghiệm, cấp bậc và kỹ năng, kèm ứng dụng web Streamlit và giao diện dòng lệnh.

- **Bộ dữ liệu chính:** [VietJobs](https://huggingface.co/datasets/dinhieufam/VietJobs) (48.092 bản ghi, 18 cột)
- **Bộ dữ liệu phụ (chỉ để khảo sát):** IT-Job-Posting (3.101 bản ghi, không có cột lương)
- **Tập sạch sau lọc IT và tạo nhãn:** 456 tin; train 370 tin, test 86 tin
- **Mô hình:** Linear Regression trên nhóm nghề, thành phố, số năm kinh nghiệm và cấp bậc (không dùng kỹ năng, xem mục *Kết quả*)
- **Sai số kỳ vọng (nested cross-validation):** MAE ≈ **5,45 triệu**, R² ≈ 0,44; baseline trung vị 7,82 triệu
- **Trên tập test:** MAE ≈ 4,34 triệu, R² ≈ 0,58 (tập test đã được chấm 5 lần, nên con số chính lấy theo nested CV; xem mục *Nguyên tắc tái lập*)

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
│   ├── train.py                  # Baseline, LR (GĐ3); Ridge, DT, RF + grid search (GĐ4)
│   ├── evaluate.py               # Metric, CV theo nhóm, bootstrap, đánh giá cuối, nhật ký chấm test
│   ├── intervals.py              # Khoảng dự đoán ~80% (cộng hoặc theo tỉ lệ lương) từ phần dư OOF
│   └── predict.py                # SalaryPredictor + CLI
├── tests/                        # 112 test (pytest)
├── .github/workflows/tests.yml   # CI: pytest + chạy thử pipeline trên dữ liệu giả (Ubuntu, Windows)
├── requirements.txt
└── README.md
```

---

## Cài đặt

Yêu cầu Python 3 (phiên bản thư viện đã được khóa trong `requirements.txt`).

```bash
git clone https://github.com/OwenDepTry/vn-it-salary-predictor-in-the-field-of-machine-learning.git
cd vn-it-salary-predictor-in-the-field-of-machine-learning

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

Lọc vị trí IT, parse lương ("0.0 triệu" được coi là thỏa thuận), kinh nghiệm, thành phố, cấp bậc, kỹ năng. Đầu ra: `data/interim/vietjobs_it_candidates.csv`, `data/processed/it_salary_clean.csv` (456 tin), `reports/gd2_cleaning_log.txt`. Các luật làm sạch được ghi trong `reports/cleaning_rules.md`.

### Bước 4 – Khám phá dữ liệu (EDA, GĐ2)

```bash
jupyter notebook notebooks/02_eda.ipynb
```

Chạy toàn bộ các ô. Đầu ra: `reports/gd2_eda_output.txt` và 7 hình `reports/figures/gd2_*.png`.

### Bước 5 – Huấn luyện baseline và Linear Regression (GĐ3)

```bash
python src/train.py
```

Dùng `data/processed/split.csv` đã lưu trong repo (chia theo nhóm tin đăng lại để chống rò rỉ; chỉ tạo mới khi chưa có file), huấn luyện DummyRegressor và Linear Regression bằng GroupKFold 5 fold. Đầu ra: `reports/model_results.csv`, `reports/gd3_train_output.txt`, `reports/gd3_lr_coefficients.csv`.

### Bước 6 – Tinh chỉnh và chọn mô hình (GĐ4)

```bash
python src/train.py --stage gd4
```

Grid search cho 4 mô hình (236 cấu hình, khoảng 2 phút):

- Mọi mô hình thử số kỹ năng `skill_top_k` = 0, 15, 30, 50 (0 = bỏ hẳn đặc trưng kỹ năng, để kiểm tra kỹ năng có thật sự giúp không).
- Linear Regression và Ridge (alpha = 0,1–300, có chuẩn hóa) thử thêm kinh nghiệm dạng số năm hoặc dạng nhóm (`experience`), và học trên lương gốc hoặc trên log(lương) (`log_target`).
- Decision Tree và Random Forest chỉnh độ sâu, số mẫu tối thiểu ở lá…

Mô hình được chọn theo **quy tắc 1 sai số chuẩn (1-SE)**: trong các cấu hình có MAE CV ≤ MAE nhỏ nhất + 1 SE (SE = độ lệch chuẩn qua 5 fold / √5), chọn cấu hình đơn giản nhất: Linear Regression < Ridge < Decision Tree < Random Forest, rồi ít cột đặc trưng hơn, rồi nhãn gốc trước log. Output vẫn in kết quả của quy tắc 1-SD cũ để đối chiếu.

Bảng `gd4_ablation.csv` so từng lựa chọn (có/không kỹ năng, kinh nghiệm số/nhóm, log/không log) theo **cặp fold**: cùng 5 fold nên lấy chênh lệch MAE từng fold, báo "rõ" khi chênh quá 2 SE.

Đầu ra: `reports/gd4_grid_results.csv`, `reports/gd4_ablation.csv`, `reports/gd4_selected_model.json`, `reports/gd4_train_output.txt`.

### Bước 6b (tùy chọn) – Nested cross-validation

```bash
python src/train.py --stage nested
```

Ước lượng sai số của **cả quy trình** chọn mô hình (grid + 1-SE) bằng cross-validation lồng nhau chỉ trên tập train: vòng ngoài GroupKFold 5, vòng trong chạy lại toàn bộ bước 6 trên phần train của vòng ngoài. Không dùng tập test, nên đây là con số sai số "chưa thấy" sạch nhất của dự án (tập test đã bị xem nhiều lần). Mất khoảng 10 phút. Đầu ra: `reports/gd4_nested_cv.csv`, `reports/gd4_nested_output.txt`.

### Bước 7 – Đánh giá cuối trên tập test (GĐ5)

```bash
python src/evaluate.py
```

Đánh giá trên tập test sau khi đã chọn xong mô hình (không dùng test để chọn mô hình), tính khoảng tin cậy 95% bằng bootstrap theo nhóm, MAE theo nhóm, permutation importance. Mỗi lần chấm test được ghi thêm một dòng vào `reports/test_set_log.csv`. Lệnh dừng nếu đã có kết quả GĐ5; xem mục *Nguyên tắc tái lập*.

Khoảng dự đoán ~80% học từ phần dư out-of-fold trên train, có hai kiểu: **cộng** (mọi đầu vào cùng độ rộng) hoặc **theo tỉ lệ** (tính trên log lương, nên lương cao thì khoảng rộng hơn). Kiểu nào cho độ phủ đều hơn ở ba mức lương dự đoán thấp/vừa/cao (vẫn trên OOF của train) thì được chọn. Đầu ra:

- `reports/gd5_final_output.txt`, `gd5_group_mae.csv`, `gd5_permutation_importance.csv`, `gd5_test_predictions.csv`, 2 hình `gd5_*.png`
- `models/salary_model.joblib`, `models/model_card.json`

> **Clone repo về chạy lại?** Repo đã có sẵn kết quả GĐ5 (`reports/gd5_final_output.txt`) nhưng không lưu file mô hình `.joblib`, nên lệnh trên sẽ dừng để bảo vệ tập test. Sau bước 6, chạy:
>
> ```bash
> python src/evaluate.py --rebuild-model
> ```
>
> Lệnh này tạo lại `models/salary_model.joblib` từ cấu hình đã chọn ở GĐ4, **chỉ fit trên tập train**, không chấm lại tập test và không ghi đè kết quả GĐ5. Trước khi fit, lệnh đối chiếu dấu vân tay dữ liệu train và cách chia train/test với `models/model_card.json`; nếu dữ liệu đã đổi thì dừng, vì metric trong model card không còn đúng. Chỉ dùng `python src/evaluate.py --force` khi thật sự muốn đánh giá lại trên tập test.

---

## Kết quả

Mọi con số dưới đây lấy từ `reports/gd4_train_output.txt`, `gd4_ablation.csv`, `gd4_nested_output.txt` và `gd5_final_output.txt` (chạy ngày 18/11/2026).

**Tốt nhất của từng họ mô hình** (GroupKFold 5 fold trên 370 tin train, 236 cấu hình):

| Mô hình | MAE CV (triệu) | SE | R² |
|---|---|---|---|
| Baseline trung vị | 7,82 | 0,15 | −0,05 |
| Linear Regression | 5,28 | 0,48 | 0,44 |
| Ridge | 5,28 | 0,48 | 0,44 |
| Decision Tree | 5,62 | 0,42 | 0,39 |
| Random Forest | 5,32 | 0,34 | 0,45 |
| **Được chọn (1-SE): Linear Regression, không kỹ năng, kinh nghiệm dạng số, lương gốc** | **5,45** | **0,34** | **0,44** |

**Mô hình được chọn.** Ngưỡng 1-SE là 5,28 + 0,48 = 5,77 triệu, có 147/236 cấu hình lọt ngưỡng; mô hình đơn giản nhất trong số đó được chọn. Quy tắc 1-SD cũ cho **cả 236/236** cấu hình lọt ngưỡng, tức là không còn tác dụng chọn lọc.

**Kỹ năng không giúp dự đoán.** Với Linear Regression và Ridge, thêm 15 kỹ năng phổ biến làm MAE CV **tăng rõ** (+0,21 ± 0,04 triệu, so theo cặp fold); 30–50 kỹ năng cũng không tốt hơn. Với Decision Tree và Random Forest, có hay không có kỹ năng không khác nhau rõ. Với 370 tin train, danh sách kỹ năng trên tin tuyển dụng không cho thêm thông tin về lương ngoài nhóm nghề, kinh nghiệm, cấp bậc và thành phố. Đây là kết luận về dữ liệu này, không phải về giá trị của kỹ năng ngoài thị trường.

**Log(lương) và kinh nghiệm dạng nhóm** cho MAE CV thấp nhất (5,28), tốt hơn mô hình được chọn 0,17 ± 0,18 triệu, chưa đủ bằng chứng (dưới 2 SE). Cấu hình này cũng dao động giữa các fold nhiều hơn (SD 1,08 so với 0,76).

**Nested cross-validation** (chạy lại toàn bộ grid + 1-SE trong từng fold ngoài): MAE **5,45 ± 0,76 triệu**, R² 0,44. Cả 5 fold ngoài đều chọn cùng một cấu hình. Vì vòng ngoài dùng đúng 5 fold của GĐ4, con số này trùng với MAE CV của mô hình được chọn: việc chọn mô hình không làm sai số bị ước lượng thấp đi. Chọn theo 1-SE (5,451) gần như bằng chọn cấu hình có MAE CV nhỏ nhất (5,442).

**Trên tập test** (86 tin, lần chấm thứ 5): MAE **4,34 triệu** (khoảng tin cậy 95% theo nhóm: 3,66–5,11), RMSE 5,49, R² 0,58; baseline trung vị 5,94. Ở cả 5 lần chấm, MAE test đều thấp hơn MAE CV, có lẽ vì tập test tập trung ở Hà Nội (66/86 tin).

**Đặc trưng quan trọng** (permutation importance trên test, MAE tăng khi xáo trộn): số năm kinh nghiệm 2,56 triệu; cấp bậc 0,57; nhóm nghề 0,40; thành phố 0,10.

**Khoảng dự đoán ~80%:** kiểu theo tỉ lệ (log) được chọn vì độ phủ OOF đều hơn ở ba mức lương thấp/vừa/cao (73% / 82% / 85%, so với 89% / 79% / 72% của kiểu cộng). Trên test, khoảng này chứa 87% tin.

### Hạn chế

- **Dữ liệu ít và lệch:** 456 tin sau khi lọc (từ 48.092 bản ghi), 77% tin test ở Hà Nội, Đà Nẵng chỉ có vài tin; 71/86 tin test không ghi cấp bậc. Mô hình chủ yếu phản ánh thị trường Hà Nội và TP.HCM, kinh nghiệm 0–6 năm.
- **Dự đoán cao hơn thực tế ở mức kinh nghiệm phổ biến:** trên test, trung vị dự đoán cao hơn trung vị thực tế ở hầu hết các nhóm (nhóm 2 năm: 22,2 so với 20,0; nhóm 3 năm: 26,8 so với 20,8). Ở nhóm 2–3 năm và nhóm Hạ tầng/DevOps, MAE còn cao hơn baseline. Nguyên nhân: hồi quy tuyến tính trên lương gốc dự đoán giá trị trung bình, trong khi lương lệch phải, và kinh nghiệm được mô hình hóa tuyến tính nên độ dốc bị các tin lương cao kéo lên. Log(lương) và kinh nghiệm dạng nhóm là hướng khắc phục, nhưng ở GĐ4 chưa đủ bằng chứng để chọn. Dự án **không đổi mô hình sau khi thấy kết quả này trên test**, để tránh chọn mô hình theo tập test.
- **Tập test đã bị xem 5 lần** (xem bên dưới), nên chỉ dùng để đối chiếu.
- Lương là lương **đăng tuyển** (trung điểm khoảng lương trên tin), không phải lương thực nhận.

---

## Sử dụng mô hình

### Ứng dụng web

```bash
streamlit run app/streamlit_app.py
```

Ứng dụng gồm 3 trang:

1. **Dự đoán** – nhập nhóm nghề, thành phố, số năm kinh nghiệm, cấp bậc để nhận mức lương dự đoán kèm khoảng ~80%. Ô kỹ năng tự ẩn khi mô hình được chọn không dùng kỹ năng (như hiện tại).
2. **Dữ liệu & Giới hạn** – mô tả dữ liệu, luật làm sạch và phạm vi áp dụng của mô hình.
3. **Phân tích mô hình** – so sánh kết quả CV và test, MAE theo nhóm, mức độ quan trọng của đặc trưng.

### Dòng lệnh

```bash
# Xem các giá trị hợp lệ (nhóm nghề, thành phố, cấp bậc, kỹ năng)
python src/predict.py --list

# Dự đoán
python src/predict.py --job "<nhóm nghề>" --city "<thành phố>" --exp 2 --skills "python,sql"
```

Giá trị đầu vào phải nằm trong danh sách của `--list`. Về kinh nghiệm:

- **Ứng dụng web** chỉ cho chọn 0–6 năm (đúng phạm vi kinh nghiệm có trong dữ liệu train).
- **CLI và hàm Python** (`SalaryPredictor.predict`) chấp nhận 0–40 năm; từ trên 6 năm vẫn trả kết quả nhưng kèm cảnh báo mô hình đang ngoại suy. Ngoài 0–40, giá trị không phải số hoặc NaN bị từ chối.

Đầu vào sai sẽ bị từ chối với lỗi `InvalidInput`. Với cùng đầu vào trong phạm vi 0–6 năm, kết quả CLI khớp với ứng dụng web.

---

## Kiểm thử

```bash
python -m pytest tests -q
```

Kết quả mong đợi: **112 passed**. GitHub Actions tự chạy bộ test và chạy thử pipeline GĐ3–GĐ5 trên dữ liệu giả mỗi lần push hoặc mở pull request (Ubuntu và Windows). Bộ test bao gồm:

- `test_data_cleaning.py` (39) – parse lương, kinh nghiệm, thành phố, chuẩn hóa kỹ năng, 38 chức danh mẫu của bộ lọc IT, an ninh mạng và lập trình nhúng
- `test_features.py` (5) – gộp nhóm tin đăng lại, không nhóm nào nằm ở cả train và test, từ vựng kỹ năng chỉ học từ dữ liệu fit
- `test_train_evaluate.py` (27) – metric, GroupKFold, bootstrap theo nhóm, MAE theo nhóm, quy tắc 1-SE (thứ tự LR < Ridge < DT < RF, đếm số cột đặc trưng, so với 1-SD cũ), Ridge co hệ số, `skill_top_k = 0`, kinh nghiệm dạng nhóm, log(lương), bảng ablation theo cặp fold, nested CV, nhật ký chấm test, dấu vân tay dữ liệu khi tạo lại mô hình
- `test_intervals.py` (5) – chọn khoảng theo tỉ lệ khi sai số tăng theo mức lương, khoảng cộng khi sai số không đổi, độ phủ ~80%, tương thích model card cũ
- `test_predict_interval.py` (3) – dự đoán với khoảng theo tỉ lệ, model card cũ, mô hình không dùng kỹ năng
- `test_predict_invalid_input.py` (33) – 15 kiểu đầu vào sai bị từ chối; đầu vào hợp lệ trả về đơn vị và câu miễn trừ; dự đoán ngoài khoảng lương của train không hiển thị con số

`tests/conftest.py` tạo một mô hình nhỏ trên dữ liệu giả, nên có thể chạy test cho phần dự đoán và ứng dụng khi chưa huấn luyện mô hình thật.

---

## Nguyên tắc tái lập và chống rò rỉ

- `random_state=42` ở mọi bước.
- `split.csv` chỉ tạo một lần, **không tạo lại**, và được lưu trong repo để mọi người chạy ra cùng kết quả; các tin đăng lại (kể cả bắc cầu) được gộp cùng nhóm và luôn nằm cùng phía train hoặc test.
- Tập test **không dùng để chọn mô hình**: mô hình được chọn chỉ dựa trên cross-validation trong tập train (GĐ4), sau đó mới chấm trên test.
- Tập test (luôn là cùng 86 tin) đã được chấm **5 lần**, ghi trong `reports/test_set_log.csv` (lần 1–4 lấy theo lịch sử git):

  | Lần | Ngày | Mô hình | MAE test | Bối cảnh |
  |---|---|---|---|---|
  | 1 | 07/11 | Decision Tree | 4,607 | Chọn theo quy tắc 1-SE, grid chỉ có DT/RF, tập sạch 449 tin |
  | 2 | 07/11 | Linear Regression | 4,641 | Đổi sang quy tắc 1-SD, thêm LR và `skill_top_k` vào grid |
  | 3 | 07/11 | Linear Regression | 4,641 | Chạy lại cùng mô hình để thêm bảng MAE theo nhóm |
  | 4 | 10/11 | Linear Regression | 4,713 | Sửa bộ lọc IT (thêm 7 tin an ninh mạng / lập trình nhúng vào train) |
  | 5 | 18/11 | Linear Regression, không kỹ năng | 4,336 | Quy tắc 1-SE + lưới mở rộng (kỹ năng 0, kinh nghiệm dạng nhóm, log lương); mô hình chọn hoàn toàn bằng CV trước khi chấm |

  Quy tắc chọn mô hình đã bị đổi **sau** lần chấm 1. Lần đổi này không làm kết quả test đẹp hơn: Decision Tree chọn ban đầu có MAE test thấp hơn. Dù vậy, tập test **không còn là tập chưa từng thấy**. Vì thế dự án quay lại quy tắc 1-SE chuẩn và thêm nested CV (bước 6b) làm ước lượng sai số không phụ thuộc tập test. Ngày 16/11, GĐ4 được chạy lại để thêm Ridge (sau lần chấm 4); mô hình được chọn không đổi nên GĐ5 không chạy lại. Ngày 18/11, GĐ4 chạy với quy tắc 1-SE và lưới mở rộng chọn mô hình mới (bỏ kỹ năng), nên GĐ5 được chấm lại (lần 5).
- Tập test không dùng để chọn mô hình hay tham số. Mỗi lần chấm lại test (`python src/evaluate.py --force`) được tự động ghi vào `reports/test_set_log.csv` và phải được nêu trong báo cáo.
- Từ vựng kỹ năng và tiền xử lý chỉ học từ dữ liệu train.
- Mọi con số trong báo cáo đều lấy từ file trong `reports/`.

---

## Quy ước làm việc nhóm

- Mỗi thành viên làm trên nhánh riêng (ví dụ `tv1-data`), tạo pull request vào `main`, cần một người khác review.
- Chạy `python -m pytest tests -q` trước mỗi lần merge; không merge khi có test lỗi.
- Muốn đổi tên cột, khóa JSON trong `model_card.json` hoặc chữ ký hàm dùng chung (`experience_group()`, `normalize_skill()`, `FEATURE_COLUMNS`, `build_preprocessor()`, `get_split()`…) phải báo người dùng trước khi merge.
