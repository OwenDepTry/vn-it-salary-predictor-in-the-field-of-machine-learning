# src/predict.py
"""GĐ5 — Dự đoán lương đăng tuyển từ pipeline đã lưu (models/salary_model.joblib + models/model_card.json).

Dùng trong code (app GĐ6):
    from predict import SalaryPredictor
    p = SalaryPredictor()
    p.predict(job_category="Phát triển phần mềm", city="Hà Nội", experience_years=2,
              seniority="unknown", skills=["java", "sql"])

Dùng từ dòng lệnh (thư mục gốc):
    python src/predict.py --list
    python src/predict.py --job "Phát triển phần mềm" --city "Hà Nội" --exp 2 --skills "java, sql"
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))  # cần để joblib tìm được lớp SkillsMultiHot trong features.py
from data_cleaning import experience_group, normalize_skill  # noqa: E402
from features import FEATURE_COLUMNS, SENIORITY_COARSE  # noqa: E402

ROOT = SRC.parent
MODEL_PATH = ROOT / "models" / "salary_model.joblib"
CARD_PATH = ROOT / "models" / "model_card.json"
MIN_GROUP_N = 10  # nhóm có < 10 tin trong train -> cảnh báo độ tin cậy thấp
MAX_EXPERIENCE_INPUT = 40


class InvalidInput(ValueError):
    """Đầu vào không hợp lệ (giá trị ngoài danh sách mô hình hỗ trợ, kiểu sai...)."""


class SalaryPredictor:
    def __init__(self, model_path: Path = MODEL_PATH, card_path: Path = CARD_PATH):
        import joblib
        if not model_path.exists() or not card_path.exists():
            raise FileNotFoundError("Chưa có mô hình. Chạy `python src/evaluate.py --rebuild-model` "
                                    "(hoặc `python src/evaluate.py` nếu chưa đánh giá GĐ5).")
        self.model = joblib.load(model_path)
        self.card = json.loads(card_path.read_text(encoding="utf-8"))
        self.allowed = self.card["allowed"]
        self.vocab = set(self.card["skills_vocabulary"])

    def validate(self, job_category, city, experience_years, seniority, skills):
        """Trả về (bản ghi đã chuẩn hóa, danh sách cảnh báo). Lỗi cứng -> InvalidInput."""
        warnings = []
        if job_category not in self.allowed["job_category"]:
            raise InvalidInput(f"Nhóm nghề không hỗ trợ: {job_category!r}. Hợp lệ: {self.allowed['job_category']}")
        if city not in self.allowed["city"]:
            raise InvalidInput(f"Thành phố không hỗ trợ: {city!r}. Hợp lệ: {self.allowed['city']}")
        if seniority not in self.allowed["seniority"]:
            raise InvalidInput(f"Cấp bậc không hỗ trợ: {seniority!r}. Hợp lệ: {self.allowed['seniority']}")
        if isinstance(experience_years, bool):
            raise InvalidInput("Số năm kinh nghiệm phải là số.")
        try:
            exp = float(experience_years)
        except (TypeError, ValueError):
            raise InvalidInput(f"Số năm kinh nghiệm phải là số, nhận được: {experience_years!r}")
        if exp != exp or exp < 0 or exp > MAX_EXPERIENCE_INPUT:  # exp != exp: NaN
            raise InvalidInput(f"Số năm kinh nghiệm phải trong [0, {MAX_EXPERIENCE_INPUT}].")

        lo, hi = self.card["experience_years_range"]
        if exp > hi:
            warnings.append(f"Kinh nghiệm {exp:g} năm vượt mức cao nhất trong dữ liệu train ({hi:g} năm): "
                            f"mô hình đang ngoại suy, độ tin cậy thấp.")
        counts = self.card["train_counts"]
        if counts["job_category"].get(job_category, 0) < MIN_GROUP_N:
            warnings.append(f"Nhóm nghề '{job_category}' có ít tin trong train ({counts['job_category'].get(job_category, 0)}).")
        if counts["city"].get(city, 0) < MIN_GROUP_N:
            warnings.append(f"Thành phố '{city}' có ít tin trong train ({counts['city'].get(city, 0)}); "
                            f"mô hình gộp vào nhóm hiếm.")
        coarse = SENIORITY_COARSE.get(seniority, "unknown")
        if counts["seniority_coarse"].get(coarse, 0) < MIN_GROUP_N:
            warnings.append(f"Cấp bậc '{seniority}' có ít tin trong train.")

        if skills is None:
            skills = []
        if isinstance(skills, str):
            skills = [s for s in skills.split(",")]
        if not isinstance(skills, (list, tuple, set)) or not all(isinstance(s, str) for s in skills):
            raise InvalidInput("Kỹ năng phải là chuỗi 'a, b' hoặc danh sách chuỗi.")
        norm = []
        for s in skills:
            k = normalize_skill(s)
            if k and k not in norm:
                norm.append(k)
        ignored = [s for s in norm if s not in self.vocab]
        if ignored:
            warnings.append(f"Mô hình không dùng các kỹ năng sau (không nằm trong {len(self.vocab)} kỹ năng "
                            f"phổ biến của dữ liệu train): {', '.join(ignored)}")
        row = {
            "job_category": job_category, "city": city, "seniority_coarse": coarse,
            "experience_years": exp, "experience_group": experience_group(exp),
            "skills": "|".join(s for s in norm),
        }
        return row, warnings

    def predict(self, job_category, city, experience_years, seniority="unknown", skills=None) -> dict:
        row, warnings = self.validate(job_category, city, experience_years, seniority, skills)
        X = pd.DataFrame([row])[FEATURE_COLUMNS]
        pred = float(self.model.predict(X)[0])
        iv = self.card["residual_interval_oof"]
        lab_lo, lab_hi = self.card["label_range_train"]
        # Chỉ trả về con số khi dự đoán nằm trong khoảng lương có trong dữ liệu train.
        # Ngoài khoảng đó (kể cả lương âm) là ngoại suy: báo "không đủ cơ sở" thay vì in số.
        reliable = lab_lo <= pred <= lab_hi
        if not reliable:
            warnings.insert(0, f"Không đủ cơ sở dự đoán: mô hình cho {pred:.1f} triệu, nằm ngoài khoảng lương "
                               f"có trong dữ liệu train ({lab_lo:g}–{lab_hi:g} triệu). Tổ hợp đầu vào này quá "
                               f"khác dữ liệu huấn luyện nên không đưa ra mức lương.")
        low = max(0.0, pred + iv["p10"])
        high = max(low, pred + iv["p90"])          # luôn đảm bảo low ≤ high
        return {
            "prediction": round(pred, 1) if reliable else None,
            "range_80": (round(low, 1), round(high, 1)) if reliable else None,
            "reliable": reliable,
            "unit": "triệu VND/tháng",
            "warnings": warnings,
            "used_skills": [s for s in row["skills"].split("|") if s and s in self.vocab],
            "disclaimer": self.card["disclaimer"],
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dự đoán lương đăng tuyển IT (triệu VND/tháng).")
    ap.add_argument("--list", action="store_true", help="In các giá trị đầu vào hợp lệ")
    ap.add_argument("--job")
    ap.add_argument("--city")
    ap.add_argument("--exp")
    ap.add_argument("--seniority", default="unknown")
    ap.add_argument("--skills", default="")
    a = ap.parse_args(argv)
    p = SalaryPredictor()
    if a.list:
        print(json.dumps(p.allowed, ensure_ascii=False, indent=2))
        print("Kỹ năng mô hình dùng:", ", ".join(p.card["skills_vocabulary"]))
        print(f"Kinh nghiệm trong train: {p.card['experience_years_range']} năm")
        return 0
    if not (a.job and a.city and a.exp is not None):
        ap.error("cần --job, --city, --exp (hoặc --list)")
    try:
        r = p.predict(a.job, a.city, a.exp, a.seniority, a.skills)
    except InvalidInput as e:
        print(f"ĐẦU VÀO KHÔNG HỢP LỆ: {e}")
        return 2
    if r["prediction"] is None:
        print("KHÔNG ĐỦ CƠ SỞ DỰ ĐOÁN.")
    else:
        print(f"Dự đoán: {r['prediction']} {r['unit']} (khoảng ~80%: {r['range_80'][0]}–{r['range_80'][1]})")
    print(f"Kỹ năng được dùng: {r['used_skills']}")
    for w in r["warnings"]:
        print("CẢNH BÁO:", w)
    print(r["disclaimer"])
    return 0 if r["prediction"] is not None else 3


if __name__ == "__main__":
    sys.exit(main())
