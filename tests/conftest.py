# tests/conftest.py
"""Cấu hình chung cho pytest: thêm src/ vào sys.path và tạo một mô hình NHỎ trên dữ liệu giả,
để test đầu vào không hợp lệ chạy được mà không cần dữ liệu thật hay mô hình đã huấn luyện."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import FEATURE_COLUMNS, build_preprocessor  # noqa: E402

JOBS = ["Phát triển phần mềm", "Kiểm thử & QA", "Dữ liệu & AI"]
CITIES = ["Hà Nội", "TP.HCM", "Đà Nẵng"]


@pytest.fixture(scope="session")
def tiny_model_files(tmp_path_factory):
    from data_cleaning import experience_group
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    import joblib

    rng = np.random.default_rng(0)
    n = 120
    exp = rng.choice([0, 1, 2, 3, 4, 5], n).astype(float)
    df = pd.DataFrame({
        "job_category": rng.choice(JOBS, n, p=[0.6, 0.3, 0.1]),
        "city": rng.choice(CITIES, n, p=[0.6, 0.35, 0.05]),
        "seniority_coarse": rng.choice(["unknown", "entry", "senior"], n),
        "experience_years": exp,
        "experience_group": [experience_group(e) for e in exp],
        "skills": rng.choice(["java|sql", "python", "", "react|javascript"], n),
    })
    y = 8 + 4 * exp + rng.normal(0, 2, n)
    model = Pipeline([("prep", build_preprocessor("numeric", skill_top_k=5, skill_min_count=1, min_frequency=10)),
                      ("model", LinearRegression())]).fit(df[FEATURE_COLUMNS], y)
    d = tmp_path_factory.mktemp("model")
    model_path, card_path = d / "salary_model.joblib", d / "model_card.json"
    joblib.dump(model, model_path)
    card = {
        "model": "tiny_test_model", "params": {},
        "allowed": {"job_category": JOBS, "city": CITIES,
                    "seniority": ["intern", "fresher", "junior", "middle", "senior", "lead", "manager", "unknown"]},
        "train_counts": {c: df[c].value_counts().to_dict() for c in ["job_category", "city", "seniority_coarse"]},
        "experience_years_range": [0.0, 5.0],
        "skills_vocabulary": list(model.named_steps["prep"].named_transformers_["skills"].vocabulary_),
        "label_range_train": [float(y.min()), float(y.max())],
        "residual_interval_oof": {"p10": -3.0, "p90": 3.0, "test_coverage": 0.8},
        "disclaimer": "Ước lượng từ tin tuyển dụng, không phải cam kết lương.",
    }
    card_path.write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
    return model_path, card_path


@pytest.fixture()
def predictor(tiny_model_files):
    from predict import SalaryPredictor
    return SalaryPredictor(*tiny_model_files)
