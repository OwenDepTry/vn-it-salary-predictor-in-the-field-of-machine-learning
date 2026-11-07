"""Test cho phần TV3 (train.py, evaluate.py). Chạy: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import (cross_validate_grouped, group_bootstrap, mae_by_group,  # noqa: E402
                      make_pipeline, regression_metrics, summarize_folds)
from features import FEATURE_COLUMNS  # noqa: E402
from train import select_one_sd  # noqa: E402


def test_regression_metrics_perfect():
    m = regression_metrics([10, 20, 30], [10, 20, 30])
    assert m["mae"] == 0 and m["rmse"] == 0 and m["r2"] == 1


def test_group_kfold_never_splits_a_group():
    from sklearn.model_selection import GroupKFold
    groups = np.repeat(np.arange(40), 3)
    for tr, va in GroupKFold(5).split(np.zeros(120), groups=groups):
        assert not set(groups[tr]) & set(groups[va])


def test_cross_validate_grouped_returns_oof_for_every_row():
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame({
        "job_category": rng.choice(["Phát triển phần mềm", "Dữ liệu & AI"], n),
        "city": rng.choice(["Hà Nội", "TP.HCM"], n),
        "seniority_coarse": rng.choice(["unknown", "entry"], n),
        "experience_years": rng.integers(0, 7, n).astype(float),
        "experience_group": rng.choice(["1 năm", "2 năm"], n),
        "skills": rng.choice(["python|sql", "java", "react|nodejs"], n),
    })[FEATURE_COLUMNS]
    y = 10 + 3 * X.experience_years + rng.normal(0, 1, n)
    folds, oof = cross_validate_grouped(make_pipeline("linear_regression"), X, y,
                                        np.arange(n) // 2)
    assert len(folds) == 5 and not np.isnan(oof).any()
    assert summarize_folds(folds)["r2_mean"] > 0.8


def test_group_bootstrap_interval_contains_mae():
    rng = np.random.default_rng(1)
    y = rng.normal(20, 5, 90)
    p = y + rng.normal(0, 3, 90)
    lo, hi = group_bootstrap(y, p, np.arange(90) // 3, n_boot=500)
    assert lo < np.mean(np.abs(y - p)) < hi


def test_mae_by_group_columns():
    df = pd.DataFrame({"y_true": [10, 20, 30], "y_pred": [12, 18, 30],
                       "city": ["A", "A", "B"]})
    out = mae_by_group(df, "city")
    assert list(out.columns) == ["column", "value", "n", "mae", "mean_error"]
    assert out.set_index("value").loc["A", "mae"] == pytest.approx(2)


def test_one_sd_rule_prefers_simpler_model():
    rows = [
        {"model": "linear_regression", "params": {"skill_top_k": 15}, "mae_mean": 5.54,
         "mae_std": 1.00, "mae_se": 0.45},
        {"model": "random_forest", "params": {"max_depth": 10, "min_samples_leaf": 5,
                                              "n_estimators": 300}, "mae_mean": 5.21,
         "mae_std": 0.92, "mae_se": 0.41},
    ]
    chosen, best, threshold, _ = select_one_sd(rows)
    assert best["model"] == "random_forest"
    assert threshold == pytest.approx(5.21 + 0.92)      # ngưỡng dùng độ lệch chuẩn, không dùng SE
    assert chosen["model"] == "linear_regression"


def test_one_sd_rule_keeps_best_when_others_too_far():
    rows = [
        {"model": "linear_regression", "params": {}, "mae_mean": 7.0, "mae_std": 0.5, "mae_se": 0.2},
        {"model": "random_forest", "params": {"max_depth": 10, "min_samples_leaf": 5,
                                              "n_estimators": 300}, "mae_mean": 5.0,
         "mae_std": 0.5, "mae_se": 0.2},
    ]
    chosen, _, _, _ = select_one_sd(rows)
    assert chosen["model"] == "random_forest"


def test_make_pipeline_passes_skill_top_k_to_preprocessor():
    from evaluate import make_pipeline
    pipe = make_pipeline("linear_regression", {"skill_top_k": 15})
    skills = {name: tr for name, tr, _ in pipe.named_steps["prep"].transformers}["skills"]
    assert skills.top_k == 15
