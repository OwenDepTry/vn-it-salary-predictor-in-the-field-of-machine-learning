"""GĐ3 và GĐ4 – Huấn luyện và chọn mô hình (TV3: Phan Thành Đạt).

GĐ3 (bước 5 của README):   python src/train.py
    DummyRegressor (baseline) và Linear Regression, đánh giá bằng GroupKFold 5 fold
    trên tập train. Xuất reports/model_results.csv, gd3_train_output.txt,
    gd3_lr_coefficients.csv. Tạo data/processed/split.csv nếu chưa có (chỉ một lần).

GĐ4 (bước 6 của README):   python src/train.py --stage gd4
    Grid search Decision Tree và Random Forest bằng cùng GroupKFold, rồi chọn mô hình
    theo quy tắc 1 độ lệch chuẩn (1-SE). Xuất gd4_grid_results.csv,
    gd4_selected_model.json, gd4_train_output.txt.

Tập test KHÔNG được dùng ở file này.
"""
import argparse
import datetime
import itertools
import json
import math

import numpy as np
import pandas as pd

from evaluate import (N_SPLITS, REPORTS, RESULTS_CSV, SELECTED_JSON, Tee,
                      cross_validate_grouped, load_train_test, make_pipeline,
                      save_results, summarize_folds)
from features import FEATURE_COLUMNS, RANDOM_STATE, TARGET

FAMILY_ORDER = {"linear_regression": 0, "decision_tree": 1, "random_forest": 2}

GRIDS = {
    "decision_tree": {
        "max_depth": [2, 3, 4, 6, 8, None],
        "min_samples_leaf": [5, 10, 20],
    },
    "random_forest": {
        "n_estimators": [300],
        "max_depth": [4, 8, None],
        "min_samples_leaf": [2, 5, 10],
        "max_features": ["sqrt", 0.5],
    },
}


def _train_frame():
    train, _ = load_train_test()  # chỉ dùng phần train
    return train[FEATURE_COLUMNS], train[TARGET].to_numpy(float), train["group_id"], train


def _evaluate(name, params, X, y, groups, stage):
    folds, _ = cross_validate_grouped(make_pipeline(name, params), X, y, groups)
    return {"model": name, "stage": stage, **summarize_folds(folds), "params": params}


def _fmt(r):
    return (f"{r['model']:<18} MAE {r['mae_mean']:.3f} ± {r['mae_std']:.3f} "
            f"(SE {r['mae_se']:.3f})  RMSE {r['rmse_mean']:.3f}  R² {r['r2_mean']:.3f}  "
            f"{json.dumps(r['params'])}")


# ---------------------------------------------------------------- GĐ3
def main():
    log = Tee(REPORTS / "gd3_train_output.txt")
    X, y, groups, train = _train_frame()
    log("GĐ3 – BASELINE VÀ LINEAR REGRESSION (GroupKFold, chỉ tập train)")
    log("Ngày chạy:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    log(f"Train: {len(train)} tin, {groups.nunique()} nhóm, {N_SPLITS} fold, "
        f"random_state={RANDOM_STATE}")
    log(f"Nhãn {TARGET}: trung vị {np.median(y):.2f}, "
        f"khoảng [{y.min():.1f}, {y.max():.1f}] triệu/tháng\n")

    rows = [_evaluate("dummy", {}, X, y, groups, "gd3"),
            _evaluate("linear_regression", {}, X, y, groups, "gd3")]
    for r in rows:
        log(_fmt(r))
    gain = rows[0]["mae_mean"] - rows[1]["mae_mean"]
    log(f"\nLR giảm MAE {gain:.3f} triệu so với baseline "
        f"({gain / rows[0]['mae_mean']:.1%}).")
    save_results(rows)

    lr = make_pipeline("linear_regression").fit(X, y)
    names = lr.named_steps["prep"].get_feature_names_out()
    coef = (pd.DataFrame({"feature": names, "coefficient": lr.named_steps["model"].coef_})
            .assign(abs_coef=lambda d: d.coefficient.abs())
            .sort_values("abs_coef", ascending=False).drop(columns="abs_coef"))
    coef.loc[len(coef)] = ["intercept", lr.named_steps["model"].intercept_]
    coef.to_csv(REPORTS / "gd3_lr_coefficients.csv", index=False)
    log("\n10 hệ số LR có trị tuyệt đối lớn nhất (triệu/tháng, fit trên toàn bộ train):")
    log(coef.head(10).round(3).to_string(index=False))
    log.save()


# ---------------------------------------------------------------- GĐ4
def _complexity_key(row):
    """Khóa sắp xếp từ ĐƠN GIẢN đến PHỨC TẠP, dùng cho quy tắc 1-SE."""
    p = row["params"]
    depth = p.get("max_depth")
    depth = math.inf if depth is None else depth
    return (FAMILY_ORDER[row["model"]], depth, -p.get("min_samples_leaf", 0),
            p.get("n_estimators", 0), str(p.get("max_features", "")))


def select_one_se(rows):
    """Quy tắc 1-SE: ngưỡng = MAE nhỏ nhất + SE của mô hình đó; trong các mô hình có
    MAE ≤ ngưỡng, chọn mô hình đơn giản nhất."""
    best = min(rows, key=lambda r: r["mae_mean"])
    threshold = best["mae_mean"] + best["mae_se"]
    eligible = [r for r in rows if r["mae_mean"] <= threshold]
    return min(eligible, key=_complexity_key), best, threshold, eligible


def gd4():
    log = Tee(REPORTS / "gd4_train_output.txt")
    X, y, groups, train = _train_frame()
    log("GĐ4 – GRID SEARCH VÀ CHỌN MÔ HÌNH (GroupKFold, chỉ tập train)")
    log("Ngày chạy:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    log(f"Train: {len(train)} tin, {groups.nunique()} nhóm, {N_SPLITS} fold\n")

    rows = [_evaluate("linear_regression", {}, X, y, groups, "gd4")]
    for name, grid in GRIDS.items():
        keys = list(grid)
        combos = list(itertools.product(*grid.values()))
        log(f"{name}: {len(combos)} tổ hợp tham số")
        for values in combos:
            rows.append(_evaluate(name, dict(zip(keys, values)), X, y, groups, "gd4"))

    grid_df = pd.DataFrame(rows).sort_values("mae_mean")
    grid_df.assign(params=grid_df.params.apply(json.dumps)).to_csv(
        REPORTS / "gd4_grid_results.csv", index=False)

    best_by_family = [min((r for r in rows if r["model"] == f), key=lambda r: r["mae_mean"])
                      for f in FAMILY_ORDER]
    log("\nTốt nhất của từng họ mô hình:")
    for r in best_by_family:
        log(_fmt(r))
    save_results(best_by_family)

    chosen, best, threshold, eligible = select_one_se(rows)
    log(f"\nQuy tắc 1-SE: MAE nhỏ nhất {best['mae_mean']:.3f} ({best['model']}) "
        f"+ SE {best['mae_se']:.3f} = ngưỡng {threshold:.3f}")
    log(f"Số cấu hình có MAE ≤ ngưỡng: {len(eligible)}")
    log(f"CHỌN (đơn giản nhất trong ngưỡng): {_fmt(chosen)}")

    selected = {
        "model": chosen["model"], "params": chosen["params"],
        "cv_mae_mean": chosen["mae_mean"], "cv_mae_std": chosen["mae_std"],
        "cv_mae_se": chosen["mae_se"], "cv_r2_mean": chosen["r2_mean"],
        "rule": "1-SE: mô hình đơn giản nhất có MAE CV ≤ MAE nhỏ nhất + 1 SE",
        "best_model": best["model"], "best_params": best["params"],
        "best_mae_mean": best["mae_mean"], "threshold": threshold,
        "n_splits": N_SPLITS, "random_state": RANDOM_STATE,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    SELECTED_JSON.write_text(json.dumps(selected, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    log(f"\nĐã ghi {SELECTED_JSON.relative_to(REPORTS.parent)} và {RESULTS_CSV.name}")
    log.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GĐ3/GĐ4: huấn luyện và chọn mô hình")
    ap.add_argument("--stage", choices=["gd3", "gd4"], default="gd3")
    args = ap.parse_args()
    gd4() if args.stage == "gd4" else main()
