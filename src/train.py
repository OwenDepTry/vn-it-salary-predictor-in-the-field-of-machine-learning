"""GĐ3 và GĐ4 – Huấn luyện và chọn mô hình (TV3: Phan Thành Đạt).

GĐ3 (bước 5 của README):   python src/train.py
    DummyRegressor (baseline) và Linear Regression, đánh giá bằng GroupKFold 5 fold
    trên tập train. Xuất reports/model_results.csv, gd3_train_output.txt,
    gd3_lr_coefficients.csv. Tạo data/processed/split.csv nếu chưa có (chỉ một lần).

GĐ4 (bước 6 của README):   python src/train.py --stage gd4
    Grid search Linear Regression, Ridge, Decision Tree và Random Forest bằng cùng GroupKFold, rồi chọn mô hình
    theo quy tắc 1 sai số chuẩn (1-SE, Breiman và cộng sự; Hastie, Tibshirani & Friedman, ESL §7.10).
    Lưới gồm cả số kỹ năng (0 = không dùng kỹ năng), kinh nghiệm dạng số/dạng nhóm và học trên log(lương)
    cho các mô hình tuyến tính. Xuất gd4_grid_results.csv, gd4_ablation.csv,
    gd4_selected_model.json, gd4_train_output.txt.

Nested CV (tùy chọn):      python src/train.py --stage nested
    Ước lượng sai số của CẢ QUY TRÌNH (grid search + quy tắc 1-SE) bằng cross-validation lồng nhau
    trên tập train: vòng ngoài GroupKFold 5, vòng trong chạy lại toàn bộ GĐ4 trên phần train của vòng ngoài.
    Không dùng tập test. Xuất gd4_nested_cv.csv, gd4_nested_output.txt.

Tập test KHÔNG được dùng ở file này.
"""
import argparse
import datetime
import itertools
import json
import math

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

from evaluate import (N_SPLITS, REPORTS, RESULTS_CSV, SELECTED_JSON, Tee,
                      cross_validate_grouped, load_train_test, make_pipeline,
                      regression_metrics, save_results, summarize_folds)
from features import FEATURE_COLUMNS, RANDOM_STATE, SKILL_TOP_K, TARGET

# Thứ tự từ đơn giản đến phức tạp cho quy tắc 1-SE. Ridge xếp sau Linear Regression vì có thêm
# siêu tham số alpha cần chỉnh; nếu LR đã nằm trong ngưỡng thì giữ LR.
FAMILY_ORDER = {"linear_regression": 0, "ridge": 1, "decision_tree": 2, "random_forest": 3}
# Số cột đặc trưng (xấp xỉ) của phần kinh nghiệm: số năm = 1 cột; dạng nhóm = one-hot ~6 bậc.
EXPERIENCE_COLUMNS = {"numeric": 1, "group": 6}

# 0 = bỏ hẳn đặc trưng kỹ năng (ablation: kỹ năng có thật sự giúp không?)
SKILL_TOP_K_GRID = [0, 15, 30, 50]

# Chỉ mô hình tuyến tính mới cần thử: kinh nghiệm dạng nhóm (cho phép quan hệ phi tuyến theo bậc)
# và log(lương). Cây quyết định tự cắt ngưỡng trên số năm, và đổi thang nhãn đơn điệu ít tác dụng với cây.
LINEAR_OPTIONS = {
    "experience": ["numeric", "group"],
    "log_target": [False, True],
}

# Lưới nhỏ có chủ đích (train chỉ ~370 tin). Mọi họ mô hình đều được chỉnh số kỹ năng
# (skill_top_k) để so sánh công bằng.
GRIDS = {
    "linear_regression": {
        "skill_top_k": SKILL_TOP_K_GRID,
        **LINEAR_OPTIONS,
    },
    "ridge": {
        "skill_top_k": SKILL_TOP_K_GRID,
        **LINEAR_OPTIONS,
        "alpha": [0.1, 1, 3, 10, 30, 100, 300],
    },
    "decision_tree": {
        "skill_top_k": SKILL_TOP_K_GRID,
        "max_depth": [3, 4, 5, 6, 8],
        "min_samples_leaf": [5, 10, 20],
    },
    "random_forest": {
        "skill_top_k": SKILL_TOP_K_GRID,
        "n_estimators": [300],
        "max_depth": [None, 10],
        "min_samples_leaf": [2, 5, 10],
        "max_features": ["sqrt", 0.5],
    },
}


def _train_frame():
    train, _ = load_train_test()  # chỉ dùng phần train
    return train[FEATURE_COLUMNS], train[TARGET].to_numpy(float), train["group_id"], train


def _evaluate(name, params, X, y, groups, stage):
    folds, _ = cross_validate_grouped(make_pipeline(name, params), X, y, groups)
    # fold_mae: MAE từng fold (GroupKFold không xáo trộn -> mọi cấu hình dùng CÙNG các fold),
    # dùng để so sánh hai cấu hình theo cặp fold trong bảng ablation.
    return {"model": name, "stage": stage, **summarize_folds(folds), "params": params,
            "fold_mae": folds["mae"].round(6).tolist()}


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

    rows = [_evaluate("dummy_median", {}, X, y, groups, "gd3"),
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
    """Khóa sắp xếp từ ĐƠN GIẢN đến PHỨC TẠP, dùng cho quy tắc 1-SE.
    Ridge: alpha lớn hơn = phạt mạnh hơn = đơn giản hơn.
    Cùng mức phức tạp: ít cột đặc trưng hơn trước (cột kinh nghiệm + số kỹ năng), rồi nhãn gốc
    trước log(lương), rồi MAE thấp hơn."""
    p = row["params"]
    depth = p.get("max_depth")
    depth = math.inf if depth is None else depth
    n_extra_columns = EXPERIENCE_COLUMNS[p.get("experience", "numeric")] + p.get("skill_top_k", SKILL_TOP_K)
    return (FAMILY_ORDER[row["model"]], -p.get("alpha", 0), depth, -p.get("min_samples_leaf", 0),
            p.get("n_estimators", 0), str(p.get("max_features", "")),
            n_extra_columns, int(bool(p.get("log_target", False))), row["mae_mean"])


def select_within(rows, width="se"):
    """Chọn mô hình đơn giản nhất có MAE CV ≤ MAE nhỏ nhất + 1 × (SE hoặc SD) của mô hình tốt nhất.
    width="se": quy tắc 1-SE chuẩn (dùng chính thức). width="sd": quy tắc 1-SD cũ, chỉ để đối chiếu."""
    key = {"se": "mae_se", "sd": "mae_std"}[width]
    best = min(rows, key=lambda r: r["mae_mean"])
    threshold = best["mae_mean"] + best[key]
    eligible = [r for r in rows if r["mae_mean"] <= threshold]
    return min(eligible, key=_complexity_key), best, threshold, eligible


def select_one_se(rows):
    """Quy tắc 1 sai số chuẩn (1-SE): ngưỡng = MAE nhỏ nhất + SE (= SD qua các fold / √số fold)
    của mô hình đó; trong các mô hình có MAE ≤ ngưỡng, chọn mô hình đơn giản nhất
    (Linear Regression < Ridge < Decision Tree < Random Forest)."""
    return select_within(rows, "se")


def select_one_sd(rows):
    """Quy tắc 1-SD cũ (ngưỡng = MAE nhỏ nhất + độ lệch chuẩn). Giữ lại để đối chiếu trong báo cáo:
    ngưỡng này rộng gấp √5 ≈ 2,2 lần 1-SE nên gần như mọi cấu hình đều lọt."""
    return select_within(rows, "sd")


def run_grid(X, y, groups, stage="gd4", grids=None, log=None):
    rows = []
    for name, grid in (grids or GRIDS).items():
        keys = list(grid)
        combos = list(itertools.product(*grid.values()))
        if log:
            log(f"{name}: {len(combos)} tổ hợp tham số")
        for values in combos:
            rows.append(_evaluate(name, dict(zip(keys, values)), X, y, groups, stage))
    return rows


def paired_diff(a, b):
    """So sánh hai cấu hình trên CÙNG các fold: trả về (chênh MAE trung bình b − a, SE của chênh lệch)."""
    d = np.asarray(b["fold_mae"]) - np.asarray(a["fold_mae"])
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


ABLATIONS = [  # (cột tham số, giá trị tham chiếu, các họ mô hình áp dụng)
    ("skill_top_k", 0, list(FAMILY_ORDER)),
    ("experience", "numeric", ["linear_regression", "ridge"]),
    ("log_target", False, ["linear_regression", "ridge"]),
]


def ablation_table(rows):
    """Với mỗi họ mô hình và mỗi lựa chọn: lấy cấu hình TỐT NHẤT ứng với từng giá trị
    (các tham số khác tự do), so với cấu hình tốt nhất ở giá trị tham chiếu theo cặp fold.
    chênh < 0: tốt hơn tham chiếu. |chênh| < 2 SE: chưa đủ bằng chứng là khác nhau."""
    out = []
    for key, ref, families in ABLATIONS:
        for fam in families:
            fam_rows = [r for r in rows if r["model"] == fam and key in r["params"]]
            values = sorted({json.dumps(r["params"][key]) for r in fam_rows})
            if len(values) < 2:
                continue
            best_of = {v: min((r for r in fam_rows if json.dumps(r["params"][key]) == v),
                              key=lambda r: r["mae_mean"]) for v in values}
            base = best_of[json.dumps(ref)]
            for v in values:
                r = best_of[v]
                diff, se = paired_diff(base, r)
                out.append({"lua_chon": key, "ho_mo_hinh": fam, "gia_tri": json.loads(v),
                            "mae_cv": round(r["mae_mean"], 3), "mae_se": round(r["mae_se"], 3),
                            "chenh_so_voi_tham_chieu": round(diff, 3), "se_chenh": round(se, 3),
                            "ket_luan": ("tham chiếu" if v == json.dumps(ref) else
                                         "tốt hơn rõ" if diff < -2 * se else
                                         "kém hơn rõ" if diff > 2 * se else "không khác rõ"),
                            "params_tot_nhat": json.dumps(r["params"], ensure_ascii=False)})
    return pd.DataFrame(out)


def gd4():
    log = Tee(REPORTS / "gd4_train_output.txt")
    X, y, groups, train = _train_frame()
    log("GĐ4 – GRID SEARCH VÀ CHỌN MÔ HÌNH (GroupKFold, chỉ tập train)")
    log("Ngày chạy:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    log(f"Train: {len(train)} tin, {groups.nunique()} nhóm, {N_SPLITS} fold\n")

    rows = run_grid(X, y, groups, log=log)
    log(f"Tổng: {len(rows)} cấu hình")

    grid_df = pd.DataFrame(rows).sort_values("mae_mean")
    grid_df.assign(params=grid_df.params.apply(json.dumps),
                   fold_mae=grid_df.fold_mae.apply(json.dumps)).to_csv(
        REPORTS / "gd4_grid_results.csv", index=False)

    families = [f for f in FAMILY_ORDER if any(r["model"] == f for r in rows)]
    best_by_family = [min((r for r in rows if r["model"] == f), key=lambda r: r["mae_mean"])
                      for f in families]
    log("\nTốt nhất của từng họ mô hình:")
    for r in best_by_family:
        log(_fmt(r))
    save_results(best_by_family)

    abl = ablation_table(rows)
    abl.to_csv(REPORTS / "gd4_ablation.csv", index=False, encoding="utf-8-sig")
    log("\nAblation (cấu hình tốt nhất ứng với từng lựa chọn, so theo cặp fold với giá trị tham chiếu;"
        " 'rõ' = chênh quá 2 SE):")
    log(abl.drop(columns="params_tot_nhat").to_string(index=False))

    chosen, best, threshold, eligible = select_one_se(rows)
    log(f"\nQuy tắc 1-SE: MAE nhỏ nhất {best['mae_mean']:.3f} ({best['model']}) "
        f"+ SE {best['mae_se']:.3f} = ngưỡng {threshold:.3f}")
    log(f"Số cấu hình có MAE ≤ ngưỡng: {len(eligible)}/{len(rows)}")
    log(f"CHỌN (đơn giản nhất trong ngưỡng): {_fmt(chosen)}")
    old, _, old_thr, old_elig = select_one_sd(rows)
    log(f"Đối chiếu – quy tắc 1-SD cũ: ngưỡng {old_thr:.3f}, {len(old_elig)}/{len(rows)} cấu hình lọt, "
        f"sẽ chọn {old['model']} {json.dumps(old['params'])}")

    selected = {
        "model": chosen["model"], "params": chosen["params"],
        "cv_mae_mean": chosen["mae_mean"], "cv_mae_std": chosen["mae_std"],
        "cv_mae_se": chosen["mae_se"], "cv_r2_mean": chosen["r2_mean"],
        "rule": "1-SE: mô hình đơn giản nhất có MAE CV ≤ MAE nhỏ nhất + 1 sai số chuẩn",
        "best_model": best["model"], "best_params": best["params"],
        "best_mae_mean": best["mae_mean"], "threshold": threshold,
        "n_eligible": len(eligible), "n_configs": len(rows),
        "n_splits": N_SPLITS, "random_state": RANDOM_STATE,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    SELECTED_JSON.write_text(json.dumps(selected, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    log(f"\nĐã ghi {SELECTED_JSON.relative_to(REPORTS.parent)}, gd4_ablation.csv và {RESULTS_CSV.name}")
    log.save()
    return selected


# ---------------------------------------------------------------- Nested CV
def nested_cv(X, y, groups, grids=None, n_outer=N_SPLITS, progress=print):
    """Cross-validation lồng nhau theo nhóm. Mỗi fold ngoài: chạy lại toàn bộ grid + 1-SE trên phần
    train của fold (GroupKFold bên trong), fit cấu hình được chọn, chấm trên phần còn lại.
    Trả về bảng theo fold: MAE của quy trình 1-SE, của cấu hình có MAE CV nhỏ nhất, và của baseline."""
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    out = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_outer).split(X, y, groups)):
        rows = run_grid(X.iloc[tr], y[tr], groups[tr], stage="nested_inner", grids=grids)
        chosen, best, _, _ = select_one_se(rows)
        rec = {"fold": k + 1, "n_val": len(va)}
        for tag, cfg in (("chon_1se", chosen), ("tot_nhat_cv", best),
                         ("dummy", {"model": "dummy_median", "params": {}})):
            pipe = make_pipeline(cfg["model"], cfg["params"]).fit(X.iloc[tr], y[tr])
            m = regression_metrics(y[va], pipe.predict(X.iloc[va]))
            rec[f"mae_{tag}"] = m["mae"]
            rec[f"r2_{tag}"] = m["r2"]
            if tag != "dummy":
                rec[f"cau_hinh_{tag}"] = f"{cfg['model']} {json.dumps(cfg['params'])}"
        out.append(rec)
        if progress:
            progress(f"  fold ngoài {k + 1}/{n_outer}: 1-SE chọn {rec['cau_hinh_chon_1se']} -> "
                     f"MAE {rec['mae_chon_1se']:.3f}")
    return pd.DataFrame(out)


def nested():
    log = Tee(REPORTS / "gd4_nested_output.txt")
    X, y, groups, train = _train_frame()
    log("NESTED CV – SAI SỐ CỦA CẢ QUY TRÌNH CHỌN MÔ HÌNH (chỉ tập train, không dùng test)")
    log("Ngày chạy:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    n_cfg = sum(len(list(itertools.product(*g.values()))) for g in GRIDS.values())
    log(f"Train: {len(train)} tin, {groups.nunique()} nhóm; ngoài {N_SPLITS} fold × trong {N_SPLITS} fold "
        f"× {n_cfg} cấu hình (có thể mất 10–20 phút)\n")
    res = nested_cv(X, y, groups, progress=log)
    res.to_csv(REPORTS / "gd4_nested_cv.csv", index=False, encoding="utf-8-sig")
    log("")
    for tag, label in (("chon_1se", "Quy trình GĐ4 (grid + 1-SE)"),
                       ("tot_nhat_cv", "Chọn cấu hình MAE CV nhỏ nhất"),
                       ("dummy", "Baseline trung vị")):
        m = res[f"mae_{tag}"]
        log(f"{label:<32} MAE {m.mean():.3f} ± {m.std(ddof=1):.3f} (SE {m.std(ddof=1) / np.sqrt(len(m)):.3f}), "
            f"R² {res[f'r2_{tag}'].mean():.3f}")
    log("\nCấu hình được chọn ở từng fold ngoài (độ ổn định của quy trình):")
    log(res["cau_hinh_chon_1se"].value_counts().to_string())
    log("\nĐây là ước lượng sai số trên dữ liệu chưa thấy KHÔNG phụ thuộc vào tập test "
        "(tập test đã được chấm nhiều lần, xem reports/test_set_log.csv).")
    log.save()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GĐ3/GĐ4: huấn luyện và chọn mô hình")
    ap.add_argument("--stage", choices=["gd3", "gd4", "nested"], default="gd3")
    args = ap.parse_args()
    {"gd3": main, "gd4": gd4, "nested": nested}[args.stage]()
