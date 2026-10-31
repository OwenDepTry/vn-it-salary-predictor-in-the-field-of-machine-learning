"""GĐ5 – Đánh giá mô hình (TV3: Phan Thành Đạt).

Gồm hai phần:
  1. Các hàm dùng chung cho train.py: regression_metrics(), make_pipeline(),
     cross_validate_grouped(), summarize_folds(), save_results(), mae_by_group(),
     group_bootstrap(), load_train_test().
  2. final_evaluation(): đánh giá MỘT LẦN DUY NHẤT trên tập test với mô hình đã
     chọn ở GĐ4, lưu models/salary_model.joblib và models/model_card.json.

Chạy (bước 7 của README):
    python src/evaluate.py
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (FEATURE_COLUMNS, RANDOM_STATE, TARGET,  # noqa: E402
                      build_preprocessor, get_split, load_clean)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"
RESULTS_CSV = REPORTS / "model_results.csv"
SELECTED_JSON = REPORTS / "gd4_selected_model.json"
DOWNLOAD_LOG = ROOT / "data/raw/download_log.json"

N_SPLITS = 5
GROUP_COLUMNS = ["seniority", "city", "job_category", "experience_group"]
DISCLAIMER = ("Kết quả chỉ mang tính tham khảo, phản ánh mức lương ĐĂNG TUYỂN trong dữ "
              "liệu VietJobs, không phải mức lương thực nhận.")

MODEL_FACTORY = {
    "dummy": lambda **p: DummyRegressor(strategy="median", **p),
    "linear_regression": lambda **p: LinearRegression(**p),
    "decision_tree": lambda **p: DecisionTreeRegressor(random_state=RANDOM_STATE, **p),
    "random_forest": lambda **p: RandomForestRegressor(random_state=RANDOM_STATE,
                                                       n_jobs=-1, **p),
}


class Tee:
    """In ra màn hình đồng thời ghi vào file output (gd3/gd4/gd5_*_output.txt)."""

    def __init__(self, path):
        self.path = Path(path)
        self.lines = []

    def __call__(self, *args):
        line = " ".join(str(a) for a in args)
        print(line)
        self.lines.append(line)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- dữ liệu
def load_train_test():
    """Ghép tập sạch với split.csv; trả về (train, test) có cột group_id."""
    df = load_clean()
    split = get_split(df)
    if "row_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "row_id"})
    data = df.merge(split[["row_id", "group_id", "split"]], on="row_id", how="inner")
    if len(data) != len(df):
        raise ValueError("split.csv không khớp tập sạch (khác row_id). "
                         "Không tạo lại split.csv; báo TV2 kiểm tra.")
    train = data[data["split"] == "train"].reset_index(drop=True)
    test = data[data["split"] == "test"].reset_index(drop=True)
    leak = set(train.group_id) & set(test.group_id)
    if leak:
        raise ValueError(f"Có {len(leak)} nhóm nằm ở cả train và test: rò rỉ dữ liệu.")
    return train, test


def make_pipeline(name, params=None):
    return Pipeline([("prep", build_preprocessor()),
                     ("model", MODEL_FACTORY[name](**(params or {})))])


# ---------------------------------------------------------------- metric
def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "r2": float(r2_score(y_true, y_pred))}


def cross_validate_grouped(pipe, X, y, groups, n_splits=N_SPLITS):
    """GroupKFold: một nhóm tin đăng lại không bao giờ nằm ở cả fold train và
    fold validation. Tiền xử lý (one-hot, từ vựng kỹ năng) fit lại trong từng fold.
    Trả về (bảng metric từng fold, dự đoán out-of-fold)."""
    y = np.asarray(y, float)
    oof = np.full(len(y), np.nan)
    rows = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        model = clone(pipe).fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[va])
        oof[va] = pred
        rows.append({"fold": k + 1, "n_val": len(va), **regression_metrics(y[va], pred)})
    return pd.DataFrame(rows), oof


def summarize_folds(folds):
    n = len(folds)
    std = float(folds["mae"].std(ddof=1))
    return {"mae_mean": float(folds["mae"].mean()), "mae_std": std,
            "mae_se": std / np.sqrt(n),
            "rmse_mean": float(folds["rmse"].mean()),
            "r2_mean": float(folds["r2"].mean()), "n_folds": n}


def save_results(rows, path=RESULTS_CSV):
    """Ghi/ghi đè các dòng (model, stage) vào reports/model_results.csv."""
    new = pd.DataFrame(rows)
    new["params"] = new["params"].apply(lambda p: json.dumps(p, sort_keys=True))
    if Path(path).exists():
        old = pd.read_csv(path)
        keys = set(zip(new.model, new.stage))
        old = old[[k not in keys for k in zip(old.model, old.stage)]]
        new = pd.concat([old, new], ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


def mae_by_group(frame, column):
    """frame có cột y_true, y_pred và column; trả về n, MAE, sai lệch TB theo nhóm."""
    err = frame["y_pred"] - frame["y_true"]
    out = (frame.assign(abs_error=err.abs(), error=err)
           .groupby(column)
           .agg(n=("abs_error", "size"), mae=("abs_error", "mean"),
                mean_error=("error", "mean"))
           .reset_index().rename(columns={column: "value"}))
    out.insert(0, "group_by", column)
    return out.sort_values(["group_by", "n"], ascending=[True, False])


def group_bootstrap(y_true, y_pred, groups, n_boot=2000, seed=RANDOM_STATE):
    """Khoảng tin cậy 95% của MAE, lấy mẫu lại theo NHÓM tin (không theo dòng),
    để các tin đăng lại không làm khoảng tin cậy hẹp giả."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    maes = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        maes[b] = mean_absolute_error(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(maes, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------- GĐ5
def _skills_vocabulary(pipe):
    names = pipe.named_steps["prep"].get_feature_names_out()
    vocab = [n.split("__", 1)[-1] for n in names if n.startswith("skills__")]
    return [v[len("skill_"):] if v.startswith("skill_") else v for v in vocab]


def _data_sha256():
    if not DOWNLOAD_LOG.exists():
        return {}
    log = json.loads(DOWNLOAD_LOG.read_text(encoding="utf-8"))
    return {f["name"]: f["sha256"] for f in log.get("files", [])}


def _plots(pred_df, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    y, p = pred_df["y_true"], pred_df["y_pred"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(p, y - p, alpha=0.7)
    ax.axhline(0, color="red", lw=1)
    ax.set_xlabel("Lương dự đoán (triệu/tháng)")
    ax.set_ylabel("Phần dư = thực tế − dự đoán")
    ax.set_title(f"Phần dư trên tập test – {title}")
    fig.tight_layout()
    fig.savefig(FIGURES / "gd5_residuals.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y, p, alpha=0.7)
    lim = [min(y.min(), p.min()) - 1, max(y.max(), p.max()) + 1]
    ax.plot(lim, lim, "r--", lw=1, label="y = x (dự đoán đúng)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Lương thực tế (triệu/tháng)")
    ax.set_ylabel("Lương dự đoán (triệu/tháng)")
    ax.set_title(f"Thực tế và dự đoán trên tập test – {title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "gd5_actual_vs_pred.png", dpi=150)
    plt.close(fig)


def final_evaluation(force=False):
    out_txt = REPORTS / "gd5_final_output.txt"
    if out_txt.exists() and not force:
        sys.exit("Tập test đã được dùng (reports/gd5_final_output.txt đã có). "
                 "Chỉ chạy lại với --force khi chắc chắn KHÔNG đổi mô hình theo kết quả test.")
    if not SELECTED_JSON.exists():
        sys.exit("Chưa có reports/gd4_selected_model.json: chạy `python src/train.py --stage gd4` trước.")

    log = Tee(out_txt)
    selected = json.loads(SELECTED_JSON.read_text(encoding="utf-8"))
    name, params = selected["model"], selected["params"]
    train, test = load_train_test()
    X_tr, y_tr, g_tr = train[FEATURE_COLUMNS], train[TARGET].to_numpy(float), train["group_id"]
    X_te, y_te, g_te = test[FEATURE_COLUMNS], test[TARGET].to_numpy(float), test["group_id"]

    log("GĐ5 – ĐÁNH GIÁ CUỐI TRÊN TẬP TEST (chạy một lần)")
    log("Ngày chạy:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    log(f"Mô hình đã chọn ở GĐ4: {name} {params}")
    log(f"Train: {len(train)} tin / {g_tr.nunique()} nhóm; "
        f"Test: {len(test)} tin / {g_te.nunique()} nhóm")

    # Phần dư OOF trên train -> khoảng sai số cho app (không dùng test).
    pipe = make_pipeline(name, params)
    folds, oof = cross_validate_grouped(pipe, X_tr, y_tr, g_tr)
    cv = summarize_folds(folds)
    resid_oof = y_tr - oof
    q10, q90 = np.percentile(resid_oof, [10, 90])

    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    metrics = regression_metrics(y_te, pred)
    ci = group_bootstrap(y_te, pred, g_te)
    dummy = make_pipeline("dummy").fit(X_tr, y_tr)
    dummy_metrics = regression_metrics(y_te, dummy.predict(X_te))

    log(f"CV train (GroupKFold {N_SPLITS}): MAE {cv['mae_mean']:.3f} ± {cv['mae_std']:.3f}, "
        f"R² {cv['r2_mean']:.3f}")
    log(f"TEST: MAE {metrics['mae']:.3f} triệu (CI95% theo nhóm {ci[0]:.3f}–{ci[1]:.3f}), "
        f"RMSE {metrics['rmse']:.3f}, R² {metrics['r2']:.3f}")
    log(f"Baseline Dummy trên test: MAE {dummy_metrics['mae']:.3f}, R² {dummy_metrics['r2']:.3f}")
    log(f"Khoảng phần dư OOF 10%–90%: [{q10:.2f}, {q90:.2f}] triệu")

    pred_df = test[["row_id", "group_id"] + FEATURE_COLUMNS].copy()
    pred_df["y_true"], pred_df["y_pred"] = y_te, pred
    pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]
    pred_df.to_csv(REPORTS / "gd5_test_predictions.csv", index=False)

    group_mae = pd.concat([mae_by_group(pred_df, c) for c in GROUP_COLUMNS
                           if c in pred_df.columns], ignore_index=True)
    group_mae.to_csv(REPORTS / "gd5_group_mae.csv", index=False)
    log("\nMAE theo nhóm (nhóm có n nhỏ thì MAE kém tin cậy):")
    log(group_mae.round(3).to_string(index=False))

    perm = permutation_importance(pipe, X_te, y_te, scoring="neg_mean_absolute_error",
                                  n_repeats=30, random_state=RANDOM_STATE)
    imp = pd.DataFrame({"feature": FEATURE_COLUMNS,
                        "mae_increase_mean": perm.importances_mean,
                        "mae_increase_std": perm.importances_std}
                       ).sort_values("mae_increase_mean", ascending=False)
    imp.to_csv(REPORTS / "gd5_permutation_importance.csv", index=False)
    log("\nPermutation importance (MAE tăng bao nhiêu khi xáo trộn cột):")
    log(imp.round(3).to_string(index=False))

    _plots(pred_df, name)

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS / "salary_model.joblib")
    card = {
        "model": {"name": name, "params": params, "selected_by": selected.get("rule")},
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "unit": "triệu VNĐ/tháng",
        "allowed": {c: sorted(train[c].dropna().astype(str).unique().tolist())
                    for c in FEATURE_COLUMNS if c not in ("experience_years", "skills")},
        "experience_years_range": [float(train["experience_years"].min()),
                                   float(train["experience_years"].max())],
        "skills_vocabulary": _skills_vocabulary(pipe),
        "train_counts": {"n_train": int(len(train)), "n_test": int(len(test)),
                         "n_train_groups": int(g_tr.nunique()),
                         **{c: train[c].astype(str).value_counts().to_dict()
                            for c in ("job_category", "city", "seniority")
                            if c in train.columns}},
        "label_range_train": [float(y_tr.min()), float(y_tr.max())],
        "residual_interval_oof": {"q10": float(q10), "q90": float(q90)},
        "test_metrics": {**metrics, "mae_ci95": list(ci)},
        "cv_metrics": cv,
        "baseline_test_metrics": dummy_metrics,
        "random_state": RANDOM_STATE,
        "data_sha256": _data_sha256(),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
    }
    (MODELS / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    log("\nĐã lưu models/salary_model.joblib và models/model_card.json")
    log.save()
    return card


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GĐ5: đánh giá cuối trên tập test")
    ap.add_argument("--force", action="store_true",
                    help="cho phép chạy lại khi đã có gd5_final_output.txt")
    final_evaluation(force=ap.parse_args().force)
