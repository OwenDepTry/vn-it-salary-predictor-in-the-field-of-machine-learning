# src/intervals.py
"""Khoảng dự đoán ~80% từ phần dư out-of-fold (OOF) trên train.

Hai kiểu khoảng:
  - "additive": [pred + q10, pred + q90] với q là phân vị của phần dư y - ŷ (triệu).
    Mọi đầu vào nhận cùng độ rộng (cách làm cũ).
  - "log":      [expm1(log1p(pred) + q10), expm1(log1p(pred) + q90)] với q là phân vị của
    phần dư log1p(y) - log1p(ŷ). Khoảng tỉ lệ với mức lương: lương cao -> khoảng rộng hơn,
    hợp với dữ liệu lương (sai số tăng theo mức lương).

Cả hai đều phủ ~80% phần dư OOF theo cách xây dựng, nên được chọn theo độ phủ CÓ ĐIỀU KIỆN:
chia các tin train thành 3 nhóm theo mức dự đoán OOF (thấp/vừa/cao), chọn kiểu khoảng có độ phủ
ở nhóm tệ nhất gần 80% hơn. Chỉ dùng train (OOF), không dùng tập test.
"""
import numpy as np

TARGET_COVERAGE = 0.80
SCALES = ("additive", "log")


def _log1p_pos(x):
    return np.log1p(np.clip(np.asarray(x, float), 0.0, None))


def _bounds(pred, scale, q10, q90):
    pred = np.asarray(pred, float)
    if scale == "additive":
        return pred + q10, pred + q90
    if scale == "log":
        base = _log1p_pos(pred)
        return np.expm1(base + q10), np.expm1(base + q90)
    raise ValueError(f"scale phải là một trong {SCALES}")


def _coverage(y, low, high):
    y = np.asarray(y, float)
    return float(np.mean((y >= low) & (y <= high)))


def fit_interval(y_true, oof_pred):
    """Học khoảng từ nhãn train và dự đoán OOF. Trả về dict lưu vào model_card.json
    (giữ các khóa cũ p10/p90 cho kiểu additive)."""
    y = np.asarray(y_true, float)
    oof = np.asarray(oof_pred, float)
    params = {
        "additive": tuple(np.percentile(y - oof, [10, 90])),
        "log": tuple(np.percentile(_log1p_pos(y) - _log1p_pos(oof), [10, 90])),
    }
    # 3 nhóm theo mức dự đoán OOF (phân vị 1/3, 2/3)
    cuts = np.quantile(oof, [1 / 3, 2 / 3])
    tercile = np.digitize(oof, cuts)
    diag = {}
    for scale, (q10, q90) in params.items():
        low, high = _bounds(oof, scale, q10, q90)
        by_t = [_coverage(y[tercile == t], low[tercile == t], high[tercile == t]) for t in range(3)]
        diag[scale] = {
            "oof_coverage": _coverage(y, low, high),
            "oof_tercile_coverage": [round(c, 3) for c in by_t],
            "worst_tercile_gap": float(max(abs(c - TARGET_COVERAGE) for c in by_t)),
            "mean_width": float(np.mean(high - low)),
        }
    chosen = min(SCALES, key=lambda s: (round(diag[s]["worst_tercile_gap"], 3), diag[s]["mean_width"]))
    return {
        "scale": chosen,
        "p10": float(params["additive"][0]), "p90": float(params["additive"][1]),
        "log_p10": float(params["log"][0]), "log_p90": float(params["log"][1]),
        "diagnostics": diag,
    }


def apply_interval(pred, iv):
    """Áp khoảng cho một hoặc nhiều dự đoán. Card cũ (không có khóa "scale") -> additive."""
    scale = iv.get("scale", "additive")
    if scale == "log":
        low, high = _bounds(pred, "log", iv["log_p10"], iv["log_p90"])
    else:
        low, high = _bounds(pred, "additive", iv["p10"], iv["p90"])
    low = np.maximum(low, 0.0)
    high = np.maximum(high, low)
    if np.ndim(pred) == 0:
        return float(low), float(high)
    return low, high


def interval_coverage(y_true, pred, iv):
    low, high = apply_interval(np.asarray(pred, float), iv)
    return _coverage(y_true, low, high)
