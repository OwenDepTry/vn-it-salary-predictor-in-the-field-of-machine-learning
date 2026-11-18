"""Test cho src/intervals.py: khoảng dự đoán ~80% từ phần dư OOF."""
import numpy as np
import pytest

from intervals import apply_interval, fit_interval, interval_coverage


def _heteroscedastic(n=3000, seed=0):
    """Lương có sai số TỈ LỆ với mức lương (như dữ liệu thật): lương cao -> sai số lớn."""
    rng = np.random.default_rng(seed)
    pred = rng.uniform(8, 60, n)
    y = pred * np.exp(rng.normal(0, 0.25, n))
    return y, pred


def test_log_interval_chosen_when_errors_grow_with_salary():
    y, pred = _heteroscedastic()
    iv = fit_interval(y, pred)
    assert iv["scale"] == "log"
    d = iv["diagnostics"]
    assert d["log"]["worst_tercile_gap"] < d["additive"]["worst_tercile_gap"]
    for s in ("additive", "log"):                      # cả hai đều phủ ~80% tổng thể
        assert d[s]["oof_coverage"] == pytest.approx(0.8, abs=0.02)


def test_additive_interval_chosen_when_errors_are_constant():
    rng = np.random.default_rng(1)
    pred = rng.uniform(8, 60, 3000)
    y = pred + rng.normal(0, 4, 3000)
    assert fit_interval(y, pred)["scale"] == "additive"


def test_log_interval_is_wider_for_higher_predictions():
    y, pred = _heteroscedastic()
    iv = fit_interval(y, pred)
    lo1, hi1 = apply_interval(10.0, iv)
    lo2, hi2 = apply_interval(50.0, iv)
    assert (hi2 - lo2) > 3 * (hi1 - lo1)
    assert lo1 < 10 < hi1 and lo2 < 50 < hi2


def test_apply_interval_old_card_is_additive_and_never_negative():
    old = {"p10": -8.0, "p90": 9.0}                    # model_card.json cũ, không có "scale"
    assert apply_interval(20.0, old) == (12.0, 29.0)
    assert apply_interval(3.0, old) == (0.0, 12.0)


def test_interval_coverage_on_new_data():
    y, pred = _heteroscedastic(seed=2)
    iv = fit_interval(*_heteroscedastic(seed=3))
    assert interval_coverage(y, pred, iv) == pytest.approx(0.8, abs=0.03)
