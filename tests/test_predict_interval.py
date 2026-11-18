"""Dự đoán với model_card.json kiểu mới: khoảng dạng log và mô hình không dùng kỹ năng."""
import json

import pytest


@pytest.fixture()
def make_predictor(tiny_model_files, tmp_path):
    from predict import SalaryPredictor
    model_path, card_path = tiny_model_files

    def _make(**changes):
        card = json.loads(card_path.read_text(encoding="utf-8"))
        card.update(changes)
        new_card = tmp_path / "model_card.json"
        new_card.write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
        return SalaryPredictor(model_path, new_card)
    return _make


def test_log_interval_scales_with_prediction(make_predictor):
    iv = {"scale": "log", "p10": -3.0, "p90": 3.0, "log_p10": -0.2, "log_p90": 0.2, "test_coverage": 0.8}
    p = make_predictor(residual_interval_oof=iv)
    low_exp = p.predict("Phát triển phần mềm", "Hà Nội", 0)
    high_exp = p.predict("Phát triển phần mềm", "Hà Nội", 5)
    w = lambda r: r["range_80"][1] - r["range_80"][0]
    assert high_exp["prediction"] > low_exp["prediction"]
    assert w(high_exp) > w(low_exp)
    for r in (low_exp, high_exp):
        assert r["range_80"][0] <= r["prediction"] <= r["range_80"][1]


def test_old_card_still_uses_additive_interval(predictor):
    r = predictor.predict("Phát triển phần mềm", "Hà Nội", 2)
    assert r["range_80"][1] - r["range_80"][0] == pytest.approx(6.0, abs=0.11)


def test_model_without_skills_warns_that_skills_are_ignored(make_predictor):
    p = make_predictor(skills_vocabulary=[])
    r = p.predict("Phát triển phần mềm", "Hà Nội", 2, skills="python, sql")
    assert r["used_skills"] == []
    assert any("không dùng kỹ năng" in w for w in r["warnings"])
