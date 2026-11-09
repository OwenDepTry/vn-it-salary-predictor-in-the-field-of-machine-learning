# tests/test_predict_invalid_input.py
"""Đầu vào không hợp lệ phải bị TỪ CHỐI (InvalidInput); đầu vào hợp lệ nhưng ngoài phạm vi phải có CẢNH BÁO."""
import math

import pytest

from predict import InvalidInput, SalaryPredictor

VALID = dict(job_category="Phát triển phần mềm", city="Hà Nội", experience_years=2,
             seniority="unknown", skills=["java", "sql"])


def call(p, **override):
    return p.predict(**{**VALID, **override})


# ----------------------------------------------------------------------------- bị từ chối
@pytest.mark.parametrize("override", [
    {"job_category": "Thiết kế đồ họa"},          # nhóm nghề ngoài phạm vi (đã loại ở GĐ2)
    {"job_category": ""},
    {"job_category": None},
    {"city": "Paris"},
    {"city": "ha noi"},                            # sai chính tả/không dấu: không tự đoán
    {"seniority": "CEO"},
    {"experience_years": -1},
    {"experience_years": 41},
    {"experience_years": "abc"},
    {"experience_years": None},
    {"experience_years": math.nan},
    {"experience_years": True},                    # bool không được coi là số
    {"experience_years": [2]},
    {"skills": 5},
    {"skills": ["java", 3]},
])
def test_invalid_input_rejected(predictor, override):
    with pytest.raises(InvalidInput):
        call(predictor, **override)


# ----------------------------------------------------------------------------- hợp lệ
def test_valid_input_returns_prediction_with_unit_and_disclaimer(predictor):
    r = call(predictor)
    assert isinstance(r["prediction"], float)
    assert r["unit"] == "triệu VND/tháng"
    assert "không phải cam kết lương" in r["disclaimer"]
    lo, hi = r["range_80"]
    assert lo <= r["prediction"] <= hi
    assert lo >= 0


def test_experience_as_numeric_string_is_accepted(predictor):
    assert call(predictor, experience_years="2")["prediction"] == call(predictor)["prediction"]


@pytest.mark.parametrize("skills", [None, [], "", "java, sql", ("java", "sql")])
def test_skill_formats_accepted(predictor, skills):
    call(predictor, skills=skills)


def test_skill_synonyms_are_normalized(predictor):
    r = call(predictor, skills=["ReactJS", "JavaScript"])
    assert set(r["used_skills"]) <= {"react", "javascript"}


# ----------------------------------------------------------------------------- cảnh báo
def test_warn_when_experience_above_train_range(predictor):
    r = call(predictor, experience_years=10)
    assert any("ngoại suy" in w for w in r["warnings"])


def test_warn_on_unknown_skill(predictor):
    r = call(predictor, skills=["cobol"])
    assert any("cobol" in w for w in r["warnings"])
    assert r["used_skills"] == []


def test_warn_on_rare_city(predictor):
    counts = predictor.card["train_counts"]["city"]
    rare = [c for c in predictor.allowed["city"] if counts.get(c, 0) < 10]
    if not rare:
        pytest.skip("dữ liệu giả không có thành phố hiếm")
    r = call(predictor, city=rare[0])
    assert any(rare[0] in w for w in r["warnings"])


def test_no_warning_for_typical_input(predictor):
    r = call(predictor, skills=[])
    assert r["warnings"] == []


def test_missing_model_files_give_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        SalaryPredictor(tmp_path / "none.joblib", tmp_path / "none.json")
