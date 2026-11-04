import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import (
    parse_salary, parse_experience_years, experience_group,
    classify_title, classify_job_category, seniority_from_title,
    map_city, normalize_skill, parse_skills,
    self_check, clean_dataset,
)


def test_salary_parser():
    assert parse_salary("10 - 20 triệu") == (10, 20)
    assert parse_salary("15 - 25 triệu") == (15, 25)
    assert np.isnan(parse_salary("0.0 triệu")[0])
    assert np.isnan(parse_salary("20 - 10 triệu")[0])
    assert np.isnan(parse_salary("thỏa thuận")[0])


def test_experience():
    assert parse_experience_years("2 năm") == 2
    assert parse_experience_years("Không yêu cầu") == 0
    assert experience_group(0) == "0 (không yêu cầu)"
    assert experience_group(2) == "2 năm"
    assert experience_group(5) == "4+ năm"


def test_it_classifier_examples():
    assert classify_job_category("Chuyên Viên Kiểm Thử Phần Mềm") == "Kiểm thử & QA"
    assert classify_job_category("Nhà phân tích kinh doanh") == "Quản lý & phân tích (PM/BA/PO)"
    assert classify_title("Backend Developer")
    assert classify_title("Data Analyst")
    assert not classify_title("Nhân Viên Kinh Doanh Phần Mềm")
    assert not classify_title("Lập Trình Máy CNC")
    assert not classify_title("Kế toán")


def test_seniority_and_city():
    assert seniority_from_title("Intern Developer") == "intern"
    assert seniority_from_title("Fresher QA") == "fresher"
    assert seniority_from_title("Senior Backend Developer") == "senior"
    assert seniority_from_title("Trưởng nhóm") == "lead"
    assert map_city("Ho Chi Minh") == "TP.HCM"
    assert map_city("Ha Noi") == "Hà Nội"
    assert map_city("Đà Nẵng") == "Đà Nẵng"
    assert map_city("Hà Nội, TP.HCM") == "Nhiều thành phố"


def test_skill_normalization():
    assert normalize_skill("Python") == "python"
    assert "python" in parse_skills("Python, Java • SQL")

def test_self_check_quality_metrics():
    df = pd.DataFrame({
        "job_title": ["Backend Developer", "Data Analyst"],
        "salary_min": [10, 15],
        "salary_max": [20, 25],
        "skills_list": [["python"], ["sql"]],
    })

    result = self_check(df)

    assert result["rows"] == 2
    assert result["duplicate_rows"] == 0
    assert result["missing_job_title"] == 0
    assert result["missing_salary_min"] == 0
    assert result["missing_salary_max"] == 0
    assert result["invalid_salary_ranges"] == 0

def test_self_check_rejects_invalid_salary_range():
    df = pd.DataFrame({
        "job_title": ["Backend Developer"],
        "salary_min": [25],
        "salary_max": [10],
    })

    with pytest.raises(ValueError):
        self_check(df)

def test_clean_dataset_pipeline():
    df = pd.DataFrame({
        "job_title": [
            "Backend Developer",
            "Data Analyst",
            "Nhân Viên Kinh Doanh",
            "Senior QA Engineer",
        ],
        "location": [
            "Ho Chi Minh",
            "Ha Noi",
            "Ho Chi Minh",
            "Da Nang",
        ],
        "country": [
            "Việt Nam",
            "Việt Nam",
            "Việt Nam",
            "Việt Nam",
        ],
        "salary": [
            "10 - 20 triệu",
            "15 - 25 triệu",
            "10 - 20 triệu",
            "20 - 10 triệu",
        ],
        "contract_type": [
            "Toàn thời gian",
            "Toàn thời gian",
            "Toàn thời gian",
            "Toàn thời gian",
        ],
        "experience_required": [
            "2 năm",
            "Không yêu cầu",
            "3 năm",
            "5 năm",
        ],
        "description": [
            "Develop backend software",
            "Analyze business data",
            "Sell software products",
            "Test software applications",
        ],
        "category": ["IT", "IT", "IT", "IT"],
        "technical_skills": [
            "Python, SQL",
            "Excel, Python",
            "Sales",
            "Java, Selenium",
        ],
    })

    interim, cleaned, checks = clean_dataset(df)

    assert interim["is_it"].sum() == 3
    assert len(cleaned) == 2

    assert "model_reason" in interim.columns
    assert interim.loc[interim["is_it"], "model_reason"].notna().all()

    assert set(cleaned["job_title"]) == {
        "Backend Developer",
        "Data Analyst",
    }

    assert checks["invalid_salary_ranges"] == 0
    assert checks["missing_salary_min"] == 0
    assert checks["missing_salary_max"] == 0

    assert "row_id" in cleaned.columns
    assert "job_category" in cleaned.columns
    assert "salary_label" in cleaned.columns
    assert "desc_group" in cleaned.columns

def test_clean_dataset_removes_duplicate_jobs():
    df = pd.DataFrame({
        "job_title": [
            "Backend Developer",
            "Backend Developer",
        ],
        "location": [
            "Ho Chi Minh",
            "Ho Chi Minh",
        ],
        "country": [
            "Việt Nam",
            "Việt Nam",
        ],
        "salary": [
            "10 - 20 triệu",
            "10 - 20 triệu",
        ],
        "contract_type": [
            "Toàn thời gian",
            "Toàn thời gian",
        ],
        "experience_required": [
            "2 năm",
            "2 năm",
        ],
        "description": [
            "Develop backend software using Python",
            "Develop backend software using Python",
        ],
        "category": ["IT", "IT"],
        "technical_skills": [
            "Python, SQL",
            "Python, SQL",
        ],
    })

    interim, cleaned, checks = clean_dataset(df)

    assert interim["is_it"].sum() == 2
    assert len(cleaned) == 1
    assert checks["duplicate_rows"] == 0

    assert "model_reason" in interim.columns
    assert interim.loc[interim["is_it"], "model_reason"].notna().all()

def test_clean_dataset_rejects_missing_required_column():
    df = pd.DataFrame({
        "job_title": ["Backend Developer"],
        "location": ["Ho Chi Minh"],
        "country": ["Việt Nam"],
        "salary": ["10 - 20 triệu"],
        "contract_type": ["Toàn thời gian"],
        "experience_required": ["2 năm"],
        "description": ["Develop backend software"],
        # Cố tình thiếu cột category
    })

    with pytest.raises(ValueError, match="Missing required source columns"):
        clean_dataset(df)