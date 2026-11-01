import pandas as pd
import pytest

from src.data_cleaning import (
    parse_salary,
    parse_experience_years,
    experience_group,
    classify_title,
    seniority_from_title,
    map_city,
    normalize_skill,
    parse_skills,
)


def test_parse_salary():
    assert parse_salary("10 - 20 triệu") == (10, 20)
    assert parse_salary("15 triệu") == (15, 15)


def test_parse_experience_years():
    assert parse_experience_years("2 năm") == 2


def test_experience_group():
    assert experience_group(1) is not None
    assert experience_group(5) is not None


def test_classify_title():
    assert classify_title("Backend Developer") is True
    assert classify_title("Data Analyst") is True
    assert classify_title("Nhân viên bán hàng") is False
    assert classify_title("Kế toán") is False


def test_seniority_from_title():
    assert seniority_from_title("Senior Backend Developer") is not None
    assert seniority_from_title("Intern Developer") is not None


def test_map_city():
    assert map_city("Ho Chi Minh") is not None
    assert map_city("Ha Noi") is not None


def test_normalize_skill():
    assert normalize_skill("Python") == "python"


def test_parse_skills():
    result = parse_skills("Python, Java, SQL")
    assert isinstance(result, list)
    assert len(result) > 0