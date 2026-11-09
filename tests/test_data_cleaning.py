# tests/test_data_cleaning.py
"""Quy tắc làm sạch (GĐ2): parse lương, kinh nghiệm, thành phố, kỹ năng và bộ lọc IT theo chức danh."""
import math

import pytest

import data_cleaning as dc


@pytest.mark.parametrize("raw, expected", [
    ("15.0 - 25.0 triệu", (15.0, 25.0, "range")),
    ("8.9 - 15.0 triệu", (8.9, 15.0, "range")),
    ("0.0 triệu", (None, None, "negotiable")),   # 0.0 = thỏa thuận (khớp 1:1 với salary_avg ở GĐ1b)
    ("15.0 triệu", (15.0, 15.0, "single_value")),
    ("0.0 - 10.0 triệu", (0.0, 10.0, "zero_lower")),
    ("30.0 - 20.0 triệu", (30.0, 20.0, "inverted")),
    ("abc", (None, None, "unparsed")),
    (None, (None, None, "missing")),
])
def test_parse_salary(raw, expected):
    lo, hi, status = dc.parse_salary(raw)
    assert status == expected[2]
    for got, exp in [(lo, expected[0]), (hi, expected[1])]:
        assert (exp is None and math.isnan(got)) or got == exp


@pytest.mark.parametrize("raw, years, group", [
    ("Không yêu cầu", 0.0, "0 (không yêu cầu)"),
    ("6 tháng", 0.5, "<1 năm"),
    ("18 tháng", 1.5, "1 năm"),
    ("3 năm", 3.0, "3 năm"),
    ("10 năm", 10.0, "4+ năm"),
])
def test_experience(raw, years, group):
    assert dc.parse_experience_years(raw) == years
    assert dc.experience_group(years) == group


def test_experience_unparseable_is_nan():
    assert math.isnan(dc.parse_experience_years("tùy vị trí"))
    assert dc.experience_group(float("nan")) == "unknown"


@pytest.mark.parametrize("location, city", [
    ("hà nội", "Hà Nội"), ("cầu giấy", "Hà Nội"), ("thủ đức", "TP.HCM"), ("quận 7", "TP.HCM"),
    ("đà nẵng", "Đà Nẵng"), ("hải phòng", "Khác"), ("hà nội, hồ chí minh", "Nhiều thành phố"),
])
def test_map_city(location, city):
    assert dc.map_city(location) == city


def test_parse_skills_normalizes_and_dedups():
    assert dc.parse_skills("['.NET', 'ASP.NET Core', 'ReactJS', 'reactjs', 'Node.js.']") == [".net", "react", "nodejs"]
    assert dc.parse_skills(None) == []
    assert dc.parse_skills("[]") == []


def test_it_title_rules_self_check():
    """38 chức danh mẫu (gồm các ca bắt nhầm tìm thấy ở GĐ1b/GĐ2) phải phân loại đúng."""
    assert dc.self_check() == []


@pytest.mark.parametrize("title", ["Nhân Viên Thiết Kế Đồ Họa", "Kế Toán Kho Kiêm Thủ Kho",
                                   "Nhân Viên Kinh Doanh Phần Mềm", "Nhân Viên Lập Trình Máy Tiện CNC"])
def test_non_it_titles_excluded(title):
    assert dc.classify_title(title, dc.IT_CATEGORY)[0] is False


@pytest.mark.parametrize("title", ["Trưởng Nhóm An Ninh Mạng/ Cyber Security Lead", "Chuyên Viên An Ninh Mạng",
                                   "Kỹ Sư An Ninh Mạng", "Đánh Giá An Toàn Thông Tin (Redteam)"])
def test_cyber_security_titles_kept_as_security(title):
    is_it, job_category, _, _ = dc.classify_title(title, dc.IT_CATEGORY)
    assert is_it is True
    assert job_category == "Hạ tầng, DevOps & bảo mật"


@pytest.mark.parametrize("title", ["Nhân Viên An Ninh", "Bảo Vệ Tòa Nhà", "Security Manager - Phụ Trách An Ninh",
                                   "Chuyên Viên Kinh Doanh Dịch Vụ An Toàn Thông Tin"])
def test_physical_security_and_sales_still_excluded(title):
    assert dc.classify_title(title, dc.IT_CATEGORY)[0] is False


@pytest.mark.parametrize("title", ["Fresher C/nhúng", "Kỹ sư Linux nhúng", "Kỹ Sư IT- Nhúng"])
def test_embedded_titles_are_software_development(title):
    is_it, job_category, role, _ = dc.classify_title(title, dc.IT_CATEGORY)
    assert (is_it, job_category, role) == (True, "Phát triển phần mềm", "embedded")


def test_name_nhung_without_accent_is_not_embedded():
    assert dc.classify_title("Nhân viên Nhung", dc.IT_CATEGORY)[0] is False
