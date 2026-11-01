"""
TV2 - Data cleaning for VietJobs IT salary prediction project.

Run from the project root:
    python src/data_cleaning.py

Input:
    data/raw/vietjobs/VietJobs.csv

Outputs:
    data/interim/vietjobs_it_candidates.csv
    data/processed/it_salary_clean.csv
    reports/gd2_cleaning_log.txt

The functions are intentionally standalone so other team members can import them.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "raw" / "vietjobs" / "VietJobs.csv"
INTERIM_FILE = ROOT / "data" / "interim" / "vietjobs_it_candidates.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "it_salary_clean.csv"
LOG_FILE = ROOT / "reports" / "gd2_cleaning_log.txt"

IT_CATEGORY = "công_nghệ_thông_tin_kỹ_thuật_số"

FALSE_POSITIVE_PATTERNS = [
    r"\bsắp xếp\b",
    r"\bkiêm thủ kho\b",
    r"\bkiêm thu kho\b",
]

SALES_PATTERNS = [
    r"\bkinh doanh\b", r"\bsales?\b", r"\btelesales\b",
    r"\bbusiness development\b", r"\bphát triển kinh doanh\b",
    r"\btư vấn\b", r"\bchăm sóc khách hàng\b",
    r"\bmarketing\b", r"\btiếp thị\b",
]

GRAPHIC_PATTERNS = [
    r"\bgraphic designer\b", r"\bgraphic design\b",
    r"\bthiết kế đồ họa\b", r"\bthiết kế đồ hoạ\b",
    r"\bchỉnh sửa hình ảnh\b", r"\bphoto editor\b",
    r"\bthumbnail\b", r"\bprint on demand\b",
    r"\bthiết kế bao bì\b", r"\bthiết kế ấn phẩm\b",
    r"\b2d artist\b", r"\b3d artist\b", r"\bmotion graphic\b",
    r"\bmultimedia designer\b",
]

IT_PATTERNS = [
    r"\bit\b", r"\bdeveloper\b", r"\bsoftware\b", r"\bengineer\b",
    r"\blập trình\b", r"\bphần mềm\b", r"\btester\b", r"\btesting\b",
    r"\bkiểm thử\b", r"\bqa\b", r"\bqc\b", r"\bdata\b",
    r"\bmachine learning\b", r"\bdeep learning\b", r"\bartificial intelligence\b",
    r"\btrí tuệ nhân tạo\b", r"\bai\b", r"\bdevops\b", r"\bcloud\b",
    r"\bnetwork\b", r"\bmạng\b", r"\bsystem\b", r"\bhệ thống\b",
    r"\bsecurity\b", r"\bbảo mật\b", r"\bcyber\b", r"\bdatabase\b",
    r"\bcơ sở dữ liệu\b", r"\bweb\b", r"\bbackend\b", r"\bfrontend\b",
    r"\bfullstack\b", r"\bmobile\b", r"\bandroid\b", r"\bios\b",
    r"\bembedded\b", r"\bblockchain\b", r"\bautomation\b",
    r"\btự động hóa\b", r"\bplc\b", r"\bcad.cam\b", r"\bsap\b",
    r"\bui/?ux\b", r"\bgame designer\b",
]


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _fold(value: object) -> str:
    """Lowercase and remove Vietnamese accents for robust matching."""
    value = _text(value).lower()
    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return value.replace("đ", "d")


def _has_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def parse_salary(value: object) -> tuple[float, float]:
    """
    Return (salary_min, salary_max) in million VND/month.
    Negotiable/unknown/zero salary is represented by (NaN, NaN), not zero.
    """
    raw = _fold(value)
    if not raw or any(x in raw for x in ("thoa thuan", "deal", "negotiable")):
        return np.nan, np.nan

    numbers = re.findall(r"\d+(?:[.,]\d+)?", raw)
    if not numbers:
        return np.nan, np.nan

    values = [float(x.replace(",", ".")) for x in numbers[:2]]
    if len(values) == 1:
        low = high = values[0]
    else:
        low, high = min(values), max(values)

    if high == 0:
        return np.nan, np.nan

    if "ty" in raw or "tỷ" in _text(value).lower():
        low, high = low * 1000, high * 1000
    elif "nghin" in raw or "ngàn" in _text(value).lower():
        low, high = low / 1000, high / 1000

    return low, high


def parse_experience_years(value: object) -> float:
    """Convert experience text to years; 'Không yêu cầu' becomes 0."""
    raw = _fold(value)
    if not raw:
        return np.nan
    if any(term in raw for term in ("khong yeu cau", "khong can", "khong doi hoi", "no experience")):
        return 0.0

    matches = re.findall(r"(\d+(?:[.,]\d+)?)\s*(nam|thang)", raw)
    if matches:
        years = []
        for number, unit in matches:
            amount = float(number.replace(",", "."))
            years.append(amount if unit == "nam" else amount / 12)
        return max(years)

    return np.nan


def experience_group(years: object) -> str:
    if pd.isna(years):
        return "unknown"
    years = float(years)
    if years == 0:
        return "no_experience"
    if years < 1:
        return "under_1_year"
    if years < 3:
        return "1_to_2_years"
    if years < 5:
        return "3_to_4_years"
    return "5_plus_years"


def classify_title(title: object, category: object = "") -> bool:
    """Return True when the title is plausibly an IT/technical role."""
    title_folded = _fold(title)
    category_folded = _fold(category)

    if not title_folded:
        return False

    NON_IT_PATTERNS = [
        r"\bcnc\b",
        r"\bmay phay\b",
        r"\bmay tien\b",
        r"\blap trinh plc\b",
        r"\bky su plc\b",
        r"\btu dong hoa cong nghiep\b",
        r"\bautomation technician\b",
        r"\bindustrial automation\b",
        r"\bke toan\b",
        r"\bnong hoc\b",
        r"\bagronomy\b",
        r"\benvironmental\b",
        r"\bmanagement system\b",
        r"\bqa/?qc\b.*\b(thuc pham|san xuat thuc pham)\b",
        r"\bphuc vu\b",
        r"\btri lieu\b",
        r"\bnhan vien sap xep\b",
        r"\bseo\b",
        r"\bgoogle ads\b",
        r"\bcontent creator\b",
        r"\bmarketing\b",
        r"\bsales\b",
        r"\btelesales\b",
        r"\btu van\b",
        r"\bban hang\b",
        r"\buser acquisition\b",
    ]

    if _has_any(title_folded, NON_IT_PATTERNS):
        return False

    if _has_any(title_folded, FALSE_POSITIVE_PATTERNS):
        return False

    if _has_any(title_folded, SALES_PATTERNS):
        return False

    if _has_any(title_folded, GRAPHIC_PATTERNS):
        if not _has_any(title_folded, [
            r"\bui/?ux\b",
            r"\bgame designer\b",
            r"\bsoftware design\b",
            r"\bthiet ke phan mem\b",
            r"\bky su thiet ke phan mem\b",
        ]):
            return False

    IT_TITLE_PATTERNS = [
        r"\bsoftware\b",
        r"\bdeveloper\b",
        r"\bprogrammer\b",
        r"\bbackend\b",
        r"\bback end\b",
        r"\bfrontend\b",
        r"\bfront end\b",
        r"\bfullstack\b",
        r"\bfull stack\b",
        r"\bweb developer\b",
        r"\bmobile developer\b",
        r"\bdata analyst\b",
        r"\bdata engineer\b",
        r"\bdata scientist\b",
        r"\bmachine learning\b",
        r"\bartificial intelligence\b",
        r"\bai engineer\b",
        r"\bqa engineer\b",
        r"\btester\b",
        r"\btest engineer\b",
        r"\bdevops\b",
        r"\bsystem engineer\b",
        r"\bsystem administrator\b",
        r"\bnetwork engineer\b",
        r"\bsecurity engineer\b",
        r"\bdatabase administrator\b",
        r"\bit support\b",
        r"\bit helpdesk\b",
        r"\btechnical support\b",
        r"\bui/?ux\b",
        r"\bgame designer\b",
        r"\bgame developer\b",
        r"\bweb administrator\b",
        r"\bquan tri web\b",
        r"\blap trinh\b",
        r"\bky su phan mem\b",
        r"\bky su cong nghe thong tin\b",
    ]

    return _has_any(title_folded, IT_TITLE_PATTERNS)


def seniority_from_title(title: object) -> str:
    raw = _fold(title)
    if _has_any(raw, [r"\bintern\b", r"\binternship\b", r"\bthuc tap\b", r"\bfresher\b", r"\bjunior\b"]):
        return "junior"
    if _has_any(raw, [r"\bsenior\b", r"\blead\b", r"\bprincipal\b", r"\btruong nhom\b", r"\btruong phong\b"]):
        return "senior"
    if _has_any(raw, [r"\bmanager\b", r"\bdirector\b", r"\bquan ly\b", r"\bgiam doc\b", r"\bhead of\b"]):
        return "manager"
    if _has_any(raw, [r"\bmiddle\b", r"\bmid-level\b", r"\bmid level\b"]):
        return "middle"
    return "unspecified"


def map_city(location: object) -> str:
    raw = _fold(location)
    if not raw:
        return "unknown"

    cities = []
    city_patterns = {
        "ha_noi": [r"\bha noi\b", r"\bha dong\b", r"\bbac tu liem\b"],
        "ho_chi_minh": [r"\bho chi minh\b", r"\btp hcm\b", r"\bthu duc\b"],
        "da_nang": [r"\bda nang\b"],
        "hai_phong": [r"\bhai phong\b"],
        "can_tho": [r"\bcan tho\b"],
    }
    for city, patterns in city_patterns.items():
        if _has_any(raw, patterns):
            cities.append(city)
    if len(cities) > 1:
        return "multiple_cities"
    if len(cities) == 1:
        return cities[0]
    return "other_vietnam"


def normalize_skill(skill: object) -> str:
    raw = _fold(skill)
    raw = re.sub(r"[^a-z0-9+#./-]+", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    aliases = {
        "js": "javascript", "ts": "typescript", "node js": "node.js",
        "nodejs": "node.js", "react js": "react", "reactjs": "react",
        "vue js": "vue", "vuejs": "vue", "c sharp": "c#",
        "dotnet": ".net", "ms sql": "sql server",
    }
    return aliases.get(raw, raw)


def parse_skills(value: object) -> list[str]:
    """Split a skill cell into normalized unique skills."""
    raw = _text(value)
    if not raw:
        return []
    parts = re.split(r"[,;|\n•]+", raw)
    skills = []
    seen = set()
    for part in parts:
        normalized = normalize_skill(part)
        if normalized and normalized not in seen:
            seen.add(normalized)
            skills.append(normalized)
    return skills


def self_check(df: pd.DataFrame) -> dict[str, object]:
    """Return basic quality checks; raises on broken salary ranges."""
    duplicate_columns = [column for column in df.columns if column != "skills_list"]
    checks: dict[str, object] = {
        "rows": int(len(df)),
        "duplicate_rows": int(df.duplicated(subset=duplicate_columns).sum()),
        "missing_job_title": int(df["job_title"].isna().sum()) if "job_title" in df else None,
        "missing_salary_min": int(df["salary_min"].isna().sum()) if "salary_min" in df else None,
        "missing_salary_max": int(df["salary_max"].isna().sum()) if "salary_max" in df else None,
    }
    if {"salary_min", "salary_max"}.issubset(df.columns):
        invalid = df["salary_min"].notna() & df["salary_max"].notna() & (
            df["salary_min"] > df["salary_max"]
        )
        checks["invalid_salary_ranges"] = int(invalid.sum())
        if invalid.any():
            raise ValueError("Found rows where salary_min > salary_max.")
    return checks


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    required = {"job_title", "salary", "experience_required", "location", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required source columns: {sorted(missing)}")

    work = df.copy()
    work["job_title"] = work["job_title"].fillna("").astype(str).str.strip()
    work["is_it_title"] = work.apply(
        lambda row: classify_title(row["job_title"], row["category"]), axis=1
    )

    candidates = work.loc[work["is_it_title"]].copy()

    salary_pairs = candidates["salary"].apply(parse_salary)
    candidates[["salary_min_parsed", "salary_max_parsed"]] = pd.DataFrame(
        salary_pairs.tolist(), index=candidates.index
    )

    valid_salary = (
        candidates["salary_min_parsed"].between(3, 200)
        & candidates["salary_max_parsed"].between(3, 200)
    )
    candidates = candidates.loc[valid_salary].copy()

    candidates["experience_years"] = candidates["experience_required"].apply(
        parse_experience_years
    )
    candidates["experience_group"] = candidates["experience_years"].apply(experience_group)
    candidates["seniority"] = candidates["job_title"].apply(seniority_from_title)
    candidates["city_group"] = candidates["location"].apply(map_city)

    skill_source = next(
        (col for col in ("technical_skills", "qualifications", "requirements_text") if col in candidates),
        None,
    )
    candidates["skills_list"] = (
        candidates[skill_source].apply(parse_skills) if skill_source else [[] for _ in range(len(candidates))]
    )

    candidates["salary_min"] = candidates["salary_min_parsed"].astype(float)
    candidates["salary_max"] = candidates["salary_max_parsed"].astype(float)
    candidates["salary_avg"] = (
        candidates["salary_min"] + candidates["salary_max"]
    ) / 2

    duplicate_columns = [column for column in candidates.columns if column != "skills_list"]
    candidates = candidates.drop_duplicates(subset=duplicate_columns).reset_index(drop=True)
    checks = self_check(candidates)
    return work.loc[work["is_it_title"]].copy(), candidates, checks


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cannot find input file: {INPUT_FILE}")

    INTERIM_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(INPUT_FILE, low_memory=False)
    interim, cleaned, checks = clean_dataset(raw)

    interim.to_csv(INTERIM_FILE, index=False, encoding="utf-8-sig")
    cleaned.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    with LOG_FILE.open("w", encoding="utf-8") as file:
        file.write("TV2 - DATA CLEANING LOG\n")
        file.write(f"Input rows: {len(raw)}\n")
        file.write(f"IT title candidates: {len(interim)}\n")
        file.write(f"Rows after salary validation and deduplication: {len(cleaned)}\n")
        for key, value in checks.items():
            file.write(f"{key}: {value}\n")
        file.write("\nSalary rule: keep 3-200 million VND/month; source 0.0 triệu means negotiated.\n")
        file.write("Title rule: exclude sales/marketing/consulting and graphic-design roles;")
        file.write(" preserve technical software-design roles.\n")

    print(f"Saved: {INTERIM_FILE.relative_to(ROOT)} ({len(interim)} rows)")
    print(f"Saved: {OUTPUT_FILE.relative_to(ROOT)} ({len(cleaned)} rows)")
    print(f"Saved: {LOG_FILE.relative_to(ROOT)}")
    print(checks)


if __name__ == "__main__":
    main()
