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
    value = "".join(
        ch for ch in value
        if unicodedata.category(ch) != "Mn"
    )
    return value.replace("đ", "d")


def _has_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def parse_salary(value: object) -> tuple[float, float]:
    """
    Return (salary_min, salary_max) in million VND/month.
    Negotiable or invalid salary is represented by (NaN, NaN).
    """
    raw = _fold(value)

    if not raw:
        return np.nan, np.nan

    if any(term in raw for term in ("thoa thuan", "deal", "negotiable")):
        return np.nan, np.nan

    numbers = re.findall(r"\d+(?:[.,]\d+)?", raw)

    if not numbers:
        return np.nan, np.nan

    values = [
        float(number.replace(",", "."))
        for number in numbers[:2]
    ]

    if len(values) == 1 and values[0] == 0 and "trieu" in raw:
        return np.nan, np.nan

    if len(values) == 1:
        low = high = values[0]
    else:
        low, high = values

        if low > high:
            return np.nan, np.nan

    if high == 0:
        return np.nan, np.nan

    if "ty" in raw:
        low *= 1000
        high *= 1000
    elif "nghin" in raw:
        low /= 1000
        high /= 1000

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
        return "0 (không yêu cầu)"
    if years < 1:
        return "<1 năm"
    if years == 1:
        return "1 năm"
    if years == 2:
        return "2 năm"  
    if years == 3:
        return "3 năm"

    return "4+ năm"



def classify_job_category(title: object) -> str | None:
    """
    Classify an IT job title into one of the 7 required categories.
    Return None for non-IT titles.
    """
    raw = _fold(title)

    if not raw:
        return None

    # Exclude known false-positive titles.
    if _has_any(raw, [
        r"\bsap xep\b",
        r"\bkiem thu kho\b",
    ]):
        return None

    # Exclude clearly non-IT roles first.
    exclude_patterns = [
        # CNC, cơ khí và tự động hóa công nghiệp
        r"\bcnc\b",
        r"\bmay phay\b",
        r"\bmay tien\b",
        r"\bphay\b",
        r"\blap trinh plc\b",
        r"\bky su plc\b",
        r"\btu dong hoa cong nghiep\b",
        r"\bindustrial automation\b",
        r"\bky su tu dong hoa\b",
        r"\blap trinh cam\b",
        r"\blap trinh may moc\b",
        r"\bmachine vision\b",

        # Xây dựng, BIM, MEP, HVAC
        r"\bhvac\b",
        r"\bbim\b",
        r"\bmep\b",
        r"\bshopdrawing\b",
        r"\bshop drawing\b",
        r"\bky su xay dung\b",
        r"\bcong trinh xay dung\b",
        r"\bthiet ke cau duong\b",
        r"\bban ve ket cau\b",
        r"\bthiet ke thi cong\b",
        r"\bdesign and build\b",
        r"\bchu dau tu\b",
        r"\bky su he thong dien\b",

        # Kế toán và tài chính
        r"\bke toan\b",
        r"\bthu ngan\b",
        r"\btai chinh ke toan\b",

        # Tuyển sinh, giảng dạy
        r"\btu van tuyen sinh\b",
        r"\bgiang vien\b",
        r"\bgiao vien\b",
        r"\bteacher\b",
        r"\blearning and development\b",

        # Sản xuất, may mặc
        r"\bnong hoc\b",
        r"\bagronomy\b",
        r"\benvironmental\b",
        r"\bdet lap trinh\b",
        r"\blap trinh mau may mac\b",

        # Bảo vệ
        r"\bbao ve\b",
        r"\bsecurity guard\b",

        # Kinh doanh, bán hàng, marketing
        r"\bnhan vien kinh doanh\b",
        r"\bchuyen vien kinh doanh\b",
        r"\bsales\b",
        r"\bsale\b",
        r"\btelesales\b",
        r"\bbusiness development\b",
        r"\bphat trien kinh doanh\b",
        r"\bban hang\b",
        r"\bmarketing\b",
        r"\btiep thi\b",
        r"\bcontent creator\b",
        r"\bgraphic design\b",
        r"\bvideo editor\b",

        # Dữ liệu không phải công việc IT
        r"\bdata labeling\b",

        # Hành chính, tư vấn không thuộc IT
        r"\bnhan vien admin\b",
        r"\badmin phong\b",
        r"\bdu lich\b",

        # Công việc QA/QC ngoài lĩnh vực phần mềm
        r"\bqa/qc.*co dien\b",
        r"\bqa qc.*co dien\b",
        r"\bqa/qc.*san xuat thuc pham\b",
        r"\bqa/qc.*phong lab\b",

        # Kiến trúc và xây dựng
        r"\bkien truc su\b",
        r"\bdesign\s*&\s*build\b",
    ]

    if _has_any(raw, exclude_patterns):
        return None

    # 1. Software development
    if _has_any(raw, [
        r"\bdeveloper\b",
        r"\bprogrammer\b",
        r"\blap trinh\b",
        r"\bky su phan mem\b",
        r"\bsoftware\b",
        r"\bbackend\b",
        r"\bback end\b",
        r"\bfrontend\b",
        r"\bfront end\b",
        r"\bfullstack\b",
        r"\bfull stack\b",
        r"\bmobile developer\b",
        r"\bweb developer\b",
        r"\bweb architect\b",
        r"\bios developer\b",
        r"\bandroid developer\b",
        r"\bblockchain\b",
        r"\bsitecore\b",
        r"\bliferay\b",
        r"\bjava\b",
        r"\bpython\b",
        r"\bphp\b",
        r"\b\.net\b",
        r"\bnodejs\b",
        r"\bunity developer\b",
        r"\bnha phat trien thong nhat\b",
        r"\bnha phat trien phu tro cao cap\b",
        r"\bsenior backend developer\b",
        r"\bky su nhung\b",
        r"\bembedded engineer\b",
        r"\bembedded software\b",
        r"\bky su blockchain\b",
        r"\bky su phat trien ung dung\b",
        r"\bky su nghien cuu thuat toan\b",
    ]):
        return "Phát triển phần mềm"

    # 2. Testing and QA
    if _has_any(raw, [
        r"\btester\b",
        r"\bkiem thu\b",
        r"\bqa engineer\b",
        r"\bqc engineer\b",
        r"\bqa intern\b",
        r"\bqa/qc\b",
        r"\bqa qc\b",
        r"\bmanual test\b",
        r"\bautomation test\b",
        r"\btest engineer\b",
        r"\btest lead\b",
        r"\bquality assurance\b",
        r"\bprocess quality assurance\b",
        r"\bquy trinh dam bao chat luong\b",
        r"\bkiem tra tu dong hoa\b",
    ]):
        return "Kiểm thử & QA"

    # 3. Infrastructure, DevOps and security
    if _has_any(raw, [
        r"\bdevops\b",
        r"\bdevsecops\b",
        r"\bsre\b",
        r"\bsite reliability\b",
        r"\bcloud\b",
        r"\bopenstack\b",
        r"\blinux\b",
        r"\bsystem administrator\b",
        r"\bsystem admin\b",
        r"\bsystem engineer\b",
        r"\bky su he thong\b",
        r"\bquan tri he thong\b",
        r"\bquan tri mang\b",
        r"\bnetwork\b",
        r"\bky su mang\b",
        r"\binfrastructure\b",
        r"\bstorage\b",
        r"\bdatacenter\b",
        r"\bdata center\b",
        r"\bvmware\b",
        r"\bkubernetes\b",
        r"\bm365\b",
        r"\bbao mat\b",
        r"\bsecurity engineer\b",
        r"\ban ninh mang\b",
        r"\ban toan thong tin\b",
        r"\bdba\b",
        r"\bdatabase administrator\b",
        r"\bvoip\b",
        r"\bky thuat van hanh voip\b",
        r"\bsystem integration engineer\b",
        r"\bnetwork consultant\b",
        r"\binfrastructure architecture\b",
    ]):
        return "Hạ tầng, DevOps & bảo mật"

    # 4. Data and AI
    if _has_any(raw, [
        r"\bdata analyst\b",
        r"\bdata engineer\b",
        r"\bdata scientist\b",
        r"\bdata science\b",
        r"\bdata platform\b",
        r"\bdata architecture\b",
        r"\bdata warehouse\b",
        r"\bbusiness intelligence\b",
        r"\bdata intelligence\b",
        r"\bphan tich du lieu\b",
        r"\bky su du lieu\b",
        r"\bmachine learning\b",
        r"\bdeep learning\b",
        r"\bartificial intelligence\b",
        r"\bai engineer\b",
        r"\bai specialist\b",
        r"\bai analyst\b",
        r"\bai leader\b",
        r"\bai/nlp\b",
        r"\bnlp engineer\b",
        r"\bllm\b",
        r"\bgenai\b",
        r"\bmlops\b",
        r"\btri tue nhan tao\b",
        r"\bky su khoa hoc du lieu\b",
    ]):
        return "Dữ liệu & AI"

    # 5. PM/BA/PO
    if _has_any(raw, [
        r"\bproject manager\b",
        r"\bquan ly du an\b",
        r"\bbusiness analyst\b",
        r"\bphan tich nghiep vu\b",
        r"\bnha phan tich kinh doanh\b",
        r"\bproduct owner\b",
        r"\bchu so huu san pham\b",
        r"\bproduct manager\b",
        r"\bscrum master\b",
        r"\bbrse\b",
        r"\bbridge system engineer\b",
    ]):
        return "Quản lý & phân tích (PM/BA/PO)"

    # 6. Software implementation and ERP
    if _has_any(raw, [
        r"\btrien khai phan mem\b",
        r"\btu van trien khai\b",
        r"\berp\b",
        r"\bsap\b",
        r"\bodoo\b",
        r"\boracle engineer\b",
        r"\boracle\b",
        r"\boig\b",
        r"\boim\b",
        r"\bsolution consultant\b",
        r"\bky su trien khai\b",
        r"\bky su giai phap\b",
        r"\bimplementation engineer\b",
    ]):
        return "Triển khai & ERP"

    # 7. IT support and general IT
    if _has_any(raw, [
        r"\bit support\b",
        r"\bit helpdesk\b",
        r"\bit help desk\b",
        r"\bhelpdesk\b",
        r"\bhelp desk\b",
        r"\bnhan vien it\b",
        r"\bky thuat it\b",
        r"\bky thuat vien it\b",
        r"\bky thuat may tinh\b",
        r"\bit engineer\b",
        r"\bit executive\b",
        r"\bit infrastructure\b",
        r"\bit specialist\b",
        r"\bit coordinator\b",
        r"\bit manager\b",
        r"\bit system\b",
        r"\bmiddle it\b",
        r"\bcntt\b",
        r"\bcong nghe thong tin\b",
        r"\bit comtor\b",
        r"\btechnical support\b",
        r"\btechnical engineer\b",
    ]):
        return "Hỗ trợ kỹ thuật & IT chung"

    return None

def classify_title(title: object, category: object = "") -> bool:
    """Return True when the title is classified into one of the IT roles."""
    return classify_job_category(title) is not None


def seniority_from_title(title: object) -> str:
    raw = _fold(title)
    if _has_any(raw, [r"\bintern\b", r"\binternship\b", r"\bthuc tap\b"]):
        return "intern"
    if _has_any(raw, [r"\bfresher\b", r"\bgraduate\b"]):
        return "fresher"
    if _has_any(raw, [r"\bjunior\b"]):
        return "junior"
    if _has_any(raw, [r"\bmiddle\b", r"\bmid-level\b", r"\bmid level\b"]):
        return "middle"
    if _has_any(raw, [r"\blead\b", r"\bprincipal\b", r"\btruong nhom\b"]):
        return "lead"
    if _has_any(raw, [r"\bmanager\b", r"\bdirector\b", r"\bquan ly\b", r"\bgiam doc\b", r"\bhead of\b"]):
        return "manager"
    if _has_any(raw, [r"\bsenior\b", r"\bsr\b", r"\bcao cap\b"]):
        return "senior"
    return "unknown"


def map_city(location: object) -> str:
    raw = re.sub(r"[^a-z0-9]+", " ", _fold(location))
    if not raw:
        return "Khác"
    city_patterns = {
        "Hà Nội": [r"\bha noi\b", r"\bha dong\b", r"\bbac tu liem\b"],
        "TP.HCM": [r"\bho chi minh\b", r"\btp hcm\b", r"\bthu duc\b", r"\bsaigon\b"],
        "Đà Nẵng": [r"\bda nang\b"],
    }
    found = [city for city, patterns in city_patterns.items() if _has_any(raw, patterns)]
    if len(found) > 1:
        return "Nhiều thành phố"
    return found[0] if found else "Khác"


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


def _normalised_description(value: object) -> str:
    return re.sub(r"\s+", " ", _fold(value)).strip()


def _classify_row(row: pd.Series) -> str | None:
    title = _text(row.get("job_title", ""))
    title_fold = _fold(title)
    category = classify_job_category(title)

    evidence = _fold(" ".join(
        _text(row.get(column, ""))
        for column in (
            "description",
            "requirements_text",
            "qualifications",
            "technical_skills",
        )
    ))

    # 1. Bridge System Engineer (BrSE):
    # Main responsibilities are requirements and project coordination.
    if _has_any(title_fold, [
        r"\bbridge system engineer\b",
        r"\bbrse\b",
    ]):
        return "Quản lý & phân tích (PM/BA/PO)"

    # 2. System Engineer:
    # Classify based on the main responsibilities, not optional skills.
    if category is None and _has_any(title_fold, [
        r"\bsystem engineer\b",
        r"\blinux system engineer\b",
    ]):
        infrastructure_title = [
            r"\blinux\b",
            r"\bnetwork\b",
            r"\bkubernetes\b",
            r"\bvmware\b",
            r"\bdata center\b",
            r"\bcloud\b",
            r"\bgenesys\b",
        ]

        infrastructure_evidence = [
            r"\blinux\b",
            r"\bnetwork\b",
            r"\bkubernetes\b",
            r"\bvmware\b",
            r"\bdata center\b",
            r"\bserver\b",
            r"\bstorage\b",
            r"\bcisco\b",
            r"\bdell\b",
            r"\bhpe\b",
            r"\bfujitsu\b",
            r"\bredhat\b",
            r"\bcloud computing\b",
            r"\bopenstack\b",
            r"\bvirtualization\b",
            r"\bha tang\b",
            r"\bquan tri he thong\b",
            r"\bvan hanh he thong\b",
            r"\bnetworking\b",
            r"\bdatacenter\b",
        ]

        software_evidence = [
            r"\bfrontend\b",
            r"\bfront end\b",
            r"\bbackend\b",
            r"\bback end\b",
            r"\bcoding\b",
            r"\bcode frontend\b",
            r"\bcode backend\b",
            r"\bphat trien chuc nang\b",
            r"\bphat trien phan mem\b",
            r"\bsoftware development\b",
            r"\brefactor\b",
            r"\bmigrate he thong\b",
        ]

        # Infrastructure signals take priority over optional programming skills.
        if _has_any(title_fold, infrastructure_title):
            category = "Hạ tầng, DevOps & bảo mật"

        elif _has_any(evidence, infrastructure_evidence):
            category = "Hạ tầng, DevOps & bảo mật"

        elif _has_any(evidence, software_evidence):
            category = "Phát triển phần mềm"

    # 3. Other Business Analyst titles:
    # The title alone is not enough; require IT-related evidence.
    if category is None:
        is_business_analyst = _has_any(title_fold, [
            r"\bnha phan tich kinh doanh\b",
            r"\bbusiness analyst\b",
            r"\bchuyen vien phan tich nghiep vu\b",
        ])

        if not is_business_analyst:
            return None

        category = "Quản lý & phân tích (PM/BA/PO)"

    elif category != "Quản lý & phân tích (PM/BA/PO)":
        return category

    # 4. Check whether PM/BA/PO work is actually related to IT.
    non_it = [
        r"\bconstruction project\b",
        r"\bdu an xay dung\b",
        r"\bthi cong cong trinh\b",
        r"\bmarketing campaign\b",
        r"\bbrand marketing\b",
        r"\bmarketing strategy\b",
        r"\btruyen thong su kien\b",
        r"\bto chuc su kien\b",
        r"\bnghien cuu thi truong\b",
        r"\bmarket research\b",
        r"\blogistics\b",
        r"\bmanufacturing\b",
        r"\bco khi\b",
        r"\bket cau thep\b",
        r"\bthang may\b",
        r"\bnoi that\b",
        r"\bdien mat troi\b",
        r"\bgiao duc\b",
        r"\bielts\b",
        r"\bban hang\b",
        r"\baccount executive\b",
    ]

    if _has_any(evidence, non_it):
        strong_it_evidence = [
            r"\bsoftware\b",
            r"\bphan mem\b",
            r"\bsaas\b",
            r"\bweb app\b",
            r"\bmobile app\b",
            r"\bphat trien ung dung\b",
            r"\bphat trien phan mem\b",
            r"\bteam dev\b",
            r"\bdeveloper\b",
            r"\bproduct backlog\b",
            r"\buser story\b",
            r"\bbrd\b",
            r"\bsrs\b",
            r"\bapi\b",
            r"\bcrm\b",
            r"\berp\b",
            r"\bdata warehouse\b",
            r"\bdoi ngu ky thuat\b",
            r"\bnhom phat trien\b",
            r"\btech company\b",
            r"\bcomputer science\b",
            r"\bengineering\b",
            r"\bagile methodologies\b",
            r"\bproduct owner\b",
            r"\bjira\b",
            r"\btrello\b",
            r"\basana\b",
            r"\btechnical teams?\b",
            r"\bnhom ky thuat\b",
            r"\bmo hinh hoa he thong\b",
            r"\bobjects\b",
            r"\bproperties\b",
            r"\buser requirements\b",
            r"\bkiem thu san pham\b",
            r"\bnghiem thu san pham\b",
            r"\bui/ux\b",
            r"\bui ux\b",
            r"\buser stories\b",
            r"\bodoo\b",
            r"\bsap\b",
            r"\bmicrosoft dynamics\b",
            r"\bphong it\b",
        ]

        if not _has_any(evidence, strong_it_evidence):
            return None

    it_evidence = [
        r"\bsoftware\b",
        r"\bphan mem\b",
        r"\bapplication\b",
        r"\bweb site\b",
        r"\bwebsite\b",
        r"\bmobile app\b",
        r"\bhe thong\b",
        r"\bsystem\b",
        r"\bphat trien phan mem\b",
        r"\bsoftware development\b",
        r"\bbrd\b",
        r"\bsrs\b",
        r"\burd\b",
        r"\buse case\b",
        r"\bwireframe\b",
        r"\bproduct backlog\b",
        r"\bscrum\b",
        r"\bsprint\b",
        r"\bproduct owner\b",
        r"\bdata warehouse\b",
        r"\bbi\b",
        r"\berp\b",
        r"\bfintech\b",
        r"\bapi\b",
        r"\bdeveloper\b",
        r"\bdoi ngu ky thuat\b",
        r"\bteam ky thuat\b",
        r"\bnhom phat trien\b",
        r"\bkiem thu\b",
        r"\btech company\b",
        r"\bcomputer science\b",
        r"\bengineering\b",
        r"\bmobile application\b",
        r"\bmobile apps?\b",
        r"\bai platform\b",
        r"\bproduct development\b",
        r"\bphat trien ung dung\b",
        r"\bcong ty cong nghe\b",
        r"\bcong ty phat trien ung dung\b",
        r"\bky su\b",
        r"\bnhom ky thuat\b",
    ]

    if _has_any(evidence, it_evidence):
        return category

    return None

def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Apply the ordered TV2 filters and return candidate rows and model-ready data."""
    required = {
        "job_title", "location", "country", "salary", "contract_type",
        "experience_required", "description", "category",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required source columns: {sorted(missing)}")

    work = df.copy()
    work["row_id"] = work.index
    work["job_title"] = work["job_title"].fillna("").astype(str).str.strip()
    counts: dict[str, object] = {"input_rows": len(work)}

    # 1. Country
    work = work.loc[work["country"].fillna("").astype(str).str.strip().eq("Việt Nam")].copy()
    counts["after_vietnam"] = len(work)

    # 2. Classify by role; PM/BA/PO titles are gated by description evidence.
    work["job_category"] = work.apply(_classify_row, axis=1)
    candidates = work.loc[work["job_category"].notna()].copy()
    counts["after_it_title"] = len(candidates)

    # 3. Negotiable / missing salary.
    salary_pairs = candidates["salary"].apply(parse_salary)
    candidates[["salary_min", "salary_max"]] = pd.DataFrame(
        salary_pairs.tolist(), index=candidates.index
    )
    candidates = candidates.loc[
        candidates["salary_min"].notna() & candidates["salary_max"].notna()
    ].copy()
    counts["after_negotiable_salary"] = len(candidates)

    # 4. The leader requires a genuine range, not a single salary number.
    candidates = candidates.loc[candidates["salary_min"].lt(candidates["salary_max"])].copy()
    counts["after_single_salary"] = len(candidates)

    # 5. Validate order and parseability; reversed endpoints are invalid, never swapped.
    candidates = candidates.loc[
        candidates["salary_min"].le(candidates["salary_max"])
    ].copy()
    counts["after_invalid_salary"] = len(candidates)

    # 6. Full-time only.
    candidates = candidates.loc[
        candidates["contract_type"].fillna("").astype(str).str.strip().eq("Toàn thời gian")
    ].copy()
    counts["after_full_time"] = len(candidates)

    # 7. Monthly salary bounds.
    candidates = candidates.loc[
        candidates["salary_min"].between(3, 200)
        & candidates["salary_max"].between(3, 200)
    ].copy()
    counts["after_salary_bounds"] = len(candidates)

    # 8. Discard implausibly broad ranges.
    candidates = candidates.loc[
        candidates["salary_max"].div(candidates["salary_min"]).le(5)
    ].copy()
    counts["after_salary_ratio"] = len(candidates)

    candidates["experience_years"] = candidates["experience_required"].apply(parse_experience_years)
    candidates["experience_group"] = candidates["experience_years"].apply(experience_group)
    candidates["seniority"] = candidates["job_title"].apply(seniority_from_title)
    candidates["city"] = candidates["location"].apply(map_city)

    skill_source = next(
        (column for column in ("technical_skills", "qualifications", "requirements_text") if column in candidates),
        None,
    )
    candidates["skills"] = (
        candidates[skill_source].apply(lambda value: "|".join(parse_skills(value)))
        if skill_source else ""
    )
    candidates["salary_raw"] = candidates["salary"].astype(str)
    candidates["salary_label"] = (candidates["salary_min"] + candidates["salary_max"]) / 2

    desc_norm = candidates["description"].apply(_normalised_description)
    candidates["desc_group"] = desc_norm.apply(
        lambda value: __import__("hashlib").sha1(value.encode("utf-8")).hexdigest()[:12]
    )

    def norm_col(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).map(_normalised_description)

    duplicate_key = pd.DataFrame({
        "title": norm_col(candidates["job_title"]),
        "location": norm_col(candidates["location"]),
        "salary": norm_col(candidates["salary"]),
        "description": desc_norm,
    }, index=candidates.index)
        # Diagnostic: inspect duplicate groups before removing them.
    duplicate_report = duplicate_key.copy()
    duplicate_report["row_id"] = candidates["row_id"]
    duplicate_report["job_title"] = candidates["job_title"]
    duplicate_report["salary"] = candidates["salary"]

    duplicate_groups = (
        duplicate_key
        .groupby(list(duplicate_key.columns), dropna=False)
        .size()
    )
    duplicate_groups = duplicate_groups[duplicate_groups > 1].sort_values(
        ascending=False
    )

    print("\n--- DUPLICATE CHECK ---")
    print("Rows before dedup:", len(candidates))
    print("Duplicate groups:", len(duplicate_groups))
    print("Rows in duplicate groups:", int(duplicate_groups.sum()))
    print("Rows removed by dedup:", int((duplicate_groups - 1).sum()))

    if not duplicate_groups.empty:
        print("\nTop 10 duplicate groups:")
        for keys, size in duplicate_groups.head(10).items():
            mask = pd.Series(True, index=duplicate_key.index)

            for column, value in zip(duplicate_key.columns, keys):
                mask &= duplicate_key[column].eq(value)

            sample = candidates.loc[mask].iloc[0]

            print(
                f"- {size} rows | "
                f"Title: {sample['job_title']} | "
                f"Salary: {sample['salary']} | "
                f"Row IDs: {candidates.loc[mask, 'row_id'].tolist()}"
            )

    candidates = candidates.loc[
        ~duplicate_key.duplicated(keep="first")
    ].copy()
    counts["after_deduplication"] = len(candidates)

    output_columns = [
        "row_id", "job_title", "job_category", "city", "experience_years",
        "experience_group", "seniority", "skills", "salary_raw", "salary_min",
        "salary_max", "salary_label", "desc_group",
    ]
    cleaned = candidates[output_columns].reset_index(drop=True)
    checks = self_check(cleaned)
    checks.update(counts)
    return work.loc[work["job_category"].notna()].copy(), cleaned, checks


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
