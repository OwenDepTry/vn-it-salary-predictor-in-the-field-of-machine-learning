# src/data_cleaning.py
"""GĐ2 — Làm sạch VietJobs thành tập IT dùng cho mô hình lương.

Chạy từ thư mục gốc:  python src/data_cleaning.py
Đầu vào : data/raw/vietjobs/VietJobs.csv (không bị sửa)
Đầu ra  : data/interim/vietjobs_it_candidates.csv  (mọi ứng viên IT + lý do giữ/loại, để rà soát)
          data/processed/it_salary_clean.csv       (tập dùng cho mô hình)
          reports/gd2_cleaning_log.txt              (log + số liệu)
Quy tắc chi tiết: reports/cleaning_rules.md

Nguyên tắc:
- Giữ cột gốc (salary_raw, location_raw, experience_raw, job_title, category_src, *_src).
- Nhãn salary_label = trung điểm (salary_min + salary_max) / 2 — GIẢ ĐỊNH xấp xỉ.
- Chỉ dùng khoảng đóng "a - b triệu"; loại thỏa thuận ("0.0 triệu"), điểm đơn, 0–b, a>b.
- Bước này KHÔNG học gì từ dữ liệu (không chọn kỹ năng theo tần suất, không impute):
  mọi bước "fit" làm ở GĐ3 sau khi chia train/test.
"""
import hashlib
import re
import sys
import unicodedata
import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "vietjobs" / "VietJobs.csv"
INTERIM_PATH = ROOT / "data" / "interim" / "vietjobs_it_candidates.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "it_salary_clean.csv"
LOG_PATH = ROOT / "reports" / "gd2_cleaning_log.txt"

IT_CATEGORY = "công_nghệ_thông_tin_kỹ_thuật_số"
MIN_PLAUSIBLE = 3.0    # triệu/tháng
MAX_PLAUSIBLE = 200.0  # triệu/tháng
MAX_WIDTH_RATIO = 5.0  # max/min > 5 (vd "5 - 200 triệu"): trung điểm không còn là ước lượng có nghĩa
SOURCE = "vietjobs"

# ----------------------------------------------------------------------------- tiện ích văn bản


def to_nfc_lower(s) -> str:
    t = "" if pd.isna(s) else str(s)
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t)).strip().lower()


def to_ascii_lower(s) -> str:
    t = "" if pd.isna(s) else str(s)
    t = t.replace("đ", "d").replace("Đ", "D")
    t = unicodedata.normalize("NFD", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", t).strip().lower()


def rx(*parts: str) -> re.Pattern:
    return re.compile("|".join(parts))


# ----------------------------------------------------------------------------- lương
SAL_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*triệu\s*$", re.I)


def parse_salary(s):
    """Trả về (min, max, status). Đơn vị: triệu VND/tháng (GĐ1a: không có USD, /năm, /giờ)."""
    if pd.isna(s):
        return np.nan, np.nan, "missing"
    m = SAL_RE.match(unicodedata.normalize("NFC", str(s)))
    if not m:
        return np.nan, np.nan, "unparsed"
    a = float(m.group(1))
    if m.group(2) is None:
        return (np.nan, np.nan, "negotiable") if a == 0 else (a, a, "single_value")
    b = float(m.group(2))
    if a == 0 and b == 0:
        return np.nan, np.nan, "negotiable"
    if a == 0:
        return a, b, "zero_lower"
    if a > b:
        return a, b, "inverted"
    return a, b, "range"


# ----------------------------------------------------------------------------- kinh nghiệm


def parse_experience_years(s) -> float:
    t = to_ascii_lower(s)
    if "khong yeu cau" in t:
        return 0.0
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(nam|thang)$", t)
    if not m:
        return np.nan
    v = float(m.group(1))
    return v if m.group(2) == "nam" else round(v / 12, 2)


def experience_group(years: float) -> str:
    if pd.isna(years):
        return "unknown"
    if years == 0:
        return "0 (không yêu cầu)"
    if years < 1:
        return "<1 năm"
    if years < 2:
        return "1 năm"
    if years < 3:
        return "2 năm"
    if years < 4:
        return "3 năm"
    return "4+ năm"


EXPERIENCE_ORDER = ["0 (không yêu cầu)", "<1 năm", "1 năm", "2 năm", "3 năm", "4+ năm", "unknown"]

# ----------------------------------------------------------------------------- bộ lọc IT
# Thứ tự: (1) loại trừ -> (2) vai trò IT theo thứ tự ưu tiên.
# a = chức danh bỏ dấu (lowercase); t = chức danh có dấu (NFC, lowercase); raw = chức danh gốc.
# Từ dễ va chạm khi bỏ dấu (kiểm thử/kiêm thủ, SAP/sắp, AI/ai) được so trên t hoặc raw.

EXCLUDE_RULES = [
    ("kinh doanh/bán hàng/marketing",
     lambda a, t, raw: bool(re.search(r"kinh doanh|ban hang|\b(pre-?)?sales?\b|telesales?|marketing|\bseo\b|cham soc khach hang|\bcskh\b"
                                       r"|phat trien khach hang|phat trien thi truong|ket noi doanh nghiep", a))
     and not re.search(r"phan tich kinh doanh", a)),
    ("tư vấn (không phải triển khai)",
     lambda a, t, raw: "tu van" in a and "trien khai" not in a),
    ("thiết kế/đồ họa/sáng tạo nội dung",
     lambda a, t, raw: bool(re.search(
         r"thiet ke|do hoa|designer|\bdesign\b|ui/?ux|\b3d\b|\b2d\b|noi that|illustrat|video|editor|bien tap"
         r"|photograph|nhiep anh|content|noi dung|livestream|kol|koc|idea creator|trinh chinh sua"
         r"|chinh sua tai lieu", a))),
    ("gán nhãn/huấn luyện dữ liệu AI",
     lambda a, t, raw: bool(re.search(r"gan nhan|annotat|huan luyen vien ai|ai trainer|ai tutor", a))),
    ("lập trình máy công cụ/tự động hóa/thiết bị",
     lambda a, t, raw: bool(re.search(r"\bcnc\b|\bplc\b|phay|máy tiện|tiện,|tiện\b|may tien|robot han|khuon"
                                      r"|smart ?home|\bknx\b", t + " " + a))),
    ("an ninh/bảo vệ",
     lambda a, t, raw: bool(re.search(r"an ninh|bao ve|giam sat camera", a))),
    ("kế toán/hành chính/văn phòng",
     lambda a, t, raw: bool(re.search(r"ke toan|thu kho|thu ngan|hanh chinh|nhap lieu|le tan", a))
     or ("van phong" in a and not re.search(r"\bIT\b", raw))),
    ("giảng dạy",
     lambda a, t, raw: bool(re.search(r"giao vien|giang vien|tro giang|teacher|tutor|mentor", a))
     or bool(re.search(r"\bdạy\b", t))),
    ("bất động sản",
     lambda a, t, raw: "bat dong san" in a),
]

# (job_category, job_role, cần category CNTT?, điều kiện)
ROLE_RULES = [
    ("Quản lý & phân tích (PM/BA/PO)", "brse_comtor", False,
     lambda a, t, raw: bool(re.search(r"\bbrse\b|cau noi|comtor", a))),
    ("Quản lý & phân tích (PM/BA/PO)", "scrum_master", False,
     lambda a, t, raw: "scrum master" in a),
    ("Quản lý & phân tích (PM/BA/PO)", "it_pm_ba", False,
     lambda a, t, raw: bool(re.search(r"\bIT\b", raw)) and bool(re.search(
         r"project manager|business analyst|quan ly du an|product owner|product manager", a))),
    ("Quản lý & phân tích (PM/BA/PO)", "pm_ba_po", True,
     lambda a, t, raw: bool(re.search(
         r"project manager|quan ly du an|product owner|chu so huu san pham|product manager|quan ly san pham"
         r"|business analyst|\bba\b|phan tich nghiep vu|phan tich kinh doanh|giam doc du an|delivery manager", a))),
    ("Triển khai & ERP", "erp_implementation", False,
     lambda a, t, raw: bool(re.search(r"\berp\b|odoo|trien khai phan mem|trien khai he thong|implementation", a))
     or bool(re.search(r"\bSAP\b", raw))),
    ("Triển khai & ERP", "implementation_generic", True,
     lambda a, t, raw: "trien khai" in a),
    ("Kiểm thử & QA", "tester", False,
     lambda a, t, raw: bool(re.search(r"tester|automation test|qa engineer|qc engineer|software qa|\bsdet\b|test lead|qa lead"
                                     r"|test engineer|performance test|manual test", a))
     or bool(re.search(r"kiểm thử|kiểm tra phần mềm|chất lượng phần mềm", t))),
    ("Kiểm thử & QA", "tester", True,  # chức danh dịch máy: "người thử nghiệm", "người kiểm tra ..."
     lambda a, t, raw: bool(re.search(r"người thử nghiệm|người kiểm tra", t))),
    ("Kiểm thử & QA", "qa_qc_generic", True,
     lambda a, t, raw: bool(re.search(r"\bqa\b|\bqc\b", a))),
    ("Dữ liệu & AI", "data_ai", False,
     lambda a, t, raw: bool(re.search(
         r"data engineer|data scientist|machine learning|\bml\b|deep learning|computer vision|\bnlp\b|\bllm\b"
         r"|big data|ky su du lieu|khoa hoc du lieu|tri tue nhan tao|\bbi developer|power bi developer", a))
     or bool(re.search(r"\bAI (Engineer|Developer|Researcher|Specialist|Architect|Intern)|Kỹ [Ss]ư AI|AI/ML|ML/AI"
                       r"|(Thực [Tt]ập [Ss]inh|Intern|Chuyên [Vv]iên) AI\b", raw))),
    ("Dữ liệu & AI", "data_analyst", True,
     lambda a, t, raw: bool(re.search(r"data analyst|phan tich du lieu|\bbi\b|business intelligence", a))),
    ("Hạ tầng, DevOps & bảo mật", "devops_infra_security", False,
     lambda a, t, raw: bool(re.search(
         r"devops|\bsre\b|\bcloud\b|sysadmin|system admin|system engineer|quan tri he thong|quan tri mang"
         r"|network|ky su mang|ky thuat mang|cyber|bao mat|security|pentest|\bdba\b|database admin"
         r"|quan tri co so du lieu|\blinux\b|kubernetes|ha tang (it|mang|cntt|cong nghe)", a))
     or bool(re.search(r"\bSOC\b", raw))),
    ("Hỗ trợ kỹ thuật & IT chung", "it_support", False,
     lambda a, t, raw: bool(re.search(r"helpdesk|it support|ky thuat vien it|ky thuat may tinh|\bcntt\b"
                                      r"|cong nghe thong tin|tin hoc", a))
     or bool(re.search(r"\bIT\b", raw))),
    ("Hỗ trợ kỹ thuật & IT chung", "tech_support_generic", True,
     lambda a, t, raw: bool(re.search(r"ho tro ky thuat|technical support", a))),
    ("Phát triển phần mềm", "dev", False,
     lambda a, t, raw: bool(re.search(
         r"developer|\bdev\b|lap trinh|programmer|software|phan mem|nha phat trien|back[\s-]?end|front[\s-]?end"
         r"|full[\s-]?stack|mobile|\bios\b|android|flutter|react|\bjava\b|python|\.net|\bphp\b|golang|node\.?js"
         r"|c\+\+|c#|ruby|unity|unreal|game dev|embedded|lap trinh nhung|firmware|blockchain|tech lead"
         r"|lanh dao ky thuat|kien truc su phan mem|kien truc su giai phap|solution architect|software architect"
         r"|\bcto\b", a))),
    ("Phát triển phần mềm", "dev", True,
     lambda a, t, raw: bool(re.search(r"engineer|ky su|chuyen vien ky thuat|ky thuat dan dau|truong nhom cong nghe", a))),
]

DEV_SUBROLES = [
    ("fullstack", r"full[\s-]?stack|day du"),
    ("mobile", r"mobile|\bios\b|android|flutter|react native|kotlin|swift"),
    ("game", r"game|unity|unreal|thong nhat"),
    ("embedded", r"embedded|nhung|firmware|\biot\b"),
    ("frontend", r"front[\s-]?end|giao dien nguoi dung|\breact(js)?\b|vue|angular|ui developer"),
    ("backend", r"back[\s-]?end|phu tro|server|\bjava\b|\.net|\bphp\b|golang|node\.?js|python|ruby|c#"),
    ("web", r"\bweb\b"),
]


def classify_title(title, category):
    """Trả về (is_it, job_category, job_role, reason)."""
    raw = "" if pd.isna(title) else unicodedata.normalize("NFC", str(title))
    a, t = to_ascii_lower(raw), to_nfc_lower(raw)
    for reason, cond in EXCLUDE_RULES:
        if cond(a, t, raw):
            return False, None, None, f"loại: {reason}"
    for job_cat, role, needs_cntt, cond in ROLE_RULES:
        if cond(a, t, raw):
            if needs_cntt and category != IT_CATEGORY:
                return False, None, None, f"mơ hồ ngoài nhóm CNTT: {role}"
            if role == "dev":
                role = next((name for name, pat in DEV_SUBROLES if re.search(pat, a)), "dev_general")
            return True, job_cat, role, "giữ"
    return False, None, None, "không khớp vai trò IT"


SENIORITY_RULES = [
    ("manager", r"(?<!project )(?<!product )(?<!account )manager|truong phong|pho phong|head of|\bhead\b|director|giam doc|\bcto\b"
                r"|quan ly(?! du an| he thong| co so| mang| san pham)"),
    ("lead", r"\blead\b|leader|truong nhom|lanh dao ky thuat|dan dau|principal"),
    ("senior", r"senior|\bsr\b|cao cap"),
    ("middle", r"\bmiddle\b|\bmid\b"),
    ("junior", r"junior|\bjr\b"),
    ("fresher", r"fresher|moi tot nghiep|moi ra truong"),
    ("intern", r"intern|thuc tap"),
]


def seniority_from_title(title) -> str:
    a = to_ascii_lower(title)
    return next((label for label, pat in SENIORITY_RULES if re.search(pat, a)), "unknown")


# ----------------------------------------------------------------------------- thành phố
# Theo địa danh ghi trong tin (KHÔNG quy đổi theo địa giới sau sáp nhập tỉnh 7/2025).
HN_KEYS = rx(
    r"\bha noi\b", r"\bhanoi\b", r"ha dong", r"cau giay", r"tu liem", r"dong da", r"hai ba trung", r"hoan kiem",
    r"ba dinh", r"thanh xuan", r"hoang mai", r"long bien", r"tay ho", r"gia lam", r"dong anh", r"soc son",
    r"thanh tri", r"hoai duc", r"dan phuong", r"thach that", r"quoc oai", r"chuong my", r"me linh", r"son tay",
    r"\bba vi\b", r"phuc tho", r"thanh oai", r"thuong tin", r"phu xuyen", r"ung hoa", r"my duc",
)
HCM_KEYS = rx(
    r"ho chi minh", r"\bhcm\b", r"tp\.? ?hcm", r"sai gon", r"saigon", r"thu duc", r"binh thanh", r"tan binh",
    r"tan phu", r"go vap", r"phu nhuan", r"binh tan", r"\bquan \d{1,2}\b", r"nha be", r"cu chi", r"hoc mon",
    r"binh chanh", r"can gio",
)
DN_KEYS = rx(r"da nang", r"hai chau", r"thanh khe", r"son tra", r"ngu hanh son", r"lien chieu", r"cam le")


def map_city(location) -> str:
    parts = [p for p in re.split(r"[,;/]| - ", to_ascii_lower(location)) if p.strip()]
    groups = set()
    for p in parts:
        if HN_KEYS.search(p):
            groups.add("Hà Nội")
        elif HCM_KEYS.search(p):
            groups.add("TP.HCM")
        elif DN_KEYS.search(p):
            groups.add("Đà Nẵng")
        else:
            groups.add("Khác")
    if not groups:
        return "Khác"
    named = groups - {"Khác"}
    if len(named) > 1:
        return "Nhiều thành phố"
    return named.pop() if named else "Khác"


# ----------------------------------------------------------------------------- kỹ năng
SKILL_SYNONYMS = {
    "reactjs": "react", "react.js": "react", "react js": "react",
    "nodejs": "nodejs", "node.js": "nodejs", "node js": "nodejs", "node": "nodejs",
    "vuejs": "vue", "vue.js": "vue", "vue js": "vue", "angularjs": "angular",
    "postgres": "postgresql", "postgre": "postgresql", "postgre sql": "postgresql",
    "ms sql": "sql server", "mssql": "sql server", "microsoft sql server": "sql server", "ms sql server": "sql server",
    "restful api": "rest api", "restful apis": "rest api", "restful": "rest api", "rest": "rest api",
    "rest apis": "rest api", "api restful": "rest api",
    "c sharp": "c#", ".net core": ".net", "asp.net": ".net", "asp.net core": ".net", "dotnet": ".net",
    ".net framework": ".net", "net core": ".net",
    "golang": "go", "k8s": "kubernetes", "amazon web services": "aws",
    "html5": "html", "css3": "css", "js": "javascript", "ts": "typescript",
    "cicd": "ci/cd", "ci cd": "ci/cd", "mongo": "mongodb",
    "github": "git", "gitlab": "git", "git/github": "git",
    "linux/unix": "linux", "ml": "machine learning",
    "adobe photoshop": "photoshop", "adobe illustrator": "illustrator", "after effect": "after effects",
    "adobe after effects": "after effects", "adobe premiere": "premiere", "premiere pro": "premiere",
    "ms excel": "excel", "microsoft excel": "excel",
}


def normalize_skill(s: str) -> str:
    k = re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip().lower()
    k = k.strip(";:,-").rstrip(".").strip()  # KHÔNG bỏ dấu chấm đầu (".net")
    if k == "net":
        k = ".net"
    a = to_ascii_lower(k)
    if re.search(r"tin hoc van phong|vi tinh van phong|microsoft office|\bms office\b", a):
        return "tin học văn phòng"
    return SKILL_SYNONYMS.get(k, k)


def parse_skills(s) -> list:
    if pd.isna(s):
        return []
    txt = str(s).strip()
    items = None
    if txt.startswith("["):
        try:
            v = ast.literal_eval(txt)
            if isinstance(v, (list, tuple)):
                items = [str(x) for x in v]
        except (ValueError, SyntaxError):
            items = None
    if items is None:
        items = txt.strip("[]").split(",")
    out = []
    for x in items:
        k = normalize_skill(x)
        if k and k not in out:
            out.append(k.replace("|", "/"))
    return out


# ----------------------------------------------------------------------------- tự kiểm tra quy tắc
SELF_CHECK = [  # (chức danh, category, kỳ vọng is_it, kỳ vọng job_role hoặc None)
    ("Kế Toán Kho Kiêm Thủ Kho", "x", False, None),
    ("Nhân Viên Pha Chế Kiêm Thu Ngân (Part-Time)", "x", False, None),
    ("Nhân Viên Sắp Xếp Phòng Trị Liệu (Có Đào Tạo) - Đi Làm Ngay", "x", False, None),
    ("Product Photographer (Biết Sử Dụng AI)", "x", False, None),
    ("Nhân Viên SEO Web Bằng AI", "x", False, None),
    ("Nhân Viên Lập Trình Phay, Tiện", "x", False, None),
    ("Nhân Viên Lập Trình Vận Hành Máy Tiện CNC", "x", False, None),
    ("Security Manager - Phụ Trách An Ninh", "x", False, None),
    ("Nhân Viên Kinh Doanh Phần Mềm", IT_CATEGORY, False, None),
    ("Nhân Viên Thiết Kế Đồ Họa", IT_CATEGORY, False, None),
    ("UI/UX Designer", IT_CATEGORY, False, None),
    ("Kỹ sư phần mềm", "x", True, "dev_general"),
    ("Nhà phát triển thống nhất", IT_CATEGORY, True, "game"),
    ("Nhà phát triển phụ trợ cao cấp", IT_CATEGORY, True, "backend"),
    ("Lãnh đạo kỹ thuật", IT_CATEGORY, True, "dev_general"),
    ("Chuyên Viên Kiểm Thử Phần Mềm", IT_CATEGORY, True, "tester"),
    ("Data Analyst", IT_CATEGORY, True, "data_analyst"),
    ("Data Analyst", "x", False, None),
    ("Nhà phân tích kinh doanh", IT_CATEGORY, True, "pm_ba_po"),
    ("Nhà phân tích kinh doanh", "x", False, None),
    ("Nhân Viên IT Helpdesk", "x", True, "it_support"),
    ("Data Scientist", "x", True, "data_ai"),
    ("Senior Java Developer", "x", True, "backend"),
    ("DevOps Engineer", "x", True, "devops_infra_security"),
    # bổ sung sau khi rà log GĐ2
    ("Test Lead", IT_CATEGORY, True, "tester"),
    ("Người thử nghiệm Fresher (lên đến 14m)", IT_CATEGORY, True, "tester"),
    ("Người kiểm tra tự động hóa", IT_CATEGORY, True, "tester"),
    ("Kiến trúc sư giải pháp", IT_CATEGORY, True, "dev_general"),
    ("Kỹ thuật dẫn đầu", IT_CATEGORY, True, "dev_general"),
    ("Thực Tập Sinh AI", IT_CATEGORY, True, "data_ai"),
    ("Huấn luyện viên AI - Âm thanh (tiếng Anh C1)", IT_CATEGORY, False, None),
    ("Nhân Viên Gán Nhãn Dữ Liệu Tiếng Anh (Cấp Độ C1)", IT_CATEGORY, False, None),
    ("Chuyên Viên Presales Giải Pháp Phần Mềm", "x", False, None),
    ("Chuyên Viên Phát Triển Khách Hàng Phần Mềm Social", "x", False, None),
    ("Junior Idea Creator Game Mobile", "x", False, None),
    ("Trình chỉnh sửa tài liệu E-Learning (Sản phẩm phần mềm)", "x", False, None),
    ("Kỹ Sư Triển Khai Hệ Thống Smarthome (KNX)", "x", False, None),
    ("Nhân Viên Giám Sát Camera", IT_CATEGORY, False, None),
]


def self_check() -> list:
    errors = []
    for title, cat, exp_it, exp_role in SELF_CHECK:
        is_it, _, role, reason = classify_title(title, cat)
        if is_it != exp_it or (exp_role is not None and role != exp_role):
            errors.append(f"  '{title}' (category={cat}): được is_it={is_it}, role={role}, lý do='{reason}' "
                          f"| kỳ vọng is_it={exp_it}, role={exp_role}")
    sen_cases = [("Nhà phát triển phụ trợ cao cấp", "senior"), ("Project Manager", "unknown"),
                 ("Trưởng Nhóm Lập Trình", "lead"), ("Thực Tập Sinh Lập Trình", "intern"),
                 ("Kỹ thuật dẫn đầu", "lead")]
    for title, exp in sen_cases:
        got = seniority_from_title(title)
        if got != exp:
            errors.append(f"  seniority '{title}': được {got}, kỳ vọng {exp}")
    got_sk = parse_skills("['.NET', 'ASP.NET Core', 'ReactJS', 'Node.js.']")
    if got_sk != [".net", "react", "nodejs"]:
        errors.append(f"  skills: được {got_sk}, kỳ vọng ['.net', 'react', 'nodejs']")
    return errors


# ----------------------------------------------------------------------------- pipeline
_lines: list = []


def log(*parts) -> None:
    text = " ".join(str(p) for p in parts)
    print(text)
    _lines.append(text)


def section(title: str) -> None:
    log("")
    log("=" * 90)
    log(title)
    log("=" * 90)


def main() -> int:
    section("0. TỰ KIỂM TRA QUY TẮC")
    errors = self_check()
    if errors:
        log("THẤT BẠI — quy tắc phân loại không như kỳ vọng:")
        for e in errors:
            log(e)
        return 1
    log(f"OK: {len(SELF_CHECK)} ca chức danh + 5 ca cấp bậc + 1 ca kỹ năng đúng kỳ vọng.")

    section("1. ĐỌC DỮ LIỆU GỐC")
    df = pd.read_csv(RAW_PATH, encoding="utf-8-sig", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"salary_min": "salary_min_src", "salary_max": "salary_max_src",
                            "salary_avg": "salary_avg_src"})  # cột int gốc (floor), giữ lại để đối chiếu
    df.insert(0, "row_id", np.arange(len(df)))
    log(f"{len(df):,} bản ghi, {df.shape[1] - 1} cột gốc")

    section("2. PHÂN LOẠI IT THEO CHỨC DANH")
    cls = [classify_title(t, c) for t, c in zip(df["job_title"], df["category"])]
    df["is_it"] = [c[0] for c in cls]
    df["job_category"] = [c[1] for c in cls]
    df["job_role"] = [c[2] for c in cls]
    df["it_reason"] = [c[3] for c in cls]
    in_cat = df["category"].eq(IT_CATEGORY)
    log(f"is_it: {int(df['is_it'].sum()):,} | trong đó thuộc category CNTT: {int((df['is_it'] & in_cat).sum()):,} "
        f"| ngoài CNTT: {int((df['is_it'] & ~in_cat).sum()):,}")
    log(f"Category CNTT nhưng bị loại: {int((~df['is_it'] & in_cat).sum()):,}. Lý do:")
    log(df.loc[~df["is_it"] & in_cat, "it_reason"].value_counts().to_string())
    log("\nTop 25 chức danh category CNTT bị loại vì 'không khớp vai trò IT' (rà soát xem có bỏ sót IT không):")
    log(df.loc[in_cat & df["it_reason"].eq("không khớp vai trò IT"), "job_title"].value_counts().head(25).to_string())
    log("\nTop 25 chức danh được giữ từ NGOÀI category CNTT (rà soát xem có bắt nhầm không):")
    log(df.loc[df["is_it"] & ~in_cat, "job_title"].value_counts().head(25).to_string())

    section("3. CHUẨN HÓA CÁC TRƯỜNG")
    sal = df["salary"].map(parse_salary)
    df["salary_raw"] = df["salary"]
    df["salary_min"] = [s[0] for s in sal]
    df["salary_max"] = [s[1] for s in sal]
    df["salary_status"] = [s[2] for s in sal]
    df["experience_raw"] = df["experience_required"]
    df["experience_years"] = df["experience_raw"].map(parse_experience_years)
    df["experience_group"] = df["experience_years"].map(experience_group)
    df["seniority"] = df["job_title"].map(seniority_from_title)
    df["location_raw"] = df["location"]
    df["city"] = df["location_raw"].map(map_city)
    skills = df["technical_skills"].map(parse_skills)
    df["skills"] = skills.map(lambda lst: "|".join(lst))
    df["n_skills"] = skills.map(len)
    df["job_title_norm"] = df["job_title"].map(to_nfc_lower)
    desc_norm = df["description"].map(to_nfc_lower)
    df["desc_group"] = desc_norm.map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:12])
    df["source"] = SOURCE
    df["date"] = pd.NA  # VietJobs không có ngày đăng (GĐ1a)
    df = df.rename(columns={"category": "category_src"})
    log("experience_years không parse được:", int(df["experience_years"].isna().sum()))
    log("salary_status:", df["salary_status"].value_counts().to_dict())

    section("4. LỌC DÒNG CHO MÔ HÌNH (lý do loại đầu tiên)")
    df["model_reason"] = "giữ"
    steps = [
        ("ngoài Việt Nam", ~df["country"].eq("Việt Nam")),
        ("không phải IT", ~df["is_it"]),
        ("lương thỏa thuận", df["salary_status"].eq("negotiable")),
        ("lương điểm đơn (không rõ cố định hay khoảng mở)", df["salary_status"].eq("single_value")),
        ("lương không hợp lệ (0–b, a>b, không parse)", df["salary_status"].isin(["zero_lower", "inverted", "unparsed", "missing"])),
        ("không phải Toàn thời gian", ~df["contract_type"].eq("Toàn thời gian")),
        (f"ngoài {MIN_PLAUSIBLE:g}–{MAX_PLAUSIBLE:g} triệu",
         (df["salary_min"] < MIN_PLAUSIBLE) | (df["salary_max"] > MAX_PLAUSIBLE)),
        (f"khoảng lương quá rộng (max/min > {MAX_WIDTH_RATIO:g})",
         df["salary_max"] > MAX_WIDTH_RATIO * df["salary_min"]),
    ]
    funnel = [("Tổng", len(df))]
    for reason, cond in steps:
        hit = df["model_reason"].eq("giữ") & cond.fillna(False)
        df.loc[hit, "model_reason"] = reason
        funnel.append((f"sau khi loại: {reason}", int(df["model_reason"].eq("giữ").sum())))
    keep = df["model_reason"].eq("giữ")
    dup_key = ["job_title_norm", "location_raw", "salary_raw", "desc_group"]
    # duplicated() tính trong tập đang giữ (giữ bản đầu tiên):
    kept_idx = df.index[keep]
    dup_in_kept = df.loc[kept_idx].duplicated(subset=dup_key, keep="first")
    df.loc[dup_in_kept[dup_in_kept].index, "model_reason"] = "trùng tin (tiêu đề+địa điểm+lương+mô tả)"
    funnel.append(("sau khi gộp trùng", int(df["model_reason"].eq("giữ").sum())))
    for name, n in funnel:
        log(f"   {name:<60} {n:>7,}")

    section("5. NHÃN")
    keep = df["model_reason"].eq("giữ")
    df["salary_label"] = np.where(keep, (df["salary_min"] + df["salary_max"]) / 2, np.nan)
    df["salary_width_ratio"] = df["salary_max"] / df["salary_min"]
    df["label_note"] = np.where(keep, "trung điểm khoảng (giả định)", "")
    log(df.loc[keep, "salary_label"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(2).to_string())

    out_cols = [
        "row_id", "source", "date", "job_title", "job_title_norm", "job_category", "job_role", "category_src",
        "city", "location_raw", "experience_raw", "experience_years", "experience_group", "seniority",
        "skills", "n_skills", "languages_required", "contract_type",
        "salary_raw", "salary_min", "salary_max", "salary_width_ratio", "salary_label", "label_note",
        "salary_min_src", "salary_max_src", "salary_avg_src", "desc_group",
    ]

    processed = df.loc[keep, out_cols].reset_index(drop=True)
    candidates_mask = df["is_it"] | df["category_src"].eq(IT_CATEGORY) | df["it_reason"].str.startswith("mơ hồ")
    interim = df.loc[candidates_mask, out_cols + ["is_it", "it_reason", "salary_status", "model_reason"]]

    section("6. TẬP SẠCH")
    log(f"Số tin dùng cho mô hình: {len(processed):,}")
    for col in ["job_category", "job_role", "city", "experience_group", "seniority"]:
        g = processed.groupby(col)["salary_label"].agg(n="size", median="median").sort_values("n", ascending=False)
        log(f"\n{col}:")
        log(g.round(1).to_string())
    log(f"\nCó kỹ năng: {100 * (processed['n_skills'] > 0).mean():.1f}% | số kỹ năng/tin trung vị: "
        f"{processed['n_skills'].median():.0f}")
    grp = processed["desc_group"].value_counts()
    log(f"Mô tả lặp lại: {int((grp > 1).sum()):,} nhóm, {int(grp[grp > 1].sum()):,} tin "
        f"-> GĐ3 chia train/test theo desc_group")
    unk_city = df.loc[keep & df["city"].eq("Khác"), "location_raw"].value_counts().head(20)
    log("\nĐịa điểm gốc được xếp 'Khác' (top 20, rà soát mapping):")
    log(unk_city.to_string())

    INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    interim.to_csv(INTERIM_PATH, index=False, encoding="utf-8-sig")
    processed.to_csv(PROCESSED_PATH, index=False, encoding="utf-8-sig")
    log(f"\nĐã ghi: {INTERIM_PATH.relative_to(ROOT)} ({len(interim):,} dòng)")
    log(f"Đã ghi: {PROCESSED_PATH.relative_to(ROOT)} ({len(processed):,} dòng, {processed.shape[1]} cột)")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(_lines), encoding="utf-8")
    print(f"Log: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
