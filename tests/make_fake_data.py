"""Tạo dữ liệu GIẢ đúng schema của data/processed/it_salary_clean.csv (đầu ra GĐ2 của TV2).

Mục đích: chạy thử train.py / evaluate.py end-to-end khi chưa tải dữ liệu thật
(`python src/download_data.py && python src/data_cleaning.py`).
KHÔNG dùng số liệu từ dữ liệu này cho báo cáo.

split.csv KHÔNG được tạo ở đây: features.get_split() tự tạo ở lần chạy train.py đầu tiên.

Chạy từ thư mục gốc repo:
    python tests/make_fake_data.py            # 444 dòng, như dữ liệu thật
    python tests/make_fake_data.py --n 200 --out data/fake
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_cleaning import experience_group  # noqa: E402

RANDOM_STATE = 42
# Phân bố lấy gần đúng theo reports/gd2_cleaning_log.txt
JOBS = {"Phát triển phần mềm": (0.37, 3), "Quản lý & phân tích (PM/BA/PO)": (0.18, 3),
        "Hỗ trợ kỹ thuật & IT chung": (0.13, -5), "Hạ tầng, DevOps & bảo mật": (0.10, 2),
        "Kiểm thử & QA": (0.09, 1), "Triển khai & ERP": (0.08, -2), "Dữ liệu & AI": (0.05, 3)}
CITIES = {"Hà Nội": (0.71, 1.5), "TP.HCM": (0.19, 0.0), "Khác": (0.08, -4.0),
          "Đà Nẵng": (0.02, -1.0)}
SENIORITY = {"unknown": (0.83, 0), "lead": (0.04, 6), "junior": (0.03, -3), "intern": (0.03, -8),
             "senior": (0.03, 4), "manager": (0.02, 6), "middle": (0.01, 2), "fresher": (0.01, -6)}
EXP_YEARS = [0, 0.5, 1, 2, 3, 4, 5]
EXP_P = [0.10, 0.04, 0.21, 0.34, 0.25, 0.04, 0.02]
SKILLS = ["java", "javascript", "sql", "python", "react", "nodejs", "php", "c#", ".net",
          "docker", "aws", "linux", "git", "android", "ios", "flutter", "golang", "excel",
          "kubernetes", "erp"]
SKILL_BONUS = {"aws": 2.5, "kubernetes": 3, "golang": 2.5, "docker": 1.5,
               "python": 1, "react": 1, "excel": -1.5, "php": -1}


def _pick(rng, table):
    keys = list(table)
    p = np.array([table[k][0] for k in keys])
    return rng.choice(keys, p=p / p.sum())


def make_rows(n, rng):
    n_groups = int(n * 0.96)          # ~4% là tin đăng lại -> cùng mô tả (desc_group)
    group_of_row = np.sort(np.concatenate(
        [np.arange(n_groups), rng.integers(0, n_groups, n - n_groups)]))
    base = {}
    for g in range(n_groups):
        years = float(rng.choice(EXP_YEARS, p=EXP_P))
        job, city, sen = _pick(rng, JOBS), _pick(rng, CITIES), _pick(rng, SENIORITY)
        skills = sorted(rng.choice(SKILLS, size=int(rng.integers(0, 7)), replace=False))
        salary = (9 + 3.5 * years + JOBS[job][1] + CITIES[city][1] + SENIORITY[sen][1]
                  + sum(SKILL_BONUS.get(s, 0) for s in skills) + rng.normal(0, 4.5))
        lo = float(np.clip(round(salary - 3), 3, 150))
        base[g] = dict(job_category=job, city=city, experience_years=years,
                       experience_group=experience_group(years), seniority=sen,
                       skills="|".join(skills), salary_raw=f"{lo:g} - {lo + 6:g} triệu",
                       salary_label=lo + 3, job_title_norm=f"fake title {g}",
                       location_raw=city, desc_group=f"g{g:05d}")
    rows = [dict(base[g], row_id=i) for i, g in enumerate(group_of_row)]
    cols = ["row_id", "job_title_norm", "job_category", "city", "location_raw",
            "experience_years", "experience_group", "seniority", "skills",
            "salary_raw", "salary_label", "desc_group"]
    return pd.DataFrame(rows)[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=444)
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    df = make_rows(a.n, np.random.default_rng(RANDOM_STATE))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "it_salary_clean.csv", index=False, encoding="utf-8-sig")
    print(f"Đã ghi {len(df)} dòng GIẢ vào {out}/it_salary_clean.csv")
    print("Lương: min %.1f, trung vị %.1f, max %.1f" % (
        df.salary_label.min(), df.salary_label.median(), df.salary_label.max()))


if __name__ == "__main__":
    main()
