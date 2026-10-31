"""Tạo dữ liệu GIẢ đúng schema của it_salary_clean.csv và split.csv.

Mục đích: TV3 chạy thử train.py / evaluate.py khi TV2 chưa bàn giao dữ liệu thật.
KHÔNG dùng số liệu từ dữ liệu này cho báo cáo.

Chạy từ thư mục gốc repo:
    python tests/make_fake_data.py            # 449 dòng, như dữ liệu thật
    python tests/make_fake_data.py --n 200 --out data/fake
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
JOBS = ["backend", "frontend", "fullstack", "mobile", "data", "devops",
        "tester_qa", "it_support", "business_analyst"]
JOB_BONUS = {"backend": 4, "frontend": 2, "fullstack": 4, "mobile": 3, "data": 5,
             "devops": 6, "tester_qa": 0, "it_support": -3, "business_analyst": 2}
CITIES = ["Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Khác"]
CITY_P = [0.48, 0.38, 0.08, 0.06]
CITY_BONUS = {"Hồ Chí Minh": 1.5, "Hà Nội": 1.0, "Đà Nẵng": 0.0, "Khác": -2.0}
SKILLS = ["python", "java", "javascript", "react", "nodejs", "sql", "php", "c#",
          ".net", "docker", "aws", "linux", "git", "android", "ios", "flutter",
          "kotlin", "golang", "kubernetes", "excel"]
SKILL_BONUS = {"aws": 2.5, "kubernetes": 3, "golang": 2.5, "docker": 1.5,
               "python": 1, "react": 1, "excel": -1.5, "php": -1}


def experience_group(years):
    # Chép theo quy ước của TV2; khi có data_cleaning.py thật thì import hàm thật.
    if years < 1:
        return "0-1"
    if years < 3:
        return "1-3"
    if years < 5:
        return "3-5"
    return "5+"


def seniority_of(years, rng):
    if years <= 1:
        return rng.choice(["intern_fresher", "junior"], p=[0.6, 0.4])
    if years <= 3:
        return rng.choice(["junior", "middle"], p=[0.5, 0.5])
    if years <= 5:
        return rng.choice(["middle", "senior"], p=[0.5, 0.5])
    return rng.choice(["senior", "lead_manager"], p=[0.6, 0.4])


def make_rows(n, rng):
    rows = []
    n_groups = int(n * 0.8)          # ~20% là tin đăng lại -> cùng desc_group
    group_of_row = np.sort(np.concatenate(
        [np.arange(n_groups), rng.integers(0, n_groups, n - n_groups)]))
    base = {}
    for g in range(n_groups):
        years = int(rng.choice(range(7), p=[.18, .2, .2, .15, .1, .09, .08]))
        job = rng.choice(JOBS)
        k = int(rng.integers(1, 6))
        skills = sorted(rng.choice(SKILLS, size=k, replace=False))
        city = rng.choice(CITIES, p=CITY_P)
        sen = seniority_of(years, rng)
        salary = (9 + 3.2 * years + JOB_BONUS[job] + CITY_BONUS[city]
                  + sum(SKILL_BONUS.get(s, 0) for s in skills)
                  + rng.normal(0, 4.5))
        base[g] = dict(job_category=job, city=city, experience_years=years,
                       experience_group=experience_group(years), seniority=sen,
                       skills=",".join(skills),
                       salary_label=round(float(np.clip(salary, 5, 80)), 1))
    for i, g in enumerate(group_of_row):
        r = dict(base[g])
        r["row_id"] = i
        r["desc_group"] = int(g)
        rows.append(r)
    cols = ["row_id", "job_category", "city", "experience_years", "experience_group",
            "seniority", "skills", "salary_label", "desc_group"]
    return pd.DataFrame(rows)[cols]


def make_split(df, test_frac, rng):
    groups = df["desc_group"].unique()
    rng.shuffle(groups)
    test_groups = set(groups[: int(round(len(groups) * test_frac))])
    split = df[["row_id"]].copy()
    split["group_id"] = df["desc_group"]
    split["split"] = np.where(df["desc_group"].isin(test_groups), "test", "train")
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=449)
    ap.add_argument("--test-frac", type=float, default=86 / 449)
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    rng = np.random.default_rng(RANDOM_STATE)
    df = make_rows(a.n, rng)
    split = make_split(df, a.test_frac, rng)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "it_salary_clean.csv", index=False, encoding="utf-8")
    split.to_csv(out / "split.csv", index=False, encoding="utf-8")
    leak = set(split.query("split=='train'").group_id) & set(split.query("split=='test'").group_id)
    print(f"Đã ghi {len(df)} dòng vào {out}/it_salary_clean.csv")
    print(split["split"].value_counts().to_string())
    print("Nhóm nằm ở cả train và test:", len(leak))
    print("Lương: min %.1f, trung vị %.1f, max %.1f" % (
        df.salary_label.min(), df.salary_label.median(), df.salary_label.max()))


if __name__ == "__main__":
    main()
