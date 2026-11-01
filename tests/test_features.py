import pandas as pd

from src.features import make_groups, get_split, SkillsMultiHot


def sample_data():
    return pd.DataFrame({
        "job_title": [
            "Backend Developer",
            "Backend Developer",
            "Data Analyst",
            "QA Engineer",
        ],
        "city_group": [
            "HCM",
            "HCM",
            "HN",
            "HN",
        ],
        "experience_years": [2, 2, 3, 1],
        "experience_group": [
            "1-3 years",
            "1-3 years",
            "3-5 years",
            "0-1 years",
        ],
        "seniority": [
            "Junior",
            "Junior",
            "Middle",
            "Junior",
        ],
        "salary_min": [10, 10, 15, 8],
        "salary_max": [20, 20, 25, 15],
        "description": [
            "Backend developer using NodeJS and PostgreSQL",
            "Backend developer using NodeJS and PostgreSQL",
            "Data analyst using SQL and Python",
            "QA engineer testing web applications",
        ],
        "requirements_text": [
            "Python NodeJS PostgreSQL",
            "Python NodeJS PostgreSQL",
            "SQL Python",
            "Selenium testing",
        ],
        "skills_list": [
            ["python", "nodejs"],
            ["python", "nodejs"],
            ["sql", "python"],
            ["selenium"],
        ],
    })
    

def test_make_groups():
    df = sample_data()
    groups = make_groups(df)

    assert len(groups) == len(df)
    assert groups.iloc[0] == groups.iloc[1]


def test_get_split_no_group_leakage():
    df = sample_data()

    groups = make_groups(df)
    split = get_split(df)

    assert len(split) == len(df)
    assert set(split.unique()) == {"train", "test"}

    assignment = pd.DataFrame({
        "group": groups,
        "split": split,
    })

    assert assignment.groupby("group")["split"].nunique().max() == 1

    assert (split == "train").sum() > 0
    assert (split == "test").sum() > 0


def test_skills_multihot():
    df = sample_data()

    encoder = SkillsMultiHot()
    encoder.fit(df["skills_list"])

    transformed = encoder.transform(df["skills_list"])

    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] > 0