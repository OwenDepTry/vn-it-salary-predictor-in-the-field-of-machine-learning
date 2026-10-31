"""BẢN TẠM (stub) của features.py, chỉ để TV3 chạy thử train.py / evaluate.py.

File thật do TV2 (Lê Ngọc Hiếu) viết. Khi TV2 bàn giao, XÓA file này và dùng
file của TV2. train.py / evaluate.py chỉ dùng các tên sau từ features.py:
    FEATURE_COLUMNS, TARGET, RANDOM_STATE, load_clean(), get_split(df),
    build_preprocessor()
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data/processed/it_salary_clean.csv"
SPLIT_PATH = ROOT / "data/processed/split.csv"

RANDOM_STATE = 42
TARGET = "salary_label"
CATEGORICAL = ["job_category", "city", "experience_group", "seniority"]
NUMERIC = ["experience_years"]
SKILLS = "skills"
FEATURE_COLUMNS = CATEGORICAL + NUMERIC + [SKILLS]


def load_clean(path=CLEAN_PATH):
    df = pd.read_csv(path)
    df["skills"] = df["skills"].fillna("")
    return df


def get_split(df, path=SPLIT_PATH, test_size=0.2):
    """Đọc split.csv nếu đã có; nếu chưa thì tạo một lần theo nhóm desc_group."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    rng = np.random.default_rng(RANDOM_STATE)
    groups = df["desc_group"].unique()
    rng.shuffle(groups)
    test_groups = set(groups[: int(round(len(groups) * test_size))])
    split = pd.DataFrame({"row_id": df["row_id"], "group_id": df["desc_group"]})
    split["split"] = np.where(split.group_id.isin(test_groups), "test", "train")
    path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(path, index=False)
    return split


class SkillsMultiHot(BaseEstimator, TransformerMixin):
    """Multi-hot cho cột kỹ năng; từ vựng chỉ học từ dữ liệu fit."""

    def __init__(self, min_count=3):
        self.min_count = min_count

    @staticmethod
    def _tokens(value):
        return [s.strip().lower() for s in str(value).split(",") if s.strip()]

    def fit(self, X, y=None):
        col = pd.DataFrame(X).iloc[:, 0]
        counts = pd.Series([t for v in col for t in set(self._tokens(v))]).value_counts()
        self.vocabulary_ = sorted(counts[counts >= self.min_count].index)
        return self

    def transform(self, X):
        col = pd.DataFrame(X).iloc[:, 0]
        index = {s: i for i, s in enumerate(self.vocabulary_)}
        out = np.zeros((len(col), len(index)))
        for r, v in enumerate(col):
            for t in self._tokens(v):
                if t in index:
                    out[r, index[t]] = 1
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"skill_{s}" for s in self.vocabulary_], dtype=object)


def build_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
        ("skills", SkillsMultiHot(), [SKILLS]),
    ])
