"""
TV2 - Feature engineering for VietJobs salary prediction.

Run from project root:
    python src/features.py

Inputs:
    data/processed/it_salary_clean.csv

Outputs:
    data/processed/split.csv

The split is group-aware to reduce duplicate-posting leakage.
"""
from __future__ import annotations

import ast
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "it_salary_clean.csv"
SPLIT_FILE = ROOT / "data" / "processed" / "split.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def _normalise_text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return ""
    text = str(value).lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#./ ]+", " ", text)).strip()


def _as_skill_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        values = list(value)
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        values = []
    else:
        raw = str(value).strip()
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
            values = list(parsed) if isinstance(parsed, (list, tuple, set)) else [raw]
        except (ValueError, SyntaxError):
            values = re.split(r"[,;|]+", raw)
    return sorted({_normalise_text(item) for item in values if _normalise_text(item)})


class UnionFind:
    """Small disjoint-set implementation used to build transitive groups."""

    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def make_groups(df: pd.DataFrame) -> pd.Series:
    """
    Assign the same group to likely duplicate postings.

    Strong duplicate keys:
    - identical normalized description/requirements text (when available)
    - identical normalized title + city + salary range + experience

    Union-Find makes the grouping transitive.
    """
    data = df.reset_index(drop=True)
    uf = UnionFind(len(data))
    seen: dict[tuple[Any, ...], int] = {}

    text_columns = [
        col for col in ("description", "requirements_text")
        if col in data.columns
    ]

    for idx, row in data.iterrows():
        keys: list[tuple[Any, ...]] = []

        for column in text_columns:
            text = _normalise_text(row.get(column, ""))
            if len(text) >= 80:
                digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
                keys.append(("text", column, digest))

        title = _normalise_text(row.get("job_title", ""))
        city = _normalise_text(row.get("city_group", row.get("location", "")))
        exp = _normalise_text(row.get("experience_group", row.get("experience_required", "")))
        salary_min = row.get("salary_min", np.nan)
        salary_max = row.get("salary_max", np.nan)

        if title:
            keys.append((
                "posting",
                title,
                city,
                exp,
                round(float(salary_min), 1) if pd.notna(salary_min) else None,
                round(float(salary_max), 1) if pd.notna(salary_max) else None,
            ))

        for key in keys:
            if key in seen:
                uf.union(idx, seen[key])
            else:
                seen[key] = idx

    roots = [uf.find(i) for i in range(len(data))]
    root_to_group: dict[int, str] = {}
    groups = []
    for root in roots:
        if root not in root_to_group:
            root_to_group[root] = f"group_{len(root_to_group):05d}"
        groups.append(root_to_group[root])

    return pd.Series(groups, index=df.index, name="group_id")


def get_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """Return train/test labels while keeping each group entirely in one split."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    groups = make_groups(df)
    if groups.nunique() < 2:
        raise ValueError("Need at least two distinct groups for a group-aware split.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(
        splitter.split(df, groups=groups.to_numpy())
    )

    split = pd.Series(index=df.index, dtype="object", name="split")
    split.iloc[train_idx] = "train"
    split.iloc[test_idx] = "test"

    # Safety assertion: no group may occur in both partitions.
    assignment = pd.DataFrame({"group": groups, "split": split})
    if assignment.groupby("group")["split"].nunique().max() > 1:
        raise RuntimeError("Group leakage detected between train and test.")
    return split


class SkillsMultiHot(BaseEstimator, TransformerMixin):
    """Convert a column of skill lists (or CSV-serialized lists) to multi-hot."""

    def __init__(self, min_frequency: int = 1):
        self.min_frequency = min_frequency

    def fit(self, X, y=None):
        frame = self._to_frame(X)
        counts: dict[str, int] = {}
        for skills in frame.iloc[:, 0].map(_as_skill_list):
            for skill in set(skills):
                counts[skill] = counts.get(skill, 0) + 1
        self.vocabulary_ = sorted(
            skill for skill, count in counts.items()
            if count >= self.min_frequency
        )
        self.feature_names_out_ = np.asarray(
            [f"skill__{skill}" for skill in self.vocabulary_], dtype=object
        )
        return self

    def transform(self, X):
        if not hasattr(self, "vocabulary_"):
            raise RuntimeError("SkillsMultiHot must be fitted before transform.")
        frame = self._to_frame(X)
        index = {skill: i for i, skill in enumerate(self.vocabulary_)}
        result = np.zeros((len(frame), len(index)), dtype=np.float64)
        for row_idx, skills in enumerate(frame.iloc[:, 0].map(_as_skill_list)):
            for skill in set(skills):
                if skill in index:
                    result[row_idx, index[skill]] = 1.0
        return result

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_.copy()

    @staticmethod
    def _to_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 0:
                raise ValueError("SkillsMultiHot received no columns.")
            return X
        array = np.asarray(X, dtype=object)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return pd.DataFrame(array)


def build_preprocessor(
    df: pd.DataFrame,
    skill_column: str = "skills_list",
) -> ColumnTransformer:
    """
    Build a preprocessing pipeline from available engineered columns.
    Fit this transformer on X_train only to avoid vocabulary/statistic leakage.
    """
    numeric_candidates = ["experience_years"]
    categorical_candidates = ["experience_group", "seniority", "city_group"]
    numeric = [c for c in numeric_candidates if c in df.columns]
    categorical = [c for c in categorical_candidates if c in df.columns]

    transformers = []
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    if categorical:
        transformers.append((
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            categorical,
        ))
    if skill_column in df.columns:
        transformers.append((
            "skills",
            SkillsMultiHot(),
            [skill_column],
        ))

    if not transformers:
        raise ValueError("No supported feature columns found in the input dataframe.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cannot find cleaned data: {INPUT_FILE}")

    if SPLIT_FILE.exists():
        raise FileExistsError(
            f"{SPLIT_FILE} already exists. It will not be overwritten. "
            "Delete it manually only if you intentionally want a new split."
        )

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    groups = make_groups(df)
    split = get_split(df)

    output = pd.DataFrame({
        "row_id": np.arange(len(df)),
        "group_id": groups,
        "split": split,
    })
    output.to_csv(SPLIT_FILE, index=False, encoding="utf-8-sig")

    print(f"Rows: {len(df)}")
    print(f"Groups: {groups.nunique()}")
    print(f"Train: {(split == 'train').sum()}")
    print(f"Test: {(split == 'test').sum()}")
    print(f"Saved: {SPLIT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
