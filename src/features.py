"""TV2 feature engineering; split is group-aware and reusable."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "it_salary_clean.csv"
SPLIT_FILE = ROOT / "data" / "processed" / "split.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET = "salary_label"
SENIORITY_COARSE = {
    "intern": "entry", "fresher": "entry", "junior": "entry",
    "middle": "senior", "senior": "senior",
    "lead": "lead_manager", "manager": "lead_manager",
    "unknown": "unknown",
}
CATEGORICAL = ["job_category", "city", "seniority_coarse"]
EXPERIENCE_NUM = ["experience_years"]
EXPERIENCE_CAT = ["experience_group"]
SKILLS = "skills"
FEATURE_COLUMNS = CATEGORICAL + EXPERIENCE_NUM + EXPERIENCE_CAT + [SKILLS]
MIN_FREQUENCY = 10
SKILL_TOP_K = 30
SKILL_MIN_COUNT = 5


def load_clean(path=INPUT_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = set(CATEGORICAL[:2] + EXPERIENCE_NUM + EXPERIENCE_CAT + [SKILLS, TARGET, "row_id", "seniority"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Clean dataset missing columns: {sorted(missing)}")
    df["seniority_coarse"] = df["seniority"].map(SENIORITY_COARSE).fillna("unknown")
    df["skills"] = df["skills"].fillna("").astype(str)
    return df


def _normalise_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def _as_skill_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [_normalise_text(v) for v in value if _normalise_text(v)]
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1



def make_groups(df: pd.DataFrame) -> pd.Series:
    """Create connected duplicate groups using Union-Find."""
    data = df.reset_index(drop=True)
    uf = UnionFind(len(data))
    seen: dict[tuple, int] = {}

    for i, row in data.iterrows():
        keys = []

        # Reuse duplicate groups created during data cleaning.
        desc_group = _normalise_text(row.get("desc_group", ""))
        if desc_group:
            keys.append(("desc_group", desc_group))

        # Match normalized job descriptions and requirements.
        for col in ("description", "requirements_text"):
            value = _normalise_text(row.get(col, ""))
            if len(value) >= 40:
                keys.append((col, hashlib.sha1(value.encode()).hexdigest()))

        # Match the complete posting identity.
        title = _normalise_text(row.get("job_title", ""))
        city = _normalise_text(row.get("city", row.get("location", "")))
        salary_min = row.get("salary_min", None)
        salary_max = row.get("salary_max", None)

        salary = _normalise_text(row.get("salary_raw", row.get("salary", "")))

        if title and salary:
            keys.append((
                "posting",
                title,
                city,
                salary,
            ))

        # Merge all records sharing at least one key.
        for key in keys:
            if key in seen:
                uf.union(i, seen[key])
            else:
                seen[key] = i

    roots = [uf.find(i) for i in range(len(data))]

    mapping = {}
    labels = []

    for root in roots:
        if root not in mapping:
            mapping[root] = f"group_{len(mapping):05d}"
        labels.append(mapping[root])

    return pd.Series(labels, index=df.index, name="group_id")

def get_split(df: pd.DataFrame, groups=None, path=SPLIT_FILE):
    """Return train/test masks and reuse a matching split."""
    from sklearn.model_selection import GroupShuffleSplit

    if "row_id" not in df:
        raise ValueError("df must contain row_id")

    row_ids = df["row_id"].tolist()

    if len(set(row_ids)) != len(row_ids):
        raise ValueError("row_id must be unique")

    group_values = groups if groups is not None else make_groups(df)
    group_values = np.asarray(group_values)

    if len(group_values) != len(df):
        raise ValueError("groups length must match df")

    if pd.isna(group_values).any():
        raise ValueError("group_id must not contain missing values")

    if pd.Series(group_values).nunique() < 2:
        raise ValueError("Need at least two distinct groups")

    path = Path(path)

    if path.exists():
        saved = pd.read_csv(path)

        if not {"row_id", "split"}.issubset(saved.columns):
            raise ValueError(
                "Existing split.csv must contain row_id and split columns"
            )

        if saved["row_id"].duplicated().any():
            raise ValueError("split.csv contains duplicate row_id")

        if set(saved["row_id"]) != set(row_ids) or len(saved) != len(row_ids):
            raise ValueError(
                "row_id set changed; remove split.csv and regenerate intentionally"
            )

        split_map = saved.set_index("row_id")["split"]
        labels = pd.Series(row_ids).map(split_map).to_numpy()

        if pd.isna(labels).any() or not set(labels).issubset({"train", "test"}):
            raise ValueError("split.csv contains invalid split labels")

        check = pd.DataFrame({
            "group_id": group_values,
            "split": labels,
        })

        if (check.groupby("group_id")["split"].nunique() > 1).any():
            raise ValueError(
                "A group appears in both train and test; "
                "remove split.csv and regenerate intentionally"
            )

        path.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({
            "row_id": row_ids,
            "group_id": group_values,
            "split": labels,
        }).to_csv(path, index=False)

        return (
            pd.Series(labels == "train", index=df.index, name="train_mask"),
            pd.Series(labels == "test", index=df.index, name="test_mask"),
            False,
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(df, groups=group_values)
    )

    train_mask = pd.Series(False, index=df.index, name="train_mask")
    test_mask = pd.Series(False, index=df.index, name="test_mask")

    train_mask.iloc[train_idx] = True
    test_mask.iloc[test_idx] = True

    path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "row_id": row_ids,
        "group_id": group_values,
        "split": np.where(
            train_mask.to_numpy(),
            "train",
            "test",
        ),
    }).to_csv(path, index=False)

    return train_mask, test_mask, True



class SkillsMultiHot(BaseEstimator, TransformerMixin):
    def __init__(self, top_k: int = SKILL_TOP_K, min_count: int = SKILL_MIN_COUNT):
        self.top_k = top_k
        self.min_count = min_count

    def fit(self, X, y=None):
        values = self._column(X)
        counts: dict[str, int] = {}
        for value in values:
            for skill in set(_as_skill_list(value)):
                counts[skill] = counts.get(skill, 0) + 1
        eligible = [(skill, count) for skill, count in counts.items() if count >= self.min_count]
        eligible.sort(key=lambda pair: (-pair[1], pair[0]))
        self.vocabulary_ = [skill for skill, _ in eligible[:self.top_k]]
        self.feature_names_out_ = np.asarray([f"skill__{s}" for s in self.vocabulary_], dtype=object)
        return self

    def transform(self, X):
        if not hasattr(self, "vocabulary_"):
            raise RuntimeError("SkillsMultiHot must be fitted before transform")
        index = {skill: i for i, skill in enumerate(self.vocabulary_)}
        result = np.zeros((len(self._column(X)), len(index)), dtype=np.float64)
        for row_i, value in enumerate(self._column(X)):
            for skill in set(_as_skill_list(value)):
                if skill in index:
                    result[row_i, index[skill]] = 1.0
        return result

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_.copy()

    @staticmethod
    def _column(X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 0:
                raise ValueError("SkillsMultiHot received no columns")
            return X.iloc[:, 0].fillna("").tolist()
        array = np.asarray(X, dtype=object)
        if array.ndim == 1:
            return array.tolist()
        return array[:, 0].tolist()


def build_preprocessor(
    experience: str = "numeric",
    skill_top_k: int = SKILL_TOP_K,
    skill_min_count: int = SKILL_MIN_COUNT,
    min_frequency: int = MIN_FREQUENCY,
) -> ColumnTransformer:

    if experience not in {"numeric", "group"}:
        raise ValueError("experience must be 'numeric' or 'group'")

    experience_columns = EXPERIENCE_NUM if experience == "numeric" else EXPERIENCE_CAT

    transformers = [
        ("categorical", OneHotEncoder(
            min_frequency=min_frequency,
            handle_unknown="infrequent_if_exist",
        ), CATEGORICAL),
    ]

    if experience == "numeric":
        transformers.append(
            ("experience_numeric", StandardScaler(), EXPERIENCE_NUM)
        )
    else:
        transformers.append(
            ("experience_group", OneHotEncoder(handle_unknown="ignore"), EXPERIENCE_CAT)
        )

    transformers.append(
        ("skills", SkillsMultiHot(
            top_k=skill_top_k,
            min_count=skill_min_count
        ), [SKILLS])
    )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


def main() -> None:
    df = load_clean()
    groups = make_groups(df)
    train_mask, test_mask, created_new = get_split(df, groups)
    print(f"Train: {int(train_mask.sum())}; test: {int(test_mask.sum())}; created_new: {created_new}")


if __name__ == "__main__":
    main()
