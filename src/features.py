# src/features.py
"""GĐ3 — Đặc trưng: nhóm chống rò rỉ, chia train/test, tiền xử lý (ColumnTransformer).

Mọi bước có "học" từ dữ liệu (từ vựng kỹ năng, gộp nhóm hiếm, imputation) nằm TRONG Pipeline,
nên chỉ được fit trên train (hoặc trên fold train khi cross-validation).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "it_salary_clean.csv"
SPLIT_PATH = ROOT / "data" / "processed" / "split.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "salary_label"

SENIORITY_COARSE = {  # 83% "unknown" (GĐ2) -> chỉ dùng dạng gộp thô
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

MIN_FREQUENCY = 10   # nhóm có < 10 tin trong train -> gộp vào "infrequent" (học trên train)
SKILL_TOP_K = 30
SKILL_MIN_COUNT = 5


# ----------------------------------------------------------------------------- dữ liệu
def load_clean(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["seniority_coarse"] = df["seniority"].map(SENIORITY_COARSE).fillna("unknown")
    df["skills"] = df["skills"].fillna("")
    return df


def make_groups(df: pd.DataFrame) -> pd.Series:
    """Gán group_id: hai tin cùng nhóm nếu trùng mô tả HOẶC trùng (chức danh, địa điểm, chuỗi lương).
    Dùng union-find để gộp bắc cầu. Mục đích: tin đăng lại không nằm ở cả train và test."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    pos = np.arange(len(df))
    keys = [df["desc_group"].astype(str),
            df["job_title_norm"].astype(str) + "||" + df["location_raw"].astype(str) + "||" + df["salary_raw"].astype(str)]
    for key in keys:
        first = {}
        for i, k in zip(pos, key):
            if k in first:
                union(first[k], i)
            else:
                first[k] = i
    roots = [find(i) for i in pos]
    return pd.Series(pd.factorize(pd.Series(roots))[0], index=df.index, name="group_id")


def get_split(df: pd.DataFrame, groups: pd.Series, path: Path = SPLIT_PATH):
    """Trả về (train_mask, test_mask, created_new). Lưu split ra file để mọi giai đoạn dùng CÙNG một split.
    Nếu file đã có và khớp đúng tập row_id hiện tại thì dùng lại; nếu không khớp thì báo lỗi
    (tránh vô tình tạo split mới sau khi đã xem test)."""
    if path.exists():
        saved = pd.read_csv(path)
        if set(saved["row_id"]) != set(df["row_id"]):
            raise RuntimeError(
                f"{path.name} không khớp dữ liệu hiện tại (dữ liệu sạch đã thay đổi). "
                f"Nếu CHƯA đánh giá trên test, xóa {path} để tạo split mới; nếu đã đánh giá, phải ghi chú lại.")
        split = df["row_id"].map(saved.set_index("row_id")["split"])
        return split.eq("train").values, split.eq("test").values, False
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    split = np.array(["train"] * len(df), dtype=object)
    split[test_idx] = "test"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": df["row_id"], "group_id": groups.values, "split": split}).to_csv(path, index=False)
    return split == "train", split == "test", True


# ----------------------------------------------------------------------------- kỹ năng
class SkillsMultiHot(BaseEstimator, TransformerMixin):
    """Multi-hot cho kỹ năng dạng chuỗi 'a|b|c'. Từ vựng = top_k kỹ năng có tần suất >= min_count
    TRÊN DỮ LIỆU FIT (train). Kỹ năng ngoài từ vựng bị bỏ qua; tin không có kỹ năng -> vector 0."""

    def __init__(self, top_k: int = SKILL_TOP_K, min_count: int = SKILL_MIN_COUNT):
        self.top_k = top_k
        self.min_count = min_count

    @staticmethod
    def _to_lists(X):
        col = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else pd.Series(np.asarray(X).ravel())
        return [[s for s in str(v).split("|") if s and s != "nan"] for v in col]

    def fit(self, X, y=None):
        counts = pd.Series([s for lst in self._to_lists(X) for s in lst], dtype=object).value_counts()
        counts = counts[counts >= self.min_count]
        self.vocabulary_ = list(counts.index[: self.top_k])
        self._index = {s: i for i, s in enumerate(self.vocabulary_)}
        return self

    def transform(self, X):
        lists = self._to_lists(X)
        out = np.zeros((len(lists), len(self.vocabulary_)), dtype=float)
        for r, lst in enumerate(lists):
            for s in lst:
                j = self._index.get(s)
                if j is not None:
                    out[r, j] = 1.0
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"skill_{s}" for s in self.vocabulary_], dtype=object)


# ----------------------------------------------------------------------------- tiền xử lý
def build_preprocessor(experience: str = "numeric", skill_top_k: int = SKILL_TOP_K,
                       skill_min_count: int = SKILL_MIN_COUNT, min_frequency: int = MIN_FREQUENCY):
    """experience='numeric': experience_years (số năm, tuyến tính);
    experience='group'  : experience_group one-hot (cho phép quan hệ phi tuyến theo bậc)."""
    def onehot():
        return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                             sparse_output=False)

    cat_pipe = Pipeline([
        # Dữ liệu sạch hiện không thiếu giá trị (GĐ2); imputer để an toàn khi dự đoán từ app.
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", onehot()),
    ])
    transformers = [("cat", cat_pipe, CATEGORICAL)]
    if experience == "numeric":
        # 0% thiếu ở GĐ2; median (học trên train) chỉ là phòng hờ cho đầu vào thiếu.
        transformers.append(("exp", SimpleImputer(strategy="median"), EXPERIENCE_NUM))
    elif experience == "group":
        transformers.append(("exp", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", onehot()),
        ]), EXPERIENCE_CAT))
    else:
        raise ValueError("experience phải là 'numeric' hoặc 'group'")
    transformers.append(("skills", SkillsMultiHot(top_k=skill_top_k, min_count=skill_min_count), [SKILLS]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
