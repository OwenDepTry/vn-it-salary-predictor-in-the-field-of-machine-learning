# tests/test_features.py
"""Chống rò rỉ dữ liệu (GĐ3): nhóm tin đăng lại, split theo nhóm, từ vựng kỹ năng chỉ học từ dữ liệu fit."""
import numpy as np
import pandas as pd
import pytest

from features import SkillsMultiHot, get_split, make_groups


def toy_frame():
    return pd.DataFrame({
        "row_id": range(6),
        "desc_group": ["d1", "d1", "d2", "d3", "d4", "d5"],          # 0-1 trùng mô tả
        "job_title_norm": ["a", "b", "c", "c", "e", "f"],            # 2-3 trùng (tiêu đề, địa điểm, lương)
        "location_raw": ["hn", "hn", "hcm", "hcm", "hn", "hn"],
        "salary_raw": ["10 - 20", "10 - 20", "15 - 25", "15 - 25", "8 - 9", "30 - 40"],
    })


def test_make_groups_links_reposts():
    g = make_groups(toy_frame())
    assert g[0] == g[1]           # cùng mô tả
    assert g[2] == g[3]           # cùng tiêu đề + địa điểm + lương
    assert len({g[0], g[2], g[4], g[5]}) == 4


def test_make_groups_is_transitive():
    df = toy_frame()
    df.loc[2, "desc_group"] = "d1"  # 2 trùng mô tả với 0-1, và trùng khóa với 3 -> cả 4 tin chung nhóm
    g = make_groups(df)
    assert g[0] == g[1] == g[2] == g[3]


def test_split_has_no_group_on_both_sides(tmp_path):
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({
        "row_id": range(n),
        "desc_group": [f"d{i // 2}" for i in range(n)],               # mỗi mô tả xuất hiện 2 lần
        "job_title_norm": rng.choice(list("abcdefghij"), n),
        "location_raw": rng.choice(["hn", "hcm"], n),
        "salary_raw": [f"{i}" for i in range(n)],
    })
    groups = make_groups(df)
    tr, te, created = get_split(df, groups, path=tmp_path / "split.csv")
    assert created
    assert not set(groups[tr]) & set(groups[te])
    assert 0.1 < te.mean() < 0.3
    tr2, te2, created2 = get_split(df, groups, path=tmp_path / "split.csv")   # dùng lại đúng split đã lưu
    assert not created2 and (tr2 == tr).all()


def test_split_refuses_when_data_changed(tmp_path):
    df = toy_frame()
    get_split(df, make_groups(df), path=tmp_path / "split.csv")
    df2 = df.iloc[:-1]
    with pytest.raises(RuntimeError):
        get_split(df2, make_groups(df2), path=tmp_path / "split.csv")


def test_skill_vocabulary_learned_only_from_fit_data():
    train = pd.DataFrame({"skills": ["java|sql", "java", "python|java", ""]})
    other = pd.DataFrame({"skills": ["rust|rust", "rust"]})
    enc = SkillsMultiHot(top_k=10, min_count=1).fit(train)
    assert "rust" not in enc.vocabulary_
    assert enc.vocabulary_[0] == "java"                               # phổ biến nhất trong train
    assert enc.transform(other).sum() == 0                            # kỹ năng lạ bị bỏ qua
    assert enc.transform(pd.DataFrame({"skills": [""]})).sum() == 0   # không kỹ năng -> vector 0
