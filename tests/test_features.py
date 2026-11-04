import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS, TARGET, SENIORITY_COARSE, SkillsMultiHot,
    build_preprocessor, get_split, make_groups,
)


def sample_data():
    return pd.DataFrame({
        "row_id": [10, 11, 12, 13, 14, 15],
        "job_title": ["A", "A", "B", "C", "D", "E"],
        "city": ["Hà Nội"] * 6,
        "salary_min": [10, 10, 12, 13, 14, 15],
        "salary_max": [20, 20, 22, 23, 24, 25],
        "description": ["same description text long enough to group"] * 2
                      + ["other description one"] * 4,
        "desc_group": ["a", "a", "b", "c", "d", "e"],
    })


def test_required_api():
    assert TARGET == "salary_label"
    assert "job_category" in FEATURE_COLUMNS
    assert SENIORITY_COARSE["intern"] == "entry"
    assert SENIORITY_COARSE["manager"] == "lead_manager"
    assert build_preprocessor(experience="numeric", skill_top_k=30) is not None


def test_skill_multihot_learns_train_vocabulary():
    x = pd.DataFrame({"skills": ["python|sql", "python|java", "sql"]})
    transformer = SkillsMultiHot(top_k=2, min_count=2).fit(x)
    assert set(transformer.vocabulary_) == {"python", "sql"}
    assert transformer.transform(pd.DataFrame({"skills": ["python|java"]})).shape == (1, 2)


def test_group_split_created_and_reused(tmp_path):
    df = sample_data()
    groups = make_groups(df)
    path = tmp_path / "split.csv"
    train, test, created = get_split(df, groups, path)
    assert created is True
    assert (train & test).sum() == 0
    assert train.sum() + test.sum() == len(df)
    train2, test2, created2 = get_split(df, groups, path)
    assert created2 is False
    assert train.tolist() == train2.tolist()
    assert test.tolist() == test2.tolist()


def test_group_split_errors_when_row_ids_change(tmp_path):
    df = sample_data()
    path = tmp_path / "split.csv"
    get_split(df, make_groups(df), path)
    changed = df.copy()
    changed.loc[0, "row_id"] = 999
    with pytest.raises(ValueError, match="row_id set changed"):
        get_split(changed, make_groups(changed), path)


def test_make_groups_transitive_relationship():
    df = pd.DataFrame({
        "job_title": ["Developer A", "Developer B", "Developer C"],
        "city": ["Ha Noi", "Da Nang", "Ho Chi Minh"],
        "salary_min": [10, 20, 30],
        "salary_max": [20, 30, 40],
        "desc_group": ["shared_ab", "shared_ab", "unique_c"],
        "description": [
            "Description A is completely different from the others",
            "Description B contains a shared long description for B and C",
            "Description B contains a shared long description for B and C",
        ],
    })

    groups = make_groups(df)

    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[1] == groups.iloc[2]
    assert groups.nunique() == 1

def test_group_split_rejects_missing_groups(tmp_path):
    df = sample_data()

    groups = pd.Series([
        "group_1",
        "group_1",
        None,
        "group_2",
        "group_3",
        "group_3",
    ])

    with pytest.raises(ValueError, match="group_id must not contain missing values"):
        get_split(df, groups, tmp_path / "split.csv")

def test_group_split_rejects_invalid_split_label(tmp_path):
    df = sample_data()
    groups = make_groups(df)

    path = tmp_path / "split.csv"

    # Tạo split hợp lệ trước.
    get_split(df, groups, path)

    # Sửa một nhãn thành giá trị không hợp lệ.
    saved = pd.read_csv(path)
    saved.loc[0, "split"] = "validation"
    saved.to_csv(path, index=False)

    with pytest.raises(ValueError, match="invalid split labels"):
        get_split(df, groups, path)
def test_skills_multihot_does_not_learn_from_test_data():
    train = pd.DataFrame({
        "skills": [
            "python|sql",
            "python|java",
            "python|sql",
            "sql|java",
        ]
    })

    test = pd.DataFrame({
        "skills": ["rust|golang"]
    })

    transformer = SkillsMultiHot(top_k=30, min_count=2)
    transformer.fit(train)

    assert "rust" not in transformer.vocabulary_
    assert "golang" not in transformer.vocabulary_

    transformed = transformer.transform(test)
    assert transformed.shape[0] == 1
    assert transformed.sum() == 0


def test_build_preprocessor_numeric_and_group_modes():
    from scipy import sparse
    import numpy as np

    df = pd.DataFrame({
        "job_category": ["Phát triển phần mềm"] * 12,
        "city": ["TP.HCM"] * 12,
        "seniority_coarse": ["entry"] * 12,
        "experience_years": list(range(12)),
        "experience_group": ["0-1 năm"] * 12,
        "skills": ["python|sql"] * 12,
    })

    for mode in ("numeric", "group"):
        preprocessor = build_preprocessor(experience=mode)
        transformed = preprocessor.fit_transform(df)

        assert transformed.shape[0] == len(df)
        assert transformed.shape[1] > 0

        if sparse.issparse(transformed):
            assert np.isfinite(transformed.data).all()
        else:
            assert np.isfinite(transformed).all()


def test_group_split_never_separates_same_group(tmp_path):
    df = sample_data()
    groups = make_groups(df)

    train, test, created = get_split(
        df,
        groups,
        tmp_path / "split.csv",
    )

    result = pd.DataFrame({
        "group_id": groups.to_numpy(),
        "split": [
            "train" if is_train else "test"
            for is_train in train
        ],
    })

    assert created is True
    assert (
        result.groupby("group_id")["split"].nunique().max()
        == 1
    )
