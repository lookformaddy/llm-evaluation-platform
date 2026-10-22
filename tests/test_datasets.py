import pytest

from llm_evaluation_platform import build_default_dataset, build_default_dataset_v2, bump_version, diff_datasets


def test_bump_version():
    assert bump_version("1.2.3") == "1.2.4"
    assert bump_version("1.2.3", "minor") == "1.3.0"
    assert bump_version("1.2.3", "major") == "2.0.0"
    with pytest.raises(ValueError):
        bump_version("1.2")
    with pytest.raises(ValueError):
        bump_version("1.2.3", "huge")


def test_diff_datasets():
    v1 = build_default_dataset()
    v2 = build_default_dataset_v2()
    diff = diff_datasets(v1, v2)
    assert diff["added"] == ["fact-05", "reason-04"]
    assert diff["removed"] == []
    assert diff["changed"] == ["fact-03"]
    reverse = diff_datasets(v2, v1)
    assert reverse["removed"] == ["fact-05", "reason-04"]


def test_diff_identical_is_empty(dataset):
    assert diff_datasets(dataset, dataset) == {"added": [], "removed": [], "changed": []}


def test_v2_has_more_cases():
    assert len(build_default_dataset_v2()) == 14
    assert build_default_dataset_v2().version == "1.1.0"
