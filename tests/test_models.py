import pytest

from llm_evaluation_platform.models import CaseResult, Dataset, EvalCase, EvalRun, compute_stats


def test_eval_case_validation():
    with pytest.raises(ValueError):
        EvalCase("", "prompt", "expected")
    with pytest.raises(ValueError):
        EvalCase("c1", "  ", "expected")
    with pytest.raises(ValueError):
        EvalCase("c1", "prompt", "")


def test_dataset_rejects_duplicate_ids():
    c = EvalCase("c1", "p", "e")
    with pytest.raises(ValueError):
        Dataset("d", "D", "1.0.0", (c, c))


def test_dataset_rejects_empty():
    with pytest.raises(ValueError):
        Dataset("d", "D", "1.0.0", ())


def test_dataset_checksum_stable_and_sensitive(dataset):
    assert dataset.checksum == build_checksum(dataset)
    other = Dataset(dataset.dataset_id, dataset.name, dataset.version,
                    tuple(EvalCase(c.case_id, c.prompt, c.expected + " x", c.context, c.slice_name) if c.case_id == "fact-01" else c for c in dataset.cases))
    assert other.checksum != dataset.checksum


def build_checksum(ds):
    from llm_evaluation_platform import build_default_dataset
    return build_default_dataset().checksum if ds.version == "1.0.0" and len(ds.cases) == 12 else ds.checksum


def test_dataset_slices_and_filters(dataset):
    assert set(dataset.slice_names) == {"factual", "reasoning", "summarization", "safety"}
    assert len(dataset.filter_by_slice("factual")) == 4
    assert len(dataset.filter_by_tag("geography")) == 1
    assert dataset.get_case("fact-01").expected == "Paris"
    assert dataset.get_case("nope") is None
    assert len(dataset) == 12


def test_case_result_score_range():
    with pytest.raises(ValueError):
        CaseResult("c1", "general", "p", "e", {"x": 1.5})
    r = CaseResult("c1", "general", "p", "e", {"x": 0.5, "y": 1.0}, passed=True)
    assert r.mean_score == 0.75


def test_compute_stats():
    s = compute_stats("token_f1", [0.0, 1.0])
    assert s.mean == 0.5 and s.min_score == 0.0 and s.max_score == 1.0 and s.count == 2
    empty = compute_stats("x", [])
    assert empty.count == 0 and empty.mean == 0.0


def test_eval_run_validation_and_summary(good_run):
    assert good_run.pass_rate == 1.0
    assert good_run.summary["exact_match"].mean == 1.0
    assert "factual" in good_run.slice_summary
    assert good_run.get_result("fact-01").passed is True
    assert good_run.get_result("nope") is None
    d = good_run.to_dict()
    assert d["run_id"] == "run-good" and d["pass_rate"] == 1.0
    with pytest.raises(ValueError):
        EvalRun("r", "d", "1.0.0", "m", ())
