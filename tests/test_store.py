import pytest

from llm_evaluation_platform import RunStore, detect_flaky_cases, flakiness_report, score_trend
from llm_evaluation_platform.errors import RunNotFoundError


def test_store_save_get_list(good_run, degraded_run):
    store = RunStore()
    store.save(good_run); store.save(degraded_run)
    assert len(store) == 2
    assert store.get("run-good") is good_run
    assert "run-bad" in store
    assert [r.run_id for r in store.list_runs(model_name="good-model")] == ["run-good"]
    assert store.list_runs(dataset_id="golden-core") != []
    assert store.latest(model_name="bad-model") is degraded_run
    with pytest.raises(RunNotFoundError):
        store.get("nope")
    store.clear()
    assert len(store) == 0


def test_score_trend_directions(good_run, mixed_run, degraded_run):
    declining = score_trend([good_run, mixed_run, degraded_run], "token_f1")
    assert declining["direction"] == "declining" and declining["slope"] < 0 and declining["delta"] < 0
    improving = score_trend([degraded_run, mixed_run, good_run], "token_f1")
    assert improving["direction"] == "improving"
    stable = score_trend([good_run, good_run], "token_f1")
    assert stable["direction"] == "stable" and stable["slope"] == 0.0
    empty = score_trend([], "token_f1")
    assert empty["points"] == [] and empty["direction"] == "stable"


def test_detect_flaky_cases(good_run, degraded_run, mixed_run):
    flaky = detect_flaky_cases([good_run, degraded_run])
    # Every case passes in good and fails in degraded -> all flip once and are flaky.
    assert len(flaky) == 12
    assert all(f.flip_count == 1 for f in flaky)
    assert detect_flaky_cases([good_run]) == []
    assert detect_flaky_cases([good_run, good_run]) == []


def test_flaky_subset_with_mixed(good_run, mixed_run):
    flaky = detect_flaky_cases([good_run, mixed_run])
    # Mixed fails odd-index cases only.
    assert 0 < len(flaky) < 12


def test_flakiness_report(good_run, degraded_run):
    report = flakiness_report([good_run, degraded_run])
    assert report["run_count"] == 2 and report["case_count"] == 12
    assert report["flaky_fraction"] == 1.0
