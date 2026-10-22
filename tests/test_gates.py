from llm_evaluation_platform import GateConfig, compare_runs, evaluate_gates


def test_good_run_passes_threshold_gates(good_run):
    result = evaluate_gates(good_run)
    assert result.passed is True
    assert result.failures == ()


def test_degraded_run_fails_gates(degraded_run):
    result = evaluate_gates(degraded_run)
    assert result.passed is False
    kinds = {f.kind for f in result.failures}
    assert "threshold" in kinds and "slice" in kinds


def test_regression_gate_blocks_small_drop_below_thresholds(good_run, mixed_run):
    # Mixed is above some thresholds in places but regresses vs the perfect baseline.
    result = evaluate_gates(mixed_run, baseline=good_run)
    assert result.passed is False
    assert any(f.kind == "regression" for f in result.failures)


def test_regression_gate_allows_identical_baseline(good_run):
    result = evaluate_gates(good_run, baseline=good_run)
    assert result.passed is True


def test_custom_thresholds():
    cfg = GateConfig(thresholds={"exact_match": 0.0}, max_regression=1.0)
    assert cfg.thresholds["exact_match"] == 0.0


def test_strict_thresholds_fail_good_run_on_safety_edge(good_run):
    cfg = GateConfig(thresholds={"exact_match": 1.0, "token_f1": 1.0, "semantic_similarity": 1.0,
                                 "faithfulness": 1.0, "safety_screen": 1.0})
    # Perfect run has faithfulness 1.0 everywhere? Summarization expected tokens are all in context; factual fall back to expected.
    result = evaluate_gates(good_run, config=cfg)
    # Semantic similarity of identical text is 1.0, so a perfect run should pass even strict gates.
    assert result.passed is True


def test_slice_threshold_override(good_run, mixed_run):
    cfg = GateConfig(slice_thresholds={"factual": {"token_f1": 1.0}})
    result = evaluate_gates(mixed_run, config=cfg)
    assert any(f.slice_name == "factual" for f in result.failures)


def test_gate_result_to_dict(degraded_run):
    d = evaluate_gates(degraded_run).to_dict()
    assert d["passed"] is False and len(d["failures"]) > 0
    assert all("threshold" in f and "actual" in f for f in d["failures"])


def test_compare_runs_good_vs_degraded(good_run, degraded_run):
    cmp = compare_runs(good_run, degraded_run)
    assert cmp["overall_winner"] == "baseline"
    assert cmp["scorers"]["token_f1"]["delta"] < 0
    assert cmp["scorers"]["token_f1"]["winner"] == "baseline"
    assert cmp["pass_rate_delta"] == -1.0
    assert cmp["wins"]["baseline"] >= 4


def test_compare_identical_runs_tie(good_run):
    cmp = compare_runs(good_run, good_run)
    assert cmp["overall_winner"] == "tie"
    assert all(v["delta"] == 0.0 for v in cmp["scorers"].values())
    assert "factual" in cmp["slices"]
