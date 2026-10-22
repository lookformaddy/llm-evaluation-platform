"""End-to-end offline demo: baseline vs candidate, gates, trends, flakiness."""
from __future__ import annotations

from typing import Any

from .datasets import build_default_dataset
from .gates import GateConfig, compare_runs, evaluate_gates
from .runner import EvalRunner, build_fake_model
from .store import RunStore, detect_flaky_cases, flakiness_report, score_trend


def run_demo() -> dict[str, Any]:
    dataset = build_default_dataset()
    runner = EvalRunner()
    store = RunStore()

    baseline = runner.run(dataset, "model-good-v1", model_fn=build_fake_model(dataset, "good"), run_id="run-baseline")
    degraded = runner.run(dataset, "model-degraded-v2", model_fn=build_fake_model(dataset, "degraded"), run_id="run-candidate")
    mixed = runner.run(dataset, "model-mixed-v2", model_fn=build_fake_model(dataset, "mixed"), run_id="run-mixed")
    for run in (baseline, degraded, mixed):
        store.save(run)

    gate_good = evaluate_gates(baseline, baseline=None)
    gate_bad = evaluate_gates(degraded, baseline=baseline)
    comparison = compare_runs(baseline, degraded)
    trend = score_trend([baseline, mixed, degraded], "token_f1")
    flaky = detect_flaky_cases([baseline, degraded, mixed])

    results: dict[str, Any] = {
        "dataset": {"id": dataset.dataset_id, "version": dataset.version, "cases": len(dataset), "checksum": dataset.checksum},
        "baseline": baseline.to_dict(),
        "candidate": degraded.to_dict(),
        "mixed": mixed.to_dict(),
        "gate_baseline_passed": gate_good.passed,
        "gate_candidate_passed": gate_bad.passed,
        "gate_candidate_failures": len(gate_bad.failures),
        "comparison": comparison,
        "trend": trend,
        "flaky_cases": [f.to_dict() for f in flaky],
        "flakiness": flakiness_report([baseline, degraded, mixed]),
        "store_runs": len(store),
    }
    assert gate_good.passed is True, "good baseline must pass its own thresholds"
    assert gate_bad.passed is False, "degraded candidate must fail gates"
    assert comparison["overall_winner"] == "baseline"

    print("LLM Evaluation Platform demo")
    print("----------------------------")
    print(f"  dataset: {dataset.dataset_id} v{dataset.version} ({len(dataset)} cases, checksum {dataset.checksum})")
    print(f"  baseline  {baseline.model_name}: pass_rate={baseline.pass_rate:.2f} mean={baseline.mean_score:.3f}")
    print(f"  candidate {degraded.model_name}: pass_rate={degraded.pass_rate:.2f} mean={degraded.mean_score:.3f}")
    print(f"  gate(baseline, thresholds only): {'PASS' if gate_good.passed else 'FAIL'}")
    print(f"  gate(candidate vs baseline): {'PASS' if gate_bad.passed else 'FAIL'} ({len(gate_bad.failures)} failing checks)")
    print(f"  comparison winner: {comparison['overall_winner']} (pass-rate delta {comparison['pass_rate_delta']:+.2f})")
    print(f"  token_f1 trend over 3 runs: {trend['direction']} (slope {trend['slope']:+.3f})")
    print(f"  flaky cases across runs: {len(flaky)}")
    return results
