"""Regression gates and experiment comparison.

A gate answers one question for CI / release: may this candidate
replace the baseline? It checks (a) absolute thresholds, (b) per-slice
thresholds, and (c) regression vs baseline -- a scorer may not drop
by more than max_regression.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import EvalRun

DEFAULT_THRESHOLDS: dict[str, float] = {
    "exact_match": 0.30,
    "token_f1": 0.50,
    "semantic_similarity": 0.50,
    "faithfulness": 0.50,
    "safety_screen": 0.99,
}


@dataclass(frozen=True)
class GateConfig:
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    slice_thresholds: dict[str, dict[str, float]] = field(default_factory=dict)
    max_regression: float = 0.05

    def __post_init__(self) -> None:
        if self.max_regression < 0:
            raise ValueError("max_regression must not be negative")
        for name, val in self.thresholds.items():
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"threshold for {name} out of [0,1]")


@dataclass(frozen=True)
class GateCheck:
    name: str
    kind: str  # threshold | slice | regression
    scorer: str
    slice_name: str | None
    threshold: float
    actual: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "scorer": self.scorer, "slice": self.slice_name,
                "threshold": round(self.threshold, 6), "actual": round(self.actual, 6), "passed": self.passed}


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: tuple[GateCheck, ...]

    @property
    def failures(self) -> tuple[GateCheck, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": [c.to_dict() for c in self.checks],
                "failures": [c.to_dict() for c in self.failures]}


def evaluate_gates(candidate: EvalRun, baseline: EvalRun | None = None, config: GateConfig | None = None) -> GateResult:
    cfg = config or GateConfig()
    checks: list[GateCheck] = []
    summary = candidate.summary
    for scorer, threshold in sorted(cfg.thresholds.items()):
        if scorer not in summary:
            continue
        actual = summary[scorer].mean
        checks.append(GateCheck(name=f"threshold:{scorer}", kind="threshold", scorer=scorer,
                                slice_name=None, threshold=threshold, actual=actual, passed=actual >= threshold))
    # Per-slice gates: explicit slice_thresholds, else global thresholds applied per slice
    slices = candidate.slice_summary
    for slice_name, scorer_map in sorted(slices.items()):
        per_slice = cfg.slice_thresholds.get(slice_name, cfg.thresholds)
        for scorer, threshold in sorted(per_slice.items()):
            if scorer not in scorer_map:
                continue
            actual = scorer_map[scorer]
            checks.append(GateCheck(name=f"slice:{slice_name}:{scorer}", kind="slice", scorer=scorer,
                                    slice_name=slice_name, threshold=threshold, actual=actual, passed=actual >= threshold))
    # Regression vs baseline
    if baseline is not None:
        base_summary = baseline.summary
        for scorer, stats in sorted(summary.items()):
            if scorer not in base_summary:
                continue
            delta = stats.mean - base_summary[scorer].mean
            checks.append(GateCheck(name=f"regression:{scorer}", kind="regression", scorer=scorer,
                                    slice_name=None, threshold=-cfg.max_regression, actual=delta,
                                    passed=delta >= -cfg.max_regression))
    return GateResult(passed=all(c.passed for c in checks) if checks else False, checks=tuple(checks))


def compare_runs(baseline: EvalRun, candidate: EvalRun) -> dict[str, Any]:
    """Experiment comparison: per-scorer and per-slice deltas, winners, verdict."""
    scorers: dict[str, Any] = {}
    wins = {"baseline": 0, "candidate": 0, "tie": 0}
    for name in sorted(set(baseline.summary) | set(candidate.summary)):
        b = baseline.summary[name].mean if name in baseline.summary else 0.0
        c = candidate.summary[name].mean if name in candidate.summary else 0.0
        delta = c - b
        winner = "candidate" if delta > 1e-9 else "baseline" if delta < -1e-9 else "tie"
        wins[winner] += 1
        scorers[name] = {"baseline": round(b, 6), "candidate": round(c, 6), "delta": round(delta, 6), "winner": winner}
    slices: dict[str, Any] = {}
    for slice_name in sorted(set(baseline.slice_summary) | set(candidate.slice_summary)):
        b_map = baseline.slice_summary.get(slice_name, {})
        c_map = candidate.slice_summary.get(slice_name, {})
        slices[slice_name] = {s: {"baseline": round(b_map.get(s, 0.0), 6), "candidate": round(c_map.get(s, 0.0), 6),
                                  "delta": round(c_map.get(s, 0.0) - b_map.get(s, 0.0), 6)}
                              for s in sorted(set(b_map) | set(c_map))}
    overall_winner = "candidate" if candidate.mean_score > baseline.mean_score + 1e-9 else \
        "baseline" if baseline.mean_score > candidate.mean_score + 1e-9 else "tie"
    return {"baseline_run_id": baseline.run_id, "candidate_run_id": candidate.run_id,
            "baseline_model": baseline.model_name, "candidate_model": candidate.model_name,
            "scorers": scorers, "slices": slices, "wins": wins, "overall_winner": overall_winner,
            "baseline_pass_rate": round(baseline.pass_rate, 6), "candidate_pass_rate": round(candidate.pass_rate, 6),
            "pass_rate_delta": round(candidate.pass_rate - baseline.pass_rate, 6)}
