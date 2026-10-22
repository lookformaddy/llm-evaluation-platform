"""Run store, trend detection, and flakiness analysis."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from .errors import RunNotFoundError
from .models import EvalRun


class RunStore:
    """Thread-safe in-memory store (a DB adapter slots behind this API)."""
    def __init__(self) -> None:
        self._runs: dict[str, EvalRun] = {}
        self._lock = threading.Lock()

    def save(self, run: EvalRun) -> EvalRun:
        with self._lock:
            self._runs[run.run_id] = run
        return run

    def get(self, run_id: str) -> EvalRun:
        try:
            return self._runs[run_id]
        except KeyError:
            raise RunNotFoundError(f"unknown run: {run_id!r}") from None

    def __len__(self) -> int:
        return len(self._runs)

    def __contains__(self, run_id: str) -> bool:
        return run_id in self._runs

    def list_runs(self, *, dataset_id: str | None = None, model_name: str | None = None) -> list[EvalRun]:
        runs = list(self._runs.values())
        if dataset_id is not None:
            runs = [r for r in runs if r.dataset_id == dataset_id]
        if model_name is not None:
            runs = [r for r in runs if r.model_name == model_name]
        return sorted(runs, key=lambda r: r.run_id)

    def latest(self, *, model_name: str | None = None, dataset_id: str | None = None) -> EvalRun | None:
        runs = self.list_runs(dataset_id=dataset_id, model_name=model_name)
        return runs[-1] if runs else None

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()


def _slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values)) / denom


def score_trend(runs: list[EvalRun], scorer: str) -> dict[str, Any]:
    """Trend of one scorer across runs (ordered as given)."""
    points = [r.summary[scorer].mean if scorer in r.summary else 0.0 for r in runs]
    if not points:
        return {"scorer": scorer, "points": [], "slope": 0.0, "direction": "stable", "delta": 0.0}
    slope = _slope(points)
    delta = points[-1] - points[0] if len(points) > 1 else 0.0
    if slope > 0.01:
        direction = "improving"
    elif slope < -0.01:
        direction = "declining"
    else:
        direction = "stable"
    return {"scorer": scorer, "points": [round(p, 6) for p in points],
            "slope": round(slope, 6), "direction": direction, "delta": round(delta, 6)}


@dataclass(frozen=True)
class FlakyCase:
    case_id: str
    flip_count: int
    pass_rate: float
    run_count: int

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "flip_count": self.flip_count,
                "pass_rate": round(self.pass_rate, 6), "run_count": self.run_count}


def detect_flaky_cases(runs: list[EvalRun], min_flips: int = 1) -> list[FlakyCase]:
    """A case is flaky when its pass/fail outcome flips across runs."""
    if len(runs) < 2:
        return []
    history: dict[str, list[bool]] = {}
    for run in runs:
        for result in run.results:
            history.setdefault(result.case_id, []).append(result.passed)
    flaky: list[FlakyCase] = []
    for case_id, outcomes in sorted(history.items()):
        flips = sum(1 for a, b in zip(outcomes, outcomes[1:]) if a != b)
        if flips >= min_flips and 0 < sum(outcomes) < len(outcomes):
            flaky.append(FlakyCase(case_id=case_id, flip_count=flips,
                                   pass_rate=sum(outcomes) / len(outcomes), run_count=len(outcomes)))
    return sorted(flaky, key=lambda f: (-f.flip_count, f.case_id))


def flakiness_report(runs: list[EvalRun]) -> dict[str, Any]:
    flaky = detect_flaky_cases(runs)
    case_ids = {r.case_id for run in runs for r in run.results}
    return {"run_count": len(runs), "case_count": len(case_ids),
            "flaky_cases": [f.to_dict() for f in flaky],
            "flaky_fraction": round(len(flaky) / len(case_ids), 6) if case_ids else 0.0}
