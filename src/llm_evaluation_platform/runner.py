"""Evaluation runner: offline and sampled-online evaluation.

A model is either a callable prompt -> prediction, a callable
case -> prediction, or an explicit {case_id: prediction} mapping.
The runner scores every case with every scorer and marks a case
passed when its mean score is >= pass_threshold (default 0.5) and
its safety score (when present) is 1.0 -- a unsafe answer never passes.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from .errors import ValidationError
from .models import CaseResult, Dataset, EvalCase, EvalRun
from .scorers import Scorer, default_scorers, score_case

_counter = itertools.count(1)


def _next_run_id(model_name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "-" for ch in model_name.lower()).strip("-") or "model"
    return f"run-{slug}-{next(_counter):04d}"


class EvalRunner:
    def __init__(self, scorers: list[Scorer] | None = None, pass_threshold: float = 0.5) -> None:
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be in [0,1]")
        self.scorers = scorers if scorers is not None else default_scorers()
        self.pass_threshold = pass_threshold

    def _predict(self, case: EvalCase, model_fn: Callable[..., str] | None, predictions: dict[str, str] | None) -> str:
        if predictions is not None:
            if case.case_id not in predictions:
                raise ValidationError(f"missing prediction for case {case.case_id}")
            return predictions[case.case_id]
        assert model_fn is not None
        try:
            out = model_fn(case)  # type: ignore[arg-type]
        except TypeError:
            out = model_fn(case.prompt)  # type: ignore[arg-type]
        if not isinstance(out, str):
            raise ValidationError(f"model_fn must return str, got {type(out).__name__}")
        return out

    def run(self, dataset: Dataset, model_name: str, *, predictions: dict[str, str] | None = None,
            model_fn: Callable[..., str] | None = None, run_id: str | None = None,
            created_at: str = "", metadata: dict[str, Any] | None = None) -> EvalRun:
        if (predictions is None) == (model_fn is None):
            raise ValidationError("provide exactly one of predictions or model_fn")
        results: list[CaseResult] = []
        for case in dataset.cases:
            predicted = self._predict(case, model_fn, predictions)
            scores = score_case(case, predicted, self.scorers)
            mean = sum(scores.values()) / len(scores) if scores else 0.0
            safety_ok = scores.get("safety_screen", 1.0) >= 1.0
            passed = mean >= self.pass_threshold and safety_ok
            results.append(CaseResult(case_id=case.case_id, slice_name=case.slice_name,
                                      predicted=predicted, expected=case.expected, scores=scores, passed=passed))
        return EvalRun(run_id=run_id or _next_run_id(model_name), dataset_id=dataset.dataset_id,
                       dataset_version=dataset.version, model_name=model_name,
                       results=tuple(results), created_at=created_at, metadata=metadata or {})

    def run_online_sample(self, dataset: Dataset, model_name: str, *, predictions: dict[str, str],
                          sample_every: int = 2, **kwargs: Any) -> EvalRun:
        """Online-style eval: score a deterministic sample of the dataset."""
        if sample_every <= 0:
            raise ValueError("sample_every must be positive")
        sampled_cases = tuple(c for i, c in enumerate(dataset.cases) if i % sample_every == 0)
        if not sampled_cases:
            raise ValidationError("sampling produced no cases")
        sampled = Dataset(dataset_id=dataset.dataset_id, name=dataset.name, version=dataset.version,
                          cases=sampled_cases, description=dataset.description)
        return self.run(sampled, model_name, predictions=predictions, **kwargs)


def build_fake_model(dataset: Dataset, quality: str = "good") -> Callable[[str], str]:
    """Deterministic fake model keyed by prompt. Qualities: good/degraded/mixed."""
    if quality not in ("good", "degraded", "mixed"):
        raise ValueError(f"unknown quality: {quality!r}")
    mapping = {c.prompt: c.expected for c in dataset.cases}
    ordered = list(dataset.cases)
    def _model(prompt: str) -> str:
        expected = mapping.get(prompt, "I do not know")
        if quality == "good":
            return expected
        if quality == "degraded":
            return "An unrelated answer about something else entirely"
        # mixed: good for even-index prompts, degraded for odd
        idx = next((i for i, c in enumerate(ordered) if c.prompt == prompt), 0)
        return expected if idx % 2 == 0 else "An unrelated answer about something else entirely"
    return _model
