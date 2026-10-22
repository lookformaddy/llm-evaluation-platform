"""Core typed models: cases, datasets, results, runs.

Dependency-free (no pydantic, no FastAPI) so core logic imports anywhere.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    """One golden case: a prompt, its expected answer, optional context."""
    case_id: str
    prompt: str
    expected: str
    context: str = ""
    slice_name: str = "general"
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if not self.expected.strip():
            raise ValueError("expected must not be empty")
        if not self.slice_name.strip():
            raise ValueError("slice_name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "prompt": self.prompt, "expected": self.expected,
                "context": self.context, "slice_name": self.slice_name,
                "tags": list(self.tags), "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class Dataset:
    """A versioned golden dataset. Version + checksum make runs reproducible."""
    dataset_id: str
    name: str
    version: str
    cases: tuple[EvalCase, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must not be empty")
        if not self.version.strip():
            raise ValueError("version must not be empty")
        if not self.cases:
            raise ValueError("dataset must contain at least one case")
        ids = [c.case_id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case_id values must be unique within a dataset")

    @property
    def checksum(self) -> str:
        payload = json.dumps([c.to_dict() for c in sorted(self.cases, key=lambda c: c.case_id)], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def slice_names(self) -> tuple[str, ...]:
        return tuple(sorted({c.slice_name for c in self.cases}))

    def __len__(self) -> int:
        return len(self.cases)

    def get_case(self, case_id: str) -> EvalCase | None:
        for c in self.cases:
            if c.case_id == case_id:
                return c
        return None

    def filter_by_slice(self, slice_name: str) -> tuple[EvalCase, ...]:
        return tuple(c for c in self.cases if c.slice_name == slice_name)

    def filter_by_tag(self, tag: str) -> tuple[EvalCase, ...]:
        return tuple(c for c in self.cases if tag in c.tags)

    def to_dict(self) -> dict[str, Any]:
        return {"dataset_id": self.dataset_id, "name": self.name, "version": self.version,
                "description": self.description, "checksum": self.checksum,
                "case_count": len(self.cases), "slices": list(self.slice_names),
                "cases": [c.to_dict() for c in self.cases]}


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    slice_name: str
    predicted: str
    expected: str
    scores: dict[str, float] = field(default_factory=dict)
    passed: bool = False

    def __post_init__(self) -> None:
        for k, v in self.scores.items():
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"score {k}={v} out of [0,1]")

    @property
    def mean_score(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "slice_name": self.slice_name, "predicted": self.predicted,
                "expected": self.expected, "scores": dict(self.scores), "passed": self.passed,
                "mean_score": round(self.mean_score, 6)}


@dataclass(frozen=True)
class ScoreStats:
    scorer: str
    mean: float
    min_score: float
    max_score: float
    count: int
    std: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"scorer": self.scorer, "mean": round(self.mean, 6), "min": round(self.min_score, 6),
                "max": round(self.max_score, 6), "count": self.count, "std": round(self.std, 6)}


def compute_stats(scorer: str, values: list[float]) -> ScoreStats:
    if not values:
        return ScoreStats(scorer=scorer, mean=0.0, min_score=0.0, max_score=0.0, count=0, std=0.0)
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return ScoreStats(scorer=scorer, mean=mean, min_score=min(values), max_score=max(values),
                      count=len(values), std=math.sqrt(var))


@dataclass(frozen=True)
class EvalRun:
    run_id: str
    dataset_id: str
    dataset_version: str
    model_name: str
    results: tuple[CaseResult, ...]
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if not self.results:
            raise ValueError("run must contain at least one result")

    @property
    def summary(self) -> dict[str, ScoreStats]:
        by: dict[str, list[float]] = {}
        for r in self.results:
            for k, v in r.scores.items():
                by.setdefault(k, []).append(v)
        return {k: compute_stats(k, v) for k, v in sorted(by.items())}

    @property
    def slice_summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, list[float]]] = {}
        for r in self.results:
            for k, v in r.scores.items():
                out.setdefault(r.slice_name, {}).setdefault(k, []).append(v)
        return {s: {k: sum(v) / len(v) for k, v in sorted(m.items())} for s, m in sorted(out.items())}

    @property
    def pass_rate(self) -> float:
        return sum(1 for r in self.results if r.passed) / len(self.results) if self.results else 0.0

    @property
    def mean_score(self) -> float:
        vals = [r.mean_score for r in self.results]
        return sum(vals) / len(vals) if vals else 0.0

    def get_result(self, case_id: str) -> CaseResult | None:
        for r in self.results:
            if r.case_id == case_id:
                return r
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "dataset_id": self.dataset_id, "dataset_version": self.dataset_version,
                "model_name": self.model_name, "created_at": self.created_at,
                "pass_rate": round(self.pass_rate, 6), "mean_score": round(self.mean_score, 6),
                "summary": {k: v.to_dict() for k, v in self.summary.items()},
                "slice_summary": {s: {k: round(v, 6) for k, v in m.items()} for s, m in self.slice_summary.items()},
                "results": [r.to_dict() for r in self.results], "metadata": dict(self.metadata)}
