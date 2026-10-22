"""FastAPI service: datasets, runs, gates, comparison, trends.

Thin layer over the same runner/store the CLI uses. FastAPI/pydantic
imports are guarded: the core works without them, and create_app
raises one explicit error if they are missing.
"""
from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised only where FastAPI is installed
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]

from .datasets import build_default_dataset
from .errors import EvalError
from .gates import GateConfig, compare_runs, evaluate_gates
from .runner import EvalRunner
from .scorers import SCORER_REGISTRY
from .store import RunStore, detect_flaky_cases, score_trend

if FastAPI is not None:  # pragma: no cover - definition only when available
    class RunPayload(BaseModel):
        model_name: str
        predictions: dict[str, str]
        dataset_id: str = "golden-core"
        run_id: str | None = None

    class GatePayload(BaseModel):
        candidate_run_id: str
        baseline_run_id: str | None = None
        thresholds: dict[str, float] | None = None
        max_regression: float = 0.05


def create_app(store: RunStore | None = None, runner: EvalRunner | None = None):  # pragma: no cover - needs FastAPI
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI is not installed; install requirements.txt to run the service. "
            "The core platform (llm_evaluation_platform) works without it."
        )
    run_store = store or RunStore()
    eval_runner = runner or EvalRunner()
    dataset = build_default_dataset()
    app = FastAPI(title="LLM Evaluation Platform", version="0.1.0")
    app.state.store = run_store
    app.state.runner = eval_runner
    app.state.dataset = dataset

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "llm-evaluation-platform"}

    @app.get("/datasets")
    def datasets() -> dict[str, Any]:
        return {"datasets": [dataset.to_dict()]}

    @app.get("/datasets/{dataset_id}")
    def get_dataset(dataset_id: str) -> dict[str, Any]:
        if dataset_id != dataset.dataset_id:
            raise HTTPException(status_code=404, detail=f"unknown dataset: {dataset_id}")
        return dataset.to_dict()

    @app.get("/scorers")
    def scorers() -> dict[str, Any]:
        return {"scorers": sorted(SCORER_REGISTRY)}

    @app.post("/runs")
    def create_run(body: "RunPayload") -> dict[str, Any]:
        if body.dataset_id != dataset.dataset_id:
            raise HTTPException(status_code=404, detail=f"unknown dataset: {body.dataset_id}")
        try:
            run = eval_runner.run(dataset, body.model_name, predictions=body.predictions, run_id=body.run_id)
        except EvalError as exc:
            raise HTTPException(status_code=exc.http_status, detail={"code": exc.code, "message": str(exc)}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        run_store.save(run)
        return run.to_dict()

    @app.get("/runs")
    def list_runs(model_name: str | None = None) -> dict[str, Any]:
        return {"runs": [r.to_dict() for r in run_store.list_runs(model_name=model_name)]}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return run_store.get(run_id).to_dict()
        except EvalError as exc:
            raise HTTPException(status_code=exc.http_status, detail={"code": exc.code, "message": str(exc)}) from exc

    @app.post("/gates/evaluate")
    def gates(body: "GatePayload") -> dict[str, Any]:
        try:
            candidate = run_store.get(body.candidate_run_id)
            baseline = run_store.get(body.baseline_run_id) if body.baseline_run_id else None
        except EvalError as exc:
            raise HTTPException(status_code=exc.http_status, detail={"code": exc.code, "message": str(exc)}) from exc
        cfg = GateConfig(thresholds=body.thresholds or GateConfig().thresholds, max_regression=body.max_regression)
        return evaluate_gates(candidate, baseline=baseline, config=cfg).to_dict()

    @app.get("/compare")
    def compare(baseline_run_id: str, candidate_run_id: str) -> dict[str, Any]:
        try:
            return compare_runs(run_store.get(baseline_run_id), run_store.get(candidate_run_id))
        except EvalError as exc:
            raise HTTPException(status_code=exc.http_status, detail={"code": exc.code, "message": str(exc)}) from exc

    @app.get("/trends/{scorer}")
    def trend(scorer: str) -> dict[str, Any]:
        return score_trend(run_store.list_runs(), scorer)

    @app.get("/flaky")
    def flaky() -> dict[str, Any]:
        return {"flaky_cases": [f.to_dict() for f in detect_flaky_cases(run_store.list_runs())]}

    return app
