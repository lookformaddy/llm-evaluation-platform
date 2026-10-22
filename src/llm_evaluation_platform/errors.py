"""Domain errors with stable codes."""
from __future__ import annotations

class EvalError(Exception):
    code = "eval_error"
    http_status = 500

class ValidationError(EvalError):
    code = "validation_error"
    http_status = 422

class DatasetNotFoundError(EvalError):
    code = "dataset_not_found"
    http_status = 404

class RunNotFoundError(EvalError):
    code = "run_not_found"
    http_status = 404

class ScorerNotFoundError(EvalError):
    code = "scorer_not_found"
    http_status = 404

class GateError(EvalError):
    code = "gate_error"
    http_status = 422
