"""LLM Evaluation Platform.

A clean-room reference implementation of an enterprise LLM evaluation
platform: versioned golden datasets, multiple deterministic scorers
(exact match, token F1, semantic similarity, faithfulness, safety
screen), offline and sampled-online runs, regression gates against a
baseline, experiment comparison, and trend / flakiness detection.
All data is synthetic and all code is original; it runs fully offline.
"""

from .datasets import build_default_dataset, build_default_dataset_v2, bump_version, diff_datasets
from .errors import DatasetNotFoundError, EvalError, GateError, RunNotFoundError, ScorerNotFoundError, ValidationError
from .gates import GateCheck, GateConfig, GateResult, compare_runs, evaluate_gates
from .models import CaseResult, Dataset, EvalCase, EvalRun, ScoreStats
from .runner import EvalRunner, build_fake_model
from .scorers import (
    ExactMatchScorer, FaithfulnessScorer, SafetyScreenScorer, SemanticSimilarityScorer, TokenF1Scorer,
    contains_pii, contains_toxicity, default_scorers, embed, get_scorer, score_case, tokenize,
)
from .store import FlakyCase, RunStore, detect_flaky_cases, flakiness_report, score_trend

__version__ = "0.1.0"

__all__ = [
    "CaseResult", "Dataset", "DatasetNotFoundError", "EvalCase", "EvalError", "EvalRun", "EvalRunner",
    "ExactMatchScorer", "FaithfulnessScorer", "FlakyCase", "GateCheck", "GateConfig", "GateError",
    "GateResult", "RunNotFoundError", "RunStore", "SafetyScreenScorer", "ScoreStats",
    "ScorerNotFoundError", "SemanticSimilarityScorer", "TokenF1Scorer", "ValidationError",
    "build_default_dataset", "build_default_dataset_v2", "build_fake_model", "bump_version",
    "compare_runs", "contains_pii", "contains_toxicity", "default_scorers", "detect_flaky_cases",
    "diff_datasets", "embed", "evaluate_gates", "flakiness_report", "get_scorer", "score_case",
    "score_trend", "tokenize",
]
