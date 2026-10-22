import pytest

from llm_evaluation_platform import (
    EvalCase, contains_pii, contains_toxicity, embed, get_scorer, score_case, tokenize,
)
from llm_evaluation_platform.scorers import (
    ExactMatchScorer, FaithfulnessScorer, SafetyScreenScorer, SemanticSimilarityScorer,
    TokenF1Scorer, cosine_similarity,
)

CASE = EvalCase("c1", "What is the capital of France?", "Paris")


def test_tokenize():
    assert tokenize("Hello, World! 42") == ["hello", "world", "42"]
    assert tokenize("") == []


def test_exact_match():
    s = ExactMatchScorer()
    assert s.score(CASE, "Paris") == 1.0
    assert s.score(CASE, " paris ") == 1.0
    assert s.score(CASE, "London") == 0.0
    assert s.score(CASE, "") == 0.0


def test_token_f1():
    s = TokenF1Scorer()
    case = EvalCase("c", "p", "the quick brown fox")
    assert s.score(case, "the quick brown fox") == 1.0
    assert s.score(case, "completely different words here") == 0.0
    partial = s.score(case, "quick brown")
    assert 0.0 < partial < 1.0
    assert s.score(case, "") == 0.0


def test_token_f1_counts_duplicates():
    s = TokenF1Scorer()
    case = EvalCase("c", "p", "a a b")
    assert s.score(case, "a b") < 1.0


def test_semantic_similarity():
    s = SemanticSimilarityScorer()
    assert s.score(CASE, "Paris") == 1.0
    assert s.score(CASE, "") == 0.0
    sim = s.score(EvalCase("c", "p", "revenue grew from new customers"), "revenue grew new customers")
    assert 0.5 < sim <= 1.0


def test_embed_deterministic_and_normalized():
    a, b = embed("hello world"), embed("hello world")
    assert a == b
    assert abs(sum(v * v for v in a) - 1.0) < 1e-9
    assert embed("") == tuple([0.0] * 64)
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_faithfulness_with_context():
    s = FaithfulnessScorer()
    case = EvalCase("c", "Summarise", "ignored expected", context="The cat sat on the warm mat in the sun")
    assert s.score(case, "cat sat warm mat sun") == 1.0
    assert s.score(case, "spaceship launched to mars") == 0.0
    assert s.score(case, "") == 0.0


def test_faithfulness_without_context_uses_expected():
    s = FaithfulnessScorer()
    assert s.score(CASE, "Paris") == 1.0
    assert s.score(CASE, "London") == 0.0


def test_safety_screen_clean_and_toxic():
    s = SafetyScreenScorer()
    assert s.score(CASE, "Welcome, glad you are here") == 1.0
    assert s.score(CASE, "you stupid idiot") == 0.0


def test_pii_detection():
    assert contains_pii("reach me at jane.doe@example.com") == ["email"]
    assert "ssn" in contains_pii("ssn 123-45-6789")
    assert "phone" in contains_pii("call 415-555-0132 today")
    assert contains_pii("a perfectly ordinary sentence") == []
    s = SafetyScreenScorer()
    assert s.score(CASE, "my email is a@b.com") == 0.0


def test_toxicity_detection():
    assert contains_toxicity("I hate this") == ["hate"]
    assert contains_toxicity("a lovely day") == []


def test_registry_and_score_case():
    assert get_scorer("exact_match").name == "exact_match"
    with pytest.raises(Exception):
        get_scorer("nope")
    scores = score_case(CASE, "Paris")
    assert set(scores) == {"exact_match", "token_f1", "semantic_similarity", "faithfulness", "safety_screen"}
    assert all(v == 1.0 for v in scores.values())
