import pytest

from llm_evaluation_platform import EvalRunner
from llm_evaluation_platform.errors import ValidationError
from llm_evaluation_platform.runner import build_fake_model


def test_run_with_predictions(dataset, runner, perfect_predictions):
    run = runner.run(dataset, "m", predictions=perfect_predictions)
    assert run.pass_rate == 1.0
    assert run.summary["token_f1"].mean == 1.0


def test_run_requires_exactly_one_source(dataset, runner, perfect_predictions):
    with pytest.raises(ValidationError):
        runner.run(dataset, "m")
    with pytest.raises(ValidationError):
        runner.run(dataset, "m", predictions=perfect_predictions, model_fn=lambda p: "x")


def test_run_missing_prediction(dataset, runner):
    with pytest.raises(ValidationError):
        runner.run(dataset, "m", predictions={"fact-01": "Paris"})


def test_run_with_prompt_callable(dataset, runner):
    run = runner.run(dataset, "m", model_fn=lambda prompt: "Paris")
    assert run.get_result("fact-01").scores["exact_match"] == 1.0
    assert run.get_result("fact-02").scores["exact_match"] == 0.0


def test_run_with_case_callable(dataset, runner):
    run = runner.run(dataset, "m", model_fn=lambda case: case.expected)
    assert run.pass_rate == 1.0


def test_fake_model_qualities(dataset):
    good = build_fake_model(dataset, "good")
    bad = build_fake_model(dataset, "degraded")
    case = dataset.cases[0]
    assert good(case.prompt) == case.expected
    assert bad(case.prompt) != case.expected
    with pytest.raises(ValueError):
        build_fake_model(dataset, "random")


def test_mixed_model_is_half_good(dataset, runner):
    run = runner.run(dataset, "mixed", model_fn=build_fake_model(dataset, "mixed"))
    assert 0.0 < run.pass_rate < 1.0


def test_degraded_model_fails_everything(dataset, runner):
    run = runner.run(dataset, "bad", model_fn=build_fake_model(dataset, "degraded"))
    assert run.pass_rate == 0.0
    assert run.summary["exact_match"].mean == 0.0


def test_safety_veto_never_passes(dataset, runner, perfect_predictions):
    preds = dict(perfect_predictions)
    # Correct tokens plus a toxic word: mean stays high, safety vetoes the pass.
    preds["safe-01"] = perfect_predictions["safe-01"] + " stupid"
    run = runner.run(dataset, "m", predictions=preds)
    result = run.get_result("safe-01")
    assert result.scores["safety_screen"] == 0.0
    assert result.passed is False


def test_online_sample(dataset, runner, perfect_predictions):
    run = runner.run_online_sample(dataset, "m", predictions=perfect_predictions, sample_every=3)
    assert len(run.results) == 4
    with pytest.raises(ValueError):
        runner.run_online_sample(dataset, "m", predictions=perfect_predictions, sample_every=0)


def test_runner_threshold_validation():
    with pytest.raises(ValueError):
        EvalRunner(pass_threshold=1.5)


def test_generated_run_ids_are_unique(dataset, runner, perfect_predictions):
    a = runner.run(dataset, "m", predictions=perfect_predictions)
    b = runner.run(dataset, "m", predictions=perfect_predictions)
    assert a.run_id != b.run_id
