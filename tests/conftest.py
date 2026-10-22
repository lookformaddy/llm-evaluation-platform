import pytest

from llm_evaluation_platform import EvalRunner, build_default_dataset
from llm_evaluation_platform.runner import build_fake_model


@pytest.fixture()
def dataset():
    return build_default_dataset()


@pytest.fixture()
def runner():
    return EvalRunner()


@pytest.fixture()
def good_run(dataset, runner):
    return runner.run(dataset, "good-model", model_fn=build_fake_model(dataset, "good"), run_id="run-good")


@pytest.fixture()
def degraded_run(dataset, runner):
    return runner.run(dataset, "bad-model", model_fn=build_fake_model(dataset, "degraded"), run_id="run-bad")


@pytest.fixture()
def mixed_run(dataset, runner):
    return runner.run(dataset, "mixed-model", model_fn=build_fake_model(dataset, "mixed"), run_id="run-mixed")


@pytest.fixture()
def perfect_predictions(dataset):
    return {c.case_id: c.expected for c in dataset.cases}
