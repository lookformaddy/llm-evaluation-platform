import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from llm_evaluation_platform.datasets import build_default_dataset  # noqa: E402
from llm_evaluation_platform.service import create_app  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(create_app())


@pytest.fixture()
def predictions():
    ds = build_default_dataset()
    return {c.case_id: c.expected for c in ds.cases}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_datasets_and_scorers(client):
    assert client.get("/datasets").json()["datasets"][0]["dataset_id"] == "golden-core"
    assert client.get("/datasets/golden-core").status_code == 200
    assert client.get("/datasets/nope").status_code == 404
    assert "exact_match" in client.get("/scorers").json()["scorers"]


def test_run_gate_compare_flow(client, predictions):
    r = client.post("/runs", json={"model_name": "svc-good", "predictions": predictions, "run_id": "svc-1"})
    assert r.status_code == 200 and r.json()["pass_rate"] == 1.0
    bad = {k: "an unrelated answer about something else entirely" for k in predictions}
    r = client.post("/runs", json={"model_name": "svc-bad", "predictions": bad, "run_id": "svc-2"})
    assert r.status_code == 200 and r.json()["pass_rate"] == 0.0
    assert client.get("/runs/svc-1").status_code == 200
    assert client.get("/runs/nope").status_code == 404
    assert len(client.get("/runs").json()["runs"]) == 2
    gate = client.post("/gates/evaluate", json={"candidate_run_id": "svc-2", "baseline_run_id": "svc-1"})
    assert gate.json()["passed"] is False
    gate = client.post("/gates/evaluate", json={"candidate_run_id": "svc-1"})
    assert gate.json()["passed"] is True
    cmp = client.get("/compare", params={"baseline_run_id": "svc-1", "candidate_run_id": "svc-2"})
    assert cmp.json()["overall_winner"] == "baseline"


def test_run_missing_prediction_is_422(client):
    r = client.post("/runs", json={"model_name": "x", "predictions": {"fact-01": "Paris"}})
    assert r.status_code == 422


def test_trends_and_flaky(client, predictions):
    client.post("/runs", json={"model_name": "a", "predictions": predictions, "run_id": "t-1"})
    bad = {k: "unrelated words here now" for k in predictions}
    client.post("/runs", json={"model_name": "b", "predictions": bad, "run_id": "t-2"})
    trend = client.get("/trends/token_f1")
    assert trend.status_code == 200 and trend.json()["direction"] == "declining"
    flaky = client.get("/flaky")
    assert flaky.status_code == 200 and len(flaky.json()["flaky_cases"]) == 12
