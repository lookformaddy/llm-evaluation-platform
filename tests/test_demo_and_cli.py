from llm_evaluation_platform.cli import main
from llm_evaluation_platform.demo import run_demo


def test_demo_runs_all_scenarios():
    result = run_demo()
    assert result["gate_baseline_passed"] is True
    assert result["gate_candidate_passed"] is False
    assert result["gate_candidate_failures"] > 0
    assert result["comparison"]["overall_winner"] == "baseline"
    assert result["store_runs"] == 3
    assert result["trend"]["direction"] == "declining"


def test_demo_dataset_info():
    result = run_demo()
    assert result["dataset"]["cases"] == 12
    assert result["dataset"]["version"] == "1.0.0"


def test_cli_scorers(capsys):
    assert main(["scorers"]) == 0
    out = capsys.readouterr().out
    assert "exact_match" in out and "safety_screen" in out


def test_cli_datasets(capsys):
    assert main(["datasets"]) == 0
    out = capsys.readouterr().out
    assert "golden-core" in out and "fact-01" in out


def test_cli_run(capsys):
    assert main(["run", "--quality", "good", "--model", "cli-model"]) == 0
    out = capsys.readouterr().out
    assert "cli-model" in out and "run-" in out


def test_cli_demo(capsys):
    assert main(["demo"]) == 0
    assert "LLM Evaluation Platform demo" in capsys.readouterr().out
