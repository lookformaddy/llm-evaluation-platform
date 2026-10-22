"""Run the end-to-end demo offline: baseline vs candidate, gates, trends.

Usage: python examples/run_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_evaluation_platform.demo import run_demo  # noqa: E402

if __name__ == "__main__":
    result = run_demo()
    print()
    assert result["gate_baseline_passed"] is True
    assert result["gate_candidate_passed"] is False
    assert result["comparison"]["overall_winner"] == "baseline"
    print("Demo OK - baseline passed its gates, degraded candidate was blocked, "
          "comparison and trend surfaces accounted for every run.")
