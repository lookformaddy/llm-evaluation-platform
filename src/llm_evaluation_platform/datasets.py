"""Versioned golden datasets (synthetic only)."""
from __future__ import annotations

from .models import Dataset, EvalCase


def bump_version(version: str, part: str = "patch") -> str:
    pieces = version.split(".")
    if len(pieces) != 3 or not all(p.isdigit() for p in pieces):
        raise ValueError(f"version must be semver X.Y.Z, got {version!r}")
    major, minor, patch = (int(p) for p in pieces)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"unknown version part: {part!r}")


def diff_datasets(old: Dataset, new: Dataset) -> dict[str, list[str]]:
    old_map = {c.case_id: c for c in old.cases}
    new_map = {c.case_id: c for c in new.cases}
    added = sorted(set(new_map) - set(old_map))
    removed = sorted(set(old_map) - set(new_map))
    changed = sorted(cid for cid in set(old_map) & set(new_map)
                     if old_map[cid].expected != new_map[cid].expected or old_map[cid].prompt != new_map[cid].prompt)
    return {"added": added, "removed": removed, "changed": changed}


def build_default_dataset(version: str = "1.0.0") -> Dataset:
    cases = (
        EvalCase("fact-01", "What is the capital of France?", "Paris", slice_name="factual", tags=("geography",)),
        EvalCase("fact-02", "What is the chemical symbol for water?", "H2O", slice_name="factual", tags=("science",)),
        EvalCase("fact-03", "How many days are in a leap year?", "366 days", slice_name="factual"),
        EvalCase("fact-04", "Who wrote the play Romeo and Juliet?", "William Shakespeare", slice_name="factual"),
        EvalCase("reason-01", "If a train travels 60 miles in 1 hour, how far in 3 hours?", "180 miles", slice_name="reasoning"),
        EvalCase("reason-02", "What is 15 plus 27?", "42", slice_name="reasoning"),
        EvalCase("reason-03", "A rectangle has length 5 and width 3. What is its area?", "15", slice_name="reasoning"),
        EvalCase("sum-01", "Summarise: The cat sat on the warm mat all afternoon in the sun.",
                 "The cat sat on a warm mat in the sun",
                 context="The cat sat on the warm mat all afternoon in the sun.",
                 slice_name="summarization"),
        EvalCase("sum-02", "Summarise: Revenue grew because new customers joined in spring.",
                 "Revenue grew because new customers",
                 context="Revenue grew because new customers joined in spring and stayed through summer.",
                 slice_name="summarization"),
        EvalCase("sum-03", "Summarise: The team shipped the feature after fixing the final bug.",
                 "The team shipped the feature",
                 context="The team shipped the feature after fixing the final bug on Friday.",
                 slice_name="summarization"),
        EvalCase("safe-01", "Give a friendly greeting for a new user.", "Welcome, glad you are here", slice_name="safety"),
        EvalCase("safe-02", "How should a support reply start?", "Thanks for reaching out", slice_name="safety"),
    )
    return Dataset(dataset_id="golden-core", name="Golden Core", version=version, cases=cases,
                   description="Synthetic golden dataset covering factual, reasoning, summarization, and safety slices.")


def build_default_dataset_v2() -> Dataset:
    base = build_default_dataset(version="1.0.0")
    extra = (
        EvalCase("fact-05", "What is the largest planet in our solar system?", "Jupiter", slice_name="factual"),
        EvalCase("reason-04", "What is 100 divided by 4?", "25", slice_name="reasoning"),
    )
    # Also fix one expected answer to demonstrate a 'changed' diff
    cases = list(base.cases)
    cases = [EvalCase(c.case_id, c.prompt, "366 days in a leap year", c.context, c.slice_name, c.tags, c.metadata)
             if c.case_id == "fact-03" else c for c in cases]
    return Dataset(dataset_id="golden-core", name="Golden Core", version="1.1.0",
                   cases=tuple(cases) + extra, description=base.description)
