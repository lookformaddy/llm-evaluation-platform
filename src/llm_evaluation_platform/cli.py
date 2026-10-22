"""Command-line entry points.

Usage:
    llm-eval-platform demo
    llm-eval-platform datasets
    llm-eval-platform scorers
    llm-eval-platform run --quality good|degraded|mixed [--model NAME]
    llm-eval-platform serve --port 8000
"""
from __future__ import annotations

import argparse
import json


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo
    run_demo()
    return 0


def _cmd_datasets(args: argparse.Namespace) -> int:
    from .datasets import build_default_dataset
    ds = build_default_dataset()
    print(f"{ds.dataset_id} v{ds.version} checksum={ds.checksum} cases={len(ds)} slices={list(ds.slice_names)}")
    for c in ds.cases:
        print(f"  {c.case_id:10s} [{c.slice_name}] {c.prompt}")
    return 0


def _cmd_scorers(args: argparse.Namespace) -> int:
    from .scorers import SCORER_REGISTRY
    for name in sorted(SCORER_REGISTRY):
        print(name)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .datasets import build_default_dataset
    from .runner import EvalRunner, build_fake_model
    ds = build_default_dataset()
    runner = EvalRunner()
    run = runner.run(ds, args.model, model_fn=build_fake_model(ds, args.quality))
    print(json.dumps(run.to_dict(), indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    from .service import create_app
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-eval-platform", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="Run the end-to-end offline demo"); p.set_defaults(func=_cmd_demo)
    p = sub.add_parser("datasets", help="List the default golden dataset"); p.set_defaults(func=_cmd_datasets)
    p = sub.add_parser("scorers", help="List available scorers"); p.set_defaults(func=_cmd_scorers)
    p = sub.add_parser("run", help="Run one evaluation with a fake model")
    p.add_argument("--quality", default="good", choices=["good", "degraded", "mixed"])
    p.add_argument("--model", default="fake-model")
    p.set_defaults(func=_cmd_run)
    p = sub.add_parser("serve", help="Run the FastAPI service")
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=_cmd_serve)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
