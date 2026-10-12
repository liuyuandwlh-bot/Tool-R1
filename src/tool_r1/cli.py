"""Command line interface; the core runs with the Python standard library."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tool-R1: reproducible tool reasoning and GRPO experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Create the deterministic synthetic curriculum")
    generate.add_argument("--data", default="data")
    for split, size in (("train", 2400), ("validation", 400), ("test", 800)):
        generate.add_argument("--" + split, type=int, default=size)
    generate.add_argument("--seed", type=int, default=42)
    audit = commands.add_parser("audit")
    audit.add_argument("--data", default="data")
    variance = commands.add_parser("sample", help="GRPO variance preflight: K trajectories per question, no training")
    variance.add_argument("--data", default="data")
    variance.add_argument("--output", required=True)
    variance.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    variance.add_argument("--model", default="tool-r1")
    variance.add_argument("--split", default="validation", choices=["validation", "test"])
    variance.add_argument("--budgets", type=int, nargs="+", default=[4])
    variance.add_argument("--limit", type=int, default=32)
    variance.add_argument("--trajectories", type=int, default=4)
    variance.add_argument("--temperature", type=float, default=0.7)
    variance.add_argument("--seed", type=int, default=42)
    variance.add_argument("--max-tokens", type=int, default=384)
    variance.add_argument("--no-budget-visible", action="store_true")
    smoke = commands.add_parser("smoke", help="Teacher replay only; no model performance claims")
    smoke.add_argument("--data", default="data")
    smoke.add_argument("--output", default="outputs/scripted-smoke")
    smoke.add_argument("--limit", type=int)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--require-a6000", action="store_true")
    preflight.add_argument("--require-stack", action="store_true")
    preflight.add_argument("--output")
    evaluation = commands.add_parser("evaluate", help="Evaluate an actual running model service")
    evaluation.add_argument("--data", default="data")
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--method", required=True, choices=[f"M{i}" for i in range(6)])
    evaluation.add_argument("--model-source", required=True, help="Served base HF model ID (M0/1) or exported local model (M2-5)")
    evaluation.add_argument("--split", default="validation", choices=["validation", "test"])
    evaluation.add_argument("--budgets", type=int, nargs="+", default=[2, 4])
    evaluation.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    evaluation.add_argument("--model", default="tool-r1", help="Serving alias")
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument("--max-tokens", type=int, default=384)
    evaluation.add_argument("--temperature", type=float, default=0.0)
    evaluation.add_argument("--backend", choices=["restricted", "docker"], default="restricted")
    compare = commands.add_parser("compare")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--iterations", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--accuracy-margin", type=float, default=0.02)
    report = commands.add_parser("report")
    report.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    from .experiments import write_json
    if args.command == "generate":
        from .data import generate_dataset
        result = generate_dataset(args.data, train=args.train, validation=args.validation, test=args.test, seed=args.seed)
    elif args.command == "audit":
        from .data import audit_dataset
        result = audit_dataset(args.data)
    elif args.command == "smoke":
        from .experiments import scripted_smoke
        result = scripted_smoke(args.data, args.output, args.limit)
    elif args.command == "preflight":
        from .preflight import inspect_environment
        result = inspect_environment(require_a6000=args.require_a6000, require_stack=args.require_stack)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
    elif args.command == "evaluate":
        from .experiments import evaluate
        options = vars(args).copy()
        options.pop("command")
        result = evaluate(**options, progress=lambda done, total: print(f"Evaluated {done}/{total}", flush=True))
    elif args.command == "sample":
        from .sample import sample as run_sample
        result = run_sample(args.data, args.output, base_url=args.base_url, model=args.model,
                            budgets=args.budgets, limit=args.limit, trajectories=args.trajectories,
                            temperature=args.temperature, split=args.split, seed_eval=args.seed,
                            max_tokens=args.max_tokens, budget_visible=not args.no_budget_visible)
    elif args.command == "compare":
        from .experiments import compare_runs
        result = compare_runs(args.left, args.right, iterations=args.iterations, seed=args.seed, accuracy_margin=args.accuracy_margin)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        if Path(args.output).exists():
            raise FileExistsError("Comparison already exists; choose a new output")
        write_json(args.output, result)
    else:
        from .experiments import load_run, write_report
        from .evaluation import summarize
        manifest, traces = load_run(args.run)
        result = summarize(traces)
        write_report(Path(args.run) / "report.md", manifest, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if isinstance(result, dict) and result.get("ok") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
