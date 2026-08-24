"""Command-line entry points for the SilentML environment.

    python -m silentml.cli generate            # build all episodes
    python -m silentml.cli generate --op ACH   # build one episode
    python -m silentml.cli demo ACH            # run the oracle policy + judge on one
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml
from pathlib import Path

from silentml.agent.llm_policy import DEFAULT_BASE_URL, DEFAULT_MODEL
from silentml.bugs.operators import operators_for
from silentml.generation import TriageError, generate_episode


def _generate(args) -> int:
    ops = [args.op] if args.op else [o.id for o in operators_for(args.pipeline)]
    rc = 0
    for op in ops:
        try:
            p = generate_episode(args.pipeline, op, out_root=args.out,
                                 gen_seeds=tuple(args.seeds), drop_margin=args.margin)
            meta = yaml.safe_load((p / "meta.yaml").read_text(encoding="utf-8"))
            print(f"[ok]     {op}: clean={meta['clean_mean_val_accuracy']:.3f} "
                  f"buggy={meta['buggy_mean_val_accuracy']:.3f} -> {p}")
        except TriageError as e:
            print(f"[reject] {op}: {e}")
            rc = 1
    return rc


def _demo(args) -> int:
    from silentml.agent.loop import run_episode, oracle_policy

    ep = Path(args.out) / f"{args.pipeline}__{args.op}"
    if not ep.exists():
        print(f"episode not found: {ep} (run `generate` first)", file=sys.stderr)
        return 2
    meta = yaml.safe_load((ep / "meta.yaml").read_text(encoding="utf-8"))
    diagnosis = f'{{"diagnosis": "{meta["operator_name"]} - {meta["bug_mechanism"]}"}}'
    policy = oracle_policy(meta["ground_truth_fix_diff"], diagnosis)
    res = run_episode(ep, policy, max_calls=10, judge_seeds=(0, 1))
    print(f"episode={res.episode_id} submitted={res.submitted} calls={res.n_tool_calls}")
    if res.judge:
        print(res.judge.render())
    return 0


def _episodes(args) -> int:
    """List the generated episode set with severity and taxonomy."""
    from silentml.benchmark import _family_of

    root = Path(args.out)
    rows = []
    for ep in sorted(root.glob("*/meta.yaml")):
        m = yaml.safe_load(ep.read_text(encoding="utf-8"))
        clean = m["clean_mean_val_accuracy"]
        buggy = m["buggy_mean_val_accuracy"]
        rows.append((m["operator_id"], m["difficulty"], clean, buggy, clean - buggy,
                     _family_of(m)))
    if not rows:
        print(f"no episodes in {root} (run `generate` first)", file=sys.stderr)
        return 2

    rows.sort(key=lambda r: -r[4])
    print(f"\n{len(rows)} episodes in {root}\n")
    print(f"  {'operator':<16}{'difficulty':<11}{'clean':>7}{'buggy':>8}{'drop':>8}  family")
    print("  " + "-" * 78)
    for op, diff, clean, buggy, drop, fam in rows:
        print(f"  {op:<16}{diff:<11}{clean:>7.3f}{buggy:>8.3f}{drop:>8.3f}  "
              f"{fam}")
    print()
    return 0


def _setup_model(args) -> int:
    from silentml.setup_model import setup

    return setup(base_url=args.base_url, model=args.model,
                 do_pull=not args.no_pull, check=not args.no_check)


def _benchmark(args) -> int:
    from silentml.benchmark import run_benchmark, save_report

    report = run_benchmark(
        episodes_dir=args.out,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key or os.environ.get("SILENTML_API_KEY"),
        max_calls=args.max_calls,
        judge_seeds=tuple(args.seeds),
        temperature=args.temperature,
        limit=args.limit,
        timeout=args.timeout,
        trajectory_dir=args.trajectory_dir,
    )
    print(report.render())
    if args.report:
        save_report(report, args.report)
        print(f"\nwrote {args.report}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="silentml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate silent-bug episodes")
    g.add_argument("--pipeline", default="transformer_text")
    g.add_argument("--op", default=None, help="single operator id (default: all applicable)")
    g.add_argument("--out", default="episodes")
    g.add_argument("--margin", type=float, default=0.05,
                   help="minimum accuracy drop for a mutant to count as a silent bug")
    g.add_argument("--seeds", type=int, nargs="+", default=[0, 1],
                   help="seeds averaged during triage; use a single seed to halve runtime")
    g.set_defaults(func=_generate)

    d = sub.add_parser("demo", help="run the oracle policy + judge on one episode")
    d.add_argument("op", help="operator id, e.g. ACH")
    d.add_argument("--pipeline", default="transformer_text")
    d.add_argument("--out", default="episodes")
    d.set_defaults(func=_demo)

    e = sub.add_parser("episodes", help="list generated episodes with severity")
    e.add_argument("--out", default="episodes")
    e.set_defaults(func=_episodes)

    s = sub.add_parser("setup-model",
                       help="pick a Qwen model for this GPU, pull it, verify tool calls")
    s.add_argument("--model", default=None,
                   help="override the automatic choice with an exact Ollama tag")
    s.add_argument("--base-url", default=DEFAULT_BASE_URL)
    s.add_argument("--no-pull", action="store_true",
                   help="skip downloading (just recommend and verify)")
    s.add_argument("--no-check", action="store_true",
                   help="skip the live endpoint/tool-call probe")
    s.set_defaults(func=_setup_model)

    b = sub.add_parser("benchmark", help="score an LLM across all episodes")
    b.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"model name on the endpoint (default: {DEFAULT_MODEL})")
    b.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help=f"OpenAI-compatible endpoint (default: {DEFAULT_BASE_URL})")
    b.add_argument("--api-key", default=None,
                   help="bearer token if the endpoint needs one (env SILENTML_API_KEY)")
    b.add_argument("--out", default="episodes", help="episodes directory")
    b.add_argument("--report", default="benchmark_report.json")
    b.add_argument("--max-calls", type=int, default=20)
    b.add_argument("--seeds", type=int, nargs="+", default=[0, 1],
                   help="judge retraining seeds")
    b.add_argument("--temperature", type=float, default=0.0)
    b.add_argument("--timeout", type=float, default=180.0,
                   help="seconds per model request; raise it for CPU inference, "
                        "where a large model needs minutes per reply")
    b.add_argument("--limit", type=int, default=None, help="only the first N episodes")
    b.add_argument("--trajectory-dir", default="trajectories",
                   help="write each episode's tool calls here for diagnosis")
    b.set_defaults(func=_benchmark)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
