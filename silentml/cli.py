"""Command-line entry points for the SilentML environment.

    python -m silentml.cli generate            # build all episodes
    python -m silentml.cli generate --op ACH   # build one episode
    python -m silentml.cli demo ACH            # run the oracle policy + judge on one
"""

from __future__ import annotations

import argparse
import sys

import yaml
from pathlib import Path

from silentml.bugs.operators import operators_for
from silentml.generation import TriageError, generate_episode


def _generate(args) -> int:
    ops = [args.op] if args.op else [o.id for o in operators_for(args.pipeline)]
    rc = 0
    for op in ops:
        try:
            p = generate_episode(args.pipeline, op, out_root=args.out,
                                 gen_seeds=(0, 1), drop_margin=args.margin)
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="silentml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate silent-bug episodes")
    g.add_argument("--pipeline", default="cnn_fashion")
    g.add_argument("--op", default=None, help="single operator id (default: all applicable)")
    g.add_argument("--out", default="episodes")
    g.add_argument("--margin", type=float, default=0.05)
    g.set_defaults(func=_generate)

    d = sub.add_parser("demo", help="run the oracle policy + judge on one episode")
    d.add_argument("op", help="operator id, e.g. ACH")
    d.add_argument("--pipeline", default="cnn_fashion")
    d.add_argument("--out", default="episodes")
    d.set_defaults(func=_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
