"""Run the benchmark through debug-gym's agent instead of our own loop.

Mirrors ``debug_gym.experiment.create_env`` rather than modifying it, so
debug-gym stays an unmodified dependency: the environment is constructed here,
tools are pulled from its Toolbox, and its agent drives the episode.

Usage (needs the Python 3.12 env with debug-gym installed):

    python -m silentml.dbg_run --only T_HLR
    python -m silentml.dbg_run --report dbg_baseline.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

DEFAULT_TOOLS = ["view", "edit", "grep", "listdir", "bash", "submit_diagnosis"]

SYSTEM_PROMPT = (
    "You are a debugging agent specialised in machine learning code. A training "
    "pipeline runs to completion without errors but the model underperforms, and "
    "exactly one bug is responsible.\n"
    "Work from evidence: read the pre-computed diagnostics in artifacts/ before "
    "reading code, and let them tell you which part of pipeline.py to inspect. Do "
    "not assume you recognise the code - a plausible-looking line may be the bug.\n"
    "You must make a tool call every turn, one at a time. Do not repeat an action "
    "that already failed or that told you something you know.\n"
    "Once you can name the faulty line, edit it. An investigation that ends "
    "without an edit scores nothing. Finish by calling submit_diagnosis with a "
    "one-sentence description of the bug."
)


def build_env(episode_dir: Path, tools: list[str], judge_seeds: tuple[int, ...],
              run_timeout: int, logger):
    from debug_gym.gym.terminals.local import LocalTerminal
    from debug_gym.gym.tools.toolbox import Toolbox

    from silentml.dbg_env import SilentMLEnv

    env = SilentMLEnv(
        task_data={"episode_dir": str(episode_dir), "env_type": "silentml"},
        terminal=LocalTerminal(),
        judge_seeds=judge_seeds,
        run_timeout=run_timeout,
        logger=logger,
    )
    for name in tools:
        env.add_tool(Toolbox.get_tool(name))
    return env


def run_one(episode_dir: Path, args, logger) -> dict:
    from debug_gym.agents.froggy_agent import FroggyAgent
    from debug_gym.llms.base import LLM

    env = build_env(episode_dir, args.tools, tuple(args.seeds),
                    args.run_timeout, logger)
    llm = LLM.instantiate(
        name=args.model,
        llm_config_file_path=args.llm_config,
        logger=logger,
        temperature=args.temperature,
    )
    agent = FroggyAgent(
        agent_args={"max_steps": args.max_steps, "system_prompt": SYSTEM_PROMPT},
        llm=llm,
        logger=logger,
    )

    started = time.time()
    try:
        agent.run(env)
        error = None
    except Exception as e:                      # keep one bad episode from
        error = f"{type(e).__name__}: {e}"      # aborting the whole sweep
        logger.warning(f"{episode_dir.name} failed: {error}")

    summary = env.summary()
    summary["wall_seconds"] = round(time.time() - started, 1)
    summary["error"] = error
    return summary


def render(rows: list[dict], model: str) -> str:
    if not rows:
        return "no episodes run"
    n = len(rows)

    def rate(key):
        return sum(bool(r[key]) for r in rows) / n * 100

    lines = [
        "",
        f"MODEL: {model} (debug-gym agent)   episodes: {n}",
        "-" * 72,
        f"  solve rate (functional gate) : {rate('functional_pass'):5.1f}%",
        f"  causal rate (fix is the cause): {rate('resolved'):5.1f}%",
        f"  diagnosis rate                : {rate('diagnosis_ok'):5.1f}%",
        f"  mean reward                   : "
        f"{statistics.mean(r['reward'] for r in rows):+.2f}",
        "",
        f"  {'episode':<18}{'solved':>8}{'ablation':>11}{'reward':>9}{'secs':>8}",
        "  " + "-" * 56,
    ]
    for r in sorted(rows, key=lambda x: x["task"]):
        mark = "yes" if r["functional_pass"] else "no"
        lines.append(f"  {r['task']:<18}{mark:>8}{r['ablation']:>11}"
                     f"{r['reward']:>+9.2f}{r['wall_seconds']:>8.0f}")
    n_err = sum(1 for r in rows if r["error"])
    if n_err:
        lines.append(f"\n  !! {n_err} episode(s) errored - see the JSON report")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="silentml.dbg_run")
    p.add_argument("--episodes", default="episodes")
    p.add_argument("--only", nargs="+", default=None,
                   help="operator ids to run, e.g. --only T_HLR")
    p.add_argument("--model", default="qwen3-coder",
                   help="key in the debug-gym llm config file")
    p.add_argument("--llm-config", default=None,
                   help="path to the debug-gym llm config yaml")
    p.add_argument("--tools", nargs="+", default=DEFAULT_TOOLS)
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1],
                   help="judge retraining seeds")
    p.add_argument("--run-timeout", type=int, default=60)
    p.add_argument("--report", default="dbg_report.json")
    args = p.parse_args(argv)

    from debug_gym.logger import DebugGymLogger

    from silentml.dbg_env import SilentMLEnv

    logger = DebugGymLogger("silentml")
    dataset = SilentMLEnv.load_dataset(args.episodes)
    names = sorted(dataset)
    if args.only:
        names = [n for n in names if n in set(args.only)]
        if not names:
            print(f"no episodes matched {args.only}; have {sorted(dataset)}",
                  file=sys.stderr)
            return 2

    rows = []
    for i, name in enumerate(names, 1):
        print(f"[{i}/{len(names)}] {name} ...", flush=True)
        row = run_one(Path(dataset[name]["episode_dir"]), args, logger)
        print(f"      solved={row['functional_pass']} ablation={row['ablation']} "
              f"reward={row['reward']:+.2f}", flush=True)
        rows.append(row)

    print(render(rows, args.model))
    if args.report:
        Path(args.report).write_text(
            json.dumps({"model": args.model, "episodes": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
