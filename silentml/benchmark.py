"""Benchmark runner: score a model across every episode and report solve rates.

Produces the headline result of the environment - a per-bug-family breakdown of
how often a model actually repairs the pipeline. Three rates are reported, and
they are deliberately distinct:

  solve_rate      - passed the functional gate (accuracy + repair rate)
  causal_rate     - passed functional AND causal ablation (the fix is the reason
                    the metric recovered, not a coincidence)
  diagnosis_rate  - correctly named the fault class

``causal_rate`` is the honest headline number: a patch can lift accuracy without
addressing the injected fault, and only ablation catches that.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from silentml.agent.llm_policy import LLMPolicy
from silentml.bugs.operators import get_operator
from silentml.agent.loop import run_episode


@dataclass
class EpisodeScore:
    episode_id: str
    operator_id: str
    family: str
    difficulty: str
    submitted: bool
    functional_pass: bool
    ablation: str
    diagnosis_ok: bool
    reward: float
    n_tool_calls: int
    parse_failures: int
    error: str | None = None

    @property
    def causal_ok(self) -> bool:
        return self.functional_pass and self.ablation == "pass"


@dataclass
class BenchmarkReport:
    model: str
    episodes: list[EpisodeScore] = field(default_factory=list)
    wall_seconds: float = 0.0

    # -- aggregates -----------------------------------------------------------
    def _rate(self, attr) -> float:
        if not self.episodes:
            return 0.0
        return sum(bool(attr(e)) for e in self.episodes) / len(self.episodes)

    def summary(self) -> dict[str, Any]:
        by_family: dict[str, dict[str, Any]] = {}
        for e in self.episodes:
            f = by_family.setdefault(e.family, {"n": 0, "solved": 0, "causal": 0, "diag": 0})
            f["n"] += 1
            f["solved"] += int(e.functional_pass)
            f["causal"] += int(e.causal_ok)
            f["diag"] += int(e.diagnosis_ok)
        return {
            "model": self.model,
            "n_episodes": len(self.episodes),
            "solve_rate": self._rate(lambda e: e.functional_pass),
            "causal_rate": self._rate(lambda e: e.causal_ok),
            "diagnosis_rate": self._rate(lambda e: e.diagnosis_ok),
            "submit_rate": self._rate(lambda e: e.submitted),
            "mean_reward": (statistics.mean(e.reward for e in self.episodes)
                            if self.episodes else 0.0),
            "mean_tool_calls": (statistics.mean(e.n_tool_calls for e in self.episodes)
                                if self.episodes else 0.0),
            "total_parse_failures": sum(e.parse_failures for e in self.episodes),
            "n_errors": sum(1 for e in self.episodes if e.error),
            "by_family": by_family,
            "wall_seconds": self.wall_seconds,
        }

    def render(self) -> str:
        s = self.summary()
        lines = [
            "",
            f"MODEL: {s['model']}   episodes: {s['n_episodes']}   "
            f"wall: {s['wall_seconds']:.0f}s",
            "-" * 72,
            f"  solve rate (functional gate) : {s['solve_rate']*100:5.1f}%",
            f"  causal rate (fix is the cause): {s['causal_rate']*100:5.1f}%",
            f"  diagnosis rate                : {s['diagnosis_rate']*100:5.1f}%",
            f"  submitted                     : {s['submit_rate']*100:5.1f}%",
            f"  mean reward                   : {s['mean_reward']:+.2f}",
            f"  mean tool calls               : {s['mean_tool_calls']:.1f}",
            f"  tool-call parse failures      : {s['total_parse_failures']}",
        ]
        if s["n_errors"]:
            lines.append(
                f"  !! {s['n_errors']} episode(s) errored (model/transport failure) - "
                f"these are scored as failures but are NOT debugging failures; "
                f"see 'error' in the JSON report"
            )
        lines += [
            "",
            f"  {'bug family':<28}{'n':>3}{'solved':>9}{'causal':>9}{'diag':>7}",
            "  " + "-" * 56,
        ]
        for fam, d in sorted(s["by_family"].items()):
            lines.append(
                f"  {fam:<28}{d['n']:>3}{d['solved']:>9}{d['causal']:>9}{d['diag']:>7}"
            )
        lines.append("")
        lines.append(f"  {'episode':<26}{'solved':>8}{'ablation':>11}{'reward':>9}")
        lines.append("  " + "-" * 56)
        for e in sorted(self.episodes, key=lambda x: x.operator_id):
            mark = "yes" if e.functional_pass else "no"
            lines.append(
                f"  {e.operator_id:<26}{mark:>8}{e.ablation:>11}{e.reward:>+9.2f}"
            )
        return "\n".join(lines)


def _family_of(meta: dict) -> str:
    """Group episodes by fault family for the breakdown table.

    The operator registry is the source of truth, so episodes generated before a
    taxonomy field existed still group correctly; meta.yaml is the fallback.
    """
    taxonomy = meta.get("taxonomy") or {}
    jahan = taxonomy.get("jahan2025") or ""
    group = taxonomy.get("deepcrime_group", "unknown")
    try:
        op = get_operator(meta["operator_id"])
        jahan = jahan or op.jahan2025
        group = op.deepcrime_group or group
    except (KeyError, TypeError):
        pass
    if jahan:
        return jahan.split("(")[0].strip()
    return group


def run_benchmark(
    episodes_dir: str | Path,
    model: str,
    base_url: str,
    api_key: str | None = None,
    max_calls: int = 20,
    judge_seeds: tuple[int, ...] = (0, 1),
    temperature: float = 0.0,
    limit: int | None = None,
    verbose: bool = True,
) -> BenchmarkReport:
    episodes_dir = Path(episodes_dir)
    eps = sorted(p for p in episodes_dir.iterdir()
                 if p.is_dir() and (p / "meta.yaml").exists())
    if limit:
        eps = eps[:limit]

    report = BenchmarkReport(model=model)
    t0 = time.time()
    for i, ep in enumerate(eps, 1):
        meta = yaml.safe_load((ep / "meta.yaml").read_text(encoding="utf-8"))
        policy = LLMPolicy(model=model, base_url=base_url, api_key=api_key,
                           temperature=temperature)
        if verbose:
            print(f"[{i}/{len(eps)}] {ep.name} ...", flush=True)

        error = None
        try:
            result = run_episode(ep, policy, max_calls=max_calls, judge_seeds=judge_seeds)
            judge = result.judge
            score = EpisodeScore(
                episode_id=result.episode_id,
                operator_id=meta["operator_id"],
                family=_family_of(meta),
                difficulty=meta.get("difficulty", "?"),
                submitted=result.submitted,
                functional_pass=bool(judge and judge.functional_pass),
                ablation=(judge.ablation if judge else "not_run"),
                diagnosis_ok=bool(judge and judge.details.get("diagnosis_ok")),
                reward=(judge.reward if judge else 0.0),
                n_tool_calls=result.n_tool_calls,
                parse_failures=policy.parse_failures,
            )
        except Exception as e:  # a model/transport failure must not abort the sweep
            error = f"{type(e).__name__}: {e}"
            score = EpisodeScore(
                episode_id=ep.name, operator_id=meta["operator_id"],
                family=_family_of(meta), difficulty=meta.get("difficulty", "?"),
                submitted=False, functional_pass=False, ablation="not_run",
                diagnosis_ok=False, reward=0.0, n_tool_calls=0,
                parse_failures=getattr(policy, "parse_failures", 0), error=error,
            )
        if verbose:
            print(f"      solved={score.functional_pass} ablation={score.ablation} "
                  f"reward={score.reward:+.2f}" + (f" ERROR={error}" if error else ""),
                  flush=True)
        report.episodes.append(score)

    report.wall_seconds = time.time() - t0
    return report


def save_report(report: BenchmarkReport, path: str | Path) -> None:
    payload = {"summary": report.summary(),
               "episodes": [asdict(e) for e in report.episodes]}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
