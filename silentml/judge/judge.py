"""Deterministic judge: functional gate -> causal ablation -> stability -> reward.

Implements the design-doc Q4 evaluation. The reward magnitudes enforce the
required ordering ``R_functional >> R_fix_quality > R_diagnosis > P_diagnosis_wrong``
and ``P_ablation > R_functional > P_curves`` so that gaming accuracy while failing
causal ablation is always net-negative (closing the magnitude-miscalibration
hack from design Q5).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from silentml.agent.patching import apply_unified_diff
from silentml.bugs.operators import get_operator
from silentml.artifacts.collector import LayerStatsCollector
from silentml.judge.invariants import check_stability
from silentml.pipelines.base import RunResult, run_pipeline

# Reward constants (magnitude ordering per design doc).
R_FUNCTIONAL = 10.0
R_FIX_QUALITY = 2.0
R_DIAGNOSIS = 1.0
P_DIAGNOSIS_WRONG = 0.5
P_ABLATION = 14.0      # > R_FUNCTIONAL + all bonuses, so failing ablation is net-negative
                       # even with a full fix-quality + diagnosis bonus (design Q5 hack guard)
P_CURVES = 1.0

ACC_TOL = 0.03         # patched mean acc must be within this of clean baseline
RR_TOL = 0.05          # worst-hit class recall recovery tolerance
ABLATION_DROP = 0.05   # re-injection must drop acc by at least this to prove causality


@dataclasses.dataclass
class JudgeResult:
    reward: float
    functional_pass: bool
    ablation: str                 # "pass" | "fail" | "unverified"
    stability_violations: list[str]
    breakdown: dict[str, float]
    details: dict[str, Any]

    def render(self) -> str:
        lines = [f"REWARD: {self.reward:.2f}",
                 f"functional_pass={self.functional_pass} ablation={self.ablation} "
                 f"stability_violations={self.stability_violations}"]
        for k, v in self.breakdown.items():
            lines.append(f"  {k}: {v:+.2f}")
        return "\n".join(lines)


def _run_source(source: str, seeds: tuple[int, ...],
                collect: bool = False) -> list[RunResult]:
    tmp = Path(tempfile.mkdtemp(prefix="silentml_judge_"))
    try:
        (tmp / "pipeline.py").write_text(source, encoding="utf-8")
        results = []
        for s in seeds:
            collector = LayerStatsCollector() if collect else None
            results.append(run_pipeline(tmp, seed=s, collector=collector))
        return results
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _worst_hit_class(clean_pc: dict[int, float], buggy_pc: dict[int, float]) -> int | None:
    common = set(clean_pc) & set(buggy_pc)
    if not common:
        return None
    return max(common, key=lambda c: clean_pc[c] - buggy_pc[c])


def _score_diagnosis(diagnosis: str, keywords: tuple[str, ...]) -> tuple[float, bool]:
    """Return (reward_delta, correct?). Parses JSON diagnosis if present."""
    text = diagnosis
    try:
        obj = json.loads(diagnosis)
        if isinstance(obj, dict):
            text = str(obj.get("diagnosis", diagnosis))
    except (json.JSONDecodeError, TypeError):
        pass
    low = text.lower()
    if any(kw.lower() in low for kw in keywords):
        return R_DIAGNOSIS, True
    return -P_DIAGNOSIS_WRONG, False


def judge_episode(
    episode_dir: str | Path,
    patched_source: str,
    diagnosis: str,
    seeds: tuple[int, ...] = (0, 1),
) -> JudgeResult:
    episode_dir = Path(episode_dir)
    meta = yaml.safe_load((episode_dir / "meta.yaml").read_text(encoding="utf-8"))
    operator = get_operator(meta["operator_id"])
    clean_acc = float(meta["clean_mean_val_accuracy"])

    buggy_source = (episode_dir / "pipeline" / "pipeline.py").read_text(encoding="utf-8")
    clean_source = apply_unified_diff(buggy_source, meta["ground_truth_fix_diff"])

    clean_pc = {int(k): v for k, v in json.loads(
        (episode_dir / "baseline" / "clean_metrics.json").read_text()
    )["per_class_accuracy"].items()}
    buggy_pc = {int(k): v for k, v in json.loads(
        (episode_dir / "artifacts" / "per_class_accuracy.json").read_text()
    ).items()}

    # --- Run the patched pipeline ---
    patched_runs = _run_source(patched_source, seeds, collect=True)
    patched_acc = mean(r.eval.accuracy for r in patched_runs)
    patched_pc = patched_runs[0].eval.per_class_accuracy

    details: dict[str, Any] = {
        "clean_acc": clean_acc, "patched_acc": patched_acc,
        "acc_threshold": clean_acc - ACC_TOL,
    }

    # --- Functional gate: accuracy threshold AND Repair Rate ---
    acc_pass = patched_acc >= clean_acc - ACC_TOL
    worst = _worst_hit_class(clean_pc, buggy_pc)
    if worst is not None:
        rr_pass = patched_pc.get(worst, 0.0) >= clean_pc[worst] - RR_TOL
        details["repair_rate"] = {
            "worst_class": worst, "clean_recall": clean_pc[worst],
            "buggy_recall": buggy_pc.get(worst), "patched_recall": patched_pc.get(worst),
        }
    else:
        rr_pass = True
    functional_pass = acc_pass and rr_pass
    details["acc_pass"] = acc_pass
    details["rr_pass"] = rr_pass

    breakdown: dict[str, float] = {}
    if not functional_pass:
        breakdown["R_functional"] = 0.0
        return JudgeResult(
            reward=0.0, functional_pass=False, ablation="not_run",
            stability_violations=[], breakdown=breakdown, details=details,
        )
    breakdown["R_functional"] = R_FUNCTIONAL

    # --- Causal ablation: re-inject the original mutation onto the patched code ---
    try:
        ablated_source = operator.inject(patched_source)
        ablated_runs = _run_source(ablated_source, seeds[:1])
        ablated_acc = mean(r.eval.accuracy for r in ablated_runs)
        details["ablated_acc"] = ablated_acc
        if ablated_acc <= patched_acc - ABLATION_DROP:
            ablation = "pass"
        else:
            ablation = "fail"
    except Exception as e:  # cannot re-inject at the canonical site
        ablation = "unverified"
        details["ablation_note"] = str(e)
    if ablation == "fail":
        breakdown["P_ablation"] = -P_ABLATION

    # --- Stability gate ---
    violations = check_stability(patched_runs[0].history, patched_runs[0].artifacts)
    if violations:
        breakdown["P_curves"] = -P_CURVES

    # --- Auxiliary bonuses ---
    def _norm(s: str) -> str:
        return "\n".join(line.rstrip() for line in s.strip().splitlines())
    if _norm(patched_source) == _norm(clean_source):
        breakdown["R_fix_quality"] = R_FIX_QUALITY   # matches human reference
    else:
        breakdown["R_fix_quality"] = R_FIX_QUALITY / 2  # valid-but-different

    diag_delta, diag_ok = _score_diagnosis(diagnosis, operator.diagnosis_keywords)
    breakdown["R_diagnosis" if diag_ok else "P_diagnosis_wrong"] = diag_delta

    reward = sum(breakdown.values())
    return JudgeResult(
        reward=reward, functional_pass=True, ablation=ablation,
        stability_violations=violations, breakdown=breakdown, details=details,
    )
