"""Episode generator: clean template + operator -> a triaged silent-bug episode.

An episode directory has the shape the design doc specifies::

    episodes/<id>/
      pipeline/pipeline.py     # buggy source (what the agent views/patches)
      artifacts/*.json         # 4-layer diagnostics, served by read_artifact
      baseline/                # clean metrics + buggy checkpoint
      meta.yaml                # HIDDEN ground truth (operator, fix diff, metrics)
      task.yaml                # VISIBLE prompt fields (bug description, manifest)

Only mutants that degrade the target metric without crashing under fixed seeds
are kept (the triage filter), so every episode is a genuine *silent* failure.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import shutil
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import yaml

from silentml.artifacts.collector import LayerStatsCollector
from silentml.bugs.operators import Operator, get_operator
from silentml.pipelines.base import RunResult, run_pipeline

PIPELINE_TEMPLATES = {
    "transformer_text": Path(__file__).resolve().parent / "pipelines" / "transformer_text",
}


class TriageError(RuntimeError):
    """Raised when an injected mutant does not form a valid silent-bug episode."""


# The clean baseline is identical for every operator on a pipeline, so it is run
# once per (pipeline, seed) and reused across the whole generation sweep.
_CLEAN_CACHE: dict[tuple[str, int], RunResult] = {}


def _clean_run(template_dir: Path, pipeline_id: str, seed: int) -> RunResult:
    key = (pipeline_id, seed)
    if key not in _CLEAN_CACHE:
        _CLEAN_CACHE[key] = run_pipeline(template_dir, seed=seed)
    return _CLEAN_CACHE[key]


# --- Layer 3 & 4 artifacts (assembled here; Layer 2 comes from the collector) -
def _data_distribution(pipeline_dir: Path, seed: int) -> dict[str, Any]:
    from silentml.pipelines.base import load_pipeline_module

    module = load_pipeline_module(pipeline_dir)
    train_loader, _val_loader, meta = module.get_dataloaders(seed)
    class_counts: dict[int, int] = {}
    n_nan = 0
    n_elems = 0
    vmin = float("inf")
    vmax = float("-inf")
    vsum = 0.0
    for inputs, targets in train_loader:
        for t in targets.tolist():
            class_counts[t] = class_counts.get(t, 0) + 1
        n_nan += torch.isnan(inputs).sum().item()
        n_elems += inputs.numel()
        vmin = min(vmin, inputs.min().item())
        vmax = max(vmax, inputs.max().item())
        vsum += inputs.sum().item()
    return {
        "num_classes": meta.num_classes,
        "class_counts": {str(k): class_counts.get(k, 0) for k in range(meta.num_classes)},
        "input_min": vmin,
        "input_max": vmax,
        "input_mean": vsum / n_elems if n_elems else None,
        "nan_fraction": n_nan / n_elems if n_elems else 0.0,
    }


def _model_architecture(result: RunResult, config: dict) -> dict[str, Any]:
    layers = []
    if result.model is not None:
        for name, module in result.model.named_modules():
            if name and len(list(module.children())) == 0:
                layers.append({"name": name, "type": type(module).__name__,
                               "repr": str(module)})
    return {"config": config, "layers": layers}


def _assemble_artifacts(pipeline_dir: Path, result: RunResult, config: dict,
                        seed: int) -> dict[str, Any]:
    return {
        "loss_curves": {"train_loss": result.history.train_loss,
                        "val_loss": result.history.val_loss},
        "accuracy_curves": {"train_acc": result.history.train_acc,
                            "val_acc": result.history.val_acc},
        "per_class_accuracy": {str(k): v for k, v in result.eval.per_class_accuracy.items()},
        "layer_gradient_stats": result.artifacts.get("layer2_gradient_stats", {}),
        "layer_activation_stats": result.artifacts.get("layer2_activation_stats", {}),
        "layer_weight_stats": result.artifacts.get("layer2_weight_stats", {}),
        "data_distribution": _data_distribution(pipeline_dir, seed),
        "model_architecture": _model_architecture(result, config),
    }


def _bug_description(clean: RunResult, buggy: RunResult) -> str:
    return (
        f"Current behavior: validation accuracy {buggy.eval.accuracy*100:.1f}% "
        f"after {len(buggy.history.val_acc)} epochs.\n"
        f"Expected behavior: ~{clean.eval.accuracy*100:.1f}% based on clean baseline."
    )


def _unified_fix_diff(buggy_src: str, clean_src: str) -> str:
    """Diff that, applied to the buggy source, recovers the clean source."""
    diff = difflib.unified_diff(
        buggy_src.splitlines(keepends=True),
        clean_src.splitlines(keepends=True),
        fromfile="pipeline.py", tofile="pipeline.py",
    )
    return "".join(diff)


def _dir_tree(root: Path) -> str:
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            lines.append(str(p.relative_to(root)).replace("\\", "/"))
    return "\n".join(lines)


def generate_episode(
    pipeline_id: str,
    operator_id: str,
    out_root: str | Path,
    episode_id: str | None = None,
    gen_seeds: tuple[int, ...] = (0, 1),
    drop_margin: float = 0.05,
) -> Path:
    """Generate one episode directory; raises TriageError if not a valid mutant."""
    template_dir = PIPELINE_TEMPLATES[pipeline_id]
    operator: Operator = get_operator(operator_id)
    if pipeline_id not in operator.applies_to:
        raise ValueError(f"operator {operator_id} not applicable to {pipeline_id}")

    episode_id = episode_id or f"{pipeline_id}__{operator_id}"
    ep_dir = Path(out_root) / episode_id
    pcode_dir = ep_dir / "pipeline"
    if ep_dir.exists():
        shutil.rmtree(ep_dir)
    pcode_dir.mkdir(parents=True)
    (ep_dir / "artifacts").mkdir()
    (ep_dir / "baseline").mkdir()

    clean_src = (template_dir / "pipeline.py").read_text(encoding="utf-8")
    buggy_src = operator.inject(clean_src)
    (pcode_dir / "pipeline.py").write_text(buggy_src, encoding="utf-8")

    # Run clean (template) and buggy (episode) across generation seeds.
    clean_runs, buggy_runs = [], []
    for s in gen_seeds:
        clean_runs.append(_clean_run(template_dir, pipeline_id, s))
        collector = LayerStatsCollector()
        try:
            buggy_runs.append(run_pipeline(pcode_dir, seed=s, collector=collector))
        except Exception as e:  # a crashing mutant is not a silent bug
            shutil.rmtree(ep_dir, ignore_errors=True)
            raise TriageError(f"{operator_id} crashed the pipeline: {e}") from e

    clean_acc = mean(r.eval.accuracy for r in clean_runs)
    buggy_acc = mean(r.eval.accuracy for r in buggy_runs)
    if clean_acc - buggy_acc < drop_margin:
        # Leave no partial directory behind: a rejected mutant is not an episode.
        shutil.rmtree(ep_dir, ignore_errors=True)
        raise TriageError(
            f"{operator_id} drop {clean_acc-buggy_acc:.3f} < margin {drop_margin} "
            f"(clean {clean_acc:.3f}, buggy {buggy_acc:.3f}) — not a measurable silent bug"
        )

    # Artifacts from the first buggy run (with collector); config reflects the
    # (possibly mutated) buggy pipeline the agent will read.
    buggy_config = _template_config(pcode_dir)
    artifacts = _assemble_artifacts(pcode_dir, buggy_runs[0], buggy_config, gen_seeds[0])
    for name, payload in artifacts.items():
        (ep_dir / "artifacts" / f"{name}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")

    # Baseline metrics + buggy checkpoint.
    (ep_dir / "baseline" / "clean_metrics.json").write_text(json.dumps({
        "mean_val_accuracy": clean_acc,
        "per_seed": [{"seed": s, "val_accuracy": r.eval.accuracy}
                     for s, r in zip(gen_seeds, clean_runs)],
        "per_class_accuracy": {str(k): v for k, v in clean_runs[0].eval.per_class_accuracy.items()},
    }, indent=2), encoding="utf-8")
    if buggy_runs[0].model is not None:
        torch.save(buggy_runs[0].model.state_dict(), ep_dir / "baseline" / "buggy_checkpoint.pt")

    # HIDDEN ground truth.
    meta = {
        "episode_id": episode_id,
        "pipeline_id": pipeline_id,
        "operator_id": operator_id,
        "operator_name": operator.name,
        "taxonomy": {
            "deepcrime_group": operator.deepcrime_group,
            "humbatova": operator.humbatova,
            "saner2024": operator.saner2024,
            "jahan2025": operator.jahan2025,
        },
        "difficulty": operator.difficulty,
        "bug_mechanism": operator.param_desc,
        "clean_mean_val_accuracy": clean_acc,
        "buggy_mean_val_accuracy": buggy_acc,
        "ground_truth_fix_diff": _unified_fix_diff(buggy_src, clean_src),
        "gen_seeds": list(gen_seeds),
    }
    (ep_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")

    # VISIBLE task spec.
    manifest = sorted(p.stem for p in (ep_dir / "artifacts").glob("*.json"))
    task = {
        "episode_id": episode_id,
        "bug_description": _bug_description(clean_runs[0], buggy_runs[0]),
        "directory_structure": _dir_tree(pcode_dir),
        "artifact_manifest": manifest,
    }
    (ep_dir / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    return ep_dir


def _template_config(template_dir: Path) -> dict:
    from silentml.pipelines.base import load_pipeline_module
    return dict(load_pipeline_module(template_dir).CONFIG)
