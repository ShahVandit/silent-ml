"""Pipeline contract and the shared training runner.

A *pipeline* is a self-contained, human-readable PyTorch training script that
lives in an episode directory as ``pipeline.py``. It is the code the agent reads
(``view_code``) and edits (``apply_patch``), and the code the judge executes.

To keep the debugged source clean, every pipeline exposes the same small API and
the harness owns orchestration (seeding, artifact hooks, metric collection). Any
bug the environment injects lives inside one of these functions/CONFIG, so it is
always visible and patchable in the source the agent sees.

Required interface for an episode's ``pipeline.py``::

    CONFIG: dict                                  # hyperparameters (lr, epochs, ...)
    def build_model() -> torch.nn.Module
    def get_dataloaders(seed: int) -> (train_loader, val_loader, DataMeta)
    def train(model, train_loader, val_loader, config, seed) -> History  # explicit, patchable loop
    def evaluate(model, val_loader) -> EvalMetrics

The shared runner below imports such a module from a directory and executes it,
optionally attaching an artifact collector via forward/backward hooks (external
to the debugged code).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Protocol

import torch

from silentml.utils.seed import set_seed


@dataclasses.dataclass
class DataMeta:
    """Metadata about a dataset needed by the judge and artifacts."""

    num_classes: int
    class_names: list[str]
    input_shape: tuple[int, ...]


@dataclasses.dataclass
class EvalMetrics:
    """Result of evaluating a model on the validation set."""

    accuracy: float
    per_class_accuracy: dict[int, float]  # class index -> recall
    loss: float


@dataclasses.dataclass
class History:
    """Epoch-by-epoch training dynamics returned by ``train``."""

    train_loss: list[float]
    val_loss: list[float]
    train_acc: list[float]
    val_acc: list[float]


@dataclasses.dataclass
class RunResult:
    """Everything the harness records from one training run."""

    history: History
    eval: EvalMetrics
    data_meta: DataMeta
    artifacts: dict[str, Any] = dataclasses.field(default_factory=dict)
    model: Optional[torch.nn.Module] = None


class ArtifactCollector(Protocol):
    """Attaches to a model to record per-layer diagnostics during training."""

    def attach(self, model: torch.nn.Module) -> None: ...
    def on_epoch_end(self, epoch: int, model: torch.nn.Module) -> None: ...
    def finalize(self) -> dict[str, Any]: ...


def load_pipeline_module(pipeline_dir: str | Path) -> ModuleType:
    """Import ``pipeline.py`` from a directory under a unique module name.

    A unique name per directory keeps buggy / patched / clean variants isolated
    so the judge can load several in one process without import caching clashes.
    """
    pipeline_dir = Path(pipeline_dir)
    path = pipeline_dir / "pipeline.py"
    if not path.exists():
        raise FileNotFoundError(f"no pipeline.py in {pipeline_dir}")
    mod_name = f"_silentml_pipeline_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pipeline from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def run_pipeline(
    pipeline_dir: str | Path,
    seed: int,
    collector: Optional[ArtifactCollector] = None,
) -> RunResult:
    """Run one pipeline deterministically and return metrics (+ artifacts).

    The bug (if any) is already baked into ``pipeline.py``; this runner is
    identical for clean, buggy, and patched variants — that identity is what
    makes the judge's comparisons fair.
    """
    set_seed(seed)
    module = load_pipeline_module(pipeline_dir)

    model = module.build_model()
    train_loader, val_loader, data_meta = module.get_dataloaders(seed)

    if collector is not None:
        collector.attach(model)

    history: History = module.train(model, train_loader, val_loader, module.CONFIG, seed)

    if collector is not None:
        collector.on_epoch_end(len(history.train_loss), model)

    eval_metrics: EvalMetrics = module.evaluate(model, val_loader)

    artifacts = collector.finalize() if collector is not None else {}
    return RunResult(
        history=history,
        eval=eval_metrics,
        data_meta=data_meta,
        artifacts=artifacts,
        model=model,
    )
