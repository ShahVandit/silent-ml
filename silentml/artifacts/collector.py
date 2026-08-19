"""Layer-2 diagnostic collector: per-layer gradient / weight / activation stats.

Attaches forward and full-backward hooks to the model BEFORE training, so the
stats are gathered from the real training pass (the runner calls ``attach`` then
``train``). These signals are what let an agent localise silent faults — e.g. a
ReLU->Sigmoid swap shows collapsing gradient norms in early layers, while a
too-small learning rate shows healthy gradients but a flat loss curve.

The DeepDiagnosis / AUTOTRAINER / DeepLocalize line of work motivates per-layer
grad/weight/activation diagnostics; this is a compact reimplementation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

# Leaf module types worth instrumenting (skip containers/activations-as-leaves).
_LEAF_TYPES = (nn.Conv2d, nn.Linear, nn.LayerNorm, nn.Embedding, nn.MultiheadAttention)


class LayerStatsCollector:
    """Implements the ArtifactCollector protocol from ``pipelines.base``."""

    def __init__(self) -> None:
        self._handles: list[Any] = []
        self._act: dict[str, dict[str, float]] = {}   # running activation stats
        self._grad: dict[str, dict[str, float]] = {}  # running gradient stats
        self._named: dict[nn.Module, str] = {}
        self._model: nn.Module | None = None

    # -- hook registration ----------------------------------------------------
    def attach(self, model: nn.Module) -> None:
        self._model = model
        for name, module in model.named_modules():
            if isinstance(module, _LEAF_TYPES):
                self._named[module] = name
                self._handles.append(module.register_forward_hook(self._fwd_hook))
                self._handles.append(module.register_full_backward_hook(self._bwd_hook))

    def _fwd_hook(self, module, _inp, out) -> None:
        if not isinstance(out, torch.Tensor):
            return
        name = self._named[module]
        with torch.no_grad():
            absmean = out.abs().mean().item()
            # Saturation: fraction of activations with |value| very large (sigmoid/tanh)
            # or exactly zero (dead ReLU) — both are silent-fault signals.
            frac_zero = (out == 0).float().mean().item()
        self._accumulate(self._act, name, "abs_mean", absmean)
        self._accumulate(self._act, name, "frac_zero", frac_zero)

    def _bwd_hook(self, module, _grad_input, grad_output) -> None:
        g = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        if not isinstance(g, torch.Tensor):
            return
        name = self._named[module]
        with torch.no_grad():
            self._accumulate(self._grad, name, "grad_out_norm", g.norm().item())

    @staticmethod
    def _accumulate(store: dict, name: str, key: str, value: float) -> None:
        d = store.setdefault(name, {})
        d[f"_{key}_sum"] = d.get(f"_{key}_sum", 0.0) + value
        d[f"_{key}_n"] = d.get(f"_{key}_n", 0) + 1

    def on_epoch_end(self, epoch: int, model: nn.Module) -> None:  # noqa: D401
        # Stats are accumulated continuously; nothing per-epoch to do here.
        pass

    # -- finalize -------------------------------------------------------------
    def finalize(self) -> dict[str, Any]:
        for h in self._handles:
            h.remove()
        self._handles.clear()

        def _means(store: dict) -> dict:
            out = {}
            for name, d in store.items():
                out[name] = {}
                keys = {k[1:-4] for k in d if k.endswith("_sum")}
                for key in keys:
                    n = d.get(f"_{key}_n", 0)
                    out[name][key] = (d[f"_{key}_sum"] / n) if n else None
            return out

        activation = _means(self._act)
        gradient = _means(self._grad)
        weights = weight_stats(self._model) if self._model is not None else {}
        return {
            "layer2_activation_stats": activation,
            "layer2_gradient_stats": gradient,
            "layer2_weight_stats": weights,
        }


def weight_stats(model: nn.Module) -> dict[str, dict[str, float]]:
    """Per-parameter weight statistics (Layer-2), read after training."""
    stats: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            stats[name] = {
                "mean": p.mean().item(),
                "std": p.std().item(),
                "norm": p.norm().item(),
            }
    return stats
