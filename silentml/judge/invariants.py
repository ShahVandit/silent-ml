"""Training-dynamics invariants for the stability gate.

Compact reimplementation of the TrainCheck (OSDI 2025) idea: a correct fix should
not only recover accuracy but keep training well-behaved. Violations here drive
the P_curves penalty so brittle fixes that pass on accuracy alone are caught.
"""

from __future__ import annotations

import math
from typing import Any

from silentml.pipelines.base import History


def check_stability(history: History, artifacts: dict[str, Any]) -> list[str]:
    """Return a list of stability-invariant violations (empty == healthy)."""
    violations: list[str] = []

    all_losses = list(history.train_loss) + list(history.val_loss)
    if any((v is None or math.isnan(v) or math.isinf(v)) for v in all_losses):
        violations.append("non-finite loss (NaN/Inf) during training")

    # Loss should make progress: final training loss below the first epoch's.
    if len(history.train_loss) >= 2 and history.train_loss[-1] >= history.train_loss[0]:
        violations.append(
            f"training loss did not decrease "
            f"({history.train_loss[0]:.3f} -> {history.train_loss[-1]:.3f})"
        )

    # Gradient norms must be finite where collected.
    grad = artifacts.get("layer2_gradient_stats", {})
    for layer, stats in grad.items():
        gn = stats.get("grad_out_norm")
        if gn is not None and (math.isnan(gn) or math.isinf(gn)):
            violations.append(f"non-finite gradient norm at {layer}")

    return violations
