"""Mutation operators = the injectable silent-bug catalog.

Operators are reimplemented in PyTorch from DeepCrime (Humbatova/Jahangirova/
Tonella, ISSTA 2021), whose 24 training-time operators remain the most complete
operationalized set; no PyTorch-native successor exists. Each operator is tagged
to three taxonomies so episodes carry literature-grounded metadata:

  * ``deepcrime``  — DeepCrime operator group (the injection semantics)
  * ``humbatova``  — Humbatova et al. ICSE 2020 leaf category (historical base)
  * ``saner2024``  — Hong et al. SANER 2024 silent-symptom category (PyTorch)

An operator injects a fault by an exact, single-site source edit whose inverse is
the ground-truth fix — this makes both injection and the reference patch precise
and reproducible.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional


@dataclasses.dataclass(frozen=True)
class Operator:
    id: str                       # DeepCrime ID, e.g. "HLR"
    name: str
    deepcrime_group: str
    humbatova: str
    saner2024: str
    applies_to: tuple[str, ...]   # pipeline ids this operator is valid for
    param_desc: str               # human description of the injected change
    difficulty: str               # "easy" | "medium" | "hard" (triage/curriculum)
    diagnosis_keywords: tuple[str, ...]  # terms a correct diagnosis should mention
    # Single-site text edit (clean -> buggy). ``inject`` asserts uniqueness.
    find: Optional[str] = None
    replace: Optional[str] = None
    # Optional richer transform for non-find/replace operators (data ops, etc.).
    transform: Optional[Callable[[str], str]] = None

    def inject(self, clean_source: str) -> str:
        """Return buggy source with this operator's fault applied."""
        if self.transform is not None:
            out = self.transform(clean_source)
            if out == clean_source:
                raise ValueError(f"operator {self.id}: transform made no change")
            return out
        assert self.find is not None and self.replace is not None
        count = clean_source.count(self.find)
        if count != 1:
            raise ValueError(
                f"operator {self.id}: anchor found {count} times (need exactly 1): {self.find!r}"
            )
        return clean_source.replace(self.find, self.replace)


# --- Operator catalog (Phase-1 slice: HLR, ACH) ------------------------------
# Broader P1-applicable operators (TCL, TUD, ARM, WCI, OCH, ...) are added in
# milestone 2 following the same pattern.

_OPERATORS: list[Operator] = [
    Operator(
        id="HLR",
        name="Decrease learning rate",
        deepcrime_group="Hyperparameters",
        humbatova="Training > Hyperparameters > learning rate",
        saner2024="Output: model underperforms (optimisation stagnation)",
        applies_to=("cnn_fashion", "transformer_text"),
        param_desc="learning rate reduced 100x (1e-3 -> 1e-5), starving optimisation",
        difficulty="medium",
        diagnosis_keywords=("learning rate", "lr"),
        find='"lr": 1e-3,',
        replace='"lr": 1e-5,',
    ),
    Operator(
        id="ACH",
        name="Change activation function",
        deepcrime_group="Activation",
        humbatova="Model > Layers > Activation Function > wrong activation",
        saner2024="Output: model underperforms (capacity loss)",
        applies_to=("cnn_fashion",),
        param_desc="hidden activation ReLU -> Sigmoid, causing vanishing gradients",
        difficulty="medium",
        diagnosis_keywords=("activation", "sigmoid", "relu"),
        find="self.act = nn.ReLU()",
        replace="self.act = nn.Sigmoid()",
    ),
]

OPERATORS: dict[str, Operator] = {op.id: op for op in _OPERATORS}


def get_operator(operator_id: str) -> Operator:
    if operator_id not in OPERATORS:
        raise KeyError(f"unknown operator {operator_id!r}; known: {sorted(OPERATORS)}")
    return OPERATORS[operator_id]


def operators_for(pipeline_id: str) -> list[Operator]:
    return [op for op in OPERATORS.values() if pipeline_id in op.applies_to]
