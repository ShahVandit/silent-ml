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
    jahan2025: str = ""           # attention-fault category (transformer ops only)
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


# --- DeepCrime operators, reimplemented in PyTorch ---------------------------
_OPERATORS: list[Operator] = [
    Operator(
        id="T_HLR",
        name="Decrease learning rate",
        deepcrime_group="Hyperparameters",
        humbatova="Training > Hyperparameters > learning rate",
        saner2024="Output: model underperforms (optimisation stagnation)",
        applies_to=("transformer_text",),
        param_desc="learning rate reduced 100x (3e-4 -> 3e-6), starving optimisation",
        difficulty="medium",
        diagnosis_keywords=("learning rate", "lr"),
        find='"lr": 3e-4,',
        replace='"lr": 3e-6,',
    ),
    Operator(
        id="T_ACH",
        name="Change activation function",
        deepcrime_group="Activation",
        humbatova="Model > Layers > Activation Function > wrong activation",
        saner2024="Output: model underperforms (capacity loss)",
        # NOT APPLICABLE here - documented negative result. Both activation
        # variants were measured and neither degrades this pipeline:
        #   ReLU -> Sigmoid   99.0% -> 98.4%   (this operator)
        #   ReLU -> Identity  99.0% -> 98.8%   (T_ARM below)
        # The task is attention routing ("which key word comes first"), which the
        # attention blocks solve on their own; the feed-forward non-linearity is
        # not load-bearing, so mutating it produces no measurable failure. The
        # DeepCrime Activation family therefore yields no episode on this pipeline.
        applies_to=(),
        param_desc="feed-forward activation ReLU -> Sigmoid, weakening the FFN block",
        difficulty="medium",
        diagnosis_keywords=("activation", "sigmoid", "relu", "ff_act"),
        find="self.ff_act = nn.ReLU()",
        replace="self.ff_act = nn.Sigmoid()",
    ),
    Operator(
        id="T_OCH",
        name="Change optimisation function",
        deepcrime_group="Optimisation Function",
        humbatova="Training > Optimiser > wrong optimiser choice",
        saner2024="Output: model underperforms",
        applies_to=("transformer_text",),
        param_desc="Adam -> plain SGD at an LR tuned for Adam, so training barely moves",
        difficulty="easy",
        diagnosis_keywords=("optimizer", "optimiser", "sgd", "adam"),
        find='"optimizer": "adam",',
        replace='"optimizer": "sgd",',
    ),
    Operator(
        id="T_RCD",
        name="Change dropout rate",
        deepcrime_group="Regularisation",
        humbatova="Model > Layers > Layer Properties > dropout rate",
        saner2024="Output: model underperforms (over-regularisation)",
        applies_to=("transformer_text",),
        param_desc="dropout raised 0.1 -> 0.75, destroying signal during training",
        difficulty="easy",
        diagnosis_keywords=("dropout", "regularis", "regulariz"),
        find='"dropout": 0.1,',
        replace='"dropout": 0.75,',
    ),
    Operator(
        id="T_WCI",
        name="Change weights initialisation",
        deepcrime_group="Weights",
        humbatova="Model > Model Type & Properties > wrong initialisation",
        saner2024="Output: model underperforms",
        applies_to=("transformer_text",),
        param_desc=(
            "embedding weights initialised with std=5.0, saturating LayerNorm and "
            "attention before training starts"
        ),
        difficulty="medium",
        diagnosis_keywords=("init", "initial", "embedding", "std", "variance"),
        find="        self.pos_encoding = PositionalEncoding(d_model, CONFIG[\"max_len\"])",
        replace=(
            "        nn.init.normal_(self.embedding.weight, std=5.0)\n"
            "        self.pos_encoding = PositionalEncoding(d_model, CONFIG[\"max_len\"])"
        ),
    ),
    Operator(
        id="T_TCL",
        name="Change labels of training data",
        deepcrime_group="Training Data",
        humbatova="Training > Training Data Quality > wrong labels",
        saner2024="Output: model underperforms (label noise)",
        applies_to=("transformer_text",),
        # DeepCrime searches for the mutation magnitude that actually kills the
        # mutant; 30% noise left accuracy at 98.8% on this task, so the rate is
        # raised until the fault is measurable.
        param_desc="75% of training labels are randomly reassigned, injecting label noise",
        difficulty="medium",
        diagnosis_keywords=("label", "target", "noise", "corrupt", "shuffl"),
        find="    train_labels = [int(train_raw.target[i]) for i in tr_idx]",
        replace=(
            "    train_labels = [int(train_raw.target[i]) for i in tr_idx]\n"
            "    _rng = torch.Generator().manual_seed(1234)\n"
            "    _n_corrupt = int(0.75 * len(train_labels))\n"
            "    _pos = torch.randperm(len(train_labels), generator=_rng)[:_n_corrupt]\n"
            "    for _i in _pos.tolist():\n"
            "        train_labels[_i] = int(torch.randint(\n"
            "            0, len(CATEGORIES), (1,), generator=_rng).item())"
        ),
    ),
    Operator(
        id="T_ARM",
        name="Remove activation function",
        deepcrime_group="Activation",
        humbatova="Model > Layers > Activation Function > missing activation",
        saner2024="Output: model underperforms (loss of non-linearity)",
        applies_to=(),   # see the note on T_ACH: the FFN non-linearity is not
                         # load-bearing on this task (99.0% -> 98.8%)
        param_desc=(
            "the feed-forward non-linearity is replaced by an identity, collapsing "
            "the two FFN projections into a single linear map"
        ),
        difficulty="medium",
        diagnosis_keywords=("activation", "relu", "linear", "non-linear", "ff_act"),
        find="self.ff_act = nn.ReLU()",
        replace="self.ff_act = nn.Identity()",
    ),
    Operator(
        id="T_EMB_FREEZE",
        name="Freeze embedding weights",
        deepcrime_group="Weights",
        humbatova="Training > Training Process > wrong parameter set optimised",
        saner2024="Output: model underperforms (parameters never updated)",
        # NOT APPLICABLE here - documented negative result (99.0% -> 98.8%).
        # With a 512-token vocabulary in 128 dimensions, random embeddings are
        # already near-orthogonal, so frozen random features remain perfectly
        # separable and the projections downstream learn everything they need.
        # Freezing embeddings only bites when the embedding must itself encode
        # learned structure (large vocabulary, or semantic similarity).
        applies_to=(),
        param_desc=(
            "the embedding matrix is frozen at its random initialisation, so token "
            "identity is never learned while the rest of the model trains normally"
        ),
        difficulty="medium",
        diagnosis_keywords=("embed", "freeze", "frozen", "requires_grad", "grad"),
        find="        self.pos_encoding = PositionalEncoding(d_model, CONFIG[\"max_len\"])",
        replace=(
            "        self.embedding.weight.requires_grad = False\n"
            "        self.pos_encoding = PositionalEncoding(d_model, CONFIG[\"max_len\"])"
        ),
    ),
    Operator(
        id="T_RESIDUAL",
        name="Drop the attention residual connection",
        deepcrime_group="(transformer extension)",
        humbatova="Model > Layers > missing skip connection",
        saner2024="Output: model underperforms",
        # NOT APPLICABLE here - documented negative result (99.0% -> 98.5%).
        # Residual connections exist to keep gradients flowing through *deep*
        # stacks; with only two encoder layers the model trains fine without one.
        # This operator would become viable on a deeper configuration.
        applies_to=(),
        param_desc=(
            "the attention sub-layer output replaces the residual stream instead of "
            "being added to it, discarding the token representation"
        ),
        difficulty="hard",
        diagnosis_keywords=("residual", "skip", "connection", "add"),
        find="        x = self.norm1(x + self.dropout(self.attn(x, pad_mask)))",
        replace="        x = self.norm1(self.dropout(self.attn(x, pad_mask)))",
    ),
    Operator(
        id="T_HNE",
        name="Change number of epochs",
        deepcrime_group="Hyperparameters",
        humbatova="Training > Hyperparameters > number of epochs",
        saner2024="Output: model underperforms (undertraining)",
        applies_to=("transformer_text",),
        param_desc="epochs cut 6 -> 1, stopping training well before convergence",
        difficulty="easy",
        diagnosis_keywords=("epoch", "undertrain", "too few", "converg"),
        find='"epochs": 6,',
        replace='"epochs": 1,',
    ),
]

# Transformer-specific attention faults (Jahan et al. 2025) live in their own
# module; imported here so the registry is the single source of truth.
from silentml.bugs.attention_ops import ATTENTION_OPERATORS  # noqa: E402

_OPERATORS.extend(ATTENTION_OPERATORS)

OPERATORS: dict[str, Operator] = {op.id: op for op in _OPERATORS}


def get_operator(operator_id: str) -> Operator:
    if operator_id not in OPERATORS:
        raise KeyError(f"unknown operator {operator_id!r}; known: {sorted(OPERATORS)}")
    return OPERATORS[operator_id]


def operators_for(pipeline_id: str) -> list[Operator]:
    return [op for op in OPERATORS.values() if pipeline_id in op.applies_to]
