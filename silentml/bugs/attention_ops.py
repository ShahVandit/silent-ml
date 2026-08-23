"""Transformer-specific mutation operators.

Grounded in Jahan et al., "Taxonomy of Faults in Attention-Based Neural Networks"
(arXiv 2508.04925, 2025), an empirical study of 555 real attention faults across
96 projects. Its top categories map directly onto the operators below: Attention
Masking (~25% of real faults), QKV Projection / Multi-Head (~22%), Score
Computation (~13%), and Positional Encoding (~12%).

These faults are only injectable because the P3 pipeline implements attention in
readable source rather than calling ``nn.TransformerEncoder`` — the faulty line
has to be something the agent can view and patch.
"""

from __future__ import annotations

from silentml.bugs.operators import Operator

ATTENTION_OPERATORS: list[Operator] = [
    Operator(
        id="ATT_SCALE",
        name="Invert attention score scaling",
        deepcrime_group="(transformer extension)",
        humbatova="Model > Layers > wrong layer computation",
        saner2024="Output: model underperforms",
        jahan2025="Score Computation (~13% of real attention faults)",
        applies_to=("transformer_text",),
        # Merely dropping the scale is self-correcting: the model just learns
        # smaller Q/K projections (measured: 99.0% -> 98.0%). Multiplying instead
        # of dividing - a '*' for '/' typo - is a far larger, non-compensable error.
        param_desc=(
            "dot-product scores are multiplied by sqrt(d_head) instead of divided "
            "by it, saturating softmax into near one-hot attention"
        ),
        difficulty="hard",
        diagnosis_keywords=("scal", "sqrt", "d_k", "d_head", "temperature"),
        find="scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)",
        replace="scores = torch.matmul(q, k.transpose(-2, -1)) * math.sqrt(self.d_head)",
    ),
    Operator(
        id="ATT_MASK",
        name="Invert the padding mask polarity",
        deepcrime_group="(transformer extension)",
        humbatova="Model > Layers > wrong layer computation",
        saner2024="Output: model underperforms",
        jahan2025="Attention Masking (~25% of real attention faults)",
        applies_to=("transformer_text",),
        # Simply deleting the mask is self-correcting here (measured 99.0% ->
        # 98.4%): padding_idx zeroes pad embeddings and the model learns to avoid
        # them. Inverting the polarity - the most common real masking bug - forces
        # attention onto padding instead of the sequence.
        param_desc=(
            "the padding mask is applied with inverted polarity, so attention is "
            "restricted to <pad> positions and real tokens are masked out"
        ),
        difficulty="hard",
        diagnosis_keywords=("mask", "pad", "invert", "polarity"),
        find='scores = scores.masked_fill(~mask, float("-inf"))',
        replace='scores = scores.masked_fill(mask, float("-inf"))',
    ),
    Operator(
        id="POS_ENC",
        name="Positional encoding not applied",
        deepcrime_group="(transformer extension)",
        humbatova="Model > Layers > missing layer",
        saner2024="Output: model underperforms",
        jahan2025="Positional Encoding (~12% of real attention faults)",
        applies_to=("transformer_text",),
        param_desc=(
            "token embeddings skip the positional encoding, making the encoder "
            "permutation-invariant and blind to word order"
        ),
        difficulty="medium",
        diagnosis_keywords=("position", "pos_encoding", "positional", "order"),
        find="        x = self.pos_encoding(x)\n",
        replace="",
    ),
    Operator(
        id="SOFTMAX_DIM",
        name="Softmax over the wrong dimension",
        deepcrime_group="(transformer extension)",
        humbatova="Model > Layers > wrong layer computation",
        saner2024="Output: model underperforms",
        jahan2025="Score Computation (~13% of real attention faults)",
        applies_to=("transformer_text",),
        param_desc=(
            "attention weights are normalised over the query axis instead of the "
            "key axis, so each row of the attention matrix no longer sums to 1"
        ),
        difficulty="hard",
        diagnosis_keywords=("softmax", "dim", "dimension", "axis"),
        find="weights = torch.softmax(scores, dim=-1)",
        replace="weights = torch.softmax(scores, dim=-2)",
    ),
    Operator(
        id="POOL_PAD",
        name="Pool the last position of a right-padded batch",
        deepcrime_group="(transformer extension)",
        humbatova="Model > Layers > wrong layer computation",
        saner2024="Output: model underperforms",
        jahan2025="Attention Masking (~25% of real attention faults)",
        # NOT APPLICABLE to this pipeline - kept as a documented negative result.
        # Two variants were measured and neither is silent-bug material here:
        #   unmasked mean pooling  99.0% -> 98.8%  (padding_idx zeroes pad embeddings)
        #   last-position pooling  99.0% -> 98.7%
        # The second is the canonical right-padding mistake and is fatal in an RNN,
        # where the final state is literally a padding step. In a bidirectional
        # encoder padding positions are still valid *queries*, so after two layers
        # the trailing <pad> position has attended over the whole sequence and
        # carries essentially the same summary as [CLS]. Injecting it would produce
        # an episode with no measurable degradation.
        applies_to=(),
        param_desc=(
            "classification reads the final sequence position instead of [CLS]; "
            "with right padding that position is always a <pad> token"
        ),
        difficulty="medium",
        diagnosis_keywords=("pool", "pad", "last", "cls", "position"),
        find="        pooled = x[:, 0, :]",
        replace="        pooled = x[:, -1, :]",
    ),
    Operator(
        id="HEAD_RESHAPE",
        name="Scramble multi-head reshape",
        deepcrime_group="(transformer extension)",
        humbatova="Model > Layers > wrong output shape",
        saner2024="Output: model underperforms",
        jahan2025="QKV Projection / Multi-Head (~22% of real attention faults)",
        applies_to=("transformer_text",),
        param_desc=(
            "heads are split by reshaping before transposing in the wrong order, "
            "mixing feature dimensions across heads while keeping shapes valid"
        ),
        difficulty="hard",
        diagnosis_keywords=("head", "reshape", "view", "transpose", "split"),
        find="        return x.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)",
        replace="        return x.view(batch, self.n_heads, seq, self.d_head)",
    ),
]
