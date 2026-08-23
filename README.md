# ML Debugging Agent Benchmark

A reinforcement-learning environment and deterministic judge for evaluating (and
later fine-tuning) **LLM agents that debug silent training failures** — bugs
where a PyTorch pipeline trains to completion with no error, yet the model
quietly underperforms.

Each episode injects exactly one known fault into a small Transformer training
pipeline, pre-computes diagnostic artifacts, and hands the agent a five-tool
interface. A compute-heavy judge then retrains the agent's patch across seeds and
applies a **causal-ablation** gate, so a patch only scores if it is demonstrably
the reason the metric recovered.

## Why the judge is the interesting part

Accuracy alone is a weak oracle: a patch can raise validation accuracy without
touching the injected fault. This judge runs four stages:

1. **Functional gate** — mean validation accuracy across fixed seeds must reach
   the clean baseline, *and* the specific failure mode must be repaired
   (Repair Rate), not merely averaged away.
2. **Causal ablation** — the original mutation is re-injected on top of the
   agent's patch. If performance does not degrade, the fix was not causal.
3. **Stability invariants** — loss curves and gradient norms must stay healthy,
   catching brittle fixes that hit the accuracy target by luck.
4. **Reward** — `R_functional + R_fix_quality + R_diagnosis − P_diagnosis_wrong
   − P_ablation − P_curves`, with `P_ablation` set above the entire positive sum
   so that passing accuracy while failing ablation is always net-negative.

## The episode set

11 episodes survive triage, spanning six DeepCrime families and four of the
attention-fault categories, with severity from catastrophic to subtle:

| Operator | Family | Clean → Buggy |
|----------|--------|---------------|
| T_OCH | Optimisation Function | 99.0% → 23.6% |
| T_HLR | Hyperparameters | 99.0% → 24.7% |
| T_RCD | Regularisation | 99.0% → 26.0% |
| ATT_MASK | Attention Masking | 99.0% → 27.3% |
| SOFTMAX_DIM | Score Computation | 99.0% → 27.3% |
| POS_ENC | Positional Encoding | 99.0% → 34.5% |
| HEAD_RESHAPE | QKV / Multi-Head | 99.0% → 75.6% |
| T_TCL | Training Data | 99.0% → 77.4% |
| T_WCI | Weights | 99.0% → 83.9% |
| T_HNE | Hyperparameters | 99.0% → 91.6% |
| ATT_SCALE | Score Computation | 99.0% → 92.3% |

The subtle end is the interesting end: a 7-point drop is a far weaker signal to
reason from than a collapse to chance.

`python -m silentml.cli episodes` prints this table from the generated set.

### What triage rejected, and why it is informative

Five operators were measured and could not produce a silent failure on this
pipeline. They are kept in the registry with `applies_to=()` and a note, because
the reasons are findings rather than bookkeeping:

- **Missing operations are self-correcting; wrong ones are not.** Deleting the
  attention scale (99.0% → 98.0%) or the padding mask (→ 98.4%) barely hurts —
  the model just learns smaller Q/K projections, or learns to avoid padding.
  *Inverting* the mask instead collapses accuracy to 27.3%. Each rejected
  operator was therefore replaced by a more realistic bug: a `*`-for-`/` typo, an
  inverted mask polarity, a wrong softmax axis.
- **The feed-forward non-linearity is not load-bearing** on an attention-routing
  task (ReLU→Sigmoid 98.4%, ReLU→Identity 98.8%), so DeepCrime's Activation
  family yields no episode here.
- **The canonical right-padding bug does not transfer to an encoder.** Pooling
  the last position is fatal in an RNN, where that state is literally a padding
  step; in a bidirectional encoder padding positions are still valid queries, so
  after two layers that position carries the same summary as `[CLS]` (98.7%).
- **Frozen embeddings** are harmless at a 512-token vocabulary in 128 dimensions
  (98.8%) — random embeddings are already near-orthogonal.
- **Dropping a residual connection** costs nothing at two layers (98.5%);
  residuals matter for depth.

## Bug families

Faults come from published fault taxonomies rather than invention:

- **DeepCrime** operators (Humbatova et al., ISSTA 2021) reimplemented in
  PyTorch: learning rate, activation, optimiser, dropout, weight initialisation,
  label noise, epoch count.
- **Attention-specific faults** (Jahan et al., 2025 — a study of 555 real
  attention bugs): inverted attention scaling, inverted padding-mask polarity,
  absent positional encoding, softmax over the wrong axis, scrambled multi-head
  reshape.

The attention faults are injectable only because the pipeline implements
attention in readable source rather than calling `nn.TransformerEncoder` — the
faulty line has to be something the agent can view and patch.

## The task

A small Transformer encoder (~330k params) classifies sequences by **which key
word appears earliest**. Every document contains the same three key words, so
presence carries no information and the decision depends entirely on order. This
matters for the benchmark: on a bag-of-words task, removing positional encoding
or the attention mask would not degrade accuracy and those episodes could not
exist. The clean baseline reaches ~99%, leaving clear headroom for a fault to
show.

## Install

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
# 1. Generate the bug episodes (each is triaged: must degrade, must not crash)
python -m silentml.cli generate

# 2. Sanity-check the environment with a scripted oracle (no LLM required)
python -m silentml.cli episodes      # list the generated set
python -m silentml.cli demo T_HLR    # scripted oracle, no LLM needed

# 3. Provision a Qwen model sized to this GPU, then verify it can call tools
ollama serve &
python -m silentml.cli setup-model

# 4. Benchmark it (free; Ollama serves an OpenAI-compatible API)
python -m silentml.cli benchmark --model qwen3-coder:30b

# Fast test suite (training mocked; ~10s)
python -m pytest -q
```

Any OpenAI-compatible endpoint works — point `--base-url` at vLLM, or at a hosted
provider with `--api-key`.

## Example benchmark output

```
MODEL: qwen3-coder:30b   episodes: 11
  solve rate (functional gate)  :  ...%
  causal rate (fix is the cause):  ...%
  diagnosis rate                :  ...%

  bug family                    n   solved   causal   diag
  Attention Masking             2      ...      ...    ...
  Hyperparameters               3      ...      ...    ...
```

`causal_rate` is the honest headline number.

## Layout

```
silentml/
  pipelines/     # pipeline contract + the Transformer pipeline (patchable source)
  bugs/          # DeepCrime operators + attention operators, with taxonomy tags
  artifacts/     # gradient / weight / activation collector (hooks)
  agent/         # 5 tools, sandbox, diff applier, episode loop, LLM policy
  judge/         # functional + causal-ablation + stability + reward
  benchmark.py   # sweep a model over all episodes -> solve-rate table
  generation.py  # clean template + operator -> triaged episode
```

An episode directory:

```
episodes/<id>/
  pipeline/pipeline.py   # buggy source the agent views and patches
  artifacts/*.json       # diagnostics served by read_artifact
  baseline/              # clean metrics + buggy checkpoint
  meta.yaml              # HIDDEN ground truth (operator, fix diff, taxonomy)
  task.yaml              # VISIBLE prompt fields
```

## Roadmap

The environment is designed so the same judge serves as a verifiable reward
function for GRPO fine-tuning (Phase 2), turning the zero-shot benchmark number
into the "before" measurement.
