# Running the benchmark on a GPU server

The judge retrains the pipeline several times per submission, so it is the
expensive part — not the LLM. Put both on the GPU box.

## 1. Check the hardware

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
df -h .            # model weights: ~20 GB for a 30B at Q4, ~16 GB for Devstral
```

Pick the model from free VRAM:

| Free VRAM | Model | Ollama tag |
|-----------|-------|-----------|
| 24 GB+ | Qwen3-Coder-30B-A3B (MoE, 3.3B active — fast) | `qwen3-coder:30b` |
| 16–24 GB | Devstral-Small (purpose-built SWE agent) | `devstral:24b` |
| 8–16 GB | Qwen2.5-Coder-7B (usable floor) | `qwen2.5-coder:7b` |

Avoid anything below 7B or any `-base` tag: they do not reliably emit tool calls
or valid unified diffs, which shows up as parse failures rather than debugging
failures.

## 2. Set up

```bash
git clone <your-repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# install a CUDA build of torch for your driver, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

The pipeline selects CUDA automatically when available. Agent `execute_code`
scripts stay pinned to CPU by design.

## 3. Generate episodes

```bash
python -m silentml.cli generate                 # all operators
python -m silentml.cli generate --seeds 0       # ~2x faster, slightly noisier triage
```

Each operator is retrained and kept only if it degrades accuracy measurably
without crashing, so some operators are expected to be rejected — that is the
triage filter, not an error.

## 4. Serve the model

```bash
curl -fsSL https://ollama.com/install.sh | sh    # if not already installed
ollama serve &                                   # OpenAI-compatible API on :11434

# Reads free VRAM, picks the largest Qwen coder model that fits, pulls it,
# and probes whether it can actually emit tool calls:
python -m silentml.cli setup-model

# or force a specific tag:
python -m silentml.cli setup-model --model qwen3-coder:30b
```

The tool-call probe is worth its few seconds. A model that cannot emit tool calls
scores 0% in a way that looks like a debugging failure but is really a plumbing
failure; `setup-model` tells you before you spend an hour on a sweep.

If Ollama runs on a different host than the benchmark, pass `--base-url
http://<host>:11434/v1`. For vLLM instead:

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --port 8000
# then --base-url http://localhost:8000/v1
```

## 5. Run the benchmark

```bash
# smoke test on two episodes first — confirms tool calls parse before a full sweep
python -m silentml.cli benchmark --model qwen3-coder:30b --limit 2

# full run
python -m silentml.cli benchmark --model qwen3-coder:30b --report qwen3.json

# compare a second model on the same episodes
python -m silentml.cli benchmark --model devstral:24b --report devstral.json
```

Read `solve rate` (repaired the pipeline) alongside `causal rate` (the repair is
demonstrably the reason the metric recovered). Watch `tool-call parse failures`:
if it is high, the model is failing to emit tool calls rather than failing to
debug, and a larger or more tool-capable model is the fix.

## Cost knobs

| Flag | Effect |
|------|--------|
| `--seeds 0` | one judge seed instead of two — roughly halves judge time |
| `--limit N` | only the first N episodes |
| `--max-calls` | cap tool calls per episode (default 20) |
