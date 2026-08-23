"""Provision a local Qwen model and verify it can drive the benchmark.

Disk is rarely the binding constraint; **VRAM** is. This module reads the GPU's
free memory, picks the largest Qwen coder model that fits, pulls it with Ollama,
and then runs a live tool-call check against the OpenAI-compatible endpoint.

That last step matters: a model that cannot emit tool calls produces a 0% score
that looks like a debugging failure but is really a plumbing failure. Catching it
here costs seconds; discovering it after a full sweep costs hours.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

# Qwen coder models, largest first. Sizes are the on-disk/VRAM footprint of the
# default (Q4) Ollama build plus headroom for KV cache at our context length.
@dataclass(frozen=True)
class ModelChoice:
    tag: str
    min_free_vram_gb: float
    approx_disk_gb: float
    note: str


QWEN_MODELS: list[ModelChoice] = [
    ModelChoice(
        "qwen3-coder:30b", 24.0, 19.0,
        "Qwen3-Coder-30B-A3B: MoE with only ~3.3B active params, so it runs "
        "far faster than its size suggests. Strong native tool calling.",
    ),
    ModelChoice(
        "qwen2.5-coder:14b", 12.0, 9.0,
        "Qwen2.5-Coder-14B: dense, reliable tool calling, good code editing.",
    ),
    ModelChoice(
        "qwen2.5-coder:7b", 6.0, 4.7,
        "Qwen2.5-Coder-7B: the smallest size that still emits usable tool "
        "calls and unified diffs. Expect weaker multi-step debugging.",
    ),
]


class SetupError(RuntimeError):
    pass


# --- hardware ----------------------------------------------------------------
def detect_free_vram_gb() -> float | None:
    """Largest single-GPU free memory in GiB, or None if nvidia-smi is absent."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    values = [float(line.strip()) / 1024 for line in out.splitlines() if line.strip()]
    return max(values) if values else None


def recommend(free_vram_gb: float | None) -> ModelChoice:
    """Largest Qwen model that fits; smallest as a fallback when VRAM is unknown."""
    if free_vram_gb is None:
        return QWEN_MODELS[-1]
    for choice in QWEN_MODELS:
        if free_vram_gb >= choice.min_free_vram_gb:
            return choice
    return QWEN_MODELS[-1]


# --- ollama ------------------------------------------------------------------
def ollama_available() -> bool:
    return shutil.which("ollama") is not None


def pull(tag: str) -> None:
    """Stream `ollama pull` so a multi-GB download shows progress."""
    if not ollama_available():
        raise SetupError(
            "ollama is not installed. Install it with:\n"
            "    curl -fsSL https://ollama.com/install.sh | sh\n"
            "then start the server with `ollama serve &` and re-run this command."
        )
    print(f"pulling {tag} (this downloads several GB) ...", flush=True)
    proc = subprocess.run(["ollama", "pull", tag])
    if proc.returncode != 0:
        raise SetupError(f"`ollama pull {tag}` failed with exit code {proc.returncode}")


# --- endpoint verification ---------------------------------------------------
def _get(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_endpoint(base_url: str, model: str, timeout: float = 180.0) -> list[str]:
    """Verify the endpoint serves the model and can emit a tool call.

    Returns a list of warnings (empty means fully healthy). Raises SetupError if
    the endpoint is unreachable, which is fatal.
    """
    warnings: list[str] = []
    root = base_url.rstrip("/")

    try:
        listing = _get(f"{root}/models")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        raise SetupError(
            f"cannot reach {root}/models ({e}).\n"
            "Is the server running? Start it with `ollama serve &`."
        ) from e

    served = {m.get("id") for m in listing.get("data", [])}
    if served and model not in served:
        warnings.append(
            f"model {model!r} is not in the endpoint's list ({sorted(served)[:5]}). "
            "Pull it, or pass the exact served name with --model."
        )

    # Live tool-call probe, using the real tool schemas the benchmark sends.
    from silentml.agent.llm_policy import TOOL_SCHEMAS, _post_json

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Call exactly one tool."},
            {"role": "user",
             "content": "Read the diagnostic artifact named 'loss_curves'."},
        ],
        "tools": TOOL_SCHEMAS,
        "temperature": 0.0,
        "stream": False,
    }
    print("probing tool-call support ...", flush=True)
    resp = _post_json(f"{root}/chat/completions", payload, None, timeout)
    message = resp["choices"][0]["message"]

    if message.get("tool_calls"):
        print("  native tool calls: OK")
    else:
        from silentml.agent.llm_policy import _extract_json_object

        if _extract_json_object(message.get("content") or ""):
            warnings.append(
                "the model did not emit native tool calls, but the JSON fallback "
                "parsed its reply. The benchmark will work, slightly less reliably."
            )
        else:
            warnings.append(
                "the model emitted neither a native tool call nor parseable JSON. "
                "Expect high 'tool-call parse failures'. Prefer a larger or more "
                "tool-capable model (see RUNNING.md)."
            )
    return warnings


# --- orchestration -----------------------------------------------------------
def setup(base_url: str, model: str | None = None, do_pull: bool = True,
          check: bool = True) -> int:
    free = detect_free_vram_gb()
    if free is None:
        print("nvidia-smi not found — cannot read VRAM. Assuming a small GPU.")
    else:
        print(f"largest GPU free VRAM: {free:.1f} GiB")

    if model:
        choice = next((c for c in QWEN_MODELS if c.tag == model), None)
        tag = model
        if choice:
            print(f"selected (explicit): {tag} — {choice.note}")
            if free is not None and free < choice.min_free_vram_gb:
                print(f"  warning: {tag} wants ~{choice.min_free_vram_gb:.0f} GiB "
                      f"free VRAM but only {free:.1f} GiB is available; it may run "
                      f"on CPU or fail to load.")
        else:
            print(f"selected (explicit, not in the built-in table): {tag}")
    else:
        choice = recommend(free)
        tag = choice.tag
        print(f"selected: {tag} (~{choice.approx_disk_gb:.0f} GB download)")
        print(f"  {choice.note}")

    if do_pull:
        try:
            pull(tag)
        except SetupError as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            return 1

    if check:
        try:
            warnings = check_endpoint(base_url, tag)
        except SetupError as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            return 1
        for w in warnings:
            print(f"  warning: {w}")
        if not warnings:
            print("endpoint healthy.")

    print("\nnext:")
    print(f"  python -m silentml.cli benchmark --model {tag} --limit 2   # smoke test")
    print(f"  python -m silentml.cli benchmark --model {tag} --report {tag.split(':')[0]}.json")
    return 0
