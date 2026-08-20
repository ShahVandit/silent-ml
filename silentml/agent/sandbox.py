"""Sandboxed execution for the ``execute_code`` tool.

The script runs in a subprocess against a throwaway *snapshot* of the current
workspace (patched pipeline source + episode artifacts/baseline). Because the
snapshot is a copy, the script has full read access to source, data, checkpoint,
and precomputed artifacts but cannot modify the real workspace source — matching
the design contract. Execution is CPU-only with a hard wall-clock timeout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_OUTPUT_CHARS = 8000


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    timed_out: bool
    returncode: int | None

    def render(self) -> str:
        parts = []
        if self.timed_out:
            parts.append("[execution timed out after the wall-clock limit]")
        if self.stdout:
            parts.append("STDOUT:\n" + self.stdout)
        if self.stderr:
            parts.append("STDERR:\n" + self.stderr)
        if not parts:
            parts.append("[no output]")
        return _truncate("\n".join(parts))


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + "\n[output truncated]"
    return text


def run_script(
    script: str,
    workspace_pipeline_dir: str | Path,
    episode_dir: str | Path,
    timeout: float = 30.0,
) -> ExecResult:
    """Run ``script`` against a snapshot of the workspace; return captured output."""
    workspace_pipeline_dir = Path(workspace_pipeline_dir)
    episode_dir = Path(episode_dir)

    tmp = Path(tempfile.mkdtemp(prefix="silentml_exec_"))
    try:
        snap = tmp / "episode"
        (snap / "pipeline").mkdir(parents=True)
        # Snapshot patched source.
        for f in workspace_pipeline_dir.glob("*.py"):
            shutil.copy2(f, snap / "pipeline" / f.name)
        # Snapshot read-only artifacts / baseline (checkpoint, metrics).
        for sub in ("artifacts", "baseline"):
            src = episode_dir / sub
            if src.exists():
                shutil.copytree(src, snap / sub)

        # Preamble makes both the repo (for `silentml`) and the snapshot pipeline
        # (for `import pipeline`) importable, regardless of env-var handling.
        preamble = (
            "import sys\n"
            f"sys.path.insert(0, {str(snap / 'pipeline')!r})\n"
            f"sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
        )
        script_path = tmp / "_script.py"
        script_path.write_text(preamble + "\n" + script, encoding="utf-8")

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only
        env["SILENTML_EPISODE"] = str(snap)

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(snap),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecResult(
                stdout=proc.stdout, stderr=proc.stderr,
                timed_out=False, returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired as e:
            return ExecResult(
                stdout=e.stdout or "", stderr=e.stderr or "",
                timed_out=True, returncode=None,
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
