"""The five agent tools, bound to a per-episode working copy.

An ``EpisodeSession`` copies the episode's buggy ``pipeline/`` into a scratch
workspace; all edits happen there so the original buggy source is preserved for
the judge. The session deliberately never reads ``meta.yaml`` — that hidden
ground truth belongs to the judge, not the agent.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from silentml.agent.patching import PatchError, apply_unified_diff
from silentml.agent.sandbox import run_script

MAX_VIEW_LINES = 400


@dataclass
class ToolCall:
    tool: str
    args: dict
    ok: bool


class ToolError(Exception):
    """A recoverable tool error surfaced back to the agent as an observation."""


@dataclass
class EpisodeSession:
    episode_dir: Path
    workspace_dir: Path = field(init=False)
    task: dict = field(init=False)
    patches_applied: int = field(default=0, init=False)
    submitted: bool = field(default=False, init=False)
    diagnosis: str | None = field(default=None, init=False)
    calls: list[ToolCall] = field(default_factory=list, init=False)

    def __init__(self, episode_dir: str | Path):
        self.episode_dir = Path(episode_dir)
        self.task = yaml.safe_load((self.episode_dir / "task.yaml").read_text(encoding="utf-8"))
        self.workspace_dir = Path(tempfile.mkdtemp(prefix="silentml_ws_"))
        shutil.copytree(self.episode_dir / "pipeline", self.workspace_dir / "pipeline")
        self.patches_applied = 0
        self.submitted = False
        self.diagnosis = None
        self.calls = []

    # -- internals ------------------------------------------------------------
    @property
    def _pipeline_dir(self) -> Path:
        return self.workspace_dir / "pipeline"

    def _log(self, tool: str, args: dict, ok: bool) -> None:
        self.calls.append(ToolCall(tool=tool, args=args, ok=ok))

    def _source_path(self, file: str) -> Path:
        p = (self._pipeline_dir / file).resolve()
        if self._pipeline_dir.resolve() not in p.parents and p != self._pipeline_dir.resolve():
            raise ToolError(f"file {file!r} is outside the pipeline directory")
        if not p.exists():
            raise ToolError(f"no such file: {file!r}")
        return p

    # -- the five tools -------------------------------------------------------
    def read_artifact(self, name: str) -> str:
        manifest = self.task.get("artifact_manifest", [])
        if name not in manifest:
            self._log("read_artifact", {"name": name}, False)
            raise ToolError(f"unknown artifact {name!r}. Available: {manifest}")
        payload = (self.episode_dir / "artifacts" / f"{name}.json").read_text(encoding="utf-8")
        self._log("read_artifact", {"name": name}, True)
        return payload

    def view_code(self, file: str = "pipeline.py", start: int | None = None,
                  end: int | None = None) -> str:
        path = self._source_path(file)
        lines = path.read_text(encoding="utf-8").splitlines()
        s = 1 if start is None else max(1, start)
        e = len(lines) if end is None else min(len(lines), end)
        if e - s + 1 > MAX_VIEW_LINES:
            e = s + MAX_VIEW_LINES - 1
        numbered = [f"{i:>4}\t{lines[i-1]}" for i in range(s, e + 1)]
        self._log("view_code", {"file": file, "start": s, "end": e}, True)
        return "\n".join(numbered)

    def apply_patch(self, diff: str) -> str:
        file = "pipeline.py"
        path = self._source_path(file)
        original = path.read_text(encoding="utf-8")
        try:
            patched = apply_unified_diff(original, diff)
        except PatchError as e:
            self._log("apply_patch", {"file": file}, False)
            raise ToolError(f"patch did not apply: {e}") from e
        try:
            compile(patched, file, "exec")
        except SyntaxError as e:
            self._log("apply_patch", {"file": file}, False)
            raise ToolError(f"patched source has a syntax error: {e}") from e
        path.write_text(patched, encoding="utf-8")
        self.patches_applied += 1
        self._log("apply_patch", {"file": file}, True)
        return f"Patch applied to {file}. Total patches this episode: {self.patches_applied}."

    def execute_code(self, script: str) -> str:
        result = run_script(script, self._pipeline_dir, self.episode_dir, timeout=30.0)
        self._log("execute_code", {"len": len(script)}, not result.timed_out)
        return result.render()

    def submit(self, diagnosis: str) -> str:
        if self.patches_applied < 1:
            self._log("submit", {}, False)
            raise ToolError("submit rejected: apply at least one patch before submitting.")
        if not diagnosis or not diagnosis.strip():
            self._log("submit", {}, False)
            raise ToolError("submit rejected: diagnosis must be a non-empty string.")
        self.submitted = True
        self.diagnosis = diagnosis
        self._log("submit", {}, True)
        return "Submission accepted. Episode ended; judge will evaluate the patched pipeline."

    # -- lifecycle ------------------------------------------------------------
    def patched_source(self) -> str:
        return (self._pipeline_dir / "pipeline.py").read_text(encoding="utf-8")

    def cleanup(self) -> None:
        shutil.rmtree(self.workspace_dir, ignore_errors=True)
