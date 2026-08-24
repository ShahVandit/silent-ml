"""A debug-gym environment wrapping our silent-bug episodes.

Our own agent layer reached the point where the model read the faulty config,
was told its remaining budget, had a one-line edit tool available, and still
never edited - so the agent loop, not the tools, was the obstacle. debug-gym
supplies a loop that many models have been driven through successfully, along
with validated `view`, `edit`, `bash`, `grep` and `pdb` tools.

What stays ours is the part that matters: ``judge_episode`` still decides whether
a fix is real, by retraining across seeds, re-injecting the mutation to prove the
fix was causal, and checking training stability. It plugs in through ``eval()``,
which ``RepoEnv`` documents as an override point.

Requires Python >= 3.12 (debug-gym's floor), so this module is imported lazily
and is not part of the default package import path.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from debug_gym.gym.entities import EvalOutput, Observation
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminals.local import LocalTerminal
from debug_gym.gym.terminals.terminal import Terminal
from debug_gym.gym.tools.submit import SubmitTool
from debug_gym.gym.tools.toolbox import Toolbox

from silentml.judge.judge import judge_episode

PIPELINE_FILE = "pipeline.py"


# The registry key is derived from the class name unless given explicitly, which
# would turn DiagnosingSubmitTool into "diagnosingsubmi".
@Toolbox.register(name="submit_diagnosis")
class DiagnosingSubmitTool(SubmitTool):
    """``submit`` that keeps the agent's message so the judge can score it.

    The stock tool passes ``message`` only into its own observation text; the
    judge needs it to award the diagnosis component of the reward.
    """

    name = "submit_diagnosis"

    def use(self, environment, message: str = None, **kwargs) -> Observation:
        environment.submit_message = message or ""
        return super().use(environment, message=message, **kwargs)


class SilentMLEnv(RepoEnv):
    """One silent-bug episode, scored by our judge instead of a test suite."""

    def __init__(
        self,
        task_data: dict,
        terminal: Terminal | None = None,
        judge_seeds: tuple[int, ...] = (0, 1),
        **kwargs,
    ):
        terminal = terminal or LocalTerminal()
        kwargs.setdefault("max_score", 1)
        # The entrypoint is never used: eval() calls the judge directly rather
        # than shelling out, because the judge retrains the model in-process.
        super().__init__(
            task_data=task_data,
            entrypoint="python -c 'pass'",
            terminal=terminal,
            **kwargs,
        )
        self.judge_seeds = tuple(judge_seeds)
        self.judge_result = None
        self.submit_message = ""

    # -- task -----------------------------------------------------------------
    def setup_task(self) -> None:
        self.episode_dir = Path(self.task_data["episode_dir"]).resolve()
        self.meta = yaml.safe_load(
            (self.episode_dir / "meta.yaml").read_text(encoding="utf-8")
        )
        self.task_spec = yaml.safe_load(
            (self.episode_dir / "task.yaml").read_text(encoding="utf-8")
        )
        self.submit_message = ""
        self.judge_result = None

    @property
    def task_name(self) -> str:
        return self.meta["operator_id"]

    @property
    def instructions(self) -> str:
        artifacts = "\n".join(
            f"  - artifacts/{name}.json"
            for name in self.task_spec.get("artifact_manifest", [])
        )
        return (
            "A supervised learning pipeline trains to completion without raising "
            "any error, but the model underperforms badly. Exactly one bug has "
            "been introduced into pipeline.py.\n\n"
            f"{self.task_spec['bug_description']}\n\n"
            "The working directory holds pipeline.py and a set of pre-computed "
            "diagnostics recorded during the failing run:\n"
            f"{artifacts}\n\n"
            "Read the diagnostics to localise the fault, edit pipeline.py to fix "
            "it, then submit with a one-sentence diagnosis naming the bug. Change "
            "only what the bug requires - do not otherwise tune the pipeline, and "
            "do not modify the diagnostics."
        )

    # -- workspace ------------------------------------------------------------
    def setup_workspace(self) -> None:
        self.workspace.reset()
        # pipeline.py at the root, diagnostics beside it. copy_content shells out
        # to `cp -r src/. target` and fails if the target is missing, so the
        # artifacts directory has to exist first.
        self.workspace.copy_content(self.episode_dir / "pipeline")
        self.terminal.run("mkdir -p artifacts")
        self.workspace.copy_content(
            self.episode_dir / "artifacts", self.workspace.working_dir / "artifacts"
        )

    def setup_terminal(self) -> None:
        self.terminal.run("git init -b main")
        self.terminal.run("git config user.name 'debug-gym'")
        self.terminal.run("git config user.email '<>'")
        self.terminal.run("git add .")
        self.terminal.run("git commit -am 'Init'")

    # -- scoring --------------------------------------------------------------
    def eval(self, **kwargs) -> EvalOutput:
        """Score the agent's edit with our judge.

        Replaces the inherited "run the entrypoint and read its exit status"
        behaviour: correctness here means the retrained model recovers, and that
        re-injecting the original mutation breaks it again.
        """
        try:
            patched_source = self.workspace.read_file(PIPELINE_FILE)
        except Exception as e:
            self.last_eval = EvalOutput(False, f"could not read {PIPELINE_FILE}: {e}")
            return self.last_eval

        result = judge_episode(
            self.episode_dir,
            patched_source,
            self.submit_message,
            seeds=self.judge_seeds,
        )
        self.judge_result = result
        success = result.functional_pass and result.ablation != "fail"
        self.last_eval = EvalOutput(success, result.render())
        return self.last_eval

    def calculate_max_score(self, eval_output: EvalOutput) -> int:
        return 1

    def calculate_score(self, eval_output: EvalOutput) -> int:
        return int(eval_output.success)

    def calculate_resolved(self, eval_output: EvalOutput) -> bool:
        return bool(eval_output.success)

    # -- dataset --------------------------------------------------------------
    @classmethod
    def load_dataset(cls, episodes_dir: str | Path = "episodes", **kwargs) -> dict:
        """Map operator id -> task_data for every generated episode."""
        episodes_dir = Path(episodes_dir)
        dataset = {}
        for meta_path in sorted(episodes_dir.glob("*/meta.yaml")):
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
            dataset[meta["operator_id"]] = {
                "env_type": "silentml",
                "episode_dir": str(meta_path.parent.resolve()),
                "task_name": meta["operator_id"],
            }
        if not dataset:
            raise FileNotFoundError(
                f"no episodes under {episodes_dir} - run `silentml.cli generate` first"
            )
        return dataset

    def summary(self) -> dict:
        """Judge detail for the benchmark report, beyond pass/fail."""
        r = self.judge_result
        return {
            "task": self.task_name,
            "resolved": bool(r and r.functional_pass and r.ablation != "fail"),
            "functional_pass": bool(r and r.functional_pass),
            "ablation": (r.ablation if r else "not_run"),
            "diagnosis_ok": bool(r and r.details.get("diagnosis_ok")),
            "reward": (r.reward if r else 0.0),
            "breakdown": (dict(r.breakdown) if r else {}),
            "submit_message": self.submit_message,
        }
