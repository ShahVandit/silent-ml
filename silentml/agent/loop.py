"""Model-agnostic episode loop: prompt -> tool calls -> judge.

A *policy* is any callable ``policy(prompt, history) -> {"tool": str, "args": dict}``.
The same loop drives a scripted oracle (for testing the harness) or an LLM-backed
policy (for the benchmark), so the environment never hard-codes a model.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable

from silentml.agent.tools import EpisodeSession, ToolError
from silentml.judge.judge import JudgeResult, judge_episode

Policy = Callable[[str, list[dict]], dict]

TOOLS_DESCRIPTION = """\
- read_artifact(name): retrieve a named precomputed diagnostic (no code change).
- view_code(file, start, end): return source lines from the pipeline.
- apply_patch(diff): apply a unified diff to pipeline.py (syntax-validated).
- execute_code(script): run a CPU-only Python script (30s limit) with read access
  to source/data/checkpoint/artifacts; cannot write source. Returns stdout/stderr.
- submit(diagnosis): end the episode; requires >=1 apply_patch and a non-empty
  diagnosis (JSON with a "diagnosis" field)."""

PROMPT_TEMPLATE = """\
You are an ML debugging agent. A supervised learning pipeline has completed
training without errors but is underperforming. Your task is to identify the
root cause and submit a fix.

[OBSERVED FAILURE]
{bug_description}

[PIPELINE DIRECTORY]
{directory_structure}

[AVAILABLE TOOLS]
{tools}

[PRE-COMPUTED ARTIFACTS]
Available via read_artifact(name):
{artifact_manifest}

[OUTPUT REQUIREMENTS]
Investigate in any order, then submit a JSON diagnosis of the form
{{"diagnosis": "<bug_type> in <file> - one-sentence causal mechanism",
  "supporting_evidence": ["exact artifact signals"]}}.
Constraints: do not optimise beyond fixing the identified bug; do not modify raw
data; execute_code is CPU-only with a 30s limit; submit is rejected without a
patch and a non-empty diagnosis. Exactly 1 bug has been injected."""


def build_prompt(session: EpisodeSession) -> str:
    task = session.task
    return PROMPT_TEMPLATE.format(
        bug_description=task["bug_description"],
        directory_structure=task["directory_structure"],
        tools=TOOLS_DESCRIPTION,
        artifact_manifest="\n".join(f"  - {a}" for a in task["artifact_manifest"]),
    )


@dataclasses.dataclass
class EpisodeResult:
    episode_id: str
    submitted: bool
    judge: JudgeResult | None
    trajectory: list[dict]
    n_tool_calls: int


def _dispatch(session: EpisodeSession, tool: str, args: dict) -> str:
    if tool == "read_artifact":
        return session.read_artifact(args["name"])
    if tool == "view_code":
        return session.view_code(args.get("file", "pipeline.py"),
                                 args.get("start"), args.get("end"))
    if tool == "apply_patch":
        return session.apply_patch(args["diff"])
    if tool == "execute_code":
        return session.execute_code(args["script"])
    if tool == "submit":
        return session.submit(args["diagnosis"])
    raise ToolError(f"unknown tool {tool!r}")


def run_episode(
    episode_dir: str | Path,
    policy: Policy,
    max_calls: int = 20,
    judge_seeds: tuple[int, ...] = (0, 1),
) -> EpisodeResult:
    session = EpisodeSession(episode_dir)
    prompt = build_prompt(session)
    history: list[dict] = []
    try:
        for _ in range(max_calls):
            action = policy(prompt, history)
            tool, args = action["tool"], action.get("args", {})
            try:
                observation = _dispatch(session, tool, args)
                ok = True
            except (ToolError, KeyError) as e:
                observation = f"ERROR: {e}"
                ok = False
            history.append({"tool": tool, "args": args, "observation": observation, "ok": ok})
            if tool == "submit" and ok:
                break

        judge_result = None
        if session.submitted:
            judge_result = judge_episode(
                session.episode_dir, session.patched_source(),
                session.diagnosis or "", seeds=judge_seeds,
            )
        return EpisodeResult(
            episode_id=session.task["episode_id"],
            submitted=session.submitted,
            judge=judge_result,
            trajectory=history,
            n_tool_calls=len(history),
        )
    finally:
        session.cleanup()


# --- A scripted oracle policy, for testing the harness end-to-end ------------
def oracle_policy(fix_diff: str, diagnosis: str) -> Policy:
    """Returns a policy that inspects one artifact, applies ``fix_diff``, submits."""
    plan = [
        {"tool": "read_artifact", "args": {"name": "layer_gradient_stats"}},
        {"tool": "view_code", "args": {"file": "pipeline.py", "start": 50, "end": 75}},
        {"tool": "apply_patch", "args": {"diff": fix_diff}},
        {"tool": "submit", "args": {"diagnosis": diagnosis}},
    ]
    step = {"i": 0}

    def policy(_prompt: str, _history: list[dict]) -> dict:
        action = plan[min(step["i"], len(plan) - 1)]
        step["i"] += 1
        return action

    return policy
