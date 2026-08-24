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
- read_artifact(name): retrieve a precomputed diagnostic. Pass a name from the
  manifest below exactly, with no .json suffix and no directory.
- view_code(file, start, end): return numbered source lines. The only file is
  "pipeline.py"; that is the default, so file may be omitted.
- replace_in_file(old, new): replace an exact snippet of pipeline.py with new
  text. The simplest way to edit - copy `old` verbatim from view_code output
  (without the line numbers) and give the corrected text as `new`.
- apply_patch(diff): apply a unified diff to pipeline.py, if you prefer diffs.
  Either this or replace_in_file must succeed before submit.
- execute_code(script): run a CPU-only Python script (30s limit). It runs in a
  throwaway copy holding pipeline/pipeline.py, artifacts/ and baseline/, so the
  paths it sees differ from the arguments the other tools take. Returns
  stdout/stderr.
- submit(diagnosis): end the episode; requires >=1 successful apply_patch and a
  non-empty diagnosis (JSON with a "diagnosis" field)."""

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

[HOW TO SPEND YOUR TURNS]
You have a limited number of tool calls, so do not spend them exploring the
filesystem - the layout above is complete. A workable order is: read the
artifacts that bear on the symptom, view the region of pipeline.py they point
at, form a hypothesis, edit with replace_in_file, then submit. Make the edit and
submit well before the budget runs out; an episode that ends without submitting
scores zero however good the analysis was. Do not re-read source you have already
seen - if you can name the faulty line, edit it.

[OUTPUT REQUIREMENTS]
Investigate in any order, then submit a JSON diagnosis of the form
{{"diagnosis": "<bug_type> in <file> - one-sentence causal mechanism",
  "supporting_evidence": ["exact artifact signals"]}}.
Constraints: do not optimise beyond fixing the identified bug; do not modify raw
data; execute_code is CPU-only with a 30s limit; submit is rejected unless an
edit has succeeded and the diagnosis is non-empty. Exactly 1 bug has been injected."""


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
    if tool == "replace_in_file":
        return session.replace_in_file(args["old"], args["new"],
                                       args.get("file", "pipeline.py"))
    if tool == "execute_code":
        return session.execute_code(args["script"])
    if tool == "submit":
        return session.submit(args["diagnosis"])
    raise ToolError(f"unknown tool {tool!r}")


def _budget_note(remaining: int, session: EpisodeSession) -> str:
    """Tell the agent how many calls are left and what still has to happen."""
    if session.patches_applied:
        todo = "You have applied a patch; call submit before the budget runs out."
    else:
        todo = "You have NOT applied a patch yet; submit is refused without one."
    if remaining <= 0:
        return "[no tool calls remaining]"
    if remaining <= 5:
        return (f"[{remaining} tool calls remaining - stop investigating and act now. "
                f"{todo}]")
    return f"[{remaining} tool calls remaining. {todo}]"


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
        for turn in range(max_calls):
            action = policy(prompt, history)
            tool, args = action["tool"], action.get("args", {})
            try:
                observation = _dispatch(session, tool, args)
                ok = True
            except (ToolError, KeyError) as e:
                observation = f"ERROR: {e}"
                ok = False
            # An agent that cannot see its remaining budget re-reads the same
            # code instead of committing to a fix, then runs out having never
            # patched. State the budget, and escalate as it gets short.
            observation = f"{observation}\n\n{_budget_note(max_calls - turn - 1, session)}"
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
