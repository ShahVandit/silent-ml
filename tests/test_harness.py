"""Fast harness tests (no model training — training is mocked where needed)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from silentml.agent.patching import PatchError, apply_unified_diff
from silentml.agent.tools import EpisodeSession, ToolError
from silentml.bugs.operators import get_operator, operators_for
import silentml.judge.judge as J
from silentml.pipelines.base import DataMeta, EvalMetrics, History, RunResult

REPO = Path(__file__).resolve().parents[1]
EP = REPO / "episodes" / "transformer_text__T_HLR"   # learning-rate episode

pytestmark = pytest.mark.skipif(
    not EP.exists(), reason="generate episodes first (python -m silentml.cli generate)"
)

DM = DataMeta(4, [], (64,))
# The injected fault for this episode is lr 3e-4 -> 3e-6; the buggy/ablated
# sources are the ones containing 3e-6.
BUGGY_MARKER = "3e-6"


def _sources():
    meta = yaml.safe_load((EP / "meta.yaml").read_text(encoding="utf-8"))
    buggy = (EP / "pipeline" / "pipeline.py").read_text(encoding="utf-8")
    return buggy, apply_unified_diff(buggy, meta["ground_truth_fix_diff"])


# --- patch applier -----------------------------------------------------------
def test_ground_truth_diff_recovers_clean():
    buggy, clean = _sources()
    assert BUGGY_MARKER in buggy
    assert '"lr": 3e-4,' in clean and BUGGY_MARKER not in clean


def test_bad_patch_raises():
    with pytest.raises(PatchError):
        apply_unified_diff("a\nb\nc\n", "@@ -1,1 +1,1 @@\n-zzz\n+q\n")


# --- operators ---------------------------------------------------------------
def test_operator_inject_is_single_site():
    op = get_operator("T_HLR")
    _buggy, clean = _sources()
    assert op.inject(clean) != clean
    assert "transformer_text" in op.applies_to


def test_registry_covers_deepcrime_and_attention_families():
    ids = {o.id for o in operators_for("transformer_text")}
    assert {"T_HLR", "T_OCH", "T_TCL", "T_RCD", "T_WCI"} <= ids        # DeepCrime
    assert {"ATT_MASK", "ATT_SCALE", "POS_ENC", "SOFTMAX_DIM"} <= ids  # Jahan 2025


def test_inactive_operators_are_documented_not_deleted():
    """Operators that provably cannot degrade this pipeline stay in the registry
    with applies_to=() so the negative result is preserved, not silently dropped."""
    from silentml.bugs.operators import OPERATORS

    inactive = {o.id for o in OPERATORS.values() if not o.applies_to}
    assert {"T_ACH", "T_ARM", "POOL_PAD", "T_EMB_FREEZE", "T_RESIDUAL"} <= inactive
    for op_id in inactive:
        assert OPERATORS[op_id].find, f"{op_id} lost its anchor"


def test_attention_operators_carry_taxonomy_tag():
    assert get_operator("ATT_MASK").jahan2025


# --- session tool gating -----------------------------------------------------
def test_session_tool_gating():
    s = EpisodeSession(EP)
    try:
        with pytest.raises(ToolError):
            s.read_artifact("does_not_exist")
        with pytest.raises(ToolError):        # submit before any patch
            s.submit("x")
        assert s.read_artifact("loss_curves")
    finally:
        s.cleanup()


# --- judge reward logic (training mocked) ------------------------------------
def _mk(acc, healthy=True):
    if healthy:
        h = History([1.0, 0.5, 0.3], [0.9, 0.5, 0.4], [.6, .8, .85], [.7, .8, acc])
    else:
        h = History([0.5, 0.8, 1.2], [0.6, 0.9, 1.3], [.8, .6, .4], [.7, .5, acc])
    return RunResult(h, EvalMetrics(acc, {i: acc for i in range(4)}, 0.4), DM,
                     {"layer2_gradient_stats": {}})


def _patch_runs(monkeypatch, fn):
    monkeypatch.setattr(J, "_run_source", fn)


def test_judge_good_fix_full_reward(monkeypatch):
    _, clean = _sources()
    _patch_runs(monkeypatch, lambda src, seeds, collect=False:
                [_mk(0.55 if BUGGY_MARKER in src else 0.99)] * len(seeds))
    r = J.judge_episode(EP, clean, '{"diagnosis":"learning rate far too low"}', seeds=(0, 1))
    assert r.functional_pass and r.ablation == "pass"
    assert r.reward == pytest.approx(13.0)


def test_judge_noop_fails_functional(monkeypatch):
    buggy, _ = _sources()
    _patch_runs(monkeypatch, lambda src, seeds, collect=False: [_mk(0.55)] * len(seeds))
    r = J.judge_episode(EP, buggy, '{"diagnosis":"learning rate"}', seeds=(0, 1))
    assert not r.functional_pass and r.reward == 0.0


def test_judge_ablation_fail_net_negative(monkeypatch):
    _, clean = _sources()
    # Re-injecting the fault does not degrade -> the fix was not the cause.
    _patch_runs(monkeypatch, lambda src, seeds, collect=False: [_mk(0.99)] * len(seeds))
    r = J.judge_episode(EP, clean, '{"diagnosis":"learning rate"}', seeds=(0, 1))
    assert r.ablation == "fail" and r.reward < 0


def test_judge_unstable_loses_curves(monkeypatch):
    _, clean = _sources()
    _patch_runs(monkeypatch, lambda src, seeds, collect=False:
                [_mk(0.55 if BUGGY_MARKER in src else 0.99,
                     healthy=BUGGY_MARKER in src)] * len(seeds))
    r = J.judge_episode(EP, clean, '{"diagnosis":"learning rate"}', seeds=(0, 1))
    assert "P_curves" in r.breakdown and r.stability_violations


def test_judge_wrong_diagnosis_penalised(monkeypatch):
    _, clean = _sources()
    _patch_runs(monkeypatch, lambda src, seeds, collect=False:
                [_mk(0.55 if BUGGY_MARKER in src else 0.99)] * len(seeds))
    r = J.judge_episode(EP, clean, '{"diagnosis":"the dataset is corrupted"}', seeds=(0, 1))
    assert "P_diagnosis_wrong" in r.breakdown


# --- execute_code sandbox ----------------------------------------------------
def test_execute_code_can_read_pipeline_but_not_mutate_workspace():
    s = EpisodeSession(EP)
    try:
        before = s.patched_source()
        out = s.execute_code(
            "import pipeline\n"
            "print('LR', pipeline.CONFIG['lr'])\n"
            "open('pipeline/pipeline.py', 'w').write('WIPED')\n"
        )
        assert "LR" in out                      # the script really ran
        assert s.patched_source() == before     # the workspace is untouched
    finally:
        s.cleanup()


def test_execute_code_times_out():
    s = EpisodeSession(EP)
    try:
        out = s.execute_code("import time\ntime.sleep(60)\n")
        assert "timed out" in out.lower()
    finally:
        s.cleanup()


def test_diagnosis_scored_even_when_fix_fails(monkeypatch):
    """'Diagnosed it but could not fix it' must be distinguishable in reporting."""
    buggy, _ = _sources()
    _patch_runs(monkeypatch, lambda src, seeds, collect=False: [_mk(0.55)] * len(seeds))
    r = J.judge_episode(EP, buggy, '{"diagnosis":"the learning rate is far too low"}',
                        seeds=(0, 1))
    assert not r.functional_pass
    assert r.reward == 0.0                    # reward semantics unchanged
    assert r.details["diagnosis_ok"] is True  # but the correct diagnosis is recorded
