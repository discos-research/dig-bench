#!/usr/bin/env python3
"""Integration / truncation-smoke tests for core/harness.py — fakes only.

    python tests/test_harness.py

Drives the real GeminiPolicy (fake genai client) + real Output + real
history through a deterministic scripted game:
- a full game completes; the debrief fires exactly once; .log + .jsonl written
  with session-first / summary-last and a terminal banner;
- invalid action -> note -> nudge cap stops "blocked";
- TRUNCATION SMOKE: a long forced-tool loop with a tiny context budget
  truncates repeatedly; on EVERY eviction the active step-pair's signature is intact
  and the head TASK DESCRIPTION is pinned; accounting accrues across truncations.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pytest
pytest.importorskip("google.genai")  # this integration test builds a real GeminiPolicy; skip w/o SDK

from core.output import Output
from core.harness import play
from clients.gemini_client import GeminiPolicy
from core import history as H
from clients.sglang_client import SglangPolicy
from test_gemini import FakeClient, FakeUsage, move_response, debrief_response, nomove_response
from test_sglang import FakePost, fake_get, MODELS_ONE, tool_resp, usage as sg_usage
from clients.anthropic_client import AnthropicPolicy
from test_anthropic import (FakeClient as AnthFakeClient, resp as anth_resp,
                            thinking_block, tool_block, text_block, usage as anth_usage)
from clients.openai_client import OpenAIPolicy
from test_openai import (FakeClient as OAFakeClient, resp as oa_resp, reasoning_item,
                         function_call as oa_call, usage as oa_usage)


# ---- Fakes -------------------------------------------------------------


class FakeBench:
    """Scripted game. `transitions` are the states reached on each VALID move."""

    def __init__(self, initial, transitions):
        self._cur = initial
        self._idx = 0
        self._script = list(transitions)
        self.on_retry = None

    def start_session(self, game, model_name, model_version):
        return {"session_id": "fake-sid", "game": game, "seed": 1, "framework_version": "fake",
                "description": "Reach the goal.", "state": self._cur, "step_index": 0, "done": False}

    def step(self, sid, step_index, action):
        assert step_index == self._idx + 1, f"index discipline: got {step_index}, want {self._idx + 1}"
        if action not in self._cur.get("actions", []):
            return {"state": self._cur, "step_index": self._idx, "invalid_action": True}
        self._cur = self._script.pop(0)
        self._idx += 1
        return {"state": self._cur, "step_index": self._idx, "invalid_action": False}


def st(obs, *, actions=("1",), score=0, status="in_progress", done=False):
    return {"observation": obs, "actions": list(actions), "score": score, "level": 1,
            "max_level": 9, "lives_left": 3, "steps_remaining": 100, "status": status, "done": done}


def make_policy(scripts, **over):
    client = FakeClient(scripts)
    kw = dict(api_key="x", model="gemini-3.1-pro-preview", thinking_level="high",
              timeout=120, max_retries=3)
    kw.update(over)
    return GeminiPolicy(client=client, **kw)


def make_args(**over):
    base = dict(model="gemini-3.1-pro-preview", max_steps=50, max_cost_usd=0.0, max_invalid_retries=5)
    base.update(over)
    return SimpleNamespace(**base)


def run(policy, bench, args, *, context_budget=None):
    with tempfile.TemporaryDirectory() as d:
        log_path = pathlib.Path(d) / "r.log"
        jsonl_path = pathlib.Path(d) / "r.jsonl"
        out = Output(summary_chars=80, verbose=False, log_path=log_path, jsonl_path=jsonl_path)
        with redirect_stdout(io.StringIO()):
            stop = play(bench, policy, out, bench.start_session("G", "L", "M"), args,
                        "http://x", "high", "L", context_budget=context_budget)
        out.close()
        records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
        return stop, records, log_path.read_text()


# ---- Tests -------------------------------------------------------------


def test_full_game_completes_with_one_debrief():
    win = st("WIN", actions=(), score=20, status="completed", done=True)
    policy = make_policy([move_response("1"), move_response("1"), debrief_response("Push the block.")])
    bench = FakeBench(st("S0"), [st("S1", score=10), win])
    stop, records, log = run(policy, bench, make_args())
    assert stop == "done"
    kinds = [r.get("type", "turn") for r in records]
    assert kinds[0] == "session" and kinds[-1] == "summary"
    turns = [r for r in records if r.get("type", "turn") == "turn"]
    assert len(turns) == 2 and turns[0]["output"]["has_continuity"] is True
    assert turns[0]["usage"]["cached"] == 800
    summary = records[-1]
    assert summary["debrief"] == "Push the block." and summary["llm_calls"] == 3  # 2 moves + debrief
    assert "★ Game completed!" in log and "SUMMARY" in log
    assert "gemini_seed" not in log and "seed None" in log  # provider-neutral seed label


def test_nudge_cap_blocks():
    # never a make_move call -> nudge each turn -> exceed the cap -> "blocked"
    policy = make_policy([nomove_response() for _ in range(4)] + [debrief_response("n/a")])
    bench = FakeBench(st("S0"), [st("S1")])
    stop, records, log = run(policy, bench, make_args(max_invalid_retries=2))
    assert stop == "blocked"
    nudges = [r for r in records if r.get("output", {}).get("nudged")]
    assert len(nudges) == 3  # cap 2 -> stops after the 3rd consecutive (>2)


def test_illegal_action_rejected_locally_without_hitting_bench():
    # Anti-cheat: an action not in the state's legal_actions is charged as invalid LOCALLY —
    # never forwarded to the bench (so a hidden/undocumented server action can't be played).
    policy = make_policy([move_response("9"), move_response("9"), move_response("9"),
                          debrief_response("n/a")])  # "9" is never legal (state offers only "1")
    bench = FakeBench(st("S0", actions=("1",)), [st("S1")])
    sent = []
    real_step = bench.step
    bench.step = lambda sid, i, a: sent.append(a) or real_step(sid, i, a)  # count/record forwarded moves
    stop, records, log = run(policy, bench, make_args(max_invalid_retries=2))
    assert stop == "blocked"
    assert sent == []  # the illegal action never reached the server
    turns = [r for r in records if r.get("type", "turn") == "turn"]
    assert len(turns) == 3 and all(t["output"]["invalid_action"] for t in turns)


def test_truncation_smoke_active_intact_every_eviction():
    N = 10
    # high reported prompt every call so we sit way over a tiny budget from the start
    big = FakeUsage(prompt=5000, cached=0, output=20, thoughts=50, total=5070)
    scripts = [move_response("1", signature=f"SIG{i}".encode(), usage=big) for i in range(N)]
    scripts.append(debrief_response("done", usage=big))
    policy = make_policy(scripts)

    # wrap eviction to assert the invariants on EVERY eviction
    real_evict = policy.evict_oldest_turn
    audit = {"evictions": 0, "min_units": 99}

    def audited_evict():
        real_evict()
        audit["evictions"] += 1
        units = policy.turns()
        audit["min_units"] = min(audit["min_units"], len(units))
        # head pinned: the first content is still the TASK DESCRIPTION
        assert "TASK DESCRIPTION" in str(policy.contents[0].parts[0].text)
        # active unit intact + carries its signature
        assert units[-1].is_active
        assert any(getattr(pt, "thought_signature", None) for c in units[-1].payload for pt in c.parts)

    policy.evict_oldest_turn = audited_evict

    bench = FakeBench(st("S0"), [st(f"S{i}", score=i) for i in range(1, N + 5)])
    stop, records, log = run(policy, bench, make_args(max_steps=N), context_budget=1000)

    assert stop == "max_steps"
    truncations = [r for r in records if r.get("type") == "truncation"]
    assert len(truncations) >= 3, f"truncation should fire repeatedly, got {len(truncations)}"
    assert audit["evictions"] >= 3
    # never evicted below the floor (active + >=1 prior)
    assert audit["min_units"] >= H.MIN_KEEP_TURNS
    # accounting accrued across all the truncations (preserved, never reset)
    assert policy.prompt_tokens == 5000 * policy.call_count
    assert "↺" in log  # truncation visible in the human trace


def test_sglang_truncation_smoke_active_intact():
    # Same smoke as Gemini, but driving the real SglangPolicy (forced-tool).
    N = 10
    big = sg_usage(prompt=5000, cached=0, output=20, reasoning=40)
    script = [tool_resp("1", call_id=f"c{i}", reasoning=f"r{i}", u=big) for i in range(N)]
    script.append(debrief_sg := {"choices": [{"finish_reason": "stop",
                  "message": {"role": "assistant", "content": "done"}}], "usage": big})
    policy = SglangPolicy(
        base_url="http://x/v1", api_key="k", model=None, timeout=30, max_retries=3,
        http_post=FakePost(script), http_get=fake_get(MODELS_ONE),
    )
    policy._reasoning_roundtrips = True

    real_evict = policy.evict_oldest_turn
    audit = {"evictions": 0, "min_units": 99}

    def audited_evict():
        real_evict()
        audit["evictions"] += 1
        units = policy.turns()
        audit["min_units"] = min(audit["min_units"], len(units))
        assert policy.messages[0]["content"].startswith("You are now playing")  # head pinned
        assert units[-1].is_active
        assert any(m.get("reasoning_content") for m in units[-1].payload)  # active reasoning intact

    policy.evict_oldest_turn = audited_evict
    bench = FakeBench(st("S0"), [st(f"S{i}", score=i) for i in range(1, N + 5)])
    stop, records, log = run(policy, bench, make_args(max_steps=N, model="kimi-k3"), context_budget=1000)

    assert stop == "max_steps"
    truncations = [r for r in records if r.get("type") == "truncation"]
    assert len(truncations) >= 3 and audit["evictions"] >= 3
    assert audit["min_units"] >= H.MIN_KEEP_TURNS
    assert policy.prompt_tokens == 5000 * policy.call_count  # accounting preserved
    # provenance recorded in the summary
    summary = records[-1]
    assert summary["move_channel"] == "forced-tool" and "channel_fell_back" not in summary
    assert "↺" in log


def test_anthropic_truncation_smoke_active_thinking_intact():
    # Same smoke as Gemini/SGLang, driving the real AnthropicPolicy (auto-tool,
    # thinking on). Each turn carries a thinking block whose signature must stay intact
    # in the active unit across repeated evictions under a tiny budget.
    N = 10
    big = anth_usage(inp=5000, out=20)
    script = [anth_resp([thinking_block(f"r{i}", f"SIG{i}"), tool_block("1", f"c{i}")], u=big) for i in range(N)]
    script.append(anth_resp([text_block("done")], u=big, stop="end_turn"))  # debrief
    policy = AnthropicPolicy(
        api_key="k", model="global.anthropic.claude-sonnet-4-6", aws_region="us-east-1",
        thinking_level="high", max_tokens=32000, timeout=30, max_retries=3,
        client=AnthFakeClient(script),
    )

    real_evict = policy.evict_oldest_turn
    audit = {"evictions": 0, "min_units": 99}

    def audited_evict():
        real_evict()
        audit["evictions"] += 1
        units = policy.turns()
        audit["min_units"] = min(audit["min_units"], len(units))
        assert policy.messages[0]["content"].startswith("You are now playing")  # head pinned
        assert units[-1].is_active
        asst = next(m for m in units[-1].payload if m.get("role") == "assistant")
        assert any(getattr(b, "type", None) == "thinking" and getattr(b, "signature", None)
                   for b in asst["content"])  # active thinking + signature intact

    policy.evict_oldest_turn = audited_evict
    bench = FakeBench(st("S0"), [st(f"S{i}", score=i) for i in range(1, N + 5)])
    stop, records, log = run(policy, bench, make_args(max_steps=N, model="global.anthropic.claude-sonnet-4-6"),
                             context_budget=1000)

    assert stop == "max_steps"
    truncations = [r for r in records if r.get("type") == "truncation"]
    assert len(truncations) >= 3 and audit["evictions"] >= 3
    assert audit["min_units"] >= H.MIN_KEEP_TURNS
    assert policy.prompt_tokens == 5000 * policy.call_count  # accounting preserved across truncations
    summary = records[-1]
    assert summary["move_channel"] == "auto-tool"  # documented: not forced (thinking on)
    assert "↺" in log


def test_included_label_shown_per_turn_sglang_not_gemini():
    # SGLang "included" mode (server omits reasoning_tokens) -> every per-turn line AND the
    # summary show "out + think N" (merged, no fake split, no "n/a"). Gemini (exact) shows the
    # disjoint "out N think M" and never "out + think".
    big = sg_usage(prompt=100, output=20, reasoning=None)  # completion_tokens_details=None
    script = [tool_resp("1", call_id=f"c{i}", reasoning="reasoning blah", u=big) for i in range(2)]
    script.append({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}], "usage": big})
    sp = SglangPolicy(base_url="http://x/v1", api_key="k", model=None, timeout=30, max_retries=3,
                          http_post=FakePost(script), http_get=fake_get(MODELS_ONE))
    sp._reasoning_roundtrips = True
    _, srecords, slog = run(sp, FakeBench(st("S0"), [st("S1"), st("S2")]),
                            make_args(max_steps=2, model="kimi-k3"))
    assert "out + think" in slog and "think n/a" not in slog and "(est)" not in slog  # every line + summary
    assert srecords[-1]["thoughts_basis"] == "included"
    turns = [r for r in srecords if r.get("type", "turn") == "turn"]
    assert all(t["output"]["thoughts_basis"] == "included" for t in turns)  # per-turn jsonl basis

    # Gemini exact: disjoint out/think, never merged.
    gp = make_policy([move_response("1"), debrief_response("ok")])
    _, grecords, glog = run(gp, FakeBench(st("S0"), [st("S1", score=10, status="completed", done=True)]), make_args())
    assert "out + think" not in glog and grecords[-1]["thoughts_basis"] == "exact"


def test_openai_truncation_smoke_active_reasoning_intact():
    # Same smoke as the others, driving the real OpenAIPolicy (Responses API, forced
    # tool). Each turn carries a reasoning item with encrypted_content that must stay intact in
    # the active unit across repeated evictions under a tiny budget.
    N = 10
    big = oa_usage(inp=5000, out=20, reasoning=10)
    script = [oa_resp([reasoning_item(f"E{i}", f"r{i}"), oa_call("1", f"c{i}")], u=big) for i in range(N)]
    script.append(oa_resp([], u=big, output_text="done"))  # debrief
    policy = OpenAIPolicy(
        api_key="k", model="gpt-5.5", effort="high", max_tokens=32000, timeout=30, max_retries=3,
        client=OAFakeClient(script),
    )

    real_evict = policy.evict_oldest_turn
    audit = {"evictions": 0, "min_units": 99}

    def audited_evict():
        real_evict()
        audit["evictions"] += 1
        units = policy.turns()
        audit["min_units"] = min(audit["min_units"], len(units))
        assert policy.input[0]["content"].startswith("You are now playing")  # head pinned
        assert units[-1].is_active
        assert any(getattr(it, "type", None) == "reasoning" and getattr(it, "encrypted_content", None)
                   for it in units[-1].payload)  # active reasoning + encrypted_content intact

    policy.evict_oldest_turn = audited_evict
    bench = FakeBench(st("S0"), [st(f"S{i}", score=i) for i in range(1, N + 5)])
    stop, records, log = run(policy, bench, make_args(max_steps=N, model="gpt-5.5"), context_budget=1000)

    assert stop == "max_steps"
    truncations = [r for r in records if r.get("type") == "truncation"]
    assert len(truncations) >= 3 and audit["evictions"] >= 3
    assert audit["min_units"] >= H.MIN_KEEP_TURNS
    assert policy.prompt_tokens == 5000 * policy.call_count  # accounting preserved
    summary = records[-1]
    assert summary["move_channel"] == "forced-tool" and summary["thoughts_basis"] == "exact"
    assert "↺" in log


def test_cost_cap_stops_gemini_and_anthropic():
    # --max-cost-usd must halt a priced run once cumulative cost crosses the cap, for
    # every priced provider. One priced move pushes cost over a tiny cap → the next loop
    # top trips "cost_cap" (and the debrief is skipped on a cost-cap stop).
    # Gemini:
    gp = make_policy([move_response("1", usage=FakeUsage(prompt=2000, cached=0, output=100, thoughts=100, total=2200))])
    gstop, grecords, _ = run(gp, FakeBench(st("S0"), [st("S1"), st("S2")]), make_args(max_cost_usd=0.001))
    assert gstop == "cost_cap" and gp.cost_usd > 0.001
    assert grecords[-1]["debrief"] is None  # debrief skipped on cost_cap (don't spend past the cap)
    gturns = [r for r in grecords if r.get("type", "turn") == "turn"]
    assert len(gturns) == 1  # stopped after the move that crossed the cap

    # Anthropic (real policy + fake client, Sonnet 4.6 priced):
    ap = AnthropicPolicy(
        api_key="k", model="global.anthropic.claude-sonnet-4-6", aws_region="us-east-1",
        thinking_level="high", max_tokens=32000, timeout=30, max_retries=3,
        client=AnthFakeClient([anth_resp([thinking_block(), tool_block("1")], u=anth_usage(inp=2000, out=200))]),
    )
    astop, arecords, _ = run(ap, FakeBench(st("S0"), [st("S1"), st("S2")]),
                             make_args(max_cost_usd=0.001, model="global.anthropic.claude-sonnet-4-6"))
    assert astop == "cost_cap" and ap.cost_usd > 0.001
    assert arecords[-1]["debrief"] is None
    assert ap.has_pricing is True and arecords[-1]["cost_usd"] is not None


def test_cost_cap_crossed_on_final_move_skips_debrief():
    # F1b: the cap is tested before each turn, so a run whose LAST move crossed it stops with a
    # normal reason ("done") — the debrief (a whole-transcript call, often the most expensive
    # of the run) must still be skipped, not overspend the cap.
    win = st("WIN", actions=(), status="completed", done=True)
    gp = make_policy([move_response("1", usage=FakeUsage(prompt=2000, cached=0, output=100,
                                                         thoughts=100, total=2200))])
    stop, records, _ = run(gp, FakeBench(st("S0"), [win]), make_args(max_cost_usd=0.001))
    assert stop == "done" and gp.cost_usd > 0.001
    assert records[-1]["debrief"] is None
    assert gp.call_count == 1  # the move only — no debrief LLM call was spent
    assert any(r.get("type") == "warn" and "skipping debrief" in r["message"] for r in records)


def test_label_three_way_reasoning_shown_vs_carried_vs_none():
    # strip-model (roundtrips False): turn 1 thinks (shown, not carried), turn 2 doesn't think
    big = sg_usage(prompt=100)
    script = [
        tool_resp("1", call_id="c1", reasoning="rrr", u=big),     # reasoning present
        tool_resp("1", call_id="c2", reasoning=None, u=big),       # no reasoning
        {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}], "usage": big},
    ]
    policy = SglangPolicy(base_url="http://x/v1", api_key="k", model=None, timeout=30, max_retries=3,
                              http_post=FakePost(script), http_get=fake_get(MODELS_ONE))
    policy._reasoning_roundtrips = False  # server strips -> carry is a no-op
    bench = FakeBench(st("S0"), [st("S1"), st("S2")])
    stop, records, log = run(policy, bench, make_args(max_steps=2, model="kimi-k3"), context_budget=None)
    assert "server STRIPS it" in log                # turn 1: thought, carry proven a no-op
    assert "no reasoning this turn" in log          # turn 2: didn't think
    assert "reasoning carried" not in log           # never claims carry when stripped
    turns = [r for r in records if r.get("type", "turn") == "turn"]
    assert [t["output"]["continuity"] for t in turns] == ["stripped", None]


class OverflowFakePolicy:
    """Minimal truncation surface for the overflow-recovery tests: units of `est` tokens
    (oldest -> newest, last active); generate_move fails with `err` until `fails` calls."""

    def __init__(self, fails, err, ests):
        from core.types import Move
        from core.clientutil import zero_usage
        self.fails, self.err, self._ests = fails, err, list(ests)
        self.calls = self.evictions = 0
        self._ok = Move("2", "", False, "tool_calls", zero_usage(), None, 0.0)
        self._fail = Move(None, "", False, None, zero_usage(), None, 0.0, error=err)

    def generate_move(self):
        self.calls += 1
        return self._fail if self.calls <= self.fails else self._ok

    def turns(self):
        from core.types import Turn
        n = len(self._ests)
        return [Turn(payload=k, is_active=(k == n - 1), est_tokens=self._ests[k]) for k in range(n)]

    def evict_oldest_turn(self):
        if len(self._ests) > 1:   # never drop the lone active unit
            self._ests.pop(0)
            self.evictions += 1


def test_overflow_deficit_evict_then_retry_recovers():
    # A context-overflow 4xx is recovered by evicting oldest turn(s) SIZED TO THE
    # DEFICIT the server reported, then retrying — not a fatal api_failure. Realistic failure
    # shape: a 1940-token deficit against ~250-token early turns needs many evictions, which a
    # fixed one-eviction-per-retry loop could never shave. One round must cover it.
    from core.harness import _generate_recovering_overflow

    err = {"message": ("Requested token count exceeds the model's maximum context length of "
                       "262144 tokens. You requested a total of 264084 tokens.")}
    p = OverflowFakePolicy(fails=1, err=err, ests=[250] * 40)
    warns = []
    move = _generate_recovering_overflow(p, SimpleNamespace(warn=warns.append))
    assert move.error is None and move.action == "2"
    # deficit 1940 -> target 1940*1.25+1000=3425 -> 14 evictions of 250 est; ONE retry round.
    assert p.calls == 2 and p.evictions == 14
    assert len(warns) == 1 and "evicted 14 oldest turn(s)" in warns[0]


def test_overflow_unparsed_doubling_fallback_recovers():
    # No counts in the wording -> blind doubling batches (1, 2, 4, ...) per retry round.
    from core.harness import _generate_recovering_overflow

    p = OverflowFakePolicy(fails=3, err={"message": "Error 400: context_length_exceeded"},
                           ests=[100] * 40)
    move = _generate_recovering_overflow(p, SimpleNamespace(warn=lambda m: None))
    assert move.error is None and move.action == "2"
    assert p.calls == 4 and p.evictions == 1 + 2 + 4   # three rounds, doubling


def test_overflow_gives_up_at_eviction_floor():
    # When eviction can't shrink further (only head + active remain), don't loop — surface the
    # error so the caller ends the run instead of spinning.
    from core.harness import _generate_recovering_overflow

    p = OverflowFakePolicy(fails=99, err={"message": "maximum context length exceeded"},
                           ests=[100])   # lone active unit — cannot shrink
    move = _generate_recovering_overflow(p, SimpleNamespace(warn=lambda m: None))
    assert move.error is not None and p.calls == 1   # single attempt, no spin


def test_stale_step_index_stops_server_protocol():
    # A valid move must advance the index by exactly one. A stale echo (server returns the index
    # we already had) would make the harness resend the same step number forever — silent replay.
    class StaleBench(FakeBench):
        def step(self, sid, step_index, action):
            out = super().step(sid, step_index, action)
            return {**out, "step_index": step_index - 1}   # stale: the index before this move

    policy = make_policy([move_response("1"), debrief_response("n/a")])
    stop, records, log = run(policy, StaleBench(st("S0"), [st("S1")]), make_args())
    assert stop == "server_protocol" and "expected 1" in log


def test_invalid_move_advancing_index_stops_server_protocol():
    # The documented invalid-move rule is NO advance (state unchanged). An advance would silently
    # skip a step. The action IS in legal_actions, so it passes the local guard and hits the bench.
    class AdvancingInvalidBench(FakeBench):
        def step(self, sid, step_index, action):
            return {"state": self._cur, "step_index": step_index, "invalid_action": True}

    policy = make_policy([move_response("1"), debrief_response("n/a")])
    stop, records, log = run(policy, AdvancingInvalidBench(st("S0"), [st("S1")]), make_args())
    assert stop == "server_protocol" and "expected 0" in log


def test_missing_step_index_stops_server_protocol():
    class NoIndexBench(FakeBench):
        def step(self, sid, step_index, action):
            out = super().step(sid, step_index, action)
            return {k: v for k, v in out.items() if k != "step_index"}

    policy = make_policy([move_response("1"), debrief_response("n/a")])
    stop, records, log = run(policy, NoIndexBench(st("S0"), [st("S1")]), make_args())
    assert stop == "server_protocol"


# ---- Runner ------------------------------------------------------------


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            import traceback
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
