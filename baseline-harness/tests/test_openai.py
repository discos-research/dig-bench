#!/usr/bin/env python3
"""Isolation tests for clients/openai_client.py — fake OpenAI Responses client, no network.

    python tests/test_openai.py

Covers:
- forced make_move parse from a function_call item; response.output appended VERBATIM;
  reasoning item's encrypted_content -> has_continuity True; request shape (forced tool,
  store=False, include reasoning.encrypted_content, reasoning.effort + summary);
- function_call_output pairing keyed by call_id;
- --no-thought-summaries -> no reasoning.summary; effort "none" -> no summary + no carry;
- accounting: prompt=input_tokens, cached (subset), out=visible (output-reasoning), think=reasoning
  (disjoint, exact), USD, last_prompt_tokens, thoughts_basis="exact";
- 4xx (incl. a reasoning/pairing validation 400) fail-fast + dump; 5xx retried;
- debrief via output_text; truncation surface (head pinned, active reasoning intact, evict oldest unit).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import clients.openai_client as O
from clients.openai_client import OpenAIPolicy


# ---- Fakes -------------------------------------------------------------


def reasoning_item(enc="ENC", summary_text="weigh the options"):
    summary = [SimpleNamespace(type="summary_text", text=summary_text)] if summary_text else []
    return SimpleNamespace(type="reasoning", encrypted_content=enc, summary=summary)


def function_call(action="2", call_id="c1"):
    return SimpleNamespace(type="function_call", name="make_move", call_id=call_id,
                           arguments=json.dumps({"action": action}))


def usage(inp=1000, out=20, reasoning=0, cached=0):
    return SimpleNamespace(
        input_tokens=inp, output_tokens=out,
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def resp(output, *, u=None, status="completed", output_text=""):
    return SimpleNamespace(output=output, usage=u or usage(), status=status, output_text=output_text)


class FakeError(Exception):
    """Mimics openai.APIStatusError: .status_code + .body."""

    def __init__(self, status_code, body=""):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


class FakeResponses:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, script):
        self.responses = FakeResponses(script)


def make_policy(script, *, effort="high", model="gpt-5.5", include_thoughts=True):
    client = FakeClient(script)
    p = OpenAIPolicy(
        api_key="k", model=model, effort=effort, max_tokens=32000, timeout=30, max_retries=3,
        include_thoughts=include_thoughts, client=client,
    )
    p._client = client
    return p


def test_parallel_tool_calls_disabled():
    p = make_policy([resp([function_call("2")])])
    p.start("d", {"observation": "S0"})
    p.generate_move()
    kw = p._client.responses.calls[0]
    assert kw["parallel_tool_calls"] is False   # >1 make_move -> unanswered call_id -> next 400


def test_null_action_parses_to_none():
    p = make_policy([resp([function_call(None)])])   # arguments {"action": null}
    p.start("d", {"observation": "S0"})
    assert p.generate_move().action is None


from core import clientutil as _cu
_cu.time.sleep = lambda *_: None  # retry backoff lives in clientutil


# ---- Tests -------------------------------------------------------------


def test_forced_tool_parse_append_and_carry():
    p = make_policy([resp([reasoning_item("ENC1", "weigh"), function_call("2", "c9")], u=usage(inp=1234))])
    p.start("desc", {"observation": "S0"})
    move = p.generate_move()
    assert move.action == "2" and move.has_continuity is True and move.reasoning_summary == "weigh"
    assert move.thoughts_basis == "exact" and p.last_prompt_tokens == 1234
    # response.output appended VERBATIM (the two items, untouched objects)
    assert [getattr(it, "type", None) for it in p.input[-2:]] == ["reasoning", "function_call"]
    assert p._last_call_id == "c9"
    # request shape
    kw = p._client.responses.calls[0]
    assert kw["store"] is False and kw["include"] == ["reasoning.encrypted_content"]
    assert kw["tool_choice"] == {"type": "function", "name": "make_move"}
    assert kw["reasoning"] == {"effort": "high", "summary": "auto"}
    assert kw["instructions"] and kw["max_output_tokens"] == 32000
    # observe -> function_call_output keyed by call_id
    p.observe({"observation": "S1"})
    fco = p.input[-1]
    assert fco == {"type": "function_call_output", "call_id": "c9", "output": json.dumps({"result": {"observation": "S1"}})}


def test_no_thought_summaries_omits_summary():
    p = make_policy([resp([reasoning_item(), function_call("1")])], include_thoughts=False)
    p.start("d", {"observation": "S0"})
    p.generate_move()
    assert "summary" not in p._client.responses.calls[0]["reasoning"]


def test_effort_none_no_summary_and_no_carry():
    # effort none -> reasoning {"effort":"none"} (no summary); model returns no reasoning item
    p = make_policy([resp([function_call("1", "c1")])], effort="none")
    p.start("d", {"observation": "S0"})
    move = p.generate_move()
    assert move.action == "1" and move.has_continuity is False
    assert p._client.responses.calls[0]["reasoning"] == {"effort": "none"}


def test_accounting_disjoint_exact_and_cost():
    # output_tokens=2200 INCLUDES reasoning=2000 -> out=visible 200, think 2000 (disjoint, exact);
    # cached (200) is a SUBSET of input_tokens (1000).
    p = make_policy([resp([reasoning_item(), function_call("1")], u=usage(inp=1000, out=2200, reasoning=2000, cached=200))])
    p.start("d", {"observation": "S0"})
    move = p.generate_move()
    assert move.usage == {"prompt": 1000, "cached": 200, "output": 200, "thoughts": 2000, "total": 3200}
    assert p.prompt_tokens == 1000 and p.cached_tokens == 200 and p.output_tokens == 200 and p.thoughts_tokens == 2000
    import core.accounting as acc
    expected = acc.compute_cost(p.model, 1000, 200, 2200, 0, p.pricing)  # full output at output rate
    assert abs(p.cost_usd - expected) < 1e-9


def test_reasoning_400_fails_fast_and_dumps():
    # A reasoning/pairing validation 400 FAILS FAST + dumps — no retry, no history rewriting;
    # a clean stop beats a degraded one.
    d = tempfile.mkdtemp()
    err = FakeError(400, "Item 'rs_x' of type 'reasoning' was provided without its required following item")
    p = make_policy([err])
    p.debug_dir = d
    p.start("desc", {"observation": "S0"})
    move = p.generate_move()
    assert move.action is None and move.error is not None
    assert len(p._client.responses.calls) == 1  # failed fast, no retry
    assert [f for f in os.listdir(d) if f.startswith("openai_4xx_")]


def test_non_continuity_4xx_failfast_and_dump():
    d = tempfile.mkdtemp()
    p = make_policy([FakeError(400, "unsupported parameter: foo")])
    p.debug_dir = d
    p.start("d", {"observation": "S0"})
    move = p.generate_move()
    assert move.action is None and move.error is not None
    assert [f for f in os.listdir(d) if f.startswith("openai_4xx_")]


def test_5xx_retried_then_success():
    p = make_policy([FakeError(503, "overloaded"), resp([reasoning_item(), function_call("1")])])
    p.start("d", {"observation": "S0"})
    move = p.generate_move()
    assert move.action == "1" and len(p._client.responses.calls) == 2


def test_debrief_uses_output_text():
    p = make_policy([resp([reasoning_item(), function_call("1", "cA")]),
                     resp([], output_text="Discovered the mechanics.")])
    p.start("d", {"observation": "S0"})
    p.generate_move()
    p.observe({"observation": "S1"})
    assert p.debrief() == "Discovered the mechanics."


def test_gpt_5_4_mini_window_distinct_from_gpt_5_4():
    # gpt-5.4-mini (400K) must NOT inherit gpt-5.4's 1.05M window via longest-substring
    # matching — it needs its own row (windows per developers.openai.com).
    import core.accounting as acc
    assert acc.match_model("gpt-5.4-mini", O.OPENAI_MAX_CONTEXT) == 400_000
    assert acc.match_model("gpt-5.4-mini-2026-01-01", O.OPENAI_MAX_CONTEXT) == 400_000
    assert acc.match_model("gpt-5.4", O.OPENAI_MAX_CONTEXT) == 1_050_000  # non-mini stays 1.05M


def test_truncation_surface_head_pinned_active_intact():
    p = make_policy([resp([reasoning_item("E1"), function_call("1", "c1")]),
                     resp([reasoning_item("E2"), function_call("2", "c2")])])
    p.start("desc", {"observation": "S0"})        # input[0] = user task (head)
    p.generate_move(); p.observe({"observation": "S1"})
    p.generate_move(); p.observe({"observation": "S2"})
    units = p.turns()
    assert len(units) == 2 and units[0].is_active is False and units[1].is_active is True
    assert p._turn_start_indices() == [1, 4]       # head = input[0:1]
    p.evict_oldest_turn()
    assert len(p.turns()) == 1
    assert p.input[0]["content"].startswith("You are now playing")  # head pinned
    active = p.turns()[-1].payload
    assert any(getattr(it, "type", None) == "reasoning" and getattr(it, "encrypted_content", None)
               for it in active)                     # active reasoning + encrypted_content intact
    p.evict_oldest_turn()                            # no-op at floor
    assert len(p.turns()) == 1


def test_incomplete_cutoff_strips_orphan_reasoning():
    # An `incomplete` cutoff returns a reasoning item but NO function_call -> action
    # None, call_id None. The trailing orphan reasoning MUST be stripped from self.input, else the
    # next store=false request 400s ("reasoning item ... without its required following item").
    p = make_policy([resp([reasoning_item("ENC", "thinking...")], status="incomplete")])
    p.start("desc", {"observation": "s0"})
    move = p.generate_move()
    assert move.action is None and p._last_call_id is None
    assert not any(getattr(it, "type", None) == "reasoning" for it in p.input)   # orphan stripped
    assert all(isinstance(it, dict) for it in p.input)   # only the pinned user head remains -> valid


def test_blank_action_answers_pending_call_on_nudge():
    # A function_call with an empty action -> action None but call_id IS set. The
    # nudge must ANSWER the pending call (function_call_output), not leave it dangling (-> 400).
    p = make_policy([resp([reasoning_item(), function_call(action="  ", call_id="c5")])])
    p.start("desc", {"observation": "s0"})
    move = p.generate_move()
    assert move.action is None and p._last_call_id == "c5"
    p.add_nudge("please call make_move")
    last = p.input[-1]
    assert isinstance(last, dict) and last["type"] == "function_call_output" and last["call_id"] == "c5"
    assert p._last_call_id is None   # cleared, so the call is answered exactly once


# ---- Runner ------------------------------------------------------------


def test_bedrock_model_id_window_and_pricing():
    # Bedrock ids (openai. prefix) get the 272K Bedrock cap, not the 1.05M direct row,
    # and still find the pricing row via substring matching.
    p = make_policy([], model="openai.gpt-5.5")
    assert p.model_max_context == 272_000
    assert p.pricing_row == O.OPENAI_PRICING["gpt-5.5"]


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
