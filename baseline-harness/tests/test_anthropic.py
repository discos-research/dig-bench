#!/usr/bin/env python3
"""Isolation tests for clients/anthropic_client.py — fake AnthropicBedrock client, no network.

    python tests/test_anthropic.py

Covers:
- thinking-ON (auto-tool): action parsed from the make_move tool_use block; the
  response content appended VERBATIM; thinking-block signature -> has_continuity True;
  request carries thinking:{adaptive}, output_config.effort, tool_choice:{auto};
  --no-thought-summaries -> display:omitted;
- redacted_thinking also counts as carried;
- manual-thinking models (Haiku 4.5 / Sonnet 4.5): thinking:{enabled, budget_tokens} from the
  shared level table, no output_config (it 400s there), budget clamped to max_tokens - 1024;
- thinking-NONE (ablation): thinking:{disabled} (explicit — Opus 5 thinks when it's omitted)
  on BOTH the turn and the tool-free debrief, tool_choice forced to make_move,
  has_continuity False; on Fable 5, where `disabled` 400s, `thinking` is omitted instead;
- Opus 5 resolves its own pricing + 1M window (a "claude-opus-4" row would not cover it);
- no tool_use this turn -> action None (harness nudges), thinking still carried;
- accounting: prompt = input + cache_read (+cache_create), cached, output, USD via table,
  last_prompt_tokens; thoughts == 0 (folded into output);
- 4xx (incl. a thinking-block validation 400) fail-fast + dump; 5xx retried;
- debrief merges its prompt into a trailing tool_result turn (no two consecutive users);
- truncation surface: head (user task) pinned, units = assistant+tool_result, latest
  active with its thinking intact, evict_oldest drops the oldest unit.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import clients.anthropic_client as A
from clients.anthropic_client import AnthropicPolicy


# ---- Fakes -------------------------------------------------------------


def thinking_block(text="weigh the options", sig="SIG"):
    return SimpleNamespace(type="thinking", thinking=text, signature=sig)


def tool_block(action="2", id="tu_1"):
    return SimpleNamespace(type="tool_use", name="make_move", id=id, input={"action": action})


def text_block(t):
    return SimpleNamespace(type="text", text=t)


def redacted_block():
    return SimpleNamespace(type="redacted_thinking", data="opaque")


def usage(inp=1000, out=20, cache_read=0, cache_create=0, thinking=0):
    # Real Anthropic returns output_tokens INCLUDING thinking, with the breakout under
    # output_tokens_details.thinking_tokens. thinking=0 -> details absent (older/None path).
    details = SimpleNamespace(thinking_tokens=thinking) if thinking else None
    return SimpleNamespace(
        input_tokens=inp, output_tokens=out,
        cache_read_input_tokens=cache_read, cache_creation_input_tokens=cache_create,
        output_tokens_details=details,
    )


def resp(content, *, u=None, stop="tool_use"):
    return SimpleNamespace(content=content, usage=u or usage(), stop_reason=stop)


class FakeError(Exception):
    """Mimics anthropic.APIStatusError: exposes .status_code and .body."""

    def __init__(self, status_code, body=""):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


class FakeMessages:
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
        self.messages = FakeMessages(script)


def make_policy(script, *, thinking_level="high", model="global.anthropic.claude-sonnet-4-6",
                include_thoughts=True):
    client = FakeClient(script)
    p = AnthropicPolicy(
        api_key="k", model=model, aws_region="us-east-1", thinking_level=thinking_level,
        max_tokens=32000, timeout=30, max_retries=3, include_thoughts=include_thoughts, client=client,
    )
    p._client = client  # test handle
    return p


from core import clientutil as _cu
_cu.time.sleep = lambda *_: None  # retry backoff lives in clientutil


# ---- Tests -------------------------------------------------------------


def test_autotool_parse_append_and_carry():
    p = make_policy([resp([thinking_block("weigh", "SIGA"), tool_block("2", "tu9")], u=usage(inp=1234))])
    p.start("desc", {"observation": "S0"})
    move = p.generate_move()
    assert move.action == "2" and move.has_continuity is True
    assert move.reasoning_summary == "weigh"
    assert p.last_prompt_tokens == 1234 and p.thoughts_tokens == 0 and p.cost_usd > 0  # sonnet priced
    # assistant appended VERBATIM (the response block objects, untouched)
    asst = p.messages[-1]
    assert asst["role"] == "assistant"
    assert [getattr(b, "type", None) for b in asst["content"]] == ["thinking", "tool_use"]
    assert p._last_tool_use_id == "tu9"
    # request shape: adaptive thinking + effort + auto tool choice
    kw = p._client.messages.calls[0]
    assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kw["output_config"] == {"effort": "high"}
    assert kw["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    # observe answers the call by id, as a tool_result block
    p.observe({"observation": "S1"})
    tr = p.messages[-1]
    assert tr["role"] == "user" and tr["content"][0]["type"] == "tool_result"
    assert tr["content"][0]["tool_use_id"] == "tu9"


def test_no_thought_summaries_sets_display_omitted():
    p = make_policy([resp([tool_block("1")])], include_thoughts=False)
    p.start("d", {"observation": "S0"})
    p.generate_move()
    assert p._client.messages.calls[0]["thinking"]["display"] == "omitted"


def test_minimal_maps_to_low_effort():
    p = make_policy([resp([tool_block("1")])], thinking_level="minimal")
    p.start("d", {"observation": "S0"})
    p.generate_move()
    assert p._client.messages.calls[0]["output_config"] == {"effort": "low"}


def test_redacted_thinking_counts_as_carry():
    p = make_policy([resp([redacted_block(), tool_block("1")])])
    p.start("d", {"observation": "S0"})
    move = p.generate_move()
    assert move.action == "1" and move.has_continuity is True and move.reasoning_summary == ""


def test_thinking_none_forces_tool_and_no_carry():
    p = make_policy([resp([tool_block("1", "tu1")], stop="tool_use")], thinking_level=None)
    assert p.thinking_on is False and p.move_channel == "forced-tool"
    p.start("d", {"observation": "S0"})
    move = p.generate_move()
    assert move.action == "1" and move.has_continuity is False
    assert p.thoughts_basis == "none"  # no thinking -> nothing to count
    kw = p._client.messages.calls[0]
    # `disabled` is EXPLICIT: on Opus 5 / Sonnet 5 an omitted `thinking` means adaptive, so the
    # ablation would otherwise depend on forced tool_choice implicitly suppressing thinking.
    # No effort rides along — on Opus 5 `disabled` is only legal at effort <= high.
    assert kw["thinking"] == {"type": "disabled"} and "output_config" not in kw
    assert kw["tool_choice"] == {"type": "tool", "name": "make_move", "disable_parallel_tool_use": True}


def test_thinking_none_debrief_also_disables_thinking():
    # The debrief drops `tools`, so there is no forced tool_choice to suppress thinking: without an
    # explicit `disabled` an Opus-5 "none" run would silently think in its debrief.
    p = make_policy([resp([tool_block("1", "tu1")]), resp([text_block("played greedily")], stop="end_turn")],
                    thinking_level=None, model="global.anthropic.claude-opus-5")
    p.start("d", {"observation": "S0"})
    p.generate_move()
    assert p.debrief() == "played greedily"
    kw = p._client.messages.calls[-1]
    assert kw["thinking"] == {"type": "disabled"}
    assert "tools" not in kw and "output_config" not in kw


def test_fable_thinking_none_omits_thinking_because_disabled_400s():
    # Fable 5 rejects thinking:{"type":"disabled"} outright, so the OFF branch must keep
    # omitting `thinking` there and rely on the forced tool to suppress it.
    model = "global.anthropic.claude-fable-5"
    p = make_policy([resp([tool_block("1", "tu1")]), resp([text_block("done")], stop="end_turn")],
                    thinking_level=None, model=model)
    assert p.can_disable_thinking is False
    p.start("d", {"observation": "S0"})
    p.generate_move()
    p.debrief()
    for kw in p._client.messages.calls:
        assert "thinking" not in kw, model
        assert p._client.messages.calls[0]["tool_choice"]["type"] == "tool"


def test_thinking_on_unaffected_by_the_always_on_gate():
    # The gate only touches the OFF branch: Fable with thinking on still gets adaptive + effort.
    p = make_policy([resp([thinking_block(), tool_block("1")])], model="global.anthropic.claude-fable-5")
    p.start("d", {"observation": "S0"})
    p.generate_move()
    kw = p._client.messages.calls[0]
    assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kw["output_config"] == {"effort": "high"}


def test_haiku_manual_thinking_budget_no_output_config():
    # Pre-effort models (Haiku 4.5 / Sonnet 4.5) 400 on output_config.effort; they take
    # MANUAL extended thinking with a budget from the SHARED
    # level table (same nominal budget per level on every budget-based family).
    p = make_policy([resp([thinking_block(), tool_block("1")]),
                     resp([text_block("done")], stop="end_turn")],
                    model="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert p.manual_thinking is True
    p.start("d", {"observation": "S0"})
    p.generate_move()
    kw = p._client.messages.calls[0]
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 16384}  # high -> 16384
    assert "output_config" not in kw
    assert kw["tool_choice"]["type"] == "auto"   # thinking still forbids forced tool_choice
    p.debrief()                                   # the tool-free debrief takes the same regime
    kw = p._client.messages.calls[-1]
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert "output_config" not in kw and "tools" not in kw


def test_manual_thinking_budget_clamped_to_max_tokens():
    # 1024 <= budget <= max_tokens - 1024 (thinking spends from the same max_tokens allowance).
    client = FakeClient([resp([tool_block("1")])])
    p = AnthropicPolicy(api_key="k", model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        aws_region="us-east-1", thinking_level="max", max_tokens=2048,
                        timeout=30, max_retries=3, client=client)
    assert p.thinking_budget == 1024  # min(32768, 2048-1024); recorded for provenance
    p.start("d", {"observation": "S0"})
    p.generate_move()
    assert client.messages.calls[0]["thinking"]["budget_tokens"] == 1024


def test_manual_thinking_infeasible_max_tokens_fails_loud():
    # max_tokens=1024 leaves NO room: bumping to the 1024 floor would send budget == max_tokens
    # (the API requires budget < max_tokens) and zero visible output. Refuse at construction.
    import pytest
    with pytest.raises(SystemExit, match="raise --max-tokens"):
        AnthropicPolicy(api_key="k", model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        aws_region="us-east-1", thinking_level="low", max_tokens=1024,
                        timeout=30, max_retries=3, client=FakeClient([]))


def test_manual_thinking_without_breakout_reports_included():
    # Bedrock manual-thinking models (Haiku 4.5) return NO
    # output_tokens_details even when the turn visibly thought: `out` keeps thinking folded in
    # and the basis demotes to "included" (rendered `out + think N`) — never a fake exact-0 split.
    p = make_policy([resp([thinking_block("hmm"), tool_block("1")], u=usage(inp=100, out=300))],
                    model="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    p.start("d", {"observation": "S0"})
    move = p.generate_move()
    assert move.usage["output"] == 300 and move.usage["thoughts"] == 0
    assert move.thoughts_basis == "included" and p.thoughts_basis == "included"
    # a later breakout-less turn with NO thinking stays exact-0 per turn, but the run-level
    # basis remains the weaker "included" claim (sticky)
    p._client.messages.script.append(resp([tool_block("2")], u=usage(inp=100, out=50)))
    p.observe({"observation": "S1"})
    move = p.generate_move()
    assert move.thoughts_basis == "exact" and p.thoughts_basis == "included"


def test_manual_thinking_model_none_still_disables():
    # thinking-OFF on a manual-thinking model is unchanged: explicit disabled + forced tool.
    p = make_policy([resp([tool_block("1", "t")])], thinking_level=None,
                    model="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    p.start("d", {"observation": "S0"})
    p.generate_move()
    kw = p._client.messages.calls[0]
    assert kw["thinking"] == {"type": "disabled"} and kw["tool_choice"]["type"] == "tool"


def test_opus_5_is_priced_and_windowed():
    # "claude-opus-4" is NOT a substring of a 5 id — without its own rows Opus 5 would run unpriced
    # with no context window. Rates match Opus 4.8 ($5 / $25 per 1M).
    p = make_policy([resp([thinking_block(), tool_block("1")], u=usage(inp=1_000_000, out=0))],
                    model="global.anthropic.claude-opus-5")
    assert p.has_pricing is True and p.model_max_context == 1_000_000
    p.start("d", {"observation": "S0"})
    p.generate_move()
    assert abs(p.cost_usd - 5.0) < 1e-6
    assert A.ANTHROPIC_BEDROCK_PRICING["claude-opus-5"]["output_per_1m"] == 25.0


def test_no_tool_call_yields_no_action_but_carries():
    # thinking + text, but no make_move tool_use -> action None (harness nudges)
    p = make_policy([resp([thinking_block(), text_block("not sure yet")], stop="end_turn")])
    p.start("d", {"observation": "S0"})
    move = p.generate_move()
    assert move.action is None and move.has_continuity is True and p._last_tool_use_id is None
    assert p.messages[-1]["role"] == "assistant"  # appended verbatim regardless


def test_accounting_cache_and_total():
    p = make_policy([resp([tool_block("1")], u=usage(inp=1000, out=50, cache_read=200))])
    p.start("d", {"observation": "S0"})
    move = p.generate_move()
    assert move.usage == {"prompt": 1200, "cached": 200, "output": 50, "thoughts": 0, "total": 1250}
    assert p.prompt_tokens == 1200 and p.cached_tokens == 200 and p.output_tokens == 50
    assert p.last_prompt_tokens == 1200 and p.thoughts_basis == "exact"


def test_thinking_tokens_split_disjoint_exact_and_cost_unchanged():
    # output_tokens=2200 INCLUDES thinking_tokens=2000 -> report out=visible 200, think 2000
    # (disjoint, Gemini convention); total = prompt + full output; cost == pricing the full
    # output at the output rate (no double count).
    p = make_policy([resp([thinking_block(), tool_block("1")], u=usage(inp=1000, out=2200, thinking=2000))])
    p.start("d", {"observation": "S0"})
    move = p.generate_move()
    assert move.usage["output"] == 200 and move.usage["thoughts"] == 2000
    assert move.usage["total"] == 1000 + 2200
    assert p.output_tokens == 200 and p.thoughts_tokens == 2000 and p.thoughts_basis == "exact"
    assert move.thoughts_basis == "exact"  # per-turn marker carrier (Anthropic is exact, never (est))
    import core.accounting as acc
    expected = acc.compute_cost(p.model, 1000, 0, 2200, 0, p.pricing)  # full output as one output block
    assert abs(p.cost_usd - expected) < 1e-9


def test_continuity_400_fails_fast_and_dumps():
    # A thinking-block validation 400 FAILS FAST + dumps — no retry, no history rewriting;
    # a clean stop beats a degraded one.
    d = tempfile.mkdtemp()
    err = FakeError(400, "`thinking` blocks in the latest assistant message cannot be modified")
    p = make_policy([err])
    p.debug_dir = d
    p.start("desc", {"observation": "S0"})
    move = p.generate_move()
    assert move.action is None and move.error is not None
    assert len(p._client.messages.calls) == 1  # failed fast, no retry
    assert [f for f in os.listdir(d) if f.startswith("anthropic_4xx_")]


def test_non_continuity_4xx_failfast_and_dump():
    d = tempfile.mkdtemp()
    p = make_policy([FakeError(422, "unprocessable: bad schema")])
    p.debug_dir = d
    p.start("d", {"observation": "S0"})
    move = p.generate_move()
    assert move.action is None and move.error is not None
    assert [f for f in os.listdir(d) if f.startswith("anthropic_4xx_")]


def test_5xx_retried_then_success():
    p = make_policy([FakeError(503, "overloaded"), resp([tool_block("1")])])
    p.start("d", {"observation": "S0"})
    move = p.generate_move()
    assert move.action == "1" and len(p._client.messages.calls) == 2


def test_null_action_parses_to_none():
    # {"action": null} must become None (harness nudges), not the literal string "None".
    p = make_policy([resp([tool_block(None, "tu1")])])
    p.start("d", {"observation": "S0"})
    assert p.generate_move().action is None


def test_debrief_answers_pending_tool_use_without_cache_write():
    # bench_failure/server_protocol break BEFORE observe() -> a make_move tool_use is still
    # pending; debrief must ANSWER it (add_nudge), not append bare user text (which 400s).
    p = make_policy([resp([tool_block("1", "tu1")]),
                     resp([text_block("Mechanics: push the block.")], stop="end_turn")])
    p.start("d", {"observation": "S0"})
    p.generate_move()                          # pending tool_use tu1; NO observe (simulated failure)
    assert p._last_tool_use_id == "tu1"
    assert p.debrief() == "Mechanics: push the block."
    assert p._last_tool_use_id is None         # the pending call was answered, not left dangling
    answer = p.messages[-1]
    assert answer["role"] == "user" and answer["content"][0]["type"] == "tool_result"
    assert answer["content"][0]["tool_use_id"] == "tu1"
    # the debrief request writes no cache (byte-0 divergence -> zero reads) and reuses messages
    kw = p._client.messages.calls[-1]
    assert "tools" not in kw and "cache_control" not in kw["system"][0]
    assert kw["messages"] is p.messages


def test_debrief_merges_into_tool_result_turn():
    p = make_policy([resp([thinking_block(), tool_block("1", "tuA")]),
                     resp([text_block("Discovered the mechanics.")], stop="end_turn")])
    p.start("d", {"observation": "S0"})
    p.generate_move()
    p.observe({"observation": "S1"})            # messages[-1] = user tool_result turn
    n_before = len(p.messages)
    text = p.debrief()
    assert text == "Discovered the mechanics."
    # the debrief prompt merged into the trailing user turn — no new user message
    assert len(p.messages) == n_before
    last = p.messages[-1]
    assert last["role"] == "user" and last["content"][-1] == {"type": "text", "text": A.prompts.DEBRIEF_PROMPT}


def test_truncation_surface_head_pinned_active_intact():
    p = make_policy([resp([thinking_block("r1", "SIG1"), tool_block("1", "c1")]),
                     resp([thinking_block("r2", "SIG2"), tool_block("2", "c2")])])
    p.start("desc", {"observation": "S0"})        # messages[0] = user task (head)
    p.generate_move(); p.observe({"observation": "S1"})
    p.generate_move(); p.observe({"observation": "S2"})
    units = p.turns()
    assert len(units) == 2 and units[0].is_active is False and units[1].is_active is True
    assert p._assistant_indices() == [1, 3]       # head = messages[0:1]
    p.evict_oldest_turn()
    assert len(p.turns()) == 1
    assert p.messages[0]["content"].startswith("You are now playing")  # head pinned
    active = p.turns()[-1].payload
    asst = next(m for m in active if m.get("role") == "assistant")
    assert any(getattr(b, "type", None) == "thinking" and getattr(b, "signature", None)
               for b in asst["content"])           # active reasoning + signature intact
    p.evict_oldest_turn()                            # no-op at floor
    assert len(p.turns()) == 1


def test_blank_action_answers_pending_tool_use_on_nudge():
    # A tool_use make_move with an empty action -> action None but tool_use_id IS
    # set. The nudge must ANSWER it with a tool_result, not leave a dangling tool_use (-> 400).
    p = make_policy([resp([thinking_block(), tool_block(action="  ", id="tu7")])])
    p.start("desc", {"observation": "s0"})
    move = p.generate_move()
    assert move.action is None and p._last_tool_use_id == "tu7"
    p.add_nudge("please call make_move")
    last = p.messages[-1]
    assert last["role"] == "user" and isinstance(last["content"], list)
    block = last["content"][0]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "tu7"
    assert p._last_tool_use_id is None   # answered exactly once


def test_prompt_caching_breakpoints_and_no_mutation():
    # Bedrock caches only with explicit cache_control: system + tools (static) + a rolling breakpoint
    # on the last message, so the append-only prefix is cached. Must be <=4 breakpoints and must NOT
    # leak into the stored history (the verbatim reasoning-carry blocks).
    p = make_policy([resp([thinking_block(), tool_block("1", "tu1")])])
    p.start("desc", {"observation": "S0"})
    p.generate_move()
    p.observe({"observation": "S1"})              # last stored message is now a tool_result (list)
    kw = p._turn_kwargs()
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}                    # system cached
    assert kw["tools"][0]["cache_control"] == {"type": "ephemeral"}                     # tools cached
    assert kw["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}  # rolling breakpoint
    bps = (sum("cache_control" in b for b in kw["system"])
           + sum("cache_control" in t for t in kw["tools"])
           + sum(isinstance(b, dict) and "cache_control" in b
                 for m in kw["messages"] if isinstance(m.get("content"), list) for b in m["content"]))
    assert bps == 3 and bps <= 4
    assert "cache_control" not in p.messages[-1]["content"][-1]   # stored history untouched


def test_cache_write_premium_added_to_cost():
    # A 5-min cache WRITE bills at 1.25x input; compute_cost charges 1x, so _account adds the 0.25x.
    p = make_policy([resp([tool_block("1")], u=usage(inp=100, cache_create=1000))])
    p.start("d", {"observation": "S0"})
    move = p.generate_move()
    import core.accounting as acc
    base = acc.compute_cost(p.model, 1100, 0, move.usage["output"], 0, p.pricing)  # all input at 1x
    premium = 1000 * 0.25 * p.pricing_row["input_per_1m"] / 1_000_000
    assert abs(p.cost_usd - (base + premium)) < 1e-9
    assert move.usage["prompt"] == 1100 and move.usage["cached"] == 0


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
