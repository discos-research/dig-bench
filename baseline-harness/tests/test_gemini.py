#!/usr/bin/env python3
"""Isolation tests for clients/gemini_client.py — fake genai client, no network.

    python tests/test_gemini.py

Covers move parsing / reasoning carry / accounting + the truncation surface:
- action parsed from the forced make_move call; no call -> action None (nudge path);
- has_continuity is honest: True iff a signature-bearing part is present;
- the model Content is appended VERBATIM (object identity preserved) and the
  signature survives an append -> turns() -> payload round-trip;
- accounting accumulates and last_prompt_tokens tracks the server prompt size;
- turns() segments into step-pairs (head pinned, latest = active);
- evict_oldest_turn drops the oldest pair, keeps the head + the active signature;
- error handling (shared clientutil): deterministic 4xx fail fast + dump; transient retries;
- the debrief fires tool-free over the function-calling history.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
pytest.importorskip("google.genai")  # GeminiPolicy builds types.* configs at construction; skip w/o SDK

from clients.gemini_client import GeminiPolicy
from core import clientutil as _cu

_cu.time.sleep = lambda *_: None  # no real backoff in retry tests


class FakeAPIError(Exception):
    """Mimics google-genai APIError: `.code` is the int HTTP status."""

    def __init__(self, code, msg=""):
        self.code = code
        super().__init__(f"{code}: {msg}")


# ---- Fakes ----------------------------------------------------------


class FakeUsage:
    def __init__(self, prompt=1000, cached=800, output=20, thoughts=100, total=1120):
        self.prompt_token_count = prompt
        self.cached_content_token_count = cached
        self.candidates_token_count = output
        self.thoughts_token_count = thoughts
        self.total_token_count = total


class FakePart:
    def __init__(self, *, text=None, thought=False, thought_signature=None, function_response=None):
        self.text = text
        self.thought = thought
        self.thought_signature = thought_signature
        self.function_response = function_response


class FakeContent:
    def __init__(self, parts):
        self.parts = parts
        self.role = "model"


class FakeCall:
    def __init__(self, args):
        self.name = "make_move"
        self.args = args


class FakeCandidate:
    def __init__(self, content, finish="STOP"):
        self.content = content
        self.finish_reason = finish


class FakeResponse:
    def __init__(self, *, content, function_calls, usage, text=""):
        self.candidates = [FakeCandidate(content)]
        self.function_calls = function_calls
        self.usage_metadata = usage
        self.text = text


def move_response(action, *, signature=b"SIG", summary="I will try this.", usage=None):
    content = FakeContent([
        FakePart(text=summary, thought=True),
        FakePart(thought_signature=signature),
    ])
    return FakeResponse(content=content, function_calls=[FakeCall({"action": action})], usage=usage or FakeUsage())


def nomove_response(usage=None):
    content = FakeContent([FakePart(text="hmm", thought=True)])
    return FakeResponse(content=content, function_calls=[], usage=usage or FakeUsage())


def debrief_response(text="Mechanics: move toward the goal.", usage=None):
    content = FakeContent([FakePart(text=text)])
    return FakeResponse(content=content, function_calls=[], usage=usage or FakeUsage(), text=text)


class FakeModels:
    """Pops the next scripted item per generate_content; an Exception item is raised."""

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append((list(contents), config))
        item = self.scripts.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, scripts):
        self.models = FakeModels(scripts)


def make_policy(scripts, **over):
    # no `pricing` passed -> the client defaults to its own GEMINI_PRICING table
    client = FakeClient(scripts)
    kw = dict(api_key="unused", model="gemini-3.1-pro-preview", thinking_level="high",
              timeout=120, max_retries=3)
    kw.update(over)
    return GeminiPolicy(client=client, **kw)


# ---- Tests -------------------------------------------------------------


def test_action_parsed_and_continuity_honest():
    p = make_policy([move_response("2", signature=b"SIG-A")])
    p.start("desc", {"observation": "S0", "legal_actions": ["1", "2"]})
    move = p.generate_move()
    assert move.action == "2"
    assert move.has_continuity is True
    assert move.finish_reason == "STOP"


def test_no_call_gives_none_action():
    p = make_policy([nomove_response()])
    p.start("desc", {"observation": "S0"})
    move = p.generate_move()
    assert move.action is None and move.has_continuity is False


def test_thinking_none_disables_thinking_budget_on_flash_only():
    # flash-class supports disabling -> thinking_budget=0 (actually off, not just summaries hidden).
    pf = make_policy([], thinking_level=None, model="gemini-2.5-flash")
    assert pf._thinking_config(include_thoughts=True).thinking_budget == 0
    # Pro models reject budget 0 ("only works in thinking mode"); silently running them at default
    # thinking would mislabel the run as "thinking none" -> refuse at construction, fail loud.
    with pytest.raises(SystemExit, match="cannot be disabled"):
        make_policy([], thinking_level=None, model="gemini-3.1-pro-preview")


def test_gemini_25_maps_levels_to_thinking_budget():
    # 2.5-family models reject thinking_level (a Gemini-3 parameter); the level maps through the
    # SHARED clientutil.THINKING_LEVEL_BUDGETS table onto thinking_budget instead (same nominal
    # budget per level as Anthropic manual thinking).
    tc = make_policy([], model="gemini-2.5-flash", thinking_level="high")._turn_config.thinking_config
    assert tc.thinking_budget == 16384 and tc.thinking_level is None
    # pro at max: min(table 32768, pro cap 32768, max_tokens 32000 - 1024) = 30976
    tc = make_policy([], model="gemini-2.5-pro", thinking_level="max")._turn_config.thinking_config
    assert tc.thinking_budget == 30976 and tc.thinking_level is None


def test_gemini_25_budget_clamped_per_family():
    # flash cap 24576 binds at level max; the resolved budget is recorded for provenance.
    p = make_policy([], model="gemini-2.5-flash", thinking_level="max")
    assert p._turn_config.thinking_config.thinking_budget == 24576
    assert p.thinking_budget == 24576


def test_gemini_25_infeasible_max_tokens_fails_loud():
    # The floor GATES, it never bumps up: max_tokens=1100 leaves only 76 tokens of budget on
    # pro (floor 128), and on flash a resolved budget of 0 would silently DISABLE thinking
    # while the run stays labeled with a level. Both refuse at construction.
    with pytest.raises(SystemExit, match="raise --max-tokens"):
        make_policy([], model="gemini-2.5-pro", thinking_level="minimal", max_tokens=1100)
    with pytest.raises(SystemExit, match="raise --max-tokens"):
        make_policy([], model="gemini-2.5-flash", thinking_level="high", max_tokens=1024)


def test_gemini_3_still_uses_thinking_level():
    tc = make_policy([], model="gemini-3.1-pro-preview", thinking_level="high")._turn_config.thinking_config
    assert tc.thinking_level is not None and tc.thinking_budget is None


def test_null_action_parses_to_none():
    r = FakeResponse(content=FakeContent([FakePart(text="x", thought=True)]),
                     function_calls=[FakeCall({"action": None})], usage=FakeUsage())
    p = make_policy([r])
    p.start("desc", {"observation": "S0"})
    assert p.generate_move().action is None


def test_parallel_calls_all_answered():
    # Two make_move calls in one turn -> observe emits TWO functionResponses (Gemini needs N
    # responses for N calls; a dangling call 400s the next request).
    r = FakeResponse(content=FakeContent([FakePart(text="go", thought=True)]),
                     function_calls=[FakeCall({"action": "1"}), FakeCall({"action": "2"})],
                     usage=FakeUsage())
    p = make_policy([r])
    p.start("desc", {"observation": "S0"})
    move = p.generate_move()
    assert move.action == "1" and p._pending_call_count == 2
    p.observe({"observation": "S1"})
    assert len(p.contents[-1].parts) == 2


def test_model_content_appended_verbatim_and_signature_survives():
    p = make_policy([move_response("1", signature=b"SIG-XYZ")])
    p.start("desc", {"observation": "S0"})
    appended_before = len(p.contents)
    move = p.generate_move()
    assert len(p.contents) == appended_before + 1
    # the exact FakeContent object the response carried is appended (no reconstruction)
    last = p.contents[-1]
    assert isinstance(last, FakeContent)
    # signature survives append -> turns() -> payload: same object reference
    active_payload = p.turns()[-1].payload
    sig_parts = [pt for c in active_payload for pt in c.parts if pt.thought_signature]
    assert sig_parts and sig_parts[0].thought_signature == b"SIG-XYZ"
    assert move.has_continuity is True


def test_accounting_and_last_prompt_tokens():
    p = make_policy([move_response("1", usage=FakeUsage(prompt=1234, cached=800, output=20, thoughts=100))])
    p.start("desc", {"observation": "S0"})
    p.generate_move()
    assert p.call_count == 1
    assert p.prompt_tokens == 1234 and p.cached_tokens == 800
    assert p.last_prompt_tokens == 1234        # the truncation trigger
    assert p.cost_usd > 0 and p.has_pricing
    # the client owns its pricing row + context window (not core/accounting)
    assert p.pricing_row["output_per_1m"] == 12.00
    assert p.model_max_context == 1_048_576


def test_turns_segmentation_head_pinned_latest_active():
    p = make_policy([move_response("1", signature=b"S1"), move_response("2", signature=b"S2")])
    p.start("desc", {"observation": "S0", "legal_actions": ["1", "2"]})   # head (user)
    p.generate_move()                                                     # model 1
    p.observe({"observation": "S1", "legal_actions": ["1", "2"]})          # funcResponse 1
    p.generate_move()                                                     # model 2
    p.observe({"observation": "S2", "legal_actions": ["1", "2"]})          # funcResponse 2
    units = p.turns()
    assert len(units) == 2                       # two step-pairs; the head is NOT a unit
    assert units[0].is_active is False and units[1].is_active is True
    assert units[0].est_tokens > 0
    # head is everything before the first model output
    assert p._model_indices() == [1, 3]


def test_evict_oldest_keeps_head_and_active_signature():
    p = make_policy([move_response("1", signature=b"S1"), move_response("2", signature=b"S2")])
    p.start("desc", {"observation": "S0"})
    head = p.contents[0]
    p.generate_move(); p.observe({"observation": "S1"})
    p.generate_move(); p.observe({"observation": "S2"})
    assert len(p.turns()) == 2
    p.evict_oldest_turn()
    assert len(p.turns()) == 1                   # oldest pair gone
    assert p.contents[0] is head                 # head pinned (same object)
    # the active unit's signature is intact
    active_sig = [pt for c in p.turns()[-1].payload for pt in c.parts if pt.thought_signature]
    assert active_sig and active_sig[0].thought_signature == b"S2"
    # evicting again is a no-op (only head + active remain)
    p.evict_oldest_turn()
    assert len(p.turns()) == 1


def test_4xx_fails_fast_and_dumps():
    # A deterministic 4xx (incl. a signature-validation 400) fails fast + dumps — no retry,
    # no history rewriting (consistent with the other providers).
    import os
    import tempfile
    d = tempfile.mkdtemp()
    p = make_policy([FakeAPIError(400, "INVALID_ARGUMENT: function call missing a thought_signature")])
    p.debug_dir = d
    p.start("desc", {"observation": "S0"})
    move = p.generate_move()
    assert move.action is None and move.error is not None
    assert len(p.client.models.calls) == 1                       # failed fast, no retry
    assert [f for f in os.listdir(d) if f.startswith("gemini_4xx_")]


def test_5xx_retried_then_success():
    # A transient 5xx retries (shared backoff) and then succeeds.
    p = make_policy([FakeAPIError(503, "UNAVAILABLE"), move_response("1")])
    p.start("desc", {"observation": "S0"})
    move = p.generate_move()
    assert move.action == "1" and len(p.client.models.calls) == 2


def test_debrief_fires_tool_free():
    p = make_policy([move_response("1"), debrief_response("Discovered: push the block.")])
    p.start("desc", {"observation": "S0"})
    p.generate_move()
    text = p.debrief()
    assert text == "Discovered: push the block."
    # the debrief used the tool-free config (no tools attached)
    cfg = p.client.models.calls[-1][1]
    assert cfg.tools is None


def test_turn_config_sets_max_output_tokens():
    # Gemini caps output (parity with the other providers).
    p = make_policy([move_response("2")])
    p.generate_move()
    cfg = p.client.models.calls[0][1]
    assert cfg.max_output_tokens == 32000


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
