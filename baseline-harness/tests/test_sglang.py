#!/usr/bin/env python3
"""Isolation tests for clients/sglang_client.py — fake HTTP transport, no network.

    python tests/test_sglang.py

Covers:
- model auto-ID + window from /v1/models (1 / 0 / many);
- forced-tool: action parsed from tool_calls; assistant appended verbatim with
  reasoning re-fed; tool result keyed by tool_call_id; accounting + last_prompt_tokens;
- guided-JSON: action from content JSON; non-object-JSON guard -> no action (nudge);
- grammar 400: a forced-tool tool/function 400 fails LOUD with a "--move-channel guided-json"
  remedy (no silent mid-run channel switch);
- 4xx (non-tool) fail-fast + dump; 5xx retried;
- reasoning round-trip probe, three explicit continuity states: equal token counts =>
  "stripped" (has_continuity False); unequal => "verified"; probe error (after one retry) =>
  "unverified" (carried -> has_continuity True, but never rendered as a bare verified carry);
- truncation surface: head (system+task) pinned, units = assistant+result, latest active,
  evict_oldest drops the oldest unit and keeps the active reasoning.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clients import sglang_client as S
from clients.sglang_client import SglangPolicy, ChatError


# ---- Fakes -------------------------------------------------------------

MODELS_ONE = {"data": [{"id": "kimi-k3", "max_model_len": 262144}]}


def usage(prompt=1000, cached=0, output=20, reasoning=50, total=None):
    # reasoning=None mimics SGLang's default: completion_tokens_details=None (unsupported).
    # reasoning=int mimics a build that reports the count (supported -> exact).
    u = {
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "total_tokens": total if total is not None else prompt + output,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": ({"reasoning_tokens": reasoning} if reasoning is not None else None),
    }
    return u


def tool_resp(action, *, call_id="call_1", reasoning="thinking...", reason_field="reasoning_content",
              u=None, finish="tool_calls"):
    msg = {"role": "assistant", "content": "",
           "tool_calls": [{"id": call_id, "type": "function",
                           "function": {"name": "make_move", "arguments": json.dumps({"action": action})}}]}
    if reasoning is not None:
        msg[reason_field] = reasoning   # SGLang emits `reasoning_content`; some vLLM builds `reasoning`
    return {"choices": [{"finish_reason": finish, "message": msg}], "usage": u or usage()}


def json_resp(action, *, reasoning="thinking...", u=None, content=None):
    body = content if content is not None else json.dumps({"action": action})
    return {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": body, "reasoning_content": reasoning}}], "usage": u or usage()}


def probe_resp(prompt_tokens):
    """A minimal /chat/completions reply carrying only a usage.prompt_tokens count
    (what probe_reasoning_roundtrip measures)."""
    return {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": ""}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1, "total_tokens": prompt_tokens + 1}}


def debrief_resp(text="Discovered the mechanics."):
    return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            "usage": usage()}


class FakePost:
    """Scriptable POST. Each script item is a dict (return) or an Exception (raise).
    Records (url, payload) per call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, url, payload, headers, timeout):
        self.calls.append((url, payload))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def fake_get(data):
    def _get(url, headers, timeout):
        return data
    return _get


def make_policy(script, *, models=MODELS_ONE, model=None, move_channel="forced-tool", preserve=True):
    post = FakePost(script)
    p = SglangPolicy(
        base_url="http://x/v1", api_key="k", model=model, timeout=30, max_retries=3,
        move_channel=move_channel, preserve_reasoning=preserve,
        http_post=post, http_get=fake_get(models),
    )
    p._post = post
    return p


# ---- Tests -------------------------------------------------------------


def test_model_autoid_and_window():
    p = make_policy([])
    assert p.model == "kimi-k3" and p.model_max_context == 262144 and p.has_pricing is False


def test_model_autoid_zero_and_many():
    for models in ({"data": []}, {"data": [{"id": "a"}, {"id": "b"}]}):
        raised = False
        try:
            make_policy([], models=models)  # no selector -> still errors on 0/many
        except SystemExit:
            raised = True
        assert raised


def test_multiple_models_select_callback():
    from clients.sglang_client import resolve_model_and_window
    data = {"data": [{"id": "a", "max_model_len": 100}, {"id": "b", "max_model_len": 200}]}
    # >=2 served + no --model + a selector -> the selector chooses (no error)
    assert resolve_model_and_window(data, None, select=lambda ids: "b") == ("b", 200)
    assert resolve_model_and_window(data, None, select=lambda ids: ids[0]) == ("a", 100)


def test_forced_tool_parse_append_and_carry():
    p = make_policy([tool_resp("2", call_id="c9", reasoning="weigh options", u=usage(prompt=1234))])
    p._reasoning_roundtrips = True  # pretend probe confirmed
    p.start("desc", {"observation": "S0"})
    move = p.generate_move()
    assert move.action == "2" and move.has_continuity is True
    assert p.last_prompt_tokens == 1234 and p.thoughts_tokens == 50 and p.cost_usd == 0.0
    # assistant appended verbatim; reasoning echoed back under the field the server emitted
    asst = p.messages[-1]
    assert asst["role"] == "assistant" and asst["reasoning_content"] == "weigh options" and asst["tool_calls"]
    assert "reasoning" not in asst  # echoed under the field the server used, not the wrong key
    assert p._last_tool_call_id == "c9"
    # observe answers the tool call by id
    p.observe({"observation": "S1"})
    assert p.messages[-1] == {"role": "tool", "tool_call_id": "c9", "content": json.dumps({"result": {"observation": "S1"}})}


def test_guided_json_parse_and_nonobject_guard():
    p = make_policy([json_resp("1"), json_resp(None, content="[1,2,3]")], move_channel="guided-json")
    p.start("desc", {"observation": "S0"})
    m1 = p.generate_move()
    assert m1.action == "1"
    p.observe({"observation": "S1"})            # guided path -> user message
    assert p.messages[-1]["role"] == "user"
    m2 = p.generate_move()                        # non-object JSON -> no action (-> nudge upstream)
    assert m2.action is None


def test_null_action_parses_to_none():
    p = make_policy([tool_resp(None)])            # arguments {"action": null}
    p.start("desc", {"observation": "S0"})
    assert p.generate_move().action is None


def test_parallel_tool_calls_history_records_only_answered():
    # Two make_move calls -> we answer the first by id; history must record ONLY that call, else
    # the unanswered one 400s the next request.
    two = {"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "",
        "tool_calls": [
            {"id": "cA", "type": "function",
             "function": {"name": "make_move", "arguments": json.dumps({"action": "1"})}},
            {"id": "cB", "type": "function",
             "function": {"name": "make_move", "arguments": json.dumps({"action": "2"})}},
        ]}}], "usage": usage()}
    p = make_policy([two])
    p.start("desc", {"observation": "S0"})
    move = p.generate_move()
    assert move.action == "1" and p._last_tool_call_id == "cA"
    assert [tc["id"] for tc in p.messages[-1]["tool_calls"]] == ["cA"]


def test_grammar_400_fails_loud_with_remedy():
    # A forced-tool grammar 400 is NOT silently switched to guided-JSON mid-run (that would make
    # this run's move channel diverge from the others). It fails fast with an actionable remedy.
    import tempfile
    err = ChatError(400, "grammar error: tool call could not be generated")
    p = make_policy([err])
    p.debug_dir = tempfile.mkdtemp()   # the fatal 400 dumps the request here — don't litter the repo
    warns = []
    p.on_retry = warns.append
    p.start("desc", {"observation": "S0"})
    move = p.generate_move()
    assert move.action is None and move.error is not None     # fatal -> caller ends with api_failure
    assert p.move_channel == "forced-tool"                    # channel unchanged (no silent flip)
    assert any("guided-json" in w.lower() for w in warns)     # actionable remedy emitted


def test_4xx_failfast_and_dump(tmp_path=None):
    import tempfile, os
    d = tempfile.mkdtemp()
    err = ChatError(422, "unprocessable: bad schema")  # non-tool 4xx
    p = make_policy([err])
    p.debug_dir = d
    p.start("desc", {"observation": "S0"})
    move = p.generate_move()
    assert move.action is None and move.error is not None
    dumps = [f for f in os.listdir(d) if f.startswith("sglang_4xx_")]
    assert dumps, "a 4xx should dump the request payload"


def test_5xx_retried_then_success():
    err = ChatError(503, "overloaded")
    p = make_policy([err, tool_resp("1")])
    S.time.sleep = lambda *_: None
    p.start("desc", {"observation": "S0"})
    move = p.generate_move()
    assert move.action == "1"


def test_programming_error_fails_fast_not_retried():
    # A TypeError from SDK/shape drift is not transport trouble: one attempt, no backoff burn
    # (same clientutil.is_retryable classification the ChatError path uses).
    p = make_policy([TypeError("unexpected keyword"), tool_resp("1")])
    p.start("desc", {"observation": "S0"})
    move = p.generate_move()
    assert move.action is None and move.error is not None
    assert move.error["type"] == "TypeError" and move.error["attempt"] == 1
    assert len(p._post.script) == 1  # the scripted success was never consumed — no retry


def test_server_info_allowlisted_never_leaks_credentials():
    # /get_server_info echoes SGLang's server_args verbatim — INCLUDING api_key /
    # admin_api_key. The stored (and published) server_info must carry ONLY the allowlist.
    info = {"model_path": "/models/m", "version": "0.5.1", "tokenizer_path": "/models/t",
            "api_key": "SECRET-KEY", "admin_api_key": "ADMIN-SECRET", "watchdog_timeout": 300}

    def get(url, headers, timeout):
        return info if url.endswith("/get_server_info") else MODELS_ONE

    p = SglangPolicy(base_url="http://x/v1", api_key="k", model=None, timeout=30,
                     max_retries=3, http_post=FakePost([]), http_get=get)
    assert p.server_info == {"model_path": "/models/m", "version": "0.5.1",
                             "tokenizer_path": "/models/t"}
    assert "SECRET" not in repr(p.server_info) and "ADMIN" not in repr(p.server_info)


def test_probe_strips_means_no_continuity():
    # carrying reasoning_content does NOT grow prompt_tokens -> server strips it on input
    p = make_policy([tool_resp("1")])
    p._post.script = [probe_resp(50), probe_resp(50)] + p._post.script  # probe consumes first two posts
    p.probe_reasoning_roundtrip()
    assert p._reasoning_roundtrips is False
    p.start("desc", {"observation": "S0"})
    move = p.generate_move()
    assert move.has_continuity is False  # carried, but proven a no-op -> honest False
    assert move.continuity == "stripped"


def test_probe_preserves_means_continuity():
    # carrying reasoning_content grows prompt_tokens by the sentinel -> server feeds it back
    p = make_policy([tool_resp("1")])
    p._post.script = [probe_resp(50), probe_resp(450)] + p._post.script
    p.probe_reasoning_roundtrip()
    assert p._reasoning_roundtrips is True
    p.start("desc", {"observation": "S0"})
    move = p.generate_move()
    assert move.has_continuity is True and move.continuity == "verified"


def test_probe_error_leaves_unverified_after_one_retry():
    # A startup hiccup gets ONE retry; only a repeated failure demotes the run to unverified —
    # and unverified is a DISTINCT state, never rendered as a bare verified carry.
    p = make_policy([tool_resp("1")])
    p._post.script = [ChatError(500, "probe failed"), ChatError(500, "probe failed again")] + p._post.script
    p.probe_reasoning_roundtrip()
    assert p._reasoning_roundtrips is None
    assert p._post.script and not isinstance(p._post.script[0], ChatError)  # both errors consumed
    p.start("desc", {"observation": "S0"})
    move = p.generate_move()
    assert move.has_continuity is True   # carried, not disproven (historical meaning)
    assert move.continuity == "unverified"


def test_probe_retry_recovers_from_transient_error():
    # First attempt dies mid-probe; the retry runs both measurements and settles VERIFIED.
    p = make_policy([tool_resp("1")])
    p._post.script = [ChatError(500, "hiccup"), probe_resp(50), probe_resp(450)] + p._post.script
    p.probe_reasoning_roundtrip()
    assert p._reasoning_roundtrips is True


def test_reasoning_field_fallback_echoed_under_same_key():
    # a server that emits `reasoning` (not reasoning_content) gets it echoed back under `reasoning`
    p = make_policy([tool_resp("1", reasoning="vllm-style", reason_field="reasoning")])
    p._reasoning_roundtrips = True
    p.start("desc", {"observation": "S0"})
    p.generate_move()
    asst = p.messages[-1]
    assert asst.get("reasoning") == "vllm-style" and "reasoning_content" not in asst


def test_thoughts_included_when_server_omits_reasoning_tokens():
    # completion_tokens_details=None (SGLang default) but the model DID reason -> we can't
    # separate it; report it merged ("included", shown as "out + think"), never estimated.
    p = make_policy([tool_resp("1", reasoning="some thinking text", u=usage(reasoning=None))])
    p._reasoning_roundtrips = True
    p.start("d", {"observation": "S0"})
    move = p.generate_move()
    assert move.usage["thoughts"] == 0 and move.thoughts_basis == "included" and p.thoughts_basis == "included"


def test_thoughts_exact_when_server_reports_reasoning_tokens():
    # A build that reports reasoning_tokens (key present) is exact AND disjoint: completion_tokens
    # INCLUDES reasoning, so out = completion - reasoning (the visible part), matching the closed
    # clients (out + think == completion). reasoning=37 of completion=100 -> out 63 / think 37.
    p = make_policy([tool_resp("1", reasoning="anything", u=usage(output=100, reasoning=37))])
    p._reasoning_roundtrips = True
    p.start("d", {"observation": "S0"})
    move = p.generate_move()
    assert move.usage["thoughts"] == 37 and move.usage["output"] == 63
    assert move.usage["output"] + move.usage["thoughts"] == 100  # == completion_tokens (disjoint)
    assert move.thoughts_basis == "exact" and p.thoughts_basis == "exact"


def test_reasoning_tokens_zero_with_key_is_exact_not_included():
    # KEY PRESENCE wins: reasoning_tokens=0 means "thought nothing" (exact 0), distinct from
    # the unsupported (details=None) case.
    p = make_policy([tool_resp("1", reasoning=None, u=usage(reasoning=0))])
    p._reasoning_roundtrips = True
    p.start("d", {"observation": "S0"})
    move = p.generate_move()
    assert move.usage["thoughts"] == 0 and move.thoughts_basis == "exact"


def test_debrief_overflow_evicts_and_retries():
    # A run whose moves just overflowed would otherwise ALWAYS lose its debrief too (the prompt
    # only grew). A debrief overflow 400 must evict deficit-sized oldest turns and re-send —
    # without appending the debrief nudge a second time.
    import tempfile
    overflow = ChatError(400, ("Requested token count exceeds the model's maximum context "
                               "length of 262144 tokens. You requested a total of 262500 tokens: "
                               "230500 tokens from the input messages and 32000 tokens for the "
                               "completion."))
    p = make_policy([tool_resp("1", call_id="c1"), tool_resp("2", call_id="c2"),
                     tool_resp("3", call_id="c3"), overflow, debrief_resp("post-mortem")])
    p.debug_dir = tempfile.mkdtemp()   # the fatal 400 dumps the request here — don't litter the repo
    p.start("desc", {"observation": "S0"})
    for i in range(3):
        p.generate_move(); p.observe({"observation": f"S{i + 1}"})
    warns = []
    p.on_retry = warns.append
    n_units = len(p.turns())
    assert p.debrief() == "post-mortem"
    assert len(p.turns()) < n_units                                  # oldest turns evicted
    assert any("debrief context overflow" in w for w in warns)       # recovery logged
    nudges = [m for m in p.messages if m.get("content") == S.prompts.DEBRIEF_PROMPT]
    assert len(nudges) == 1                                          # nudge appended exactly once


def test_debrief_overflow_gives_up_when_cannot_shrink():
    # Only head + the lone active unit remain -> eviction can't shrink; give up (None), no spin.
    import tempfile
    overflow = ChatError(400, "maximum context length of 100 tokens. requested a total of 999 tokens")
    p = make_policy([tool_resp("1", call_id="c1"), overflow])
    p.debug_dir = tempfile.mkdtemp()
    p.start("desc", {"observation": "S0"})
    p.generate_move(); p.observe({"observation": "S1"})
    assert p.debrief() is None
    assert p._post.script == []   # exactly one debrief POST — no blind retries


def test_truncation_surface_head_pinned_active_intact():
    p = make_policy([tool_resp("1", call_id="c1", reasoning="r1"),
                     tool_resp("2", call_id="c2", reasoning="r2")])
    p.start("desc", {"observation": "S0"})          # messages[0] = user task (head)
    p.generate_move(); p.observe({"observation": "S1"})   # assistant1 + tool1
    p.generate_move(); p.observe({"observation": "S2"})   # assistant2 + tool2
    units = p.turns()
    assert len(units) == 2 and units[0].is_active is False and units[1].is_active is True
    assert p._assistant_indices() == [1, 3]         # head = messages[0:1]
    p.evict_oldest_turn()
    assert len(p.turns()) == 1
    assert p.messages[0]["content"].startswith("You are now playing")  # head pinned
    active = p.turns()[-1].payload
    assert any(m.get("reasoning_content") == "r2" for m in active)  # active reasoning intact
    p.evict_oldest_turn()                            # no-op at floor
    assert len(p.turns()) == 1


def test_unserved_model_exits_loud():
    # An explicit --model not among the served ids must raise, not silently proceed with
    # window=None (which would disable truncation).
    try:
        make_policy([], model="not-served-xyz")
        assert False, "expected SystemExit for an unserved --model"
    except SystemExit as e:
        assert "not served" in str(e)


def test_turn_payload_sends_max_tokens():
    # SGLang sends a max_tokens cap (parity with the closed providers).
    p = make_policy([tool_resp("1")])
    p.start("desc", {"observation": "S0"})
    p.generate_move()
    assert p._post.calls[0][1]["max_tokens"] == 32000


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
