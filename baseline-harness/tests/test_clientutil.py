#!/usr/bin/env python3
"""Isolation tests for core/clientutil.py — the shared error classifier + retry loop.

    python tests/test_clientutil.py

Covers the one provider-agnostic policy: is_retryable across 4xx (fatal) / 5xx+529 (retry) /
429 rate-limit (retry) vs 429 insufficient_quota (fatal) / no-status (retry); status extracted
from either `.status_code` (OpenAI/Anthropic) or `.code` (google-genai/sglang); run_request
success / fail-fast+dump / transient-retry / give-up.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import clientutil as cu

cu.time.sleep = lambda *_: None  # no real backoff


class Err(Exception):
    """Status carried via `.status_code` (default) or `.code`; optional permanent-429 marker."""

    def __init__(self, status=None, *, code=None, body="", attr="status_code"):
        setattr(self, attr, status)
        if code is not None:
            self.code = code
        self.body = body
        super().__init__(f"{status}: {body}")


# ---- classifier --------------------------------------------------------


def test_is_retryable_fatal_4xx():
    for s in (400, 401, 403, 404, 413, 422):
        assert cu.is_retryable(Err(s)) is False


def test_is_retryable_transient_5xx_and_529():
    for s in (500, 502, 503, 504, 529):
        assert cu.is_retryable(Err(s)) is True


def test_429_rate_limit_retries_but_insufficient_quota_fails_fast():
    assert cu.is_retryable(Err(429)) is True                                   # rate-limit -> retry
    assert cu.is_retryable(Err(429, code="insufficient_quota")) is False       # billing -> fatal
    assert cu.is_retryable(Err(429, body="...exceeded quota: insufficient_quota...")) is False


def test_no_status_is_transient():
    assert cu.is_retryable(Exception("connection reset by peer")) is True


def test_programming_errors_fail_fast_not_retried():
    # SDK drift / our bugs (TypeError & friends) are never transport trouble — retrying only
    # hides the traceback behind minutes of backoff. ValueError is exempt: json/decode paths
    # raise it for genuinely transient read garbage.
    for exc in (TypeError("bad kwargs"), AttributeError("no such attr"), KeyError("k"),
                IndexError("i"), NameError("n")):
        assert cu.is_retryable(exc) is False, type(exc).__name__
    assert cu.is_retryable(ValueError("Expecting value: line 1")) is True


def test_run_request_fails_fast_on_programming_error():
    import tempfile
    calls = []

    def bad_call():
        calls.append(1)
        raise TypeError("create() got an unexpected keyword argument 'output_config'")

    with tempfile.TemporaryDirectory() as d:
        result, err = cu.run_request(bad_call, provider="X", max_retries=5,
                                     request={"model": "m"}, debug_dir=d)
        assert result is None and err["type"] == "TypeError"
        assert len(calls) == 1                      # first attempt only — no backoff burn
        assert err.get("payload_dump")              # dumped for offline debugging


def test_408_request_timeout_is_retryable():
    assert cu.is_retryable(Err(408)) is True  # transient timeout, not a deterministic 4xx


def test_3xx_refused_redirect_is_fatal():
    # urlopen_no_redirect surfaces a 3xx as an HTTPError; the endpoint deterministically
    # redirects, so retrying can't help.
    for s in (301, 302, 307, 308):
        assert cu.is_retryable(Err(s)) is False


def test_redirect_handler_refuses_and_never_resends_credentials():
    # Returning None from redirect_request makes urllib raise the 3xx instead of issuing a
    # second request — the Authorization header can never be replayed to another host.
    h = cu._RefuseRedirect()
    assert h.redirect_request(None, None, 302, "Found", {}, "https://evil.example") is None


def test_gemini_context_overflow_marker_recognized():
    msg = "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576)."
    assert cu.is_context_overflow({"message": msg}) is True   # Gemini overflow -> evict-and-retry
    assert cu.is_context_overflow({"message": "invalid request: bad field"}) is False


def test_parse_overflow_tokens_litellm_sglang_wording():
    # The exact litellm/SGLang wording emitted on context overflow.
    msg = ('litellm.BadRequestError: OpenAIException - Requested token count exceeds the '
           "model's maximum context length of 262144 tokens. You requested a total of 264084 "
           'tokens: 232084 tokens from the input messages and 32000 tokens for the completion.')
    assert cu.parse_overflow_tokens({"message": msg}) == (264084, 262144)


def test_parse_overflow_tokens_openai_classic_wording():
    msg = ("This model's maximum context length is 8193 tokens, "
           "however you requested 10001 tokens.")
    assert cu.parse_overflow_tokens({"message": msg}) == (10001, 8193)


def test_parse_overflow_tokens_unparseable_or_incoherent():
    assert cu.parse_overflow_tokens(None) is None
    assert cu.parse_overflow_tokens({"message": "context_length_exceeded"}) is None  # no counts
    assert cu.parse_overflow_tokens({"message": "maximum context length of 100 tokens"}) is None
    incoherent = "maximum context length of 300 tokens. You requested a total of 200 tokens"
    assert cu.parse_overflow_tokens({"message": incoherent}) is None  # total <= limit -> no deficit


def test_status_extracted_from_code_attr():
    # google-genai APIError / sglang ChatError carry the int status under `.code`.
    assert cu.is_retryable(Err(400, attr="code")) is False
    assert cu.is_retryable(Err(503, attr="code")) is True


# ---- run_request -------------------------------------------------------


def test_run_request_success_first_try():
    res, err = cu.run_request(lambda: "ok", provider="X", max_retries=3, request={}, debug_dir=".")
    assert res == "ok" and err is None


def test_run_request_fatal_dumps_and_returns():
    d = tempfile.mkdtemp()
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        raise Err(400, body="bad request")

    events = []
    res, err = cu.run_request(call, provider="X", max_retries=5, request={"k": 1},
                              debug_dir=d, on_event=events.append)
    assert res is None and err is not None and err["type"] == "Err"
    assert calls["n"] == 1  # failed fast — no retry
    assert "payload_dump" in err and [f for f in os.listdir(d) if f.startswith("x_4xx_")]
    assert any("not retrying" in e for e in events)


def test_run_request_transient_retries_then_succeeds():
    seq = [Err(503), Err(503), "ok"]

    def call():
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    events = []
    res, err = cu.run_request(call, provider="X", max_retries=5, request={}, debug_dir=".",
                              on_event=events.append)
    assert res == "ok" and err is None
    assert sum("retrying" in e for e in events) == 2


def test_run_request_gives_up_after_max_retries():
    def call():
        raise Err(503)

    res, err = cu.run_request(call, provider="X", max_retries=3, request={}, debug_dir=".")
    assert res is None and err is not None and err["attempt"] == 3 and err["max_retries"] == 3


# ---- SSE stream accumulation -------------------------------------------


def _sse(*items):
    """Build a list of SSE byte lines. A dict item -> a `data: {json}` chunk line; a str item
    -> a raw line verbatim (blank, `:` keep-alive comment, or `data: [DONE]`)."""
    out = []
    for it in items:
        line = ("data: " + json.dumps(it)) if isinstance(it, dict) else it
        out.append(line.encode("utf-8"))
    return out


def _delta(**kw):
    return {"choices": [{"index": 0, "delta": kw, "finish_reason": None}]}


def test_sse_accumulate_tool_call_multidelta():
    # name in one delta; arguments split across three; reasoning across two; then finish + usage.
    u = {"prompt_tokens": 285, "completion_tokens": 77, "total_tokens": 362,
         "prompt_tokens_details": {"cached_tokens": 0},
         "completion_tokens_details": {"reasoning_tokens": 47}}
    lines = _sse(
        _delta(role="assistant"),
        _delta(reasoning_content="think"),
        _delta(reasoning_content="ing..."),
        _delta(tool_calls=[{"index": 0, "id": "call_1", "type": "function",
                            "function": {"name": "make_move", "arguments": ""}}]),
        _delta(tool_calls=[{"index": 0, "function": {"arguments": '{"action": '}}]),
        _delta(tool_calls=[{"index": 0, "function": {"arguments": '"2"}'}}]),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": u},
        "data: [DONE]",
    )
    got = cu.accumulate_chat_stream(lines)
    expected = {"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "", "reasoning_content": "thinking...",
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "make_move", "arguments": '{"action": "2"}'}}]}}],
        "usage": u}
    assert got == expected, got


def test_sse_accumulate_reasoning_field_variant():
    # Server streams reasoning under `reasoning` (not `reasoning_content`): emit only that key.
    lines = _sse(
        _delta(reasoning="abc"),
        _delta(content="hi"),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
        "data: [DONE]",
    )
    msg = cu.accumulate_chat_stream(lines)["choices"][0]["message"]
    assert msg["reasoning"] == "abc" and "reasoning_content" not in msg
    assert msg["content"] == "hi" and "tool_calls" not in msg  # no tool deltas -> key omitted


def test_sse_skips_keepalive_blank_emptydata_and_stops_at_done():
    lines = _sse(
        ": keep-alive comment",
        "",
        "data:",                                 # bare keep-alive data line -> skipped, no crash
        _delta(content="x"),
        "",
        ": ping",
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        "data: [DONE]",
        _delta(content="SHOULD_BE_IGNORED"),     # after [DONE] -> never read
    )
    got = cu.accumulate_chat_stream(lines)
    assert got["choices"][0]["message"]["content"] == "x"          # comment/blank/empty-data skipped
    assert got["choices"][0]["finish_reason"] == "stop"            # post-[DONE] chunk ignored


def test_sse_truncated_stream_raises_retryable():
    # Server closes mid-stream: content but no finish_reason and no usage -> retryable StreamError
    # (not a silent partial that would misread as an invalid move / lose token accounting).
    lines = _sse(_delta(role="assistant"), _delta(content="par"), _delta(content="tial"))
    try:
        cu.accumulate_chat_stream(lines)
        assert False, "expected StreamError on a truncated stream"
    except cu.StreamError as e:
        assert e.code is None and cu.is_retryable(e) is True
    # Usage present but NO finish_reason is ALSO truncated.
    lines2 = _sse(_delta(content="x"), {"choices": [], "usage": {"prompt_tokens": 5, "total_tokens": 6}})
    try:
        cu.accumulate_chat_stream(lines2)
        assert False, "expected StreamError when usage present but finish_reason missing"
    except cu.StreamError:
        pass       # code None => transient => retried


def test_sse_finish_reason_without_requested_usage_is_truncated():
    # Connection cut BETWEEN finish_reason and the include_usage trailer: without the guard the
    # move would be committed with zero token/cost accounting. require_usage => StreamError.
    lines = _sse(
        _delta(content="ok"),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
    try:
        cu.accumulate_chat_stream(lines, require_usage=True)
        assert False, "expected StreamError when the requested usage trailer never arrived"
    except cu.StreamError as e:
        assert e.code is None and cu.is_retryable(e) is True
    # Without the flag (caller didn't request usage), usage simply stays absent.
    lines2 = _sse(
        _delta(content="ok"),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
    got = cu.accumulate_chat_stream(lines2)
    assert got["choices"][0]["finish_reason"] == "stop" and "usage" not in got


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
