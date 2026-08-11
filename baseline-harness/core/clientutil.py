"""Shared, provider-agnostic helpers for the LLM client classes.

Reusable infrastructure only — no game logic and no single-provider knowledge. It holds
the small helpers and the retry/error machinery shared by the four clients:

- ``zero_usage`` / ``est_tokens`` — trivial shared utilities. (Model→table matching
  lives in ``accounting.match_model`` — one matcher for pricing + context-window lookups.)
- ``extract_status`` / ``is_retryable`` — one HTTP-error classification policy for every
  provider (the SDKs expose the status differently; see ``extract_status``).
- ``dump_request`` — the 4xx request dump for offline debugging.
- ``run_request`` — the retry loop with exponential backoff used by the closed-model clients
  (Gemini/Anthropic/OpenAI). SGLang keeps its own loop (it accumulates a streamed SSE response and
  has provider-specific move-channel handling) but reuses the classifier + dump here.
- ``iter_sse_data`` / ``accumulate_chat_stream`` (+ ``StreamError``) — fold an OpenAI-compatible
  chat-completions SSE stream back into the same response dict a non-streamed call returns. Only
  SGLang streams today; kept here so it is socket-free unit-testable and reusable by any
  OpenAI-compatible client. Iterating the live response makes the socket read timeout act as a
  per-chunk idle timeout (no watchdog thread).

Error taxonomy:
- retryable: HTTP 500/502/503/504, Anthropic 529 (overloaded), 429 rate-limit, and
  network/timeout errors (no HTTP status);
- fatal (fail-fast): 400/401/403/404/413/422, AND the one permanent 429 — OpenAI
  ``insufficient_quota`` (a billing state; backoff cannot create quota). Anthropic billing is a
  400 (already fatal); Gemini has no clean permanent-429 marker, so its 429 stays retryable.
- fatal (fail-fast): programming errors (TypeError/AttributeError/KeyError/IndexError/NameError
  — SDK drift or our bugs), whatever their status; retrying them only hides the traceback
  behind minutes of backoff. ValueError stays transient (json/decode paths raise it for
  genuinely transient read garbage).
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.request
from typing import Callable

# 429s that are permanent (billing/quota) rather than transient rate-limiting. Matched against
# the exception's code/body/str — only OpenAI emits this, but the check is harmless everywhere.
_PERMANENT_429_MARKERS = ("insufficient_quota",)

# Max exponential-backoff wait (seconds) between retries; shared by every retry loop.
BACKOFF_CAP = 60

# --- Manual thinking budgets --------------------------------------------------------
# One shared --thinking-level -> token-budget table for every model family whose thinking is
# configured by an explicit token budget instead of a level/effort knob (Anthropic manual
# extended thinking on pre-effort models, Gemini 2.5 `thinking_budget`). Sharing the table is
# a fairness property: the same level buys the same NOMINAL budget on every provider; each
# client applies only its API-mandated floor/cap on top via ``clamp_thinking_budget``.
THINKING_LEVEL_BUDGETS: dict[str, int] = {
    "minimal": 1024, "low": 2048, "medium": 8192,
    "high": 16384, "xhigh": 24576, "max": 32768,
}


def clamp_thinking_budget(level: str, max_tokens: int, *, floor: int, cap: int | None = None) -> int:
    """The manual thinking budget for ``level``, clamped to a family's API bounds. On every
    budget-based family the thinking spends from the SAME ``max_tokens`` output allowance, so
    at most ``max_tokens - 1024`` is granted (1024 tokens of visible output always remain);
    ``floor``/``cap`` are the family's hard bounds (e.g. Anthropic floor 1024; Gemini 2.5
    flash cap 24576, pro floor 128 / cap 32768).

    The floor GATES, it never bumps up: when ``max_tokens`` leaves less room than the floor
    (or would resolve to a 0 budget — thinking silently OFF while the run is labeled with a
    level), the combination is INFEASIBLE and raises ValueError. Bumping up instead would
    either violate the API (Anthropic requires budget < max_tokens) or mislabel the run."""
    budget = min(THINKING_LEVEL_BUDGETS[level], max_tokens - 1024)
    if cap is not None:
        budget = min(budget, cap)
    if budget < max(floor, 1):
        raise ValueError(
            f"--max-tokens {max_tokens} leaves no room for a thinking budget at level "
            f"{level!r} (needs >= {max(floor, 1) + 1024} total): raise --max-tokens or use "
            "--thinking-level none"
        )
    return budget


def backoff_sleep(attempt: int, cap: int = BACKOFF_CAP) -> None:
    """Sleep an exponential backoff (``2**attempt``, capped at ``cap``) plus jitter. The jitter
    desynchronizes concurrent clients so their retries don't form a thundering herd against one
    server. Calls ``time.sleep`` at call time so tests can neutralize the wait."""
    time.sleep(min(2 ** attempt, cap) + random.uniform(0, 1 + attempt))


# --- HTTP without redirects ---------------------------------------------------------
# Every stdlib-urllib request we make carries a bearer token, and urllib's default redirect
# handler re-sends ALL headers — Authorization included — to whatever host a 3xx points at
# (cross-origin credential leak). API endpoints never legitimately redirect, so refuse outright:
# returning None from redirect_request makes urllib raise the 3xx as an HTTPError, which flows
# through each caller's existing error branch.


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_no_redirect_opener = urllib.request.build_opener(_RefuseRedirect)


def urlopen_no_redirect(req, timeout):
    """``urllib.request.urlopen`` that refuses to follow redirects (3xx raises ``HTTPError``).
    Use for every request that carries credentials."""
    return _no_redirect_opener.open(req, timeout=timeout)


# --- Streaming (SSE) accumulation -------------------------------------------------
# An OpenAI-compatible /chat/completions stream (stream:true) returns Server-Sent Events:
# `data: {json}` lines, blanks and `:` keep-alive comments interspersed, ending at
# `data: [DONE]`. We fold the deltas back into the SAME dict a non-streamed call returns, so
# every downstream reader (and every test mocking the response) is unchanged.


class StreamError(RuntimeError):
    """A streamed response carried an error chunk. ``code`` is the HTTP-ish status if the server
    gave one (else None -> treated as transient by ``is_retryable``). Provider code maps this to
    its own error type (keeps this module provider-agnostic)."""

    def __init__(self, code, body):
        super().__init__(str(body))
        self.code = code
        self.body = body


def iter_sse_data(line_iterable):
    """Yield the ``data:`` payload string of each SSE event from a line iterable (a urllib
    response iterates as ``bytes`` lines; tests can pass a list/``BytesIO``). Skips blanks and
    ``:`` keep-alive comments; stops at the ``[DONE]`` sentinel. The blocking read happens here,
    so the socket's read timeout fires naturally on a stalled stream."""
    for raw in line_iterable:
        line = (raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw).strip()
        if not line or line.startswith(":"):          # blank line or keep-alive comment
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()                        # strip "data:" + optional leading space
        if not data:                                   # bare "data:" keep-alive — nothing to parse
            continue
        if data == "[DONE]":
            return
        yield data


def accumulate_chat_stream(line_iterable, *, require_usage: bool = False) -> dict:
    """Fold a chat-completions SSE stream into the non-streamed response shape:
    ``{"choices":[{"message":{...}, "finish_reason":...}], "usage":{...}}``. Concatenates
    content / reasoning / tool-call-argument deltas, preserves WHICH reasoning key the server
    used (``reasoning_content`` vs ``reasoning`` — emit only that one), and takes ``usage`` from
    the final (choices-empty) chunk. Raises ``StreamError`` on an error chunk. With
    ``require_usage`` (the caller requested ``stream_options.include_usage``), a stream that
    ends without the usage trailer is treated as truncated too."""
    content: list = []
    reasoning: list = []
    reasoning_key = None                               # emit only the key the deltas actually used
    tool_calls: dict = {}                              # keyed by delta index
    finish_reason = None
    usage = None
    model = None                                       # echoed on every chunk; kept for parity
    for data in iter_sse_data(line_iterable):
        chunk = json.loads(data)
        if chunk.get("error"):                         # mid-stream server error
            err = chunk["error"]
            code = err.get("code") if isinstance(err, dict) else None
            raise StreamError(code if isinstance(code, int) else None, err)
        if chunk.get("model"):
            model = chunk["model"]
        if chunk.get("usage"):                         # final include_usage chunk (choices empty)
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            for key in ("reasoning_content", "reasoning"):
                if delta.get(key):
                    reasoning.append(delta[key])
                    reasoning_key = reasoning_key or key
                    break
            for tc in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(
                    tc.get("index", 0),
                    {"id": None, "type": "function", "function": {"name": None, "arguments": ""}},
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                if tc.get("type"):
                    slot["type"] = tc["type"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    # A complete OpenAI-compatible stream ALWAYS terminates with a finish_reason (it precedes the
    # final include_usage chunk). Its absence => the server closed mid-stream; surface it as a
    # retryable error rather than silently returning a partial response (which would misread as an
    # invalid/empty move and lose this turn's token accounting).
    if finish_reason is None:
        raise StreamError(None, "truncated stream (no finish_reason)")
    # A finish_reason alone is NOT proof of completeness: the requested include_usage trailer
    # follows it, and a connection cut in between would commit the move with ZERO token/cost
    # accounting (and never advance last_prompt_tokens, blinding truncation). Treat it as a
    # truncated stream — StreamError(None) is transient, so the caller retries.
    if require_usage and usage is None:
        raise StreamError(None, "truncated stream (finish_reason without requested usage)")
    message: dict = {"role": "assistant", "content": "".join(content)}
    if reasoning_key is not None:                      # never default-create a reasoning key
        message[reasoning_key] = "".join(reasoning)
    if tool_calls:                                     # omit entirely when no tool-call deltas
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    result: dict = {"choices": [{"message": message, "finish_reason": finish_reason}]}
    if usage is not None:                              # pass through verbatim (absent stays absent)
        result["usage"] = usage
    if model is not None:                              # parity with the non-streamed shape
        result["model"] = model
    return result


def zero_usage() -> dict:
    return {"prompt": 0, "cached": 0, "output": 0, "thoughts": 0, "total": 0}


def est_tokens(chunk) -> int:
    """Cheap size proxy (chars // 4) over a unit's serialized form — for MINIMAL eviction, not
    billing. Provider payload objects stringify to their repr, which is fine as a proxy."""
    return sum(len(str(c)) for c in chunk) // 4


def extract_status(exc) -> int | None:
    """The HTTP status from a provider exception, or None (network/timeout/unknown).
    OpenAI/Anthropic SDK errors expose `.status_code`; google-genai `APIError.code` and the
    SGLang `ChatError.code` are the int status under `.code`."""
    for attr in ("status_code", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    return None


def _is_permanent_429(exc) -> bool:
    blob = (str(getattr(exc, "code", "")) + " " + str(getattr(exc, "body", "")) + " " + str(exc)).lower()
    return any(m in blob for m in _PERMANENT_429_MARKERS)


def is_retryable(exc) -> bool:
    """Transient (retry) vs deterministic (fail-fast). See the module docstring for the policy."""
    # Programming errors are never transport trouble: retrying a TypeError from SDK drift or a
    # bug of ours burns the whole backoff budget (minutes) to fail identically N times, and the
    # backoff masks the traceback. Fail fast so the real error surfaces immediately.
    # (ValueError is NOT here: json/decode paths raise it for genuinely transient read garbage.)
    if isinstance(exc, (TypeError, AttributeError, KeyError, IndexError, NameError)):
        return False
    status = extract_status(exc)
    if status is None:                       # network / timeout / unknown -> transient
        return True
    if status == 429:                        # rate-limit retry; permanent quota fails fast
        return not _is_permanent_429(exc)
    if status == 408:                        # request timeout -> transient
        return True
    if 300 <= status < 400:                  # refused redirect -> deterministic (see urlopen_no_redirect)
        return False
    if 400 <= status < 500:                  # 400/401/403/404/413/422 -> deterministic
        return False
    return True                              # 5xx / 529 -> transient


# Context-window overflow markers — a 400 that EVICTING history can fix, unlike a generic fatal
# 4xx. OpenAI: "context_length_exceeded" / "maximum context length"; Anthropic: "...exceed context
# limit: X + Y > Z"; SGLang/vLLM: "maximum context length" / "longer than the maximum"; Gemini:
# "The input token count ... exceeds the maximum number of tokens allowed".
_CONTEXT_OVERFLOW_MARKERS = (
    "context_length_exceeded", "maximum context length", "exceed context limit",
    "context window", "reduce the length", "longer than the maximum",
    "exceeds the maximum number of tokens allowed",
)


def is_context_overflow(err) -> bool:
    """True if ``err`` (a failed-Move error dict, or an exception) is a context-window overflow —
    recoverable by evicting history and retrying, vs a generic fatal 4xx."""
    if not err:
        return False
    blob = (str(err.get("message", "")) if isinstance(err, dict) else str(err)).lower()
    return any(m in blob for m in _CONTEXT_OVERFLOW_MARKERS)


# OpenAI/litellm/SGLang overflow wording carries the exact numbers — "maximum context length
# of 262144 tokens. You requested a total of 264084 tokens" (litellm/SGLang) and "maximum
# context length is 8193 tokens, however you requested 10001 tokens" (OpenAI classic).
_OVERFLOW_LIMIT_RE = re.compile(r"maximum context length (?:is |of )?(\d+)")
_OVERFLOW_TOTAL_RE = re.compile(r"requested (?:a total of )?(\d+) tokens")


def parse_overflow_tokens(err) -> tuple[int, int] | None:
    """``(requested_total, limit)`` parsed from a context-overflow error message, or None when
    the wording carries no usable counts (then the caller falls back to blind eviction).
    Only a coherent pair (total > limit > 0) is returned — never a negative deficit."""
    if not err:
        return None
    blob = (str(err.get("message", "")) if isinstance(err, dict) else str(err)).lower()
    limit_m = _OVERFLOW_LIMIT_RE.search(blob)
    total_m = _OVERFLOW_TOTAL_RE.search(blob)
    if not (limit_m and total_m):
        return None
    limit, total = int(limit_m.group(1)), int(total_m.group(1))
    if not (0 < limit < total):
        return None
    return total, limit


def _error_dict(exc, attempt: int, max_retries: int, dump: str | None = None) -> dict:
    err = {"type": type(exc).__name__, "message": str(exc),
           "attempt": attempt + 1, "max_retries": max_retries}
    if dump:
        err["payload_dump"] = dump
    return err


def dump_request(debug_dir: str, provider: str, exc, request, call_count: int) -> str | None:
    """Write the failing request to ``<provider>_4xx_<ts>_<n>.json`` for offline debugging.
    The request is dumped verbatim when JSON-serializable (e.g. SGLang payloads); otherwise we
    record just its keys (SDK payloads carry non-serializable block/item objects)."""
    fname = os.path.join(debug_dir, f"{provider.lower()}_4xx_{time.strftime('%Y%m%d-%H%M%S')}_{call_count}.json")
    try:
        req = json.loads(json.dumps(request))
    except (TypeError, ValueError):
        req = {"request_keys": sorted(request.keys())} if isinstance(request, dict) else {"request_type": str(type(request))}
    try:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump({"http_status": extract_status(exc), "error": str(exc), "request": req},
                      f, indent=2, ensure_ascii=False)
        return fname
    except Exception:
        return None


def run_request(
    call: Callable,
    *,
    provider: str,
    max_retries: int,
    request,
    debug_dir: str,
    call_count: int = 0,
    on_event: Callable | None = None,
):
    """Call ``call()`` with retries. Returns ``(result, None)`` on success, else
    ``(None, error_dict)``. Deterministic errors (``is_retryable`` False) fail fast and dump the
    request; transient errors retry with exponential backoff + jitter (see ``backoff_sleep``,
    capped at ``BACKOFF_CAP`` s). Backoff uses ``time.sleep`` at call time (so tests can patch it)."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return call(), None
        except Exception as exc:
            if not is_retryable(exc):
                dump = dump_request(debug_dir, provider, exc, request, call_count)
                last_error = _error_dict(exc, attempt, max_retries, dump)
                if on_event:
                    on_event(f"{provider} HTTP {extract_status(exc)} (not retrying): {str(exc)[:160]}"
                             + (f"; request dumped to {dump}" if dump else ""))
                return None, last_error
            last_error = _error_dict(exc, attempt, max_retries)
            if on_event:
                more = attempt < max_retries - 1
                on_event(f"{provider} attempt {attempt + 1}/{max_retries} failed: "
                         f"{type(exc).__name__}: {str(exc)[:140]}" + ("; retrying" if more else "; giving up"))
        if attempt < max_retries - 1:
            backoff_sleep(attempt)
    return None, last_error
