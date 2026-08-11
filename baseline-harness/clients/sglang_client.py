"""SGLang client — SglangPolicy, implements core.types.Policy.

Talks to an SGLang OpenAI-compatible Chat Completions endpoint (open models served
on a server).
Append-only `messages`,
the forced/named make_move tool call (`--move-channel`; `guided-json` is selectable for builds
whose grammar backend rejects forced tool decoding — a forced-tool grammar 400 fails loud with
that remedy rather than switching channel mid-run), the
`reasoning`-field round-trip (the open-model analog of Gemini thought signatures),
4xx fail-fast + dump. Transport is stdlib HTTP with SSE streaming, so the socket read
timeout acts as a per-chunk IDLE timeout (`--stream-idle-timeout`): a silent/wedged server
is caught and retried, while a slow-but-progressing generation runs to completion. The
streamed deltas are reassembled into the same response shape a non-streamed call returns.

Parser-agnostic: the `--tool-call-parser` / `--reasoning-parser` are SGLang
server-launch flags (the operator's job), NOT this client's. We only use stable
framework features: `tool_choice` named fn, `response_format` json_schema,
`reasoning_content`, SSE streaming (`stream` + `stream_options.include_usage`),
`/v1/models`->`max_model_len`.

Behind the Policy truncation surface:
- `model_max_context` from the server (`/v1/models` `max_model_len`) -> feeds the
  shared truncation budget; no per-model table (unlike closed models).
- step-pair truncation surface over `messages` (head = system+task pinned, unit =
  assistant + its result, latest = active) — core/history.py reused unchanged.
- a `/chat/completions` reasoning round-trip probe: detect a template that silently
  strips carried reasoning so `has_continuity` is reported HONESTLY (not a no-op claim).
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Callable

from core import clientutil, history, prompts
from core.types import Move, Turn


# /get_server_info returns SGLang's full server_args verbatim — INCLUDING credentials
# (api_key, admin_api_key, ...). Published run artifacts must never carry those, so only
# these result-affecting launch fields are kept. Allowlist, not redaction: unknown/future
# fields stay out by default instead of leaking until someone notices.
SGLANG_SERVER_INFO_FIELDS: tuple[str, ...] = (
    "version", "model_path", "served_model_name", "tokenizer_path", "tokenizer_mode",
    "chat_template", "completion_template", "revision", "dtype", "quantization",
    "kv_cache_dtype", "context_length", "max_model_len", "tool_call_parser",
    "reasoning_parser", "grammar_backend", "random_seed", "tp_size", "dp_size",
    "speculative_algorithm", "speculative_draft_model_path",
)


class ChatError(RuntimeError):
    def __init__(self, code: int | None, body: str):
        self.code = code
        self.body = body
        super().__init__(f"HTTP {code}: {body}" if code else f"network: {body}")


def _post_json(url: str, payload: dict, headers: dict, timeout: float | None) -> dict:
    # Stream the response so urllib's socket timeout acts as a per-CHUNK IDLE timeout: a silent
    # (wedged) server trips `timeout` on a stalled read and is retried, while a generation that
    # keeps emitting tokens resets the clock each chunk and runs to completion however long. The
    # SSE deltas are reassembled into the SAME dict a non-streamed call returns (clientutil), so
    # every downstream reader is byte-identical. include_usage puts full usage on the final chunk.
    body = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with clientutil.urlopen_no_redirect(req, timeout=timeout) as resp:
            return clientutil.accumulate_chat_stream(resp, require_usage=True)
    except urllib.error.HTTPError as exc:                # pre-stream 4xx/5xx (incl. forced-tool 400)
        raise ChatError(exc.code, exc.read().decode("utf-8", "replace")) from exc
    except clientutil.StreamError as exc:                # mid-stream error chunk
        raise ChatError(exc.code, str(exc.body)) from exc
    except (TimeoutError, socket.timeout) as exc:        # idle read timeout mid-stream (wedge)
        raise ChatError(None, "timed out") from exc
    except urllib.error.URLError as exc:                 # connect-phase failure (timeout/network)
        raise ChatError(None, str(exc.reason)) from exc


def _get_json(url: str, headers: dict, timeout: float | None) -> dict:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with clientutil.urlopen_no_redirect(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ChatError(exc.code, exc.read().decode("utf-8", "replace")) from exc
    except urllib.error.URLError as exc:
        raise ChatError(None, str(exc.reason)) from exc


def _make_move_tool() -> dict:
    s = prompts.MAKE_MOVE_SPEC
    return {"type": "function", "function": {
        "name": s["name"], "description": s["description"], "parameters": s["parameters"]}}


def _action_schema() -> dict:
    s = prompts.MAKE_MOVE_SPEC
    return {**s["parameters"], "additionalProperties": False}


def resolve_model_and_window(data: dict, model: str | None, *, select=None) -> tuple[str, int | None]:
    """From a GET /v1/models response pick (model_id, max_model_len). If `model` is given,
    match it; with exactly one served model auto-select it; with several, call `select(ids)`
    to choose (or raise if no selector is provided)."""
    entries = [e for e in (data.get("data") or []) if isinstance(e, dict) and e.get("id")]
    ids = [str(e["id"]) for e in entries]
    if model:
        entry = next((e for e in entries if str(e["id"]) == model), None)
        if entry is None:  # an unserved --model must fail loud, not silently disable truncation
            raise SystemExit(f"--model {model!r} is not served at /v1/models. served: "
                             f"{', '.join(sorted(ids)) or '(none)'}")
    elif len(ids) == 1:
        entry = entries[0]
    elif not ids:
        raise SystemExit("no models served at /v1/models — is the server up?")
    elif select is not None:
        chosen_id = select(sorted(ids))
        entry = next((e for e in entries if str(e["id"]) == chosen_id), None)
        if entry is None:  # selector returned an id not in the served list — fail loud
            raise SystemExit(f"selected model {chosen_id!r} is not served at /v1/models. "
                             f"served: {', '.join(sorted(ids))}")
    else:
        raise SystemExit(f"multiple models served; pass --model. served: {', '.join(sorted(ids))}")
    chosen = model or str(entry["id"])
    window = None
    if entry is not None and entry.get("max_model_len"):
        try:
            window = int(entry["max_model_len"])
        except (TypeError, ValueError):
            window = None
    return chosen, window


class SglangPolicy:
    """Owns the OpenAI-compatible chat: append-only `messages`, the move channel
    (forced-tool by default; guided-json/auto-tool selectable per run), the reasoning round-trip,
    exact token accounting (no USD), and the step-pair truncation surface."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str | None,
        timeout: int = 60,                  # one-time /v1/models discovery GET ceiling (NOT generation)
        stream_idle_timeout: int = 180,     # per-chunk idle timeout for the streaming generation
        max_retries: int,
        max_tokens: int = 32000,            # universal output cap (parity across providers)
        move_channel: str = "forced-tool",   # forced-tool | guided-json | auto-tool
        preserve_reasoning: bool = True,
        seed: int | None = None,
        debug_dir: str = ".",
        select_model: Callable | None = None,  # called with served ids when >=2 and no --model
        http_post: Callable = _post_json,
        http_get: Callable = _get_json,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout                        # /v1/models discovery GET only
        self.stream_idle_timeout = stream_idle_timeout  # streaming generation + probe (idle, per-chunk)
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.debrief_retries = max(1, min(2, max_retries))
        self.move_channel = move_channel
        self.preserve_reasoning = preserve_reasoning
        self.include_thoughts = preserve_reasoning   # display parity with Gemini
        self.seed = seed
        self.debug_dir = debug_dir
        self._http_post = http_post
        self._http_get = http_get

        # Startup discovery: model id + context window from the server.
        headers = {"Authorization": f"Bearer {api_key or '-'}"}
        data = http_get(f"{self.base_url}/models", headers, timeout)
        self.model, self.model_max_context = resolve_model_and_window(data, model, select=select_model)
        # Provenance: the raw /v1/models entry for the chosen model, plus the server's own
        # launch configuration (SGLang /get_server_info sits at the root, not under /v1) —
        # model revision, tokenizer/template, dtype, parsers all change results. Best-effort:
        # a server without the endpoint (or any error) just records None. The response is
        # NEVER stored verbatim — it embeds server_args credentials (api_key/admin_api_key);
        # only the SGLANG_SERVER_INFO_FIELDS allowlist survives into the artifact.
        self.models_entry = next(
            (e for e in (data.get("data") or [])
             if isinstance(e, dict) and str(e.get("id")) == self.model), None)
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        try:
            info = http_get(f"{root}/get_server_info", headers, timeout)
            self.server_info = {k: info[k] for k in SGLANG_SERVER_INFO_FIELDS if k in info} or None
        except Exception:
            self.server_info = None

        self.has_pricing = False                      # self-hosted -> USD n/a
        self._reasoning_roundtrips: bool | None = None  # set by the reasoning round-trip probe
        self.thoughts_basis: str | None = None        # run-level (sticky): exact|included|none
        self._last_thoughts_basis = "none"             # this turn's basis (per-turn rendering)

        self.messages: list[dict] = []
        self._last_tool_call_id: str | None = None
        self.call_count = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.thoughts_tokens = 0
        self.total_tokens = 0
        self.cost_usd = 0.0
        self.elapsed_s = 0.0
        self.last_prompt_tokens = 0
        self.reported_model: str | None = None  # model id echoed on responses (provenance)
        self._last_error: dict | None = None
        self.on_retry = None
        self.on_note = None   # neutral diagnostics (probe verdict) — not warnings

    # --- conversation construction ---

    def start(self, description: str, state_slice: dict) -> None:
        self.messages.append({"role": "user", "content": prompts.seed_text(description, state_slice)})

    def observe(self, result: dict) -> None:
        payload = json.dumps({"result": result})
        if self._last_tool_call_id is not None:  # proper tool protocol: answer the call by id
            self.messages.append(
                {"role": "tool", "tool_call_id": self._last_tool_call_id, "content": payload}
            )
            self._last_tool_call_id = None
        else:  # guided-JSON / no-tool-call path
            self.messages.append({"role": "user", "content": payload})

    def add_nudge(self, text: str) -> None:
        # Answer a pending make_move tool call (empty/unparseable action) to keep a valid tool
        # chain; otherwise a plain user nudge. Reuses observe()'s construction.
        if self._last_tool_call_id is not None:
            self.messages.append({"role": "tool", "tool_call_id": self._last_tool_call_id, "content": text})
            self._last_tool_call_id = None
        else:
            self.messages.append({"role": "user", "content": text})

    # --- payloads ---

    def _turn_payload(self) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": prompts.SYSTEM_INSTRUCTION}] + self.messages,
            "max_tokens": self.max_tokens,   # universal output cap (parity across providers)
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.move_channel == "guided-json":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "make_move", "schema": _action_schema()},
            }
        else:
            payload["tools"] = [_make_move_tool()]
            payload["tool_choice"] = (
                {"type": "function", "function": {"name": "make_move"}}
                if self.move_channel == "forced-tool" else "auto"
            )
        return payload

    def _debrief_payload(self) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": prompts.DEBRIEF_SYSTEM_INSTRUCTION}] + self.messages,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload

    # --- LLM call ---

    def _generate(self, build_payload, *, max_retries=None, timeout=None) -> dict | None:
        retries = self.max_retries if max_retries is None else max_retries
        timeout = self.stream_idle_timeout if timeout is None else timeout  # streaming POST -> idle timeout
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key or '-'}"}
        url = f"{self.base_url}/chat/completions"
        self._last_error = None
        for attempt in range(retries):
            payload = build_payload()
            try:
                return self._http_post(url, payload, headers, timeout)
            except ChatError as exc:
                body = (exc.body or "").lower()
                # Forced/auto tool choice rejected by the grammar backend (HTTP 400). We do NOT
                # silently switch to guided-JSON mid-run — that would make this run's move channel
                # differ from the others (a hidden cross-run fairness confound). Fail loud with the
                # remedy instead; the operator picks one channel for the whole run.
                if (exc.code == 400 and self.move_channel in ("forced-tool", "auto-tool")
                        and ("tool" in body or "function" in body) and self.on_retry):
                    self.on_retry(
                        "forced-tool rejected by the grammar backend (HTTP 400) — this server build "
                        "needs guided-JSON. Re-run with --move-channel guided-json."
                    )
                if not clientutil.is_retryable(exc):  # deterministic client error -> fail fast
                    dump = clientutil.dump_request(self.debug_dir, "SGLang", exc, payload, self.call_count)
                    self._record_error(exc, attempt, retries, fatal=True, dump=dump)
                    return None
                self._record_error(exc, attempt, retries)  # transient (5xx/429/network) -> retry
            except Exception as exc:
                # Same classification as the ChatError path: programming errors (TypeError &
                # friends, per clientutil.is_retryable) fail fast instead of burning the whole
                # backoff budget to fail identically N times.
                if not clientutil.is_retryable(exc):
                    self._record_error(exc, attempt, retries, fatal=True)
                    return None
                self._record_error(exc, attempt, retries)
            if attempt < retries - 1:
                clientutil.backoff_sleep(attempt)
        return None

    def _record_error(self, exc, attempt, retries, *, fatal: bool = False, dump: str | None = None) -> None:
        self._last_error = {"type": type(exc).__name__, "message": str(exc),
                            "attempt": attempt + 1, "max_retries": retries}
        if dump:
            self._last_error["payload_dump"] = dump
        if not self.on_retry:
            return
        if fatal:
            code = getattr(exc, "code", None)
            self.on_retry(f"SGLang HTTP {code} client error (not retrying): {str(exc)[:160]}"
                          + (f"; request dumped to {dump}" if dump else ""))
        else:
            more = attempt < retries - 1
            self.on_retry(f"SGLang attempt {attempt + 1}/{retries} failed: "
                          f"{type(exc).__name__}: {str(exc)[:140]}" + ("; retrying" if more else "; giving up"))

    @staticmethod
    def _message(data: dict) -> dict:
        choices = data.get("choices") or [{}]
        return (choices[0] or {}).get("message") or {}

    @staticmethod
    def _extract_action(msg: dict) -> tuple[str | None, str | None]:
        """Action + tool_call_id. Prefer the make_move tool call; else parse an
        {"action": ...} JSON object from message content (guided-JSON path)."""
        for call in msg.get("tool_calls") or []:
            fn = (call or {}).get("function") or {}
            if fn.get("name") != "make_move":
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (TypeError, ValueError):
                args = {}
            if not isinstance(args, dict):  # guard non-object JSON args
                args = {}
            return str(args.get("action") or "").strip() or None, call.get("id")  # JSON null -> None
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):  # guard non-object JSON (list/str/num)
                    return str(parsed.get("action") or "").strip() or None, None  # JSON null -> None
            except (TypeError, ValueError):
                pass
        return None, None

    def _account(self, usage: dict, has_reasoning: bool = False) -> tuple[dict, None]:
        prompt = int(usage.get("prompt_tokens") or 0)
        output = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + output))
        cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
        # Thinking accounting. Detect server support by KEY PRESENCE, not value: SGLang
        # usually returns completion_tokens_details=None, but a build that supports it reports
        # reasoning_tokens (0 there means "thought nothing", which differs from unsupported).
        # `completion_tokens` always INCLUDES reasoning, so:
        #  - key present  -> exact: report DISJOINT, out = completion - reasoning (= visible),
        #    matching the closed clients so out + think == completion (no double-count in the
        #    "out N think M" display).
        #  - absent but the model DID reason (reasoning_content present) -> can't split; keep
        #    out = completion and show it MERGED ("out + think N").
        #  - no reasoning at all -> none.
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
            thoughts, basis = int(details["reasoning_tokens"]), "exact"
            output = max(0, output - thoughts)  # disjoint: out = visible (like Gemini/Anthropic/OpenAI)
        elif has_reasoning:
            thoughts, basis = 0, "included"  # folded into `output`; not separately counted
        else:
            thoughts, basis = 0, "none"
        self._last_thoughts_basis = basis  # this turn's basis (per-turn rendering)
        if basis == "included":            # weaker claim than exact — make it sticky
            self.thoughts_basis = "included"
        elif basis == "exact" and self.thoughts_basis != "included":
            self.thoughts_basis = "exact"
        elif self.thoughts_basis is None:  # no reasoning observed yet -> "none", like the closed clients
            self.thoughts_basis = "none"
        counts = {"prompt": prompt, "cached": cached, "output": output, "thoughts": thoughts, "total": total}
        self.call_count += 1
        self.prompt_tokens += prompt
        self.cached_tokens += cached
        self.output_tokens += output
        self.thoughts_tokens += thoughts
        self.total_tokens += total
        if prompt:  # the truncation trigger — server's reported prompt size
            self.last_prompt_tokens = prompt
        return counts, None  # self-hosted -> no USD

    def generate_move(self) -> Move:
        start = time.time()
        data = self._generate(self._turn_payload)
        elapsed = time.time() - start
        self.elapsed_s += elapsed
        if data is None:
            return Move(None, "", False, None, clientutil.zero_usage(), None, elapsed, error=self._last_error)

        if self.reported_model is None:  # provenance: the id the server echoes on responses
            self.reported_model = data.get("model")
        msg = self._message(data)
        action, tool_call_id = self._extract_action(msg)
        # Echo reasoning back under the SAME field the server emitted it on — most SGLang builds
        # use `reasoning_content` (the field the chat template re-injects); some vLLM
        # builds use `reasoning`. Sending it under the wrong key = silently dropped.
        reasoning, reasoning_key = "", "reasoning_content"
        for key in ("reasoning_content", "reasoning"):
            val = msg.get(key)
            if val:
                reasoning = val.strip() if isinstance(val, str) else str(val)
                reasoning_key = key
                break

        # Append the assistant turn VERBATIM; re-send the reasoning so the model
        # appends to its trace rather than re-thinking (the open-model analog of a
        # thought signature).
        assistant: dict = {"role": "assistant", "content": msg.get("content") or ""}
        # Record only the make_move call we answer by id (observe/add_nudge reply to one id);
        # keeping unanswered parallel calls in history 400s the next request. An id-less call
        # can't be answered at all, so no tool_calls are stored then (content/reasoning still
        # carry) rather than leaving a call dangling.
        if msg.get("tool_calls") and tool_call_id is not None:
            assistant["tool_calls"] = [tc for tc in msg["tool_calls"] if tc.get("id") == tool_call_id]
        carried = bool(self.preserve_reasoning and reasoning)
        if carried:
            assistant[reasoning_key] = reasoning
        self.messages.append(assistant)
        self._last_tool_call_id = tool_call_id

        # Tell _account whether the model reasoned this turn, so it can mark the count
        # "included" (folded into output) when the server doesn't report reasoning_tokens.
        counts, cost = self._account(data.get("usage") or {}, has_reasoning=bool(reasoning))
        finish = ((data.get("choices") or [{}])[0] or {}).get("finish_reason")
        # Continuity is reported in three EXPLICIT states (never a positive claim on an
        # unproven carry): probe True -> verified; probe never ran / errored -> unverified;
        # probe proved the template strips carried reasoning on input -> stripped.
        # has_continuity keeps its historical meaning (carried and not disproven).
        if not carried:
            continuity = None
        elif self._reasoning_roundtrips is True:
            continuity = "verified"
        elif self._reasoning_roundtrips is None:
            continuity = "unverified"
        else:
            continuity = "stripped"
        has_continuity = bool(carried and self._reasoning_roundtrips is not False)
        return Move(action, reasoning, has_continuity, finish, counts, cost, elapsed,
                    thoughts_basis=self._last_thoughts_basis, continuity=continuity)

    def debrief(self) -> str | None:
        self.add_nudge(prompts.DEBRIEF_PROMPT)
        start = time.time()
        data = self._generate(self._debrief_payload, max_retries=self.debrief_retries,
                               timeout=self.stream_idle_timeout)  # idle timeout (2x is meaningless when streaming)
        # Context-overflow recovery, mirroring the harness's move path: a run whose move just
        # overflowed would otherwise ALWAYS lose its debrief too (the prompt only grew). The
        # debrief nudge is already appended — evict deficit-sized oldest turns and re-send.
        rounds = 0
        while (data is None and clientutil.is_context_overflow(self._last_error)
               and rounds < history.MAX_OVERFLOW_ROUNDS):
            evicted = history.evict_for_overflow(self, self._last_error, round_idx=rounds)
            if not evicted:                   # nothing evicted — cannot recover
                break
            rounds += 1
            if self.on_retry:
                self.on_retry(f"debrief context overflow — evicted {evicted} oldest turn(s), "
                              f"retrying ({rounds})")
            data = self._generate(self._debrief_payload, max_retries=self.debrief_retries,
                                   timeout=self.stream_idle_timeout)
        self.elapsed_s += time.time() - start
        if data is None:
            return None
        self._account(data.get("usage") or {})
        return (self._message(data).get("content") or "").strip() or "(empty debrief)"

    # --- reasoning round-trip probe ---

    def probe_reasoning_roundtrip(self) -> None:
        """Does the server actually feed carried reasoning back into the prompt?

        Send two minimal /chat/completions requests that differ ONLY by a fat
        reasoning block (carried under both `reasoning_content` and `reasoning`, since
        builds vary in which key they inject) on a prior assistant message, and compare the
        server's reported `prompt_tokens`. If the one carrying reasoning isn't
        bigger, the server strips it on input => carry is a silent no-op =>
        `has_continuity` is reported False. Uses /chat/completions (the endpoint we
        know works), not a guessed /tokenize. Best-effort: on error -> unverified.

        This IS the 2-turn round-trip check, run once at startup.
        """
        if not self.preserve_reasoning:
            return
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key or '-'}"}
        url = f"{self.base_url}/chat/completions"
        sentinel = "SENTINEL_REASONING_TOKEN " * 200  # a few hundred tokens
        # MIRROR the move channel's continuation shape. Some templates (e.g. DeepSeek V4's)
        # keep reasoning only WITHIN a tool-call chain and DROP it after a fresh USER turn,
        # so a user-trailing probe is a false negative for the forced-tool loop (which
        # continues via tool messages). Build the prior assistant + the SAME kind of
        # follower the loop appends.
        extra: dict = {}
        if self.move_channel == "guided-json":
            asst = {"role": "assistant", "content": json.dumps({"action": "1"})}
            follow = {"role": "user", "content": "continue"}
        else:
            call = {"id": "probe_call", "type": "function",
                    "function": {"name": "make_move", "arguments": json.dumps({"action": "1"})}}
            asst = {"role": "assistant", "content": "", "tool_calls": [call]}
            follow = {"role": "tool", "tool_call_id": "probe_call", "content": json.dumps({"result": "ok"})}
            extra = {"tools": [_make_move_tool()]}
        base = [{"role": "user", "content": "ping"}, asst, follow]
        # Carry the sentinel under BOTH reasoning keys: most SGLang builds echo/consume
        # `reasoning_content`, some vLLM builds use `reasoning` (the same fork we handle
        # on the generation path). We don't know which the server's chat template injects
        # until it runs, so inflate both — whichever it feeds back grows prompt_tokens.
        withr = [base[0], {**asst, "reasoning_content": sentinel, "reasoning": sentinel}, follow]
        report = self.on_note or self.on_retry  # neutral channel; the verdict isn't a warning

        def prompt_tokens(msgs):
            payload = {"model": self.model, "messages": msgs, "max_tokens": 1, **extra}
            data = self._http_post(url, payload, headers, self.stream_idle_timeout)  # streams now
            return int((data.get("usage") or {}).get("prompt_tokens") or 0)

        last_exc = None
        for _ in range(2):  # one retry: a startup hiccup must not demote the whole run to unverified
            try:
                n0 = prompt_tokens(base)
                n1 = prompt_tokens(withr)
                break
            except Exception as exc:
                last_exc = exc
        else:
            self._reasoning_roundtrips = None
            if report:
                report(f"reasoning round-trip probe skipped ({type(last_exc).__name__}); "
                       "continuity reported UNVERIFIED")
            return
        self._reasoning_roundtrips = bool(n0 and n1 and (n1 - n0) > 20)
        if report:
            verdict = "PRESERVES" if self._reasoning_roundtrips else "STRIPS"
            report(f"reasoning round-trip probe ({self.move_channel}): server {verdict} "
                   f"carried reasoning_content (prompt_tokens {n0} -> {n1})")

    # --- truncation surface (used only by core/history.py) ---

    def _assistant_indices(self) -> list[int]:
        return [i for i, m in enumerate(self.messages) if m.get("role") == "assistant"]

    def turns(self):
        """Step-pair units over `messages`, oldest -> newest. Head (everything before
        the first assistant: the system is separate + the pinned user task) is NOT a
        unit. A unit = an assistant message + the tool/user result(s) up to the next
        assistant. The last unit is ACTIVE."""
        aidx = self._assistant_indices()
        if not aidx:
            return []
        bounds = aidx + [len(self.messages)]
        n = len(aidx)
        out = []
        for k in range(n):
            chunk = self.messages[bounds[k]:bounds[k + 1]]
            out.append(Turn(payload=chunk, is_active=(k == n - 1), est_tokens=clientutil.est_tokens(chunk)))
        return out

    def evict_oldest_turn(self) -> None:
        """Drop the oldest completed step-pair: from the first assistant message up to
        (not incl.) the second. The pinned head (before the first assistant) and the
        active (latest) unit are never touched. No-op when only head + active remain."""
        aidx = self._assistant_indices()
        if len(aidx) <= 1:
            return
        del self.messages[aidx[0]:aidx[1]]
