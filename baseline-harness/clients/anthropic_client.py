"""Anthropic client — AnthropicPolicy, implements core.types.Policy.

Talks to Claude on AWS Bedrock via the official ``anthropic`` SDK's
``AnthropicBedrock`` client, authenticated with a Bedrock bearer token
(``AWS_BEARER_TOKEN_BEDROCK`` / ``--api-key``). Closed-model sibling of the Gemini
client: append-only ``messages``, the model's response content appended VERBATIM
each turn (so the ``thinking`` block's ``signature`` rides forward and Claude keeps
its reasoning), exact token/USD accounting, a tool-free debrief, and the step-pair
truncation surface.

One Anthropic constraint shapes the thinking/move-channel mode: **forced
tool_choice is INCOMPATIBLE with extended/adaptive thinking** (the API 400s), and
on Sonnet 4.6/Opus 4.6/Haiku 4.5 the API runs WITHOUT thinking unless ``thinking``
is sent. So the two are mutually exclusive and both map onto one knob — the
thinking level:

  - thinking ON  (level low/medium/high/xhigh/max) -> ``thinking:{type:"adaptive"}``
    + ``output_config.effort`` + ``tool_choice:{type:"auto"}`` (+ nudge fallback);
    reasoning CARRIED (signature) — the default.
  - thinking ON, pre-effort model (``ANTHROPIC_MANUAL_THINKING``, e.g. Haiku 4.5) ->
    ``thinking:{type:"enabled", budget_tokens:N}`` (manual extended thinking; these models
    400 on ``output_config.effort``) with N from the
    SHARED level->budget table ``clientutil.THINKING_LEVEL_BUDGETS`` (same nominal budget per
    level across providers), clamped to ``1024 <= N <= max_tokens - 1024``. Same auto-tool
    + nudge channel (forced tool_choice is illegal with manual thinking too).
  - thinking OFF (level "none") -> ``thinking:{type:"disabled"}`` +
    ``tool_choice:{type:"tool"}`` (forced make_move); reasoning NOT carried — a clean
    ablation / forced-tool mode.

The OFF branch sends ``disabled`` explicitly: on **Opus 5** and Sonnet 5 an omitted
``thinking`` means ADAPTIVE, not off. Forced tool_choice suppresses thinking on move
turns, but the tool-free debrief has no forced tool to lean on, so mere omission
would let the debrief silently think. Exception: on Fable 5 ``disabled``
itself 400s (``ANTHROPIC_THINKING_ALWAYS_ON``), so there the OFF branch omits
``thinking``.

Reasoning carry is SELF-CHECKING like Gemini: re-feeding a modified thinking block
400s ("blocks in the latest assistant message cannot be modified"). Prior-turn
thinking is auto-ignored by the API, so evicting whole old step-pairs is safe. Retry/
error handling is the shared ``core.clientutil`` policy: such a validation 400 fails
fast.
"""

from __future__ import annotations

import json
import time
from typing import Any

from core import accounting, clientutil, prompts
from core.types import Move, Turn

# Pricing (USD per 1M tokens), keyed by a distinctive model-id SUBSTRING (Bedrock
# ids look like "global.anthropic.claude-sonnet-4-6"). First-party list rates —
# may not match Bedrock billing (regional endpoints add a ~10% premium). Token
# counts are exact (from usage); only these rates carry the uncertainty. Thinking
# tokens are billed within output, so thoughts_per_1m == output_per_1m.
ANTHROPIC_BEDROCK_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5":   {"input_per_1m": 5.0,  "cached_input_per_1m": 0.50, "output_per_1m": 25.0, "thoughts_per_1m": 25.0},
    "claude-opus-4":   {"input_per_1m": 5.0,  "cached_input_per_1m": 0.50, "output_per_1m": 25.0, "thoughts_per_1m": 25.0},
    "claude-sonnet-5": {"input_per_1m": 3.0, "cached_input_per_1m": 0.30, "output_per_1m": 15.0, "thoughts_per_1m": 15.0},  # standard rate (intro $2/$10 through 2026-08-31)
    "claude-sonnet-4-6": {"input_per_1m": 3.0, "cached_input_per_1m": 0.30, "output_per_1m": 15.0, "thoughts_per_1m": 15.0},
    "claude-sonnet-4-5": {"input_per_1m": 3.0, "cached_input_per_1m": 0.30, "output_per_1m": 15.0, "thoughts_per_1m": 15.0},
    "claude-haiku-4-5": {"input_per_1m": 1.0,  "cached_input_per_1m": 0.10, "output_per_1m": 5.0,  "thoughts_per_1m": 5.0},
    "claude-fable-5":   {"input_per_1m": 10.0, "cached_input_per_1m": 1.00, "output_per_1m": 50.0, "thoughts_per_1m": 50.0},
}

# Context windows (tokens) by model-id substring: 1M for Opus 5, Opus 4.6/4.7/4.8, Sonnet 5,
# Sonnet 4.6, and Fable 5; 200K for Sonnet 4.5 and Haiku 4.5 — each has its OWN row, so it
# does not inherit a 1M family window. NOTE the opus rows are "claude-opus-5" and
# "claude-opus-4" — the 4.x row is NOT a prefix of the 5 id, so every new Opus generation
# needs its own row here and above. --context-budget overrides.
ANTHROPIC_MAX_CONTEXT: dict[str, int] = {
    "claude-opus-5":     1_000_000,
    "claude-opus-4":     1_000_000,
    "claude-sonnet-5":   1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-sonnet-4-5":   200_000,
    "claude-haiku-4-5":    200_000,
    "claude-fable-5":    1_000_000,
}

# Models whose thinking CANNOT be switched off: an explicit `thinking:{"type":"disabled"}`
# 400s with '"thinking.type.disabled" is not supported for this model'. For these the
# thinking-OFF branch OMITS `thinking` and leans on forced tool_choice to suppress it — but
# the tool-free DEBRIEF then unavoidably thinks. Substring match, like the tables above.
ANTHROPIC_THINKING_ALWAYS_ON: tuple[str, ...] = ("claude-fable-5",)

# Models that predate the adaptive-thinking/effort API and take MANUAL extended thinking
# (`thinking:{type:"enabled", budget_tokens:N}`) instead: they 400 on `output_config.effort`
# ("Extra inputs are not permitted"). Substring match, like the tables above. Sonnet 4.5
# shares Haiku 4.5's generation, so it is gated too.
ANTHROPIC_MANUAL_THINKING: tuple[str, ...] = ("claude-haiku-4-5", "claude-sonnet-4-5")


def _make_move_tool() -> dict:
    """Anthropic tool from the neutral MAKE_MOVE_SPEC (its `parameters` JSON Schema
    becomes `input_schema`)."""
    s = prompts.MAKE_MOVE_SPEC
    return {"name": s["name"], "description": s["description"], "input_schema": s["parameters"]}


class AnthropicPolicy:
    """Owns the Claude-on-Bedrock chat: append-only `messages`, verbatim thinking+
    tool_use refeed, the thinking-vs-forced-tool mode, exact accounting, the
    tool-free debrief, and the step-pair truncation surface."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        aws_region: str,
        thinking_level: str | None,
        max_tokens: int,
        timeout: int,
        max_retries: int,
        pricing: dict | None = None,
        include_thoughts: bool = True,
        client=None,
    ):
        self.model = model
        self.thinking_level = thinking_level          # None => thinking off (forced-tool mode)
        self.thinking_on = thinking_level is not None
        self.effort = {"minimal": "low"}.get(thinking_level, thinking_level)  # minimal->low; else identity
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.debrief_retries = max(1, min(2, max_retries))
        self.pricing = pricing or ANTHROPIC_BEDROCK_PRICING
        self.include_thoughts = include_thoughts
        # Provenance for the run summary (mirrors SGLang's fields via getattr).
        self.move_channel = "auto-tool" if self.thinking_on else "forced-tool"
        # Opus 5 (and Sonnet 5 / Fable 5) THINK BY DEFAULT when `thinking` is omitted, so the
        # thinking-OFF branch must say `disabled` out loud instead of just leaving it out —
        # except on the always-on models, where `disabled` itself 400s.
        self.can_disable_thinking = not any(k in model for k in ANTHROPIC_THINKING_ALWAYS_ON)
        # Pre-effort models take manual extended thinking (budget_tokens) — effort 400s there.
        # The budget is resolved (and its feasibility checked) ONCE here, before any session
        # starts: an infeasible --max-tokens/--thinking-level pair must fail loud up front,
        # not 400 mid-run. Recorded in provenance as `thinking_budget`.
        self.manual_thinking = any(k in model for k in ANTHROPIC_MANUAL_THINKING)
        self.thinking_budget: int | None = None
        if self.thinking_on and self.manual_thinking:
            try:
                self.thinking_budget = clientutil.clamp_thinking_budget(
                    thinking_level, max_tokens, floor=1024)
            except ValueError as exc:
                raise SystemExit(f"anthropic manual thinking on {model}: {exc}")
        # `think` count basis (read by core/output.py): thinking_tokens is exact from the
        # API when the usage breakout is present; demoted to "included" (sticky) when a turn
        # thinks without a breakout (Bedrock manual-thinking models); "none" with thinking off.
        self.thoughts_basis = "exact" if self.thinking_on else "none"
        self._last_thoughts_basis = self.thoughts_basis  # this turn's basis (per-turn rendering)

        self.pricing_row = accounting.pricing_for(model, self.pricing)
        self.has_pricing = self.pricing_row is not None
        self.model_max_context = accounting.match_model(model, ANTHROPIC_MAX_CONTEXT)

        if client is not None:
            self.client = client
        elif "anthropic." in model:  # Bedrock id (us./global./eu. + anthropic. prefix)
            from anthropic import AnthropicBedrock  # deferred: heavy SDK import
            self.client = AnthropicBedrock(
                api_key=api_key, aws_region=aws_region, timeout=timeout, max_retries=0,
            )
        else:  # bare claude-* id → direct api.anthropic.com (aws_region unused)
            from anthropic import Anthropic  # deferred: heavy SDK import
            self.client = Anthropic(api_key=api_key, timeout=timeout, max_retries=0)

        self.messages: list[dict] = []
        self._last_tool_use_id: str | None = None
        self.call_count = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.thoughts_tokens = 0
        self.total_tokens = 0
        self.cost_usd = 0.0
        self.elapsed_s = 0.0
        self.last_prompt_tokens = 0
        self.reported_model: str | None = None  # server-reported model id (exact revision)
        self._last_error: dict | None = None
        self.on_retry = None
        self.debug_dir = "."

    # --- conversation construction ---

    def start(self, description: str, state_slice: dict) -> None:
        self.messages.append({"role": "user", "content": prompts.seed_text(description, state_slice)})

    def observe(self, result: dict) -> None:
        block = {"type": "tool_result", "content": json.dumps({"result": result})}
        if self._last_tool_use_id is not None:  # answer the make_move call by id
            block["tool_use_id"] = self._last_tool_use_id
            self.messages.append({"role": "user", "content": [block]})
            self._last_tool_use_id = None
        else:  # no tool call this turn — plain user message
            self.messages.append({"role": "user", "content": json.dumps({"result": result})})

    def add_nudge(self, text: str) -> None:
        # If a make_move tool_use is pending (e.g. emitted with an empty/unparseable action),
        # ANSWER it — an unanswered tool_use 400s the next request. Reuses observe()'s
        # construction; otherwise merge as plain user text.
        if self._last_tool_use_id is not None:
            self.messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": self._last_tool_use_id, "content": text}]})
            self._last_tool_use_id = None
        else:
            self._append_user_text(text)

    def _append_user_text(self, text: str) -> None:
        """Append user text, MERGING into a trailing user message (e.g. a tool_result
        turn) so we never emit two consecutive user messages."""
        if self.messages and self.messages[-1].get("role") == "user":
            content = self.messages[-1]["content"]
            if isinstance(content, list):
                content.append({"type": "text", "text": text})
            else:
                self.messages[-1]["content"] = [
                    {"type": "text", "text": str(content)}, {"type": "text", "text": text}
                ]
        else:
            self.messages.append({"role": "user", "content": text})

    # --- request building ---

    def _thinking_param(self) -> dict:
        """The `thinking` param for the thinking-ON regimes. Adaptive models take a display
        knob; manual (pre-effort) models take the token budget resolved at construction
        (shared level table, 1024 <= budget < max_tokens — feasibility already checked)."""
        if self.manual_thinking:
            return {"type": "enabled", "budget_tokens": self.thinking_budget}
        return {"type": "adaptive", "display": "summarized" if self.include_thoughts else "omitted"}

    def _cache_messages(self) -> list:
        """`self.messages` with a cache_control breakpoint on the LAST block of the LAST message,
        so the append-only prefix is cached for the next turn (read ~0.1x). NEVER mutates the
        stored history (the verbatim reasoning-carry blocks) — the marker rides only on the
        request copy. Skips safely if the tail isn't a plain str/dict (e.g. SDK objects), in
        which case the static system+tools breakpoints still cache."""
        if not self.messages:
            return self.messages
        msgs = list(self.messages)
        last = dict(msgs[-1])
        content = last.get("content")
        bp = {"type": "ephemeral"}
        if isinstance(content, str):
            last["content"] = [{"type": "text", "text": content, "cache_control": bp}]
        elif isinstance(content, list) and content and isinstance(content[-1], dict):
            blocks = list(content)
            blocks[-1] = {**blocks[-1], "cache_control": bp}
            last["content"] = blocks
        else:
            return self.messages
        msgs[-1] = last
        return msgs

    def _turn_kwargs(self) -> dict:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # Prompt caching (Bedrock has no auto-caching -> explicit cache_control). Static
            # breakpoints on system + tools, plus a rolling one on the last message (_cache_messages),
            # cache the append-only prefix: read at ~0.1x, write only each turn's delta at 1.25x.
            "system": [{"type": "text", "text": prompts.SYSTEM_INSTRUCTION,
                        "cache_control": {"type": "ephemeral"}}],
            "tools": [{**_make_move_tool(), "cache_control": {"type": "ephemeral"}}],
            "messages": self._cache_messages(),
        }
        if self.thinking_on:
            kwargs["thinking"] = self._thinking_param()
            if not self.manual_thinking:  # manual (budget_tokens) models 400 on output_config
                kwargs["output_config"] = {"effort": self.effort}
            # forced tool is illegal with thinking; disable_parallel_tool_use keeps Claude to ONE
            # make_move per turn (else extra tool_use blocks go unanswered -> next request 400s).
            kwargs["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        else:
            # Say `disabled` explicitly: on Opus 5 / Sonnet 5 an OMITTED `thinking` means ADAPTIVE
            # (thinking on), so omission would make the "none" ablation depend on forced tool_choice
            # implicitly suppressing it. No `output_config.effort` here — on Opus 5 `disabled` is
            # only legal at effort <= high, and the unset default is high.
            if self.can_disable_thinking:
                kwargs["thinking"] = {"type": "disabled"}
            # disable_parallel_tool_use keeps Claude to ONE make_move even when forced (parity with
            # the thinking branch); extra tool_use blocks would go unanswered -> next request 400s.
            kwargs["tool_choice"] = {"type": "tool", "name": "make_move", "disable_parallel_tool_use": True}
        return kwargs

    def _debrief_kwargs(self) -> dict:
        # No cache_control here: the debrief drops `tools` and swaps in DEBRIEF_SYSTEM_INSTRUCTION, so
        # its prefix diverges from the turn prefix at byte 0 and gets ZERO cache reads. A breakpoint
        # would only bill a 1.25x cache WRITE that nothing ever reads (one debrief per run).
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": prompts.DEBRIEF_SYSTEM_INSTRUCTION}],
            "messages": self.messages,
        }
        if self.thinking_on:
            kwargs["thinking"] = self._thinking_param()
            if not self.manual_thinking:  # manual (budget_tokens) models 400 on output_config
                kwargs["output_config"] = {"effort": self.effort}
        elif self.can_disable_thinking:
            # MUST be explicit here: the debrief carries no `tools`, so there is no forced
            # tool_choice to implicitly suppress thinking, and on Opus 5 an omitted `thinking`
            # is ADAPTIVE — the "none" ablation would silently think and bill the thinking
            # as output.
            kwargs["thinking"] = {"type": "disabled"}
        return kwargs

    # --- LLM call ---

    def _create(self, build_kwargs, max_retries: int, timeout: int | None = None):
        kwargs = build_kwargs()
        opts = {} if timeout is None else {"timeout": timeout}
        resp, err = clientutil.run_request(
            lambda: self.client.messages.create(**kwargs, **opts),
            provider="Anthropic", max_retries=max_retries, request=kwargs,
            debug_dir=self.debug_dir, call_count=self.call_count, on_event=self.on_retry,
        )
        self._last_error = err
        return resp

    @staticmethod
    def _extract(resp) -> tuple[str | None, str | None, str, bool, str | None]:
        """Return (action, tool_use_id, reasoning_summary, has_signature, stop_reason)."""
        action = tool_use_id = None
        summary_parts, has_sig = [], False
        for block in getattr(resp, "content", None) or []:
            btype = getattr(block, "type", None)
            if btype == "tool_use" and getattr(block, "name", None) == "make_move":
                args = getattr(block, "input", None)
                if isinstance(args, dict):
                    action = str(args.get("action") or "").strip() or None  # JSON null -> None, not "None"
                tool_use_id = getattr(block, "id", None)
            elif btype == "thinking":
                if getattr(block, "signature", None):
                    has_sig = True
                text = getattr(block, "thinking", None)
                if text:
                    summary_parts.append(text.strip())
            elif btype == "redacted_thinking":
                has_sig = True  # opaque carried reasoning, no readable text
        return action, tool_use_id, " ".join(p for p in summary_parts if p), has_sig, getattr(resp, "stop_reason", None)

    def _account(self, usage, *, has_thinking: bool = False) -> tuple[dict, float | None]:
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out_total = int(getattr(usage, "output_tokens", 0) or 0)  # INCLUDES thinking (billed as output)
        # Thinking-count basis, by KEY PRESENCE (mirrors the SGLang client): effort-capable
        # models report output_tokens_details.thinking_tokens (exact, disjoint), but Bedrock
        # manual-thinking models (e.g. Haiku 4.5) return NO breakout
        # even when the turn visibly thought — there `out` keeps the thinking folded in and the
        # basis is "included" (rendered `out + think N`), never a fake exact-0 split.
        details = getattr(usage, "output_tokens_details", None)
        counted = getattr(details, "thinking_tokens", None) if details is not None else None
        if counted is not None:
            thinking, basis = int(counted or 0), "exact"
        elif has_thinking:
            thinking, basis = 0, "included"  # folded into `out`; not separately counted
        else:
            thinking, basis = 0, ("exact" if self.thinking_on else "none")  # truly no thinking
        self._last_thoughts_basis = basis
        if basis == "included":  # weaker claim than exact — sticky for the run summary
            self.thoughts_basis = "included"
        visible = max(0, out_total - thinking)  # report out/think DISJOINT (Gemini convention)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_create = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        prompt = inp + cache_read + cache_create   # full prompt size (inp=uncached + cached read + write)
        counts = {"prompt": prompt, "cached": cache_read, "output": visible, "thoughts": thinking,
                  "total": prompt + out_total}
        # Thinking bills at the output rate and already sits inside out_total; pricing the
        # disjoint visible+thinking both at the output rate reproduces out_total*rate exactly
        # (thoughts_per_1m == output_per_1m in the table) — no double count.
        cost = accounting.compute_cost(self.model, prompt, cache_read, visible, thinking, self.pricing)
        if cost is not None and cache_create and self.pricing_row:
            # compute_cost billed cache_create at the 1x input rate; a 5-min cache WRITE is 1.25x,
            # so add the 0.25x premium (cache_read is already billed at the 0.1x cached rate).
            cost += cache_create * 0.25 * self.pricing_row["input_per_1m"] / 1_000_000
        self.call_count += 1
        self.prompt_tokens += prompt
        self.cached_tokens += cache_read
        self.output_tokens += visible
        self.thoughts_tokens += thinking
        self.total_tokens += counts["total"]
        if prompt:  # the truncation trigger — server-reported prompt size
            self.last_prompt_tokens = prompt
        if cost is not None:
            self.cost_usd += cost
        return counts, cost

    def generate_move(self) -> Move:
        start = time.time()
        resp = self._create(self._turn_kwargs, self.max_retries)
        elapsed = time.time() - start
        self.elapsed_s += elapsed
        if resp is None:
            return Move(None, "", False, None, clientutil.zero_usage(), None, elapsed, error=self._last_error)

        if self.reported_model is None:  # provenance: the exact revision behind the alias
            self.reported_model = getattr(resp, "model", None)
        # CRITICAL: append the response content VERBATIM (thinking + signature +
        # tool_use) so reasoning rides forward. Never reconstruct or type-filter it.
        self.messages.append({"role": "assistant", "content": getattr(resp, "content", [])})
        action, tool_use_id, summary, has_sig, finish = self._extract(resp)
        self._last_tool_use_id = tool_use_id
        counts, cost = self._account(getattr(resp, "usage", None),
                                     has_thinking=bool(has_sig or summary))
        return Move(action, summary, has_sig, finish, counts, cost, elapsed,
                    thoughts_basis=self._last_thoughts_basis,
                    continuity="verified" if has_sig else None)  # API-validated signature

    def debrief(self) -> str | None:
        # add_nudge (not _append_user_text): on bench_failure/server_protocol stops a make_move
        # tool_use is still pending, and an unanswered tool_use 400s the debrief request.
        self.add_nudge(prompts.DEBRIEF_PROMPT)
        start = time.time()
        resp = self._create(self._debrief_kwargs, self.debrief_retries, timeout=2 * self.timeout)
        self.elapsed_s += time.time() - start
        if resp is None:
            return None
        blocks = getattr(resp, "content", None) or []
        self._account(getattr(resp, "usage", None),
                      has_thinking=any(getattr(b, "type", None) in ("thinking", "redacted_thinking")
                                       for b in blocks))
        text = " ".join(
            getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text"
        ).strip()
        return text or "(empty debrief)"

    # --- truncation surface (used only by core/history.py) ---

    def _assistant_indices(self) -> list[int]:
        return [i for i, m in enumerate(self.messages) if m.get("role") == "assistant"]

    def turns(self):
        """Step-pair units over `messages`, oldest -> newest. The head (the pinned
        user task before the first assistant) is NOT a unit. A unit = an assistant
        message + the tool_result/user message(s) up to the next assistant. The last
        unit is ACTIVE (its thinking blocks must stay intact -> never evicted)."""
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
        (not incl.) the second. Pinned head + active (latest) unit are never touched.
        No-op when only head + active remain. (Prior-turn thinking is auto-ignored by
        the API, so dropping whole old pairs needs no continuity bookkeeping.)"""
        aidx = self._assistant_indices()
        if len(aidx) <= 1:
            return
        del self.messages[aidx[0]:aidx[1]]
