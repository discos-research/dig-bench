"""OpenAI client — OpenAIPolicy, implements core.types.Policy.

Talks to OpenAI reasoning models (gpt-5.5 / o-series) via the official ``openai`` SDK's
**Responses API** (``client.responses.create``) — NOT Chat Completions (a different
surface from the SGLang OpenAI-compatible Chat Completions client).
Closed-reasoning-model sibling of the Anthropic client: append-only ``input`` (a list of
typed Responses items), the model's ``response.output`` appended VERBATIM each turn (so the
reasoning item's ``encrypted_content`` rides forward), exact token/USD accounting, a
tool-free debrief, and the step-pair truncation surface.

Reasoning carry uses **stateless** mode — ``store=False`` +
``include=["reasoning.encrypted_content"]`` returns an encrypted reasoning item; pass the
whole ``response.output`` (reasoning item + function_call) back UNCHANGED, then a
``function_call_output`` item. The reasoning item must accompany its function_call (the API
400s / degrades otherwise) → self-checking like Gemini/Anthropic; such a 400 fails fast under
the shared ``core.clientutil`` policy. Unlike Anthropic, OpenAI reasoning models keep reasoning
WITH forced tool calls, so the move channel is **forced make_move** (parity with Gemini/SGLang).
``encrypted_content`` is a dedicated field separate from the visible ``summary`` — the cleanest
carry of the four. Seed: the Responses API has no seed parameter (warned no-op, like Anthropic).
"""

from __future__ import annotations

import json
import time
from typing import Any

from core import accounting, clientutil, prompts
from core.types import Move, Turn

# Pricing (USD per 1M tokens), keyed by a distinctive model-id SUBSTRING. First-party list
# rates — may drift from current OpenAI pricing. Token counts are exact (from usage); only
# these rates carry uncertainty (o4-mini output rate unverified). Reasoning bills within output, so
# thoughts_per_1m == output_per_1m. gpt-5.5's >272K-input tier (2x input / 1.5x output) is
# APPLIED per call via the "long_context" sub-row (accounting.compute_cost). The Bedrock id
# (`openai.gpt-5.5`) matches the same row via substring, but its window is capped at 272K so
# the tier can never trigger there; rates assume Bedrock = OpenAI list.
OPENAI_PRICING: dict[str, dict[str, float]] = {
    "gpt-5.5":  {"input_per_1m": 5.0,  "cached_input_per_1m": 0.50, "output_per_1m": 30.0, "thoughts_per_1m": 30.0,
                 "long_context": {"threshold": 272_000,  # >272K: 2x in / 1.5x out
                                  "input_per_1m": 10.0, "cached_input_per_1m": 1.0,
                                  "output_per_1m": 45.0, "thoughts_per_1m": 45.0}},
    "gpt-5.4-mini": {"input_per_1m": 0.75, "cached_input_per_1m": 0.075, "output_per_1m": 4.50, "thoughts_per_1m": 4.50},
    "gpt-5.4":  {"input_per_1m": 2.50, "cached_input_per_1m": 0.25, "output_per_1m": 15.0, "thoughts_per_1m": 15.0},
    "o3":       {"input_per_1m": 2.0,  "cached_input_per_1m": 0.50, "output_per_1m": 8.0,  "thoughts_per_1m": 8.0},
    "o4-mini":  {"input_per_1m": 1.10, "cached_input_per_1m": 0.275, "output_per_1m": 4.40, "thoughts_per_1m": 4.40},
}

# Context windows (tokens) by model-id substring, per developers.openai.com.
# gpt-5.4-mini needs its OWN row (400K): it is a smaller window than gpt-5.4 (1.05M),
# and longest-substring matching would otherwise resolve it to the "gpt-5.4" row. --context-budget
# overrides any of these.
OPENAI_MAX_CONTEXT: dict[str, int] = {
    "openai.gpt-5.5":  272_000,  # Bedrock ids (openai. prefix): window capped at 272K
    "openai.gpt-5.4":  272_000,
    "gpt-5.5":       1_050_000,
    "gpt-5.4-mini":    400_000,
    "gpt-5.4":       1_050_000,
    "gpt-4.1":       1_000_000,
    "o3":              200_000,
    "o4-mini":         200_000,
}

# gpt-5.5 prices input >272K at 2x and output at 1.5x; the tier is applied per call via the
# pricing row's "long_context" sub-row, and we note the crossing once in the run log
# (analogous to Gemini's long-context tier).
OPENAI_LONG_CONTEXT_THRESHOLD = 272_000


def _make_move_tool() -> dict:
    """Responses-API function tool from the neutral MAKE_MOVE_SPEC."""
    s = prompts.MAKE_MOVE_SPEC
    return {"type": "function", "name": s["name"], "description": s["description"], "parameters": s["parameters"]}


def _item_type(item) -> str | None:
    if isinstance(item, dict):
        return item.get("type") or ("message" if "role" in item else None)
    return getattr(item, "type", None)


class OpenAIPolicy:
    """Owns the Responses-API loop: append-only `input` items, verbatim reasoning+
    function_call refeed, forced make_move, exact accounting, the tool-free debrief,
    and the step-pair truncation surface."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        effort: str,                 # none|minimal|low|medium|high|xhigh (raw --thinking-level; "max" 400s)
        max_tokens: int,
        timeout: int,
        max_retries: int,
        pricing: dict | None = None,
        include_thoughts: bool = True,
        base_url: str | None = None,     # None = api.openai.com; Bedrock passes its openai/v1 endpoint
        client=None,
    ):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.debrief_retries = max(1, min(2, max_retries))
        self.pricing = pricing or OPENAI_PRICING
        self.include_thoughts = include_thoughts
        self.move_channel = "forced-tool"        # provenance (parity with Gemini/SGLang)
        self.thoughts_basis = "exact"            # reasoning_tokens is server-exact (0 when effort none)

        self.pricing_row = accounting.pricing_for(model, self.pricing)
        self.has_pricing = self.pricing_row is not None
        self.model_max_context = accounting.match_model(model, OPENAI_MAX_CONTEXT)

        if client is not None:
            self.client = client
        else:
            from openai import OpenAI  # deferred: heavy SDK import
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

        self.input: list = []
        self._last_call_id: str | None = None
        self.call_count = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.thoughts_tokens = 0
        self.total_tokens = 0
        self.cost_usd = 0.0
        self.elapsed_s = 0.0
        self.last_prompt_tokens = 0
        self.base_url = base_url                 # provenance (None = api.openai.com)
        self.reported_model: str | None = None   # server-reported model id (exact revision)
        self._warned_long_context = False
        self._last_error: dict | None = None
        self.on_retry = None
        self.debug_dir = "."

    # --- conversation construction ---

    def start(self, description: str, state_slice: dict) -> None:
        self.input.append({"role": "user", "content": prompts.seed_text(description, state_slice)})

    def observe(self, result: dict) -> None:
        payload = json.dumps({"result": result})
        if self._last_call_id is not None:  # answer the make_move call by id
            self.input.append({"type": "function_call_output", "call_id": self._last_call_id, "output": payload})
            self._last_call_id = None
        else:  # no function call this turn — plain user message
            self.input.append({"role": "user", "content": payload})

    def add_nudge(self, text: str) -> None:
        # If a make_move call is pending (e.g. the model emitted the call but with an empty/
        # unparseable action), ANSWER it — an unanswered function_call 400s the next request.
        # Reuses observe()'s construction; falls back to a plain user message otherwise.
        if self._last_call_id is not None:
            self.input.append({"type": "function_call_output", "call_id": self._last_call_id, "output": text})
            self._last_call_id = None
        else:
            self.input.append({"role": "user", "content": text})

    # --- request building ---

    def _reasoning_param(self) -> dict:
        r: dict[str, Any] = {"effort": self.effort}
        if self.effort != "none" and self.include_thoughts:
            r["summary"] = "auto"   # free visible reasoning summary
        return r

    def _turn_kwargs(self) -> dict:
        return {
            "model": self.model,
            "instructions": prompts.SYSTEM_INSTRUCTION,
            "input": self.input,
            "tools": [_make_move_tool()],
            "tool_choice": {"type": "function", "name": "make_move"},  # forced; OpenAI keeps reasoning
            "parallel_tool_calls": False,  # >1 make_move -> an unanswered call_id -> next request 400s
            "reasoning": self._reasoning_param(),
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": self.max_tokens,
        }

    def _debrief_kwargs(self) -> dict:
        return {
            "model": self.model,
            "instructions": prompts.DEBRIEF_SYSTEM_INSTRUCTION,
            "input": self.input,
            "reasoning": self._reasoning_param(),
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": self.max_tokens,
        }

    # --- LLM call ---

    def _create(self, build_kwargs, max_retries: int, timeout: int | None = None):
        kwargs = build_kwargs()
        opts = {} if timeout is None else {"timeout": timeout}
        resp, err = clientutil.run_request(
            lambda: self.client.responses.create(**kwargs, **opts),
            provider="OpenAI", max_retries=max_retries, request=kwargs,
            debug_dir=self.debug_dir, call_count=self.call_count, on_event=self.on_retry,
        )
        self._last_error = err
        return resp

    @staticmethod
    def _extract(resp) -> tuple[str | None, str | None, str, bool, str | None]:
        """Return (action, call_id, reasoning_summary, has_encrypted_reasoning, status)."""
        action = call_id = None
        summary_parts, has_enc = [], False
        for item in getattr(resp, "output", None) or []:
            itype = getattr(item, "type", None)
            if itype == "function_call" and getattr(item, "name", None) == "make_move":
                try:
                    args = json.loads(getattr(item, "arguments", "") or "{}")
                except (TypeError, ValueError):
                    args = {}
                if isinstance(args, dict):
                    action = str(args.get("action") or "").strip() or None  # JSON null -> None, not "None"
                call_id = getattr(item, "call_id", None)
            elif itype == "reasoning":
                if getattr(item, "encrypted_content", None):
                    has_enc = True
                for s in getattr(item, "summary", None) or []:
                    text = getattr(s, "text", None)
                    if text:
                        summary_parts.append(text.strip())
        return action, call_id, " ".join(p for p in summary_parts if p), has_enc, getattr(resp, "status", None)

    def _account(self, usage) -> tuple[dict, float | None]:
        inp = int(getattr(usage, "input_tokens", 0) or 0)          # full input (cached is a SUBSET)
        out_total = int(getattr(usage, "output_tokens", 0) or 0)   # INCLUDES reasoning
        odetails = getattr(usage, "output_tokens_details", None)
        reasoning = int(getattr(odetails, "reasoning_tokens", 0) or 0)   # exact
        idetails = getattr(usage, "input_tokens_details", None)
        cached = int(getattr(idetails, "cached_tokens", 0) or 0)
        visible = max(0, out_total - reasoning)   # report out/think DISJOINT
        counts = {"prompt": inp, "cached": cached, "output": visible, "thoughts": reasoning,
                  "total": inp + out_total}
        # cached ⊆ input_tokens (uncached = inp - cached); thinking bills at the output rate and
        # already sits inside out_total, so pricing visible+thinking both at the output rate
        # reproduces out_total*rate (thoughts_per_1m == output_per_1m) — no double count.
        cost = accounting.compute_cost(self.model, inp, cached, visible, reasoning, self.pricing)
        self.call_count += 1
        self.prompt_tokens += inp
        self.cached_tokens += cached
        self.output_tokens += visible
        self.thoughts_tokens += reasoning
        self.total_tokens += counts["total"]
        if inp:  # the truncation trigger — server-reported prompt size
            self.last_prompt_tokens = inp
        if cost is not None:
            self.cost_usd += cost
        if inp > OPENAI_LONG_CONTEXT_THRESHOLD and not self._warned_long_context:
            self._warned_long_context = True
            if self.on_retry:
                if self.pricing_row and self.pricing_row.get("long_context"):
                    self.on_retry(f"prompt {inp} > {OPENAI_LONG_CONTEXT_THRESHOLD}: "
                                  "long-context tier rates now apply (per call, priced)")
                else:  # a row with no tier — still flat-priced
                    self.on_retry(f"prompt {inp} > {OPENAI_LONG_CONTEXT_THRESHOLD}: cost may be "
                                  "underestimated (no long-context tier priced for this model)")
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
        # CRITICAL: append the response output VERBATIM (reasoning item + encrypted_content +
        # function_call) so reasoning rides forward. Never reconstruct or type-filter it.
        self.input += list(getattr(resp, "output", None) or [])
        action, call_id, summary, has_enc, status = self._extract(resp)
        self._last_call_id = call_id
        if call_id is None:
            # No make_move function_call this turn (e.g. status="incomplete" — a reasoning cutoff
            # produced reasoning but no tool call). Under store=false a TRAILING reasoning item with
            # no following item 400s the next request, so strip the trailing reasoning we just
            # appended. Safe: completed turns always end in a function_call_output (from observe),
            # so a trailing reasoning item is only ever this orphan response's.
            while self.input and _item_type(self.input[-1]) == "reasoning":
                self.input.pop()
        counts, cost = self._account(getattr(resp, "usage", None))
        return Move(action, summary, has_enc, status, counts, cost, elapsed,
                    thoughts_basis=self.thoughts_basis,
                    continuity="verified" if has_enc else None)  # API-validated encrypted item

    def debrief(self) -> str | None:
        self.add_nudge(prompts.DEBRIEF_PROMPT)
        start = time.time()
        resp = self._create(self._debrief_kwargs, self.debrief_retries, timeout=2 * self.timeout)
        self.elapsed_s += time.time() - start
        if resp is None:
            return None
        self._account(getattr(resp, "usage", None))
        text = (getattr(resp, "output_text", None) or "").strip()
        return text or "(empty debrief)"

    # --- truncation surface (used only by core/history.py) ---

    @staticmethod
    def _is_model_item(item) -> bool:
        """A model-generated output item (reasoning / function_call / assistant message) —
        as opposed to a head/user message or a function_call_output (the result we add)."""
        t = _item_type(item)
        if t in ("reasoning", "function_call"):
            return True
        if t == "message":
            role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
            return role == "assistant"
        return False

    def _turn_start_indices(self) -> list[int]:
        """Indices where a model turn begins — a model-output item whose predecessor is not
        one (i.e. it follows the head, a function_call_output, or a user message)."""
        out = []
        for i, it in enumerate(self.input):
            if self._is_model_item(it) and (i == 0 or not self._is_model_item(self.input[i - 1])):
                out.append(i)
        return out

    def turns(self):
        """Step-pair units over `input`, oldest -> newest. The head (the pinned user task
        before the first model item) is NOT a unit. A unit = a model turn's items
        (reasoning + function_call) + the function_call_output that follows. The last unit
        is ACTIVE (its reasoning+call must stay paired/intact -> never evicted)."""
        starts = self._turn_start_indices()
        if not starts:
            return []
        bounds = starts + [len(self.input)]
        n = len(starts)
        out = []
        for k in range(n):
            chunk = self.input[bounds[k]:bounds[k + 1]]
            out.append(Turn(payload=chunk, is_active=(k == n - 1), est_tokens=clientutil.est_tokens(chunk)))
        return out

    def evict_oldest_turn(self) -> None:
        """Drop the oldest completed unit (its reasoning + function_call + function_call_output
        together). Pinned head + active (latest) unit are never touched. No-op when only head +
        active remain. (The API harmlessly discards reasoning items that aren't relevant.)"""
        starts = self._turn_start_indices()
        if len(starts) <= 1:
            return
        del self.input[starts[0]:starts[1]]
