"""Gemini client — GeminiPolicy, implements core.types.Policy.

Append-only `contents`,
the model's Content appended VERBATIM each turn (so the thought signature rides
forward and Gemini keeps its reasoning), the forced `make_move` tool (mode=ANY),
exact token/USD accounting, and the tool-free debrief.

The Policy truncation surface:
- `last_prompt_tokens` — the server-reported prompt size, the truncation trigger;
- `turns()` / `evict_oldest_turn()` — segment the flat
  `contents` into evictable STEP-PAIR units. A unit = one model output
  (functionCall + signature) plus the user functionResponse that follows it. The
  head TASK DESCRIPTION (everything before the first model output) is PINNED and
  never a unit; the latest unit is the ACTIVE unit and is never evicted.

Retry/error handling is the shared `core.clientutil` policy: transient errors
(5xx/429-rate/network) retry with backoff; deterministic 4xx fail fast and dump the
request. (A signature-validation 400 thus fails fast.)
"""

from __future__ import annotations

import time
from typing import Any

try:  # google-genai is an optional dependency — importing this module must not hard-fail without
    from google import genai            # it (so the fake-client test suites for OTHER providers,
    from google.genai import types      # which transitively import this module, still run). A real
except ImportError:                     # GeminiPolicy needs the SDK at construction (it builds
    genai = types = None                # types.* configs); tests that do so skip when it's absent.

from core import accounting, clientutil, prompts
from core.types import Move, Turn

# Gemini 3 Pro switches to a higher input-price tier above this prompt size.
GEMINI_LONG_CONTEXT_THRESHOLD = 200_000

# Pricing (USD per 1M tokens); cached ≈ 10% of input. First-party list rates — token counts
# are exact (from usage_metadata); only these rates carry uncertainty. Pro models add a
# >200K-input tier (2x input / 1.5x output) — APPLIED per call via the row's "long_context"
# sub-row (accounting.compute_cost); flash rows have no tier.
GEMINI_PRICING: dict[str, dict[str, float]] = {
    "gemini-3.1-pro":  {"input_per_1m": 2.00, "cached_input_per_1m": 0.20,  "output_per_1m": 12.00, "thoughts_per_1m": 12.00,
                        "long_context": {"threshold": GEMINI_LONG_CONTEXT_THRESHOLD,  # >200K: 2x in / 1.5x out
                                         "input_per_1m": 4.00, "cached_input_per_1m": 0.40,
                                         "output_per_1m": 18.00, "thoughts_per_1m": 18.00}},
    "gemini-3.5-flash":{"input_per_1m": 1.50, "cached_input_per_1m": 0.15,  "output_per_1m": 9.00,  "thoughts_per_1m": 9.00},
    "gemini-3-flash":  {"input_per_1m": 0.50, "cached_input_per_1m": 0.05,  "output_per_1m": 3.00,  "thoughts_per_1m": 3.00},
    "gemini-2.5-pro":  {"input_per_1m": 1.25, "cached_input_per_1m": 0.125, "output_per_1m": 10.00, "thoughts_per_1m": 10.00},
    "gemini-2.5-flash":{"input_per_1m": 0.30, "cached_input_per_1m": 0.03,  "output_per_1m": 2.50,  "thoughts_per_1m": 2.50},
}

# Context windows (tokens). Closed models can't be queried — keep a per-model table.
# Gemini 3.x Pro/Flash, 3.5 Flash, and 2.5 Pro/Flash are all 1,048,576 — the flashes
# resolve via the "gemini-3"/"gemini-2.5" substrings to the same window, so no separate
# rows are needed. These DRIFT between generations; re-verify against current provider
# docs when adding models. `--context-budget` overrides.
GEMINI_MAX_CONTEXT: dict[str, int] = {
    "gemini-3.1-pro-preview": 1_048_576,
    "gemini-3": 1_048_576,
    "gemini-2.5": 1_048_576,
}

# Models configured by an explicit token budget (`thinking_budget`) instead of
# `thinking_level` — thinkingLevel is a Gemini-3-family parameter; 2.5 models reject it.
# --thinking-level maps through the SHARED clientutil.THINKING_LEVEL_BUDGETS table (same
# nominal budget per level as Anthropic manual thinking), clamped to the 2.5 API bounds:
# flash 0..24576, pro 128..32768 (pro can never fully disable thinking — the `none` branch
# below already refuses non-flash). Substring match, like the tables above.
GEMINI_BUDGET_THINKING: tuple[str, ...] = ("gemini-2.5",)


def _md(metadata: Any, field: str) -> int:
    if metadata is None:
        return 0
    return getattr(metadata, field, 0) or 0


def _make_move_declaration() -> types.FunctionDeclaration:
    """Build the Gemini FunctionDeclaration from the neutral MAKE_MOVE_SPEC so the
    wording stays shared across providers (all current properties are strings)."""
    spec = prompts.MAKE_MOVE_SPEC
    params = spec["parameters"]
    return types.FunctionDeclaration(
        name=spec["name"],
        description=spec["description"],
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                key: types.Schema(type=types.Type.STRING, description=prop.get("description", ""))
                for key, prop in params["properties"].items()
            },
            required=list(params.get("required", [])),
        ),
    )


class GeminiPolicy:
    """Owns the Gemini chat: the append-only `contents`, verbatim signature
    refeed, the forced make_move tool, exact accounting, and the step-pair
    truncation surface."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        thinking_level: str | None,
        timeout: int,
        max_retries: int,
        max_tokens: int = 32000,
        pricing: dict | None = None,
        include_thoughts: bool = True,
        seed: int | None = None,
        client=None,
    ):
        self.model = model
        self.thinking_level = thinking_level
        self.max_retries = max_retries
        self.max_tokens = max_tokens          # universal output cap (consistency across providers)
        self.timeout = timeout
        # The debrief is one heavy call (whole transcript + thinking + long output):
        # a smaller retry budget and a longer per-attempt timeout (see debrief config).
        self.debrief_retries = max(1, min(2, max_retries))
        self.pricing = pricing or GEMINI_PRICING
        self.include_thoughts = include_thoughts
        self.seed = seed
        self.pricing_row = accounting.pricing_for(model, self.pricing)
        self.has_pricing = self.pricing_row is not None
        self.model_max_context = accounting.match_model(model, GEMINI_MAX_CONTEXT)
        self.client = client or genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=timeout * 1000)
        )
        self.thinking_budget: int | None = None  # resolved by _thinking_config (2.5 family); provenance
        self._turn_config = self._build_turn_config()
        self._debrief_config = self._build_debrief_config()

        self.contents: list[types.Content] = []
        self.call_count = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.thoughts_tokens = 0
        self.total_tokens = 0
        self.cost_usd = 0.0
        self.elapsed_s = 0.0
        self.last_prompt_tokens = 0  # server-reported prompt size of the last call
        self.reported_model: str | None = None  # server-reported model version (exact revision)
        self.thoughts_basis = "exact"  # thoughts_token_count is exact + disjoint from output
        self._warned_long_context = False
        self._last_error: dict | None = None
        self._call_pending = False        # a make_move functionCall emitted but not yet answered
        self._pending_call_count = 0      # how many make_move calls this turn (all must be answered)
        self.on_retry = None  # optional callable(msg) to log failed attempts / events
        self.debug_dir = "."  # 4xx request dumps land here (set by the CLI)

    # --- config ---

    def _thinking_config(self, *, include_thoughts: bool) -> types.ThinkingConfig:
        if self.thinking_level is None:  # --thinking-level none
            # thinking_budget=0 truly disables thinking on flash-class models; Pro models REJECT it
            # (400 "Budget 0 is invalid. This model only works in thinking mode."). Running Pro at
            # its default thinking instead would mislabel the run as an ablation that never
            # happened, so refuse rather than silently substitute.
            if "flash" in self.model.lower():
                return types.ThinkingConfig(thinking_budget=0, include_thoughts=include_thoughts)
            raise SystemExit(
                f"--thinking-level none: thinking cannot be disabled on {self.model} "
                "(non-flash models reject thinking_budget=0) — use a flash model or a low level."
            )
        if any(k in self.model for k in GEMINI_BUDGET_THINKING):
            # 2.5-family: thinking_level is rejected; translate the level through the SHARED
            # budget table. Thinking spends from max_output_tokens here too, so the same
            # max_tokens - 1024 guard applies; an infeasible pair (e.g. a budget that would
            # resolve to 0 — thinking OFF while the run is labeled with a level) fails loud
            # at construction. Resolved budget recorded for provenance.
            flash = "flash" in self.model.lower()
            try:
                budget = clientutil.clamp_thinking_budget(
                    self.thinking_level, self.max_tokens,
                    floor=0 if flash else 128, cap=24_576 if flash else 32_768,
                )
            except ValueError as exc:
                raise SystemExit(f"gemini 2.5 thinking on {self.model}: {exc}")
            self.thinking_budget = budget
            return types.ThinkingConfig(thinking_budget=budget, include_thoughts=include_thoughts)
        level = getattr(types.ThinkingLevel, self.thinking_level.upper(), self.thinking_level)
        return types.ThinkingConfig(thinking_level=level, include_thoughts=include_thoughts)

    def _build_turn_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=prompts.SYSTEM_INSTRUCTION,
            tools=[types.Tool(function_declarations=[_make_move_declaration()])],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=["make_move"],
                )
            ),
            thinking_config=self._thinking_config(include_thoughts=self.include_thoughts),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            max_output_tokens=self.max_tokens,   # universal output cap (parity across providers)
            seed=self.seed,
        )

    def _build_debrief_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=prompts.DEBRIEF_SYSTEM_INSTRUCTION,
            thinking_config=self._thinking_config(include_thoughts=False),
            max_output_tokens=self.max_tokens,
            seed=self.seed,
            http_options=types.HttpOptions(timeout=2 * self.timeout * 1000),
        )

    # --- conversation construction ---

    def start(self, description: str, state_slice: dict) -> None:
        text = prompts.seed_text(description, state_slice)
        self.contents.append(types.Content(role="user", parts=[types.Part(text=text)]))

    def observe(self, result: dict) -> None:
        # Answer EVERY make_move call emitted this turn: Gemini has no disable-parallel flag, and N
        # functionCalls require N functionResponses or the next request 400s. All map to one result.
        n = max(1, self._pending_call_count)
        self.contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_function_response(name="make_move", response={"result": result})
                       for _ in range(n)],
            )
        )
        self._call_pending = False
        self._pending_call_count = 0

    def add_nudge(self, text: str) -> None:
        # Under forced function-calling (mode=ANY) the model emits a make_move call every turn; if
        # this turn's action was empty/unparseable, ANSWER the pending call with a functionResponse
        # so it isn't left dangling, otherwise a plain user nudge. Reuses observe()'s construction.
        if self._call_pending:
            n = max(1, self._pending_call_count)
            self.contents.append(types.Content(role="user", parts=[
                types.Part.from_function_response(name="make_move", response={"result": text})
                for _ in range(n)]))
            self._call_pending = False
            self._pending_call_count = 0
        else:
            self.contents.append(types.Content(role="user", parts=[types.Part(text=text)]))

    # --- LLM call ---

    def _generate(self, config: types.GenerateContentConfig, *, max_retries: int):
        resp, err = clientutil.run_request(
            lambda: self.client.models.generate_content(
                model=self.model, contents=self.contents, config=config),
            provider="Gemini", max_retries=max_retries,
            request={"model": self.model, "contents_len": len(self.contents)},
            debug_dir=self.debug_dir, call_count=self.call_count, on_event=self.on_retry,
        )
        self._last_error = err
        return resp

    def _account(self, metadata: Any) -> tuple[dict, float | None]:
        counts = {
            "prompt": _md(metadata, "prompt_token_count"),
            "cached": _md(metadata, "cached_content_token_count"),
            "output": _md(metadata, "candidates_token_count"),
            "thoughts": _md(metadata, "thoughts_token_count"),
            "total": _md(metadata, "total_token_count"),
        }
        cost = accounting.compute_cost(
            self.model, counts["prompt"], counts["cached"], counts["output"], counts["thoughts"], self.pricing
        )
        self.call_count += 1
        self.prompt_tokens += counts["prompt"]
        self.cached_tokens += counts["cached"]
        self.output_tokens += counts["output"]
        self.thoughts_tokens += counts["thoughts"]
        self.total_tokens += counts["total"]
        if counts["prompt"]:  # the truncation trigger — server's reported prompt size
            self.last_prompt_tokens = counts["prompt"]
        if cost is not None:
            self.cost_usd += cost
        if counts["prompt"] > GEMINI_LONG_CONTEXT_THRESHOLD and not self._warned_long_context:
            self._warned_long_context = True
            if self.on_retry:  # route through the run log, not bare stdout (parity with OpenAI)
                if self.pricing_row and self.pricing_row.get("long_context"):
                    self.on_retry(
                        f"prompt {counts['prompt']} > {GEMINI_LONG_CONTEXT_THRESHOLD}: "
                        "long-context tier rates now apply (per call, priced)"
                    )
                else:  # a >200K prompt on a row with no tier (e.g. flash) — still flat-priced
                    self.on_retry(
                        f"prompt {counts['prompt']} > {GEMINI_LONG_CONTEXT_THRESHOLD}: cost may be "
                        "underestimated (no long-context tier priced for this model)"
                    )
        return counts, cost

    @staticmethod
    def _extract(resp, content) -> tuple[str | None, str, bool, str | None]:
        action = None
        calls = resp.function_calls or []
        if calls:
            action = str((calls[0].args or {}).get("action") or "").strip() or None  # JSON null -> None
        summary_parts, has_sig = [], False
        for part in (content.parts or []) if content is not None else []:
            if getattr(part, "thought_signature", None):
                has_sig = True
            if getattr(part, "thought", False):
                text = getattr(part, "text", None)
                if text:
                    summary_parts.append(text.strip())
        finish = None
        if resp.candidates:
            fr = resp.candidates[0].finish_reason
            finish = str(fr) if fr is not None else None
        return action, " ".join(p for p in summary_parts if p), has_sig, finish

    def generate_move(self) -> Move:
        start = time.time()
        resp = self._generate(self._turn_config, max_retries=self.max_retries)
        elapsed = time.time() - start
        self.elapsed_s += elapsed
        if resp is None:
            return Move(None, "", False, None, clientutil.zero_usage(), None, elapsed, error=self._last_error)

        if self.reported_model is None:  # provenance: the exact revision behind the alias
            self.reported_model = getattr(resp, "model_version", None)
        content = resp.candidates[0].content if resp.candidates else None
        if content is not None:
            # CRITICAL: append the model's Content VERBATIM so the thought
            # signature rides forward. Never reconstruct the parts.
            self.contents.append(content)
        fc = getattr(resp, "function_calls", None) or []
        self._call_pending = bool(fc)
        self._pending_call_count = len(fc)  # answered (all of them) by observe/add_nudge
        counts, cost = self._account(resp.usage_metadata)
        action, summary, has_sig, finish = self._extract(resp, content)
        return Move(action, summary, has_sig, finish, counts, cost, elapsed, thoughts_basis="exact",
                    continuity="verified" if has_sig else None)  # API-validated thought signature

    def debrief(self) -> str | None:
        self.add_nudge(prompts.DEBRIEF_PROMPT)
        start = time.time()
        resp = self._generate(self._debrief_config, max_retries=self.debrief_retries)
        self.elapsed_s += time.time() - start
        if resp is None:
            return None
        self._account(resp.usage_metadata)
        return (resp.text or "").strip() or "(empty debrief)"

    # --- truncation surface (used only by core/history.py) ---

    def _model_indices(self) -> list[int]:
        return [i for i, c in enumerate(self.contents) if getattr(c, "role", None) == "model"]

    def turns(self):
        """Segment `contents` into evictable step-pair units, oldest -> newest.
        Everything before the first model output is the PINNED head (not a unit).
        Each unit runs from a model output up to (not incl.) the next model output,
        so it captures the functionResponse (and any nudge) that follows. The last
        unit is the ACTIVE unit."""
        midx = self._model_indices()
        if not midx:
            return []
        bounds = midx + [len(self.contents)]
        units = []
        n = len(midx)
        for k in range(n):
            s, e = bounds[k], bounds[k + 1]
            chunk = self.contents[s:e]
            units.append(Turn(payload=chunk, is_active=(k == n - 1), est_tokens=clientutil.est_tokens(chunk)))
        return units

    def evict_oldest_turn(self) -> None:
        """Drop the oldest completed step-pair: the contents from the first model
        output up to (not incl.) the second. The pinned head (before the first
        model output) and the active (latest) unit are never touched. No-op when
        only the head + the active unit remain."""
        midx = self._model_indices()
        if len(midx) <= 1:  # only the active unit (or none) — nothing to evict
            return
        del self.contents[midx[0]:midx[1]]
