# Gemini — operator reference

How to run the baseline harness (`--provider gemini`) against Google Gemini models. The client
(`clients/gemini_client.py`) uses the official **`google-genai`** SDK (the Gemini Developer API).

> Model ids, context windows, and pricing drift between generations — verify against the
> [current Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing). Token counts are exact
> (from `usage_metadata`); only the USD rates carry uncertainty.

## Setup
- **Dependency:** `google-genai`.
- **Auth:** `GEMINI_API_KEY` (or `GOOGLE_API_KEY` / `GEMINI_KEY`), or `--api-key`. Export it,
  or have a secrets manager of your choice inject it into the environment.
- **Default model:** `gemini-3.1-pro-preview`.

## Models & pricing
USD per 1M tokens. Cached input ≈ 10% of input. All current Gemini models
have a ~1M-token context window. **Pro** models use tiered pricing: standard rates up to 200K
input tokens, then **2× input / 1.5× output** above 200K — the tier is **applied per call**
(see *Accounting*).

| Model | `--model` | context | input | output |
|---|---|---|---|---|
| Gemini 3.1 Pro (default) | `gemini-3.1-pro-preview` | ~1M | $2.00 | $12.00 |
| Gemini 3.5 Flash | `gemini-3.5-flash` | ~1M | $1.50 | $9.00 |
| Gemini 3 Flash | `gemini-3-flash-preview` | ~1M | $0.50 | $3.00 |
| Gemini 2.5 Pro | `gemini-2.5-pro` | ~1M | $1.25 | $10.00 |
| Gemini 2.5 Flash | `gemini-2.5-flash` | ~1M | $0.30 | $2.50 |

Windows come from a per-model table (`GEMINI_MAX_CONTEXT`); override with `--context-budget`.

## Move channel + reasoning carry
- **Move channel = forced `make_move`** via `FunctionCallingConfig(mode=ANY,
  allowed_function_names=["make_move"])` — Gemini is constrained to emit exactly the `make_move`
  call each turn. (It may occasionally emit *two* calls in one response; the client acts on the
  first but answers **every** emitted call — N `functionResponse`s — since a dangling call 400s the
  next request. Gemini has no disable-parallel flag.)
- **Reasoning carry = the `thought_signature`** on the model's `Content`, appended **verbatim**
  each turn so the model continues its chain instead of re-deriving. **Self-checking:** on Gemini 3
  the current-turn signature is mandatory and a tampered/missing one 400s — so `has_continuity` is
  honest. Never reconstruct the parts (that strips the signature).
- **`--thinking-level`** ∈ `minimal|low|medium|high|xhigh|max` → `ThinkingConfig.thinking_level`;
  `none` sets `thinking_budget=0` (**actually disables** thinking) on **flash-class** models. **Pro**
  models reject `budget=0` (`400 "Budget 0 is invalid. This model only works in thinking mode."`),
  so on Pro `none` is refused at construction — a true no-thinking ablation needs a flash model.
  Unsupported levels surface as an API error.
- **Gemini 2.5 models take a token budget, not a level** (`thinkingLevel` is a Gemini-3-family
  parameter; 2.5 rejects it). For `gemini-2.5-*` ids the level maps through the **shared** budget
  table (`core/clientutil.THINKING_LEVEL_BUDGETS` — the same nominal budget per level Anthropic
  manual thinking uses): `minimal 1024 · low 2048 · medium 8192 · high 16384 · xhigh 24576 ·
  max 32768`, clamped to the family bounds (flash 0–24 576, pro 128–32 768) and to
  `max_tokens − 1024` (on 2.5, thinking spends from `max_output_tokens`).
- **`--no-thought-summaries`** sets `include_thoughts=False`. Summaries are ON by default for
  inspectability — but the summary text is welded into the same `Content` as the signature, so
  verbatim carry re-feeds it as input each turn (an accepted cost; it can't be stripped).

## Accounting
- Gemini reports **disjoint** counts natively: `out = candidates_token_count`,
  `think = thoughts_token_count` (no overlap) → `thoughts_basis = "exact"`.
- `cached = cached_content_token_count`; USD via the per-model pricing table above.
- Above ~200K prompt tokens the Pro price tier rises (2× input / 1.5× output); the tier is
  **applied per call** via the pricing row's `long_context` sub-row, and the client notes the
  crossing once in the run log. Flash rows have no tier (a >200K flash prompt warns that cost
  may be underestimated).
- **`--seed`** is honored (`GenerateContentConfig.seed`) for reproducible sampling.

## Smoke-test (from inside the repo)
```bash
GEMINI_API_KEY=... python -m core.cli P-1 \
  --provider gemini --model gemini-3.1-pro-preview --thinking-level medium \
  --max-steps 30 --context-budget 2000 --verbose
```
Confirm: `🔑 reasoning carried` every turn (signature round-trips; zero continuity-400s); `think`
exact and disjoint from `out`; exact USD; `↺` truncations fire with the active signature intact;
debrief + playback link present.

## Notes
- The ~1M window means truncation rarely fires in real runs (still wired + tested).
- Forced-tool + signature is the most robust of the four providers — Gemini handles forced
  function calls and reasoning together natively.
