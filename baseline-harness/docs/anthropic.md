# Anthropic — operator reference

How to run the baseline harness (`--provider anthropic`) against Claude models on **AWS Bedrock**. The client
(`clients/anthropic_client.py`) uses the official **`anthropic`** SDK's `AnthropicBedrock` client
with **bearer-token** auth. The model id picks the backend: ids containing `anthropic.` (the
default) go to Bedrock; a bare `claude-*` id goes direct to api.anthropic.com via the plain
`Anthropic` client with `ANTHROPIC_API_KEY` (same Messages API, caching, and thinking behavior).

> Model ids, context windows, and pricing drift between generations — verify against the
> [current Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing). Token counts
> are exact; USD rates are first-party list rates (Bedrock regional endpoints add ~10%).

## Setup
- **Dependency:** `anthropic[bedrock]`.
- **Auth (bearer token, NOT SigV4):** set `AWS_BEARER_TOKEN_BEDROCK` or pass `--api-key`. The SDK
  detects it automatically; no AWS access-key/secret needed. Export it, or have a secrets
  manager of your choice inject it into the environment.
- **Region:** `--aws-region` (default `us-east-1`; falls back to `AWS_REGION`).
- **Default model:** `global.anthropic.claude-opus-5`.
- **Model id = a Bedrock inference profile:** `global.` prefix = global routing (recommended, no
  premium); `us.`/`eu.`/… = regional endpoint (CRIS, +10% pricing). A **bare** `claude-*` id
  (no `anthropic.` segment) instead selects the **direct Anthropic API** (`ANTHROPIC_API_KEY`
  or `--api-key`; `--aws-region` unused).
- **Opus 5 is `global.`-only for now**: `us.anthropic.claude-opus-5` returns
  503 and the bare `anthropic.claude-opus-5` returns "on-demand throughput isn't supported —
  retry with an inference profile". Use `global.anthropic.claude-opus-5`.

## Models & pricing
USD per 1M tokens. Cached read ≈ 10% of input. The client sets explicit `cache_control`
breakpoints (Bedrock has no *automatic* caching), so `cached` is populated here and
turns reuse the cached prefix. Thinking
tokens bill at the **output** rate.

| Model | `--model` | context | input | output |
|---|---|---|---|---|
| Claude Opus 5 (default) | `global.anthropic.claude-opus-5` | 1M | $5.00 | $25.00 |
| Claude Opus 4.8 | `global.anthropic.claude-opus-4-8` | 1M | $5.00 | $25.00 |
| Claude Opus 4.7 | `global.anthropic.claude-opus-4-7` | 1M | $5.00 | $25.00 |
| Claude Opus 4.6 | `global.anthropic.claude-opus-4-6-v1` | 1M | $5.00 | $25.00 |
| Claude Sonnet 4.6 | `global.anthropic.claude-sonnet-4-6` | 1M | $3.00 | $15.00 |
| Claude Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 200K | $1.00 | $5.00 |
| Claude Fable 5 | `global.anthropic.claude-fable-5` | 1M | $10.00 | $50.00 |

Windows come from a substring table (`ANTHROPIC_MAX_CONTEXT`); override with `--context-budget`.
Both tables key on the longest matching id substring, and the Opus rows are `claude-opus-5` /
`claude-opus-4` — a 4.x row does **not** cover a 5 id, so each new Opus generation needs its own
row in both tables (an unpriced model reports `—` for USD instead of failing).

## Move channel + reasoning carry
**Forced `tool_choice` is INCOMPATIBLE with extended/adaptive thinking** (the API 400s), and on
Sonnet 4.6 / Opus 4.6 / Haiku 4.5 the API runs **without thinking unless `thinking` is sent**. So
thinking-carry and forced-tool are mutually exclusive, and both map onto `--thinking-level`:

| `--thinking-level` | request | move channel | reasoning carried? |
|---|---|---|---|
| `high` (default) / `medium` / `low` / `minimal`→`low` / `xhigh` / `max` | `thinking:{type:"adaptive"}` + `output_config.effort` | **auto-tool** make_move (+ nudge) | **yes** — `thinking` block `signature`, re-fed verbatim |
| same levels on a **pre-effort model** (Haiku 4.5 / Sonnet 4.5) | `thinking:{type:"enabled", budget_tokens:N}` — **manual** extended thinking; no `output_config` | **auto-tool** make_move (+ nudge) | **yes** — same `signature` carry |
| `none` | `thinking:{type:"disabled"}` | **forced** make_move | no — ablation / forced-tool mode |

- This is a **documented fairness difference** from Gemini's forced `mode=ANY` (Anthropic can't
  force the tool while thinking). The run header/summary records the resolved `move_channel`.
- **Pre-effort models take a manual budget** (`ANTHROPIC_MANUAL_THINKING`: `claude-haiku-4-5`,
  `claude-sonnet-4-5`): they 400 on `output_config.effort` (`"Extra inputs are not permitted"`),
  so the level maps through the **shared** budget table
  (`core/clientutil.THINKING_LEVEL_BUDGETS` — the same nominal budget per level Gemini 2.5 uses):
  `minimal 1024 · low 2048 · medium 8192 · high 16384 · xhigh 24576 · max 32768`, clamped to
  `1024 ≤ N ≤ max_tokens − 1024` (thinking spends from the same `max_tokens` allowance).
- **`effort`** ∈ `low|medium|high|xhigh|max`. Opus 5 takes the whole ladder;
  `xhigh` is Opus-4.7+/Opus-5/Fable only, and Sonnet 4.6 has `max` but not `xhigh`. **No
  client-side allow-listing** — an unsupported level 400s (fail-loud).
- **`none` sends `disabled` explicitly, and that matters from Opus 5 on.** Opus 5 and Sonnet 5
  **think by default when `thinking` is omitted**. Forced `tool_choice` suppresses thinking on
  move turns, but the **debrief carries no tools** — with `thinking` merely omitted, the `none`
  ablation would silently think there (billed as output). The client therefore sends
  `thinking:{type:"disabled"}` on both the turn and the debrief. No `effort` rides along:
  on Opus 5 `disabled` is only legal at effort ≤ `high`, and the unset default is `high`.
- **`--thinking-level none` is not available on Fable 5** — `disabled` 400s there
  (`"thinking.type.disabled" is not supported for this model`), so the client keeps omitting
  `thinking` for those ids (`ANTHROPIC_THINKING_ALWAYS_ON`): forced-tool turns still come back with
  zero thinking tokens, but their debrief thinks and there is no way to stop it.
- **`--no-thought-summaries`** → `display:"omitted"` (signature still carried; cheaper).
- **Self-checking:** re-feeding a modified thinking block 400s, so `has_continuity` is honest and
  no probe is needed; prior-turn thinking is auto-ignored, so old step-pairs are safe to evict.

## Accounting
- `usage.output_tokens` **includes** thinking, broken out under `output_tokens_details.thinking_tokens`
  (exact). Reported **disjoint** (`out = output_tokens − thinking`, `think = thinking`,
  `thoughts_basis = "exact"`); the full output is priced at the output rate (no double count).
- `cached = cache_read_input_tokens` (populated on Bedrock via explicit `cache_control`); `prompt = input + cache_read + cache_creation`.
- `--max-tokens` (default 32000) caps thinking + output; raise it (≥64k) at `xhigh`/`max` effort.
- **`--seed`** is a warned no-op — the Messages API has no seed parameter.

## Smoke-test (from inside the repo)
```bash
AWS_BEARER_TOKEN_BEDROCK=... python -m core.cli P-1 \
  --provider anthropic --model global.anthropic.claude-opus-5 \
  --aws-region us-east-1 --thinking-level medium --max-steps 30 --context-budget 2000 --verbose
```
Confirm: `🔑 reasoning carried` (signature round-trips; zero continuity-400s); `move_channel=auto-tool`
with make_move reliably called; `think` exact and disjoint; exact USD; `↺` truncations fire;
debrief + playback link present.

## Notes
- Not supported on Bedrock (irrelevant to the baseline harness): Files API, server-side tools, Message
  Batches, automatic prompt caching, server-side refusal fallback. Extended thinking + tool use +
  the Messages API **are** supported.
- **Refusals (Opus 5 / Fable 5).** These carry elevated cyber/bio safeguards and can decline a turn
  with HTTP 200 + `stop_reason: "refusal"` — not an exception. There is no server-side `fallbacks`
  on Bedrock, so such a turn simply arrives with no make_move: it lands in the harness's normal
  no-action path (`finish_reason=refusal` in the log, generic nudge, counted invalid, and
  `blocked` past `--max-invalid-retries`). Refusals are rare in this setting; if one fires, the
  finish reason in the per-turn log line is the tell.
