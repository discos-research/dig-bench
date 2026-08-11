# OpenAI — operator reference

How to run the baseline harness (`--provider openai`) against OpenAI reasoning models. The client
(`clients/openai_client.py`) uses the official **`openai`** SDK's **Responses API**
(`client.responses.create`) — **not** Chat Completions. The model id picks the backend:
`openai.`-prefixed ids (the default) go to **AWS Bedrock**'s OpenAI-compatible endpoint
(`https://bedrock-mantle.{region}.api.aws/openai/v1`); bare ids go direct to api.openai.com.

> Model ids, context windows, and pricing drift between generations — verify against the
> [current OpenAI pricing](https://openai.com/api/pricing/). Token counts are exact; only the USD
> rates carry uncertainty.

## Setup
- **Dependency:** `openai>=2.24`.
- **Auth:** Bedrock ids (`openai.` prefix) use `AWS_BEARER_TOKEN_BEDROCK` (bearer token, not
  SigV4); bare ids use `OPENAI_API_KEY`. `--api-key` overrides either. Export it, or have a
  secrets manager of your choice inject it into the environment.
- **Region (Bedrock ids only):** `--aws-region` (default `us-east-2` — GPT-5.5 lives in Ohio;
  falls back to `AWS_REGION`).
- **Default model:** `openai.gpt-5.5` (GPT-5.5 on Bedrock).

## Models & pricing
USD per 1M tokens. Cached input ≈ 10% of input (gpt-5.5: $0.50). **gpt-5.5**
uses tiered pricing: standard rates up to 272K input tokens, then **2× input / 1.5× output** above
272K — the tier is **applied per call** via the pricing row's `long_context` sub-row, and the
client notes the crossing once in the run log.

> **Reproducibility tip:** on the direct API prefer an **immutable snapshot id** (e.g.
> `gpt-5.5-2026-04-23`) over the mutable `gpt-5.5` alias — snapshots exist to lock behavior.
> The pricing/window rows match snapshot ids via substring, and the run summary records the
> server-reported model (`reported_model`) either way.

| Model | `--model` | context | input | output |
|---|---|---|---|---|
| GPT-5.5 on Bedrock (default) | `openai.gpt-5.5` | 272K¹ | $5.00 | $30.00 |
| GPT-5.4 on Bedrock | `openai.gpt-5.4` | 272K¹ | $2.50 | $15.00 |
| GPT-5.5 (direct) | `gpt-5.5` | ~1.05M | $5.00 | $30.00 |
| GPT-5.4 | `gpt-5.4` | ~1M | $2.50 | $15.00 |
| GPT-5.4 mini | `gpt-5.4-mini` | 400K | $0.75 | $4.50 |
| o3 | `o3` | 200K | $2.00 | $8.00 |
| o4-mini | `o4-mini` | 200K | $1.10 | $4.40 |

Windows come from a substring table (`OPENAI_MAX_CONTEXT`); override with `--context-budget`.

¹ Bedrock caps the GPT-5.4/5.5 window at 272K (so the >272K direct-API price tier can never
trigger there). Rates assume Bedrock = OpenAI list price; the pricing rows match both id forms
via substring.

## Move channel + reasoning carry
This client speaks the **Responses API**: the conversation is a list of typed **items** in `input`
(a `{role,content}` message, `reasoning` items, `function_call` items, `function_call_output`
items); the system prompt goes in the top-level `instructions`. (Different surface from the SGLang
OpenAI-*compatible* Chat Completions client — don't conflate them.)

- **Move channel = forced `make_move`** (`tool_choice={"type":"function","name":"make_move"}`) —
  parity with Gemini/SGLang. OpenAI reasoning models **still emit a reasoning item with the forced
  tool call**, so forcing does not break carry (unlike Anthropic). `move_channel=forced-tool`.
- **Reasoning carry = encrypted reasoning items, stateless:** `store=False` +
  `include=["reasoning.encrypted_content"]`. Each turn the client appends `response.output` (the
  reasoning item carrying `encrypted_content`, plus the `function_call`) **verbatim** to `input`,
  then a `function_call_output`. The reasoning item must accompany its function_call (the API 400s
  otherwise) → **self-checking**, `has_continuity` honest, no probe. `encrypted_content` is a
  dedicated field separate from the visible `summary` — the cleanest carry of the four providers.
- **`--thinking-level`** → `reasoning.effort` ∈ `none|minimal|low|medium|high|xhigh`. `none`
  disables reasoning (no carry). `max` is not an OpenAI value → 400 (fail-loud, no allow-listing).
- **`--no-thought-summaries`** omits `reasoning.summary:"auto"` (the visible 🧠 summary;
  `encrypted_content` is still carried).

## Accounting
- `usage.output_tokens` **includes** `output_tokens_details.reasoning_tokens` (exact). Reported
  **disjoint** (`out = output_tokens − reasoning`, `think = reasoning`, `thoughts_basis="exact"`).
- `usage.input_tokens_details.cached_tokens` is a **subset of** `input_tokens` (so `prompt =
  input_tokens`, `cached = cached_tokens`) — unlike Anthropic, where cache_read is additive.
- `--max-tokens` (default 32000) → `max_output_tokens`; raise it at high/xhigh effort.
- **`--seed`** is a warned no-op — the Responses API has no seed parameter.

## Smoke-test (from inside the repo)
```bash
# Bedrock (default)
AWS_BEARER_TOKEN_BEDROCK=... python -m core.cli P-1 \
  --provider openai --thinking-level medium \
  --max-steps 30 --context-budget 2000 --verbose

# Direct API (bare model id)
OPENAI_API_KEY=... python -m core.cli P-1 \
  --provider openai --model gpt-5.5 --thinking-level medium \
  --max-steps 30 --context-budget 2000 --verbose
```
Confirm: forced `make_move` every turn (no nudges); `🔑 reasoning carried` (encrypted_content
round-trips; zero pairing-400s); `think` exact and disjoint from `out`; exact USD; `↺` truncations
fire; debrief + playback link present.

## Notes
- Out of scope: Chat Completions; `previous_response_id` stateful mode (we use stateless
  `store=false` for parity); non-reasoning gpt models.
