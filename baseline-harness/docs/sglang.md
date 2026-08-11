# SGLang (open models) — operator reference

How to run the baseline harness (`--provider sglang`) against open models on an **SGLang / any
OpenAI-compatible Chat Completions endpoint**. The client (`clients/sglang_client.py`) is
**parser-agnostic**, uses only the standard library (no SDK, no per-model code), and **streams**
responses — so its timeout is a per-chunk *idle* timeout, not a total one (see *Transport & timeouts*).

> Open-model windows/behaviours depend on how the operator launched the server; the per-model
> rows below are typical and should be confirmed at launch (see *Notes*). No USD pricing
> (self-hosted → cost reported as `n/a`).

## Setup
- **Dependency:** none (stdlib HTTP).
- **Auth:** `--api-key`, or `SGLANG_API_KEY` / `OPENAI_API_KEY` (optional; many endpoints need none).
  Export any key directly, or have a secrets manager of your choice inject it into the environment.
- **Endpoint:** `--base-url http://host:PORT/v1` (**required**).
- **Model:** auto-IDed from `GET /v1/models` when one is served; with several served and no
  `--model`, you're prompted to pick one. The `--tool-call-parser` / `--reasoning-parser` are
  **server-launch flags (the operator's job)**, not client options — the client only uses stable
  framework features (`tool_choice` named fn, `response_format` json_schema, `reasoning_content`,
  and SSE streaming with `stream_options.include_usage`).

## Models
The open-weights models of the DiG-bench report (Table 2), with the served-model identifiers the
report records ("the checkpoint we served rather than an API identifier") and their served
context sizes. Served context = whatever the operator configured at launch. Reasoning carry is
**template-dependent** and verified per run by the startup probe (below) — `has_continuity` is
reported honestly.

| Model | served id | served context |
|---|---|---|
| Kimi K3 | `kimi-k3` | 262K |
| GLM-5.2 | `glm-5.2` | ~1M |
| DeepSeek V4 Pro | `deepseek-v4-pro` | ~1M |
| DeepSeek V4 Flash | `deepseek-v4-flash` | ~1M |
| Qwen3.6 27B | `qwen3.6-27b` | 262K |

Any other reasoning-capable open model on an OpenAI-compatible endpoint works the same way — the
client is parser-agnostic and takes the id and window from `/v1/models`.

The window normally comes from `/v1/models` (`max_model_len`); **set `--context-budget` when it
isn't reported** (e.g. behind a proxy — see *Notes*).

## Move channel + reasoning carry
- **Move channel = forced/named `make_move` tool call.** `guided-json` (`response_format`
  json_schema) and `auto-tool` are **selectable** via `--move-channel {forced-tool,guided-json,auto-tool}`,
  chosen once for the **whole run** (so the channel is consistent across models). If a grammar
  backend rejects forced tool decoding (a tool-related HTTP 400), the run **fails loud with a
  remedy** ("re-run with `--move-channel guided-json`") rather than silently switching channel
  mid-run — a silent switch would make that run's channel differ from the others (a hidden
  fairness confound).
- **Reasoning carry = the visible `reasoning_content`**, echoed back under the **same field the
  server emitted** so the model appends to its trace. Preservation is **template-dependent**, so a
  startup **round-trip probe** (two `/chat/completions` prompt-token calls that mirror the move
  channel; one retry on error) checks whether the server actually feeds carried reasoning back,
  and reports `PRESERVES` / `STRIPS` / `UNVERIFIED`. Every turn then carries an explicit
  three-state `continuity` (`verified` / `unverified` / `stripped`) in the terminal line and the
  JSONL — an unverified carry renders as `reasoning carried (unverified)`, never as a bare
  verified claim, and a proven no-op says `server STRIPS it`.
- **Server provenance:** at startup the client records the chosen model's raw `/v1/models` entry
  and (best-effort) `GET /get_server_info` — SGLang's own launch configuration (model revision,
  tokenizer/chat template, dtype/quantization, parsers) — into the session JSONL row, since all
  of those can change results.
- **`--no-reasoning-carry`** disables the carry (ablation; carry is ON by default).
- `--thinking-level` does **not** apply (reasoning depth is a server-launch concern).

## Transport & timeouts
The client **streams** the response (`stream:true`, `stream_options.include_usage`) and reassembles
the SSE deltas into the same response a non-streamed call returns — streaming changes only error
handling, never the model's output, move, or token accounting.

Streaming makes the socket read timeout a **per-chunk idle timeout** — `--stream-idle-timeout`
(default **180 s**), reset on every chunk:
- a **silent / wedged** server (no bytes arriving) trips it and the call is retried;
- a **slow-but-progressing** generation (a long chain over a large context, decoding at a few
  tokens/s) is **never** cut off — each token resets the clock.

Because of this, `--api-timeout-seconds` (a single *total* timeout) does **not** apply to SGLang: a
total timeout can't distinguish a stalled server from a healthy-but-slow one. Set
`--stream-idle-timeout` above the worst-case **cold prefill time-to-first-token** at your context
size — a cold ~1M-token prefill can take tens of seconds before the first token, though prefix
caching keeps later turns fast. (The one-time `/v1/models` discovery GET uses a fixed short ceiling
and is unaffected.)

Transient failures (5xx / 429 / network / idle-timeout) retry with exponential backoff + jitter, up
to `--max-api-retries`; each idle-timeout attempt is cheap, so a generous retry budget rides out a
brief server reload without crashing a long run. Deterministic 4xx fail fast and dump the request to
the run dir.

## Accounting
- **USD `n/a`** (self-hosted). Token counts are exact from `usage`.
- If the server reports `completion_tokens_details.reasoning_tokens`, `out`/`think` are **disjoint
  and exact** (`out = completion − reasoning`, `thoughts_basis="exact"`). If it doesn't (older
  builds) but the model reasoned, the total can't be split, so it's shown **merged** as
  `out + think N` (`thoughts_basis="included"`) — never an invented split.

## Smoke-test (from inside the repo)
```bash
python -m core.cli P-1 --provider sglang --base-url http://host:30000/v1 \
  --model kimi-k3 --context-budget 2000 --max-steps 30 --verbose
```
Confirm: the `🔎 reasoning round-trip probe (…): server PRESERVES/STRIPS …` line; whether
forced-tool held (a grammar-backend 400 instead fails loud with a `--move-channel guided-json` hint — no silent fallback); `↺` truncations under the tiny budget;
`out + think`/disjoint accounting (USD `n/a`); debrief + playback link present.

## Notes
- **Forced tool-choice + a reasoning model** can be fragile on some grammar
  backends (xgrammar mid-decode rejects → 400). If your build rejects it, the 400 fails loud and
  you re-run with `--move-channel guided-json` (a whole-run choice — the client does not switch
  channel mid-run).
- **`--move-channel auto-tool`** (`tool_choice:auto`) lets a model think *then* call the tool.
  Some templates emit a `<think>` block *before* the tool call (e.g. DeepSeek V4 with
  `chat_template_kwargs.thinking:true`); the **streaming** client reassembles the tool call from
  deltas even after a think block, so these run on the default forced-tool. `auto-tool` remains
  available for a build that drops the call anyway. (Record the channel if you ever switch, for
  cross-run fairness.)
- **Context window behind a proxy:** an OpenAI-compatible router (e.g. LiteLLM) often strips
  `max_model_len` from `/v1/models`, so auto-detect comes back empty → pass `--context-budget`. To
  read the *served* window through such a proxy, send an oversized `max_tokens` and parse the limit
  from the error:
  ```bash
  curl -s $BASE/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"<m>","messages":[{"role":"user","content":"hi"}],"max_tokens":100000000}'
  # -> "...maximum context length of N tokens..."
  ```
- `completion_tokens` includes reasoning tokens — don't compare `out` across providers naively.
