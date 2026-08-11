# DiG-bench baseline harness

The agent behind the **baseline harness** results in the DiG-bench report (Figure 1;
Section 4.1.1). One agent plays [DiG-bench](https://digbench.ai/) games across **Gemini,
Anthropic, OpenAI, and open models (served via SGLang / any OpenAI-compatible endpoint)** —
with each model's **reasoning carried across turns** and a **rolling-window truncation sized
to the model's own context window**. 

As in the report, the agent is told each game's objective and any special actions, but **not
the rules** — it must discover the mechanics through interaction and experimentation, turn
by turn.

---

## Relation to the report

This is the minimal harness of Section 4.1.1: it tests the ability of base LLMs to make
discoveries **without additional scaffolding** — no tools, no memory files, nothing beyond
the game loop itself.

- **The loop.** At the start of a run the harness sends the model the task description and
  initial game state. Then, each turn: the updated state is appended to the conversation,
  the model submits exactly one legal action via a `make_move` tool call, and the action is
  applied — repeating until the game ends.
- **Prompts.** The system instruction and first-turn text reproduced in Appendix A live
  verbatim in [`core/prompts.py`](core/prompts.py). Every provider receives identical
  wording.
- **Fairness.** Comparing models is only fair if everything *except the model* is held
  constant: identical prompt, task framing, state slice, and move channel, and a history
  budget set from **each model's own context window minus the same absolute output
  reserve** (so every model retains as much history as its window allows). **Sampling
  parameters are never sent** — no `temperature`/`top_p` on any provider, so every model
  runs its own defaults (deliberate: reasoning models generally fix their sampling;
  forcing one value across providers would be a different intervention per model). What
  legitimately differs — and is documented per provider, not forced — is each provider's
  native reasoning-continuity mechanism: Gemini thought signatures, Anthropic thinking
  signatures, OpenAI encrypted reasoning items, visible reasoning in SGLang-served open
  models. Preserving these gives each model the best chance of performing well.
- **Truncation.** In very long games the observation-action history is truncated to fit the
  context window by dropping the oldest whole step-pairs — truncation, not summarization
  (a minimal harness must not help the model remember). This is the mechanism behind the
  truncated runs counted in Section 4.1.1 (34 of 554).
- **Models.** The baseline-harness conditions of Table 2 map onto the providers below:
  Opus 5 and GPT-5.5 through Amazon Bedrock (`--provider anthropic` / `openai`), Gemini 3.1
  Pro Preview through Google's API (`--provider gemini`) — the provider defaults are the
  report's recorded identifiers (`global.anthropic.claude-opus-5`, `openai.gpt-5.5`,
  `gemini-3.1-pro-preview`) — and the open-weights models (Kimi K3, GLM-5.2, DeepSeek V4,
  Qwen3.6 27B) served with SGLang (`--provider sglang`). Reasoning effort was `high` —
  `max` where an open model exposed one.

## Install

Python **3.10+**.

```bash
pip install -r requirements.txt          # or install just the provider(s) you need:
# pip install "google-genai>=2.8"        # gemini
# pip install "anthropic[bedrock]>=0.111"# anthropic (AWS Bedrock)
# pip install "openai>=2.24"             # openai (Responses API)
# sglang needs no extra dependency (stdlib HTTP)

# reproduce the validated environment exactly (pinned SDK versions):
pip install -r requirements.txt -c constraints.txt
```

The harness and the SGLang client use only the standard library; each closed-provider client
imports **only its own SDK, lazily**, so you install what you use.

## The benchmark API

The baseline harness talks to the DiG-bench Agent API — the same interface the report's runs used,
publicly available for the 21 public games (`POST /api/agent/sessions`,
`POST /api/agent/sessions/{id}/step`; full reference at
[digbench.ai/api](https://digbench.ai/api)). It points at the public server
`https://api.digbench.ai` by default, so all you need is a token:

```bash
export DIGBENCH_API_TOKEN="<token>"   # mint at https://digbench.ai/account/tokens
```

## Quickstart

Run from **inside this directory** (`core/` and `clients/` are top-level packages). The game
argument is any name from `GET /api/agent/games` — omit it to pick from a menu.

```bash
# Gemini
GEMINI_API_KEY=...  python3 -m core.cli P-1 --provider gemini --thinking-level high --verbose

# Anthropic on AWS Bedrock (bearer token)
AWS_BEARER_TOKEN_BEDROCK=...  python3 -m core.cli P-1 --provider anthropic \
    --model global.anthropic.claude-opus-5 --aws-region us-east-1 --verbose

# OpenAI (Responses API; default model openai.gpt-5.5 routes via AWS Bedrock)
AWS_BEARER_TOKEN_BEDROCK=...  python3 -m core.cli P-1 --provider openai --verbose
# …or direct to api.openai.com with a bare model id
OPENAI_API_KEY=...  python3 -m core.cli P-1 --provider openai --model gpt-5.5 --verbose

# Open model via an SGLang / OpenAI-compatible endpoint
python3 -m core.cli P-1 --provider sglang --base-url http://host:30000/v1 --verbose
```

> **Credentials.** The inline `KEY=...` prefix above is just one way. You can also `export`
> the variables once, or have a secrets manager of your choice inject them into the
> environment — anything that sets the variables works.

Omit `--provider` (or the game, or `--model` on SGLang when several models are served) and you'll
be prompted to pick from a menu.

## Providers

| `--provider` | Surface | Move channel | Reasoning carried as | Doc |
|---|---|---|---|---|
| `gemini` | google-genai | forced `make_move` (`mode=ANY`) | `thought_signature` (verbatim) | [docs/gemini.md](docs/gemini.md) |
| `anthropic` | Messages API — Bedrock default, direct API for bare `claude-*` ids | auto-tool + nudge¹ | `thinking` block `signature` | [docs/anthropic.md](docs/anthropic.md) |
| `openai` | **Responses API** (not Chat Completions) — Bedrock default, direct for bare ids | forced `make_move` | encrypted reasoning items (`store=false`) | [docs/openai.md](docs/openai.md) |
| `sglang` | OpenAI-compatible Chat Completions | forced tool; guided-JSON selectable | `reasoning_content` (probe-verified) | [docs/sglang.md](docs/sglang.md) |

¹ Anthropic forbids forced `tool_choice` while extended thinking is on, so the thinking path uses
`tool_choice:auto` + a nudge fallback (a documented, API-mandated fairness difference).

Model families whose thinking is configured by a **token budget** rather than a level/effort knob
(Anthropic pre-effort models like Haiku 4.5, and Gemini 2.5) translate `--thinking-level` through
**one shared level→budget table** (`core/clientutil.THINKING_LEVEL_BUDGETS`) — the same nominal
budget per level on every provider, clamped only by each API's own bounds (see the provider docs).

## How it works

- **Provider-agnostic core** (`core/`): the turn loop (`harness.py`), rolling-window truncation
  (`history.py`), token/USD accounting (`accounting.py`), shared retry/error policy
  (`clientutil.py`), terminal+JSONL output (`output.py`), prompts (`prompts.py`), CLI (`cli.py`).
  It carries **no game-solving logic and no provider specifics** — it drives any client through
  the `Policy` protocol in `types.py`.
- **One client per provider** (`clients/*_client.py`): owns only that provider's request shape,
  where its continuity token lives, turn-boundary detection, and **verbatim** re-serialization.
- **Reasoning carry** — the model's output is appended **verbatim** each turn so it *continues*
  its chain instead of re-deriving. On Gemini/Anthropic/OpenAI the continuity token is
  cryptographically validated by the API (a tampered/missing one 400s), so carry is self-checking
  and `has_continuity` is honest; on SGLang it's the plaintext `reasoning_content`, verified once
  at startup by a round-trip probe.
- **Context truncation** — *truncation, not summarization*: drop only the earliest **whole
  step-pairs**, never touch the active turn, keep the pinned task. Reactive on the
  server-reported prompt size; budget = **each model's own window minus a fixed reserve for one
  turn's output** (~1.5 × `--max-tokens`, since output counts toward the hard window and truncation
  lags one turn) — so every model keeps as much history as its window allows while reserving the
  same absolute output room. `--context-proportion` (default 1.0) optionally adds a larger flat
  buffer; `--context-budget` pins an absolute value.
- **Honest accounting** — `out`/`think` are reported **disjoint** (exact reasoning tokens) on
  every provider that exposes the count; when a server reports a total that includes reasoning but
  no breakdown, it's shown merged as `out + think N` rather than guessing a split. Long-context
  price tiers (Gemini Pro >200K, direct gpt-5.5 >272K) are **applied per call**, not just warned
  about. Reasoning-continuity is likewise three-state honest: `verified` (API-validated or
  probe-proven) / `unverified` / `stripped`, in the terminal and the JSONL.
- **Self-describing artifacts** — every `.jsonl` session row carries full provenance
  (`core/provenance.py`): git commit + dirty flag, Python/platform + provider-SDK versions, the
  complete resolved CLI config (secrets excluded), the endpoint actually hit, and a sha256 of the
  prompt contract; the summary row records the **server-reported model revision**
  (`reported_model`). SGLang runs additionally capture the server's `/get_server_info` launch
  configuration.
- **Error handling** — transient errors (5xx / 529, 429 rate-limit, 408, network / timeout, and
  read/decode failures) retry with exponential backoff + jitter; deterministic ones (other 4xx,
  plus a permanent `insufficient_quota` 429, plus programming errors like `TypeError` from SDK
  drift) fail fast and dump the request to `runs/` for
  inspection. SGLang responses are **streamed**, so
  its timeout is a per-chunk *idle* timeout — a stalled server is caught fast while a
  slow-but-progressing generation is never cut off (see [docs/sglang.md](docs/sglang.md)).

## Key CLI flags

| flag | meaning |
|---|---|
| `--provider {gemini,sglang,anthropic,openai}` | required (prompts if omitted) |
| `--model` | model id (provider default applied; SGLang auto-IDs / prompts) |
| `--thinking-level {minimal,low,medium,high,xhigh,max,none}` | reasoning depth (maps to each provider's level/effort knob, or through the shared budget table on budget-based families; `none` disables) |
| `--context-budget N` / `--context-proportion F` | absolute budget, or an optional extra flat buffer `(1−F)×window` on top of the output reserve (default F=1.0 → reserve only) |
| `--max-tokens` | universal per-call output-token cap (all providers) |
| `--max-steps`, `--max-cost-usd` | turn cap / cost cap |
| `--api-timeout-seconds` / `--stream-idle-timeout` | per-call timeout (gemini/anthropic/openai) / SGLang inter-chunk idle timeout |
| `--max-api-retries` | transient-error retry budget (exponential backoff + jitter) |
| `--no-thought-summaries` | drop the human-readable reasoning summary (cheaper) |
| `--seed` | sampling seed (honored on Gemini/SGLang; warned no-op on Anthropic/OpenAI, which have none) |
| `--no-save-run`, `--run-dir`, `--verbose` | artifacts + output |

Full `--help` lists every flag with defaults.

## Layout

```
core/        provider-agnostic harness, truncation, accounting, retry, output, CLI, Policy
clients/     one client per provider (gemini / anthropic / openai / sglang)
docs/        per-provider operator docs + the API reference (API.md, api/index.html)
tools/       gen_api_docs.py — regenerates the API reference from source
tests/       isolation tests (fakes only; no network) — run any test_*.py directly
```

## Tests

Fakes only — no network or API keys required:

```bash
python3 -m pytest -q        # aggregates results across files
```

Provider SDKs are optional: the OpenAI/Anthropic/SGLang suites run with no SDK installed
(fake clients are injected). The Gemini suite and the 4-provider integration test build a
real `GeminiPolicy`, which needs `google-genai`, so they **skip cleanly** when it's absent.
CI (`.github/workflows/ci.yml`) runs the suite on Python 3.10 + 3.13 with all SDKs, once with
**no** SDKs (proving the skip claim), and fails if the generated API docs drift from source.

## Reproducing results

Game instances are randomized by the bench server: `POST /sessions` assigns a **game seed**,
which every run records (session JSONL row, terminal header). The client cannot currently
*request* a seed, so two runs — even of the same model — play different instances; requesting a
seed for paired runs is future server-side work (the recorded seed is already sufficient to pair
runs once the server honors one).

Cross-model comparisons should therefore use a **repeated-run protocol**:

- Run **N repetitions per model per game** (same flags; the model id, thinking level, and every
  resolved flag are in each artifact's `provenance.config`).
- Report the **mean ± a confidence interval of `levels_beaten`** (the headline metric, in every
  summary row) across repetitions, not a single run.
- **Failed runs are data, not noise**: runs stopping `api_failure` / `blocked` /
  `server_protocol` are reported alongside (with their stop reason), never silently dropped;
  exclude a run only for an infrastructure cause outside the model's control (e.g. the bench
  server died), and say so.
- Each artifact is self-describing (provenance: git commit, SDK versions, resolved config,
  prompt-contract hash; summary: `reported_model`) — two runs are comparable iff those match.
- Pin the environment with `constraints.txt`; on the direct OpenAI API prefer immutable snapshot
  ids (see [docs/openai.md](docs/openai.md)). `--seed` fixes *sampling* where supported
  (Gemini/SGLang) but not the game instance.

## Documentation

- Per-provider operator guides: [`docs/`](docs/) (gemini, anthropic, openai, sglang).
- **Full API reference** (every module / class / method): [`docs/API.md`](docs/API.md) and the
  browsable HTML at [`docs/api/index.html`](docs/api/index.html) — regenerate with
  `python3 tools/gen_api_docs.py`.
- Per-provider model + pricing tables live in each provider doc (above) and in the client
  source (`clients/<provider>_client.py`).

## License

MIT — see [LICENSE](LICENSE). Cite via [CITATION.cff](CITATION.cff).

## Caveats

- **Cost figures are approximate.** Per-model pricing rows are first-party list rates and may
  not match a given platform's billing (e.g. Bedrock regional premiums). Token counts are
  exact; only the USD rates carry uncertainty. Long-context tiers (Gemini Pro >200K, direct
  gpt-5.5 >272K) are applied per call.
- Model ids, context windows, and pricing **drift between generations** — verify against current
  provider docs.
