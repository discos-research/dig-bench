# DiG-bench baseline harness — API reference

> Auto-generated from source docstrings + signatures by `tools/gen_api_docs.py`. Re-run that script to refresh.

## Modules

- [`core.types`](#coretypes) — Shared dataclasses + the Policy interface the harness drives and the
- [`core.prompts`](#coreprompts) — Shared, provider-neutral prompt text and the make_move tool contract.
- [`core.accounting`](#coreaccounting) — Generic token + USD accounting mechanism: longest-substring model matching and
- [`core.clientutil`](#coreclientutil) — Shared, provider-agnostic helpers for the LLM client classes.
- [`core.bench`](#corebench) — Agent Benchmark API HTTP client (stdlib only) + the game-state slice helpers.
- [`core.history`](#corehistory) — The shared turn-aware rolling-window context truncation.
- [`core.output`](#coreoutput) — Output — streams one turn at a time to the terminal and, when run artifacts are
- [`core.harness`](#coreharness) — The provider-agnostic game loop.
- [`core.cli`](#corecli) — CLI + wiring: parse args, resolve credentials, build the provider policy + bench,
- [`core.provenance`](#coreprovenance) — Run provenance — everything a published trace needs to be reproducible.
- [`clients.gemini_client`](#clientsgeminiclient) — Gemini client — GeminiPolicy, implements core.types.Policy.
- [`clients.anthropic_client`](#clientsanthropicclient) — Anthropic client — AnthropicPolicy, implements core.types.Policy.
- [`clients.openai_client`](#clientsopenaiclient) — OpenAI client — OpenAIPolicy, implements core.types.Policy.
- [`clients.sglang_client`](#clientssglangclient) — SGLang client — SglangPolicy, implements core.types.Policy.

---

## `core.types`

> Shared dataclasses + the Policy interface the harness drives and the
> rolling-window context truncation operates through. No provider or game logic here.
> 
> The truncation surface (turns / evict_oldest_turn) is the seam: core/history.py
> uses ONLY these methods and never reconstructs a provider payload, so each
> provider's verbatim re-append stays inside its own client.

### class `Move`

One LLM turn's outcome. `action` is None when the model made no move.

- **`__init__(self, action: 'str | None', reasoning_summary: 'str', has_continuity: 'bool', finish_reason: 'str | None', usage: 'dict', cost: 'float | None', elapsed_s: 'float', error: 'dict | None' = None, thoughts_basis: 'str' = 'none', continuity: 'str | None' = None) -> None`**  
  Initialize self.  See help(type(self)) for accurate signature.

### class `Turn`

One evictable unit. For the forced-tool bench loop a unit is a completed
step-pair (model functionCall+signature + its functionResponse).
The payload is the provider-native chunk(s), kept VERBATIM — never rebuilt.

- **`__init__(self, payload: 'object', is_active: 'bool', est_tokens: 'int') -> None`**  
  Initialize self.  See help(type(self)) for accurate signature.

### class `Policy`

What harness.play() drives and history.truncate_if_needed() truncates.

- **`start(self, description: 'str', state_slice: 'dict') -> 'None'`**  
  _(no docstring)_

- **`generate_move(self) -> 'Move'`**  
  _(no docstring)_

- **`observe(self, result: 'dict') -> 'None'`**  
  _(no docstring)_

- **`add_nudge(self, text: 'str') -> 'None'`**  
  _(no docstring)_

- **`debrief(self) -> 'str | None'`**  
  _(no docstring)_

- **`turns(self) -> 'list[Turn]'`**  
  _(no docstring)_

- **`evict_oldest_turn(self) -> 'None'`**  
  _(no docstring)_

- **`__init__(self, *args, **kwargs)`**  
  _(no docstring)_

---

## `core.prompts`

> Shared, provider-neutral prompt text and the make_move tool contract.
> 
> Every provider uses the SAME wording (fairness) and builds its own
> native tool from one neutral spec. Deliberately NO `google.genai` import here —
> each client translates MAKE_MOVE_SPEC into its own tool object (Gemini
> FunctionDeclaration, OpenAI/SGLang function tool, Anthropic tool), keeping this
> module dependency-free.

### Functions

#### `creative_mode_instruction(toggle: 'str | None') -> 'str'`

Human-facing creative-mode guidance plus the game's exact action token.

#### `seed_text(description: 'str', state_slice: 'dict') -> 'str'`

The first user turn: the task description (objective + special actions; NOT the rules) + the
initial state. IDENTICAL wording for every provider (fairness).

---

## `core.accounting`

> Generic token + USD accounting mechanism: longest-substring model matching and
> the per-call cost formula.
> 
> This module holds **no provider data**. Each client owns its own pricing table,
> context-window table, and any provider-specific tier thresholds
> (clients/<provider>.py) and passes its pricing table into compute_cost here — so
> adding a provider touches only that client, never this file.

### Functions

#### `match_model(model: 'str', table: 'dict')`

Value under the LONGEST table key that is a substring of `model`; None if no
key matches. An exact id wins (it is its own longest substring). The one matcher
for both pricing rows and context-window lookups across every provider — substring
(not prefix) so it tolerates regional id prefixes like `us.`/`global.` on
Bedrock/Vertex model ids (e.g. `us.anthropic.claude-...` still matches `claude-...`).

#### `pricing_for(model: 'str', table: 'dict') -> 'dict | None'`

_(no docstring)_

#### `compute_cost(model: 'str', prompt: 'int', cached: 'int', output: 'int', thoughts: 'int', table: 'dict') -> 'float | None'`

Per-call USD. Cached tokens are billed at the cheaper cached rate; the
rest of the prompt at full input rate. Returns None if the model is unpriced.

A pricing row may carry an optional ``"long_context"`` sub-row
(``{"threshold": N, <same rate keys>}``): calls whose prompt exceeds the threshold are
billed at the sub-row's rates instead (per call — providers price the whole request at
the higher tier once the input crosses it). Rows without the key stay flat.

---

## `core.clientutil`

> Shared, provider-agnostic helpers for the LLM client classes.
> 
> Reusable infrastructure only — no game logic and no single-provider knowledge. It holds
> the small helpers and the retry/error machinery shared by the four clients:
> 
> - ``zero_usage`` / ``est_tokens`` — trivial shared utilities. (Model→table matching
>   lives in ``accounting.match_model`` — one matcher for pricing + context-window lookups.)
> - ``extract_status`` / ``is_retryable`` — one HTTP-error classification policy for every
>   provider (the SDKs expose the status differently; see ``extract_status``).
> - ``dump_request`` — the 4xx request dump for offline debugging.
> - ``run_request`` — the retry loop with exponential backoff used by the closed-model clients
>   (Gemini/Anthropic/OpenAI). SGLang keeps its own loop (it accumulates a streamed SSE response and
>   has provider-specific move-channel handling) but reuses the classifier + dump here.
> - ``iter_sse_data`` / ``accumulate_chat_stream`` (+ ``StreamError``) — fold an OpenAI-compatible
>   chat-completions SSE stream back into the same response dict a non-streamed call returns. Only
>   SGLang streams today; kept here so it is socket-free unit-testable and reusable by any
>   OpenAI-compatible client. Iterating the live response makes the socket read timeout act as a
>   per-chunk idle timeout (no watchdog thread).
> 
> Error taxonomy:
> - retryable: HTTP 500/502/503/504, Anthropic 529 (overloaded), 429 rate-limit, and
>   network/timeout errors (no HTTP status);
> - fatal (fail-fast): 400/401/403/404/413/422, AND the one permanent 429 — OpenAI
>   ``insufficient_quota`` (a billing state; backoff cannot create quota). Anthropic billing is a
>   400 (already fatal); Gemini has no clean permanent-429 marker, so its 429 stays retryable.
> - fatal (fail-fast): programming errors (TypeError/AttributeError/KeyError/IndexError/NameError
>   — SDK drift or our bugs), whatever their status; retrying them only hides the traceback
>   behind minutes of backoff. ValueError stays transient (json/decode paths raise it for
>   genuinely transient read garbage).

### Functions

#### `clamp_thinking_budget(level: 'str', max_tokens: 'int', *, floor: 'int', cap: 'int | None' = None) -> 'int'`

The manual thinking budget for ``level``, clamped to a family's API bounds. On every
budget-based family the thinking spends from the SAME ``max_tokens`` output allowance, so
at most ``max_tokens - 1024`` is granted (1024 tokens of visible output always remain);
``floor``/``cap`` are the family's hard bounds (e.g. Anthropic floor 1024; Gemini 2.5
flash cap 24576, pro floor 128 / cap 32768).

The floor GATES, it never bumps up: when ``max_tokens`` leaves less room than the floor
(or would resolve to a 0 budget — thinking silently OFF while the run is labeled with a
level), the combination is INFEASIBLE and raises ValueError. Bumping up instead would
either violate the API (Anthropic requires budget < max_tokens) or mislabel the run.

#### `backoff_sleep(attempt: 'int', cap: 'int' = 60) -> 'None'`

Sleep an exponential backoff (``2**attempt``, capped at ``cap``) plus jitter. The jitter
desynchronizes concurrent clients so their retries don't form a thundering herd against one
server. Calls ``time.sleep`` at call time so tests can neutralize the wait.

#### `urlopen_no_redirect(req, timeout)`

``urllib.request.urlopen`` that refuses to follow redirects (3xx raises ``HTTPError``).
Use for every request that carries credentials.

#### `iter_sse_data(line_iterable)`

Yield the ``data:`` payload string of each SSE event from a line iterable (a urllib
response iterates as ``bytes`` lines; tests can pass a list/``BytesIO``). Skips blanks and
``:`` keep-alive comments; stops at the ``[DONE]`` sentinel. The blocking read happens here,
so the socket's read timeout fires naturally on a stalled stream.

#### `accumulate_chat_stream(line_iterable, *, require_usage: 'bool' = False) -> 'dict'`

Fold a chat-completions SSE stream into the non-streamed response shape:
``{"choices":[{"message":{...}, "finish_reason":...}], "usage":{...}}``. Concatenates
content / reasoning / tool-call-argument deltas, preserves WHICH reasoning key the server
used (``reasoning_content`` vs ``reasoning`` — emit only that one), and takes ``usage`` from
the final (choices-empty) chunk. Raises ``StreamError`` on an error chunk. With
``require_usage`` (the caller requested ``stream_options.include_usage``), a stream that
ends without the usage trailer is treated as truncated too.

#### `zero_usage() -> 'dict'`

_(no docstring)_

#### `est_tokens(chunk) -> 'int'`

Cheap size proxy (chars // 4) over a unit's serialized form — for MINIMAL eviction, not
billing. Provider payload objects stringify to their repr, which is fine as a proxy.

#### `extract_status(exc) -> 'int | None'`

The HTTP status from a provider exception, or None (network/timeout/unknown).
OpenAI/Anthropic SDK errors expose `.status_code`; google-genai `APIError.code` and the
SGLang `ChatError.code` are the int status under `.code`.

#### `is_retryable(exc) -> 'bool'`

Transient (retry) vs deterministic (fail-fast). See the module docstring for the policy.

#### `is_context_overflow(err) -> 'bool'`

True if ``err`` (a failed-Move error dict, or an exception) is a context-window overflow —
recoverable by evicting history and retrying, vs a generic fatal 4xx.

#### `parse_overflow_tokens(err) -> 'tuple[int, int] | None'`

``(requested_total, limit)`` parsed from a context-overflow error message, or None when
the wording carries no usable counts (then the caller falls back to blind eviction).
Only a coherent pair (total > limit > 0) is returned — never a negative deficit.

#### `dump_request(debug_dir: 'str', provider: 'str', exc, request, call_count: 'int') -> 'str | None'`

Write the failing request to ``<provider>_4xx_<ts>_<n>.json`` for offline debugging.
The request is dumped verbatim when JSON-serializable (e.g. SGLang payloads); otherwise we
record just its keys (SDK payloads carry non-serializable block/item objects).

#### `run_request(call: 'Callable', *, provider: 'str', max_retries: 'int', request, debug_dir: 'str', call_count: 'int' = 0, on_event: 'Callable | None' = None)`

Call ``call()`` with retries. Returns ``(result, None)`` on success, else
``(None, error_dict)``. Deterministic errors (``is_retryable`` False) fail fast and dump the
request; transient errors retry with exponential backoff + jitter (see ``backoff_sleep``,
capped at ``BACKOFF_CAP`` s). Backoff uses ``time.sleep`` at call time (so tests can patch it).

### class `_RefuseRedirect`

_(no docstring)_

- **`redirect_request(self, req, fp, code, msg, headers, newurl)`**  
  Return a Request or None in response to a redirect.
  
  This is called by the http_error_30x methods when a
  redirection response is received.  If a redirection should
  take place, return a new Request to allow http_error_30x to
  perform the redirect.  Otherwise, raise HTTPError if no-one
  else should try to handle this url.  Return None if you can't
  but another Handler might.

### class `StreamError`

A streamed response carried an error chunk. ``code`` is the HTTP-ish status if the server
gave one (else None -> treated as transient by ``is_retryable``). Provider code maps this to
its own error type (keeps this module provider-agnostic).

- **`__init__(self, code, body)`**  
  Initialize self.  See help(type(self)) for accurate signature.

---

## `core.bench`

> Agent Benchmark API HTTP client (stdlib only) + the game-state slice helpers.
> 
> This is the sole game
> interface and its contract (idempotent stepping off the server's returned
> ``step_index``) is identical across every provider, so it lives in the
> provider-agnostic core. Carries no game-solving intelligence.

### Functions

#### `state_for_model(state: 'dict') -> 'dict'`

The slice of game state handed to the model (seed text / function result).

#### `fmt_level(state: 'dict') -> 'str'`

_(no docstring)_

#### `levels_beaten(state: 'dict') -> 'int | None'`

Levels cleared so far — the run's headline performance metric, DERIVED from
`level` (the bench carries no per-turn score): mid-game you have cleared
`level - 1`, and a fully `completed` game has beaten every level (`max_level`).
Returns None if `level` is absent (the bench always sends it; this is purely
defensive).

Display/analysis ONLY: recorded in the trace and shown in the operator terminal,
but NEVER part of `state_for_model` — the model tracks progress via
`level`/`transition` exactly as a human does (parity), and is handed no score.

#### `terminal_banner(state: 'dict') -> 'str'`

_(no docstring)_

### class `BenchError`

A bench-API call failed (after retries, or a non-retryable 4xx).

### class `Bench`

Thin HTTP client for the Agent Benchmark API (stdlib only).

Retries transient (5xx / network) errors with backoff; 4xx are raised
immediately. Stepping is idempotent: callers send the server's returned
step_index + 1, and resending an applied index returns the cached response.

- **`__init__(self, base: 'str', token: 'str', *, timeout: 'int', max_retries: 'int')`**  
  Initialize self.  See help(type(self)) for accurate signature.

- **`_call(self, method: 'str', path: 'str', payload: 'dict | None' = None) -> 'dict'`**  
  _(no docstring)_

- **`list_games(self) -> 'list[str]'`**  
  _(no docstring)_

- **`start_session(self, game: 'str', model_name: 'str', model_version: 'str') -> 'dict'`**  
  _(no docstring)_

- **`step(self, sid: 'str', step_index: 'int', action: 'str') -> 'dict'`**  
  _(no docstring)_

---

## `core.history`

> The shared turn-aware rolling-window context truncation.
> 
> This is the ONLY place the rolling window lives. It operates purely through the
> Policy truncation surface (turns / evict_oldest_turn; the active unit is the one
> Turn with is_active=True, never evicted) and never reconstructs a provider
> payload, so it is identical for every provider. ``evict_for_overflow`` is the
> reactive companion: deficit-sized eviction after a server overflow 4xx (used by
> the harness's move retry and the SGLang debrief retry).
> 
> Note: this is *truncation* — drop the earliest whole step-pairs — NOT
> consolidation/summarization. By design we never summarize evicted
> turns: this is a minimal benchmark and must not help the model beat the games.
> 
> Invariants:
> - evict only WHOLE, oldest, completed units (for the bench loop: step-pairs);
> - NEVER touch the active unit; the client also pins the head task;
> - keep >= MIN_KEEP_TURNS units;
> - evict the FEWEST units needed to get just under budget (minimal eviction).
> 
> Note: the trigger, `policy.last_prompt_tokens`, is each provider's own server-reported,
> cache-inclusive prompt size, and tokenizers differ — so "the same proportion of the window"
> is consistent in intent but only approximate in absolute tokens ACROSS providers.
> 
> Fairness note (interpreting cross-model results): the output reserve (~1.5 × max_tokens, see
> cli.resolve_budget) is an ABSOLUTE token count, not a proportion — a model retains
> `window − reserve` of history, which is a LARGER fraction of a big window than a small one
> (e.g. ~95% of a 1M window vs ~81% of 256K at the default 32K cap). This is intentional (output
> needs an absolute reserve), but small-window models keep proportionally less context per turn;
> lower `--max-tokens` to shrink the reserve, or set `--context-proportion < 1` for a proportional
> buffer, if tighter per-window parity is wanted.

### Functions

#### `truncate_if_needed(policy: 'Policy', budget: 'int | None', *, log=None) -> 'int'`

Reactively shrink the next request if the LAST call's server-reported
prompt size exceeded `budget`. Returns the number of units evicted.

The trigger is reactive: we only learn `prompt_tokens` after a call, so we
evict before the next one. Eviction is minimal — we project the prompt size
down by each evicted unit's `est_tokens` and stop as soon as the projection
is under budget (never reflexively down to the floor). The next call's real
`last_prompt_tokens` corrects any estimate drift.

#### `evict_for_overflow(policy: 'Policy', err, *, round_idx: 'int' = 0) -> 'int'`

Reactive recovery for a context-overflow 4xx: evict enough oldest units to cover the
deficit the server itself reported. Returns the number of units evicted; 0 means eviction
cannot shrink further (only the pinned head + active unit remain) — the caller gives up.

The overflow wording states the exact numbers ("requested a total of T tokens ...
maximum context length of L"); evict oldest units until their summed ``est_tokens``
covers ``(T - L) * _OVERFLOW_MARGIN + _OVERFLOW_SLACK``. Sizing to the deficit matters:
the oldest units are the smallest (early turns carry little/no reasoning), so a fixed
per-retry count starves against a multi-thousand-token overshoot. When the wording has
no counts, fall back to a doubling batch (``2**round_idx``) so repeated rounds still
converge. No MIN_KEEP_TURNS floor here — this mirrors the recovery path's pre-existing
semantics (the client itself refuses to evict the pinned head / lone active unit).

---

## `core.output`

> Output — streams one turn at a time to the terminal and, when run artifacts are
> saved (the default; disable with --no-save-run), to a `.log` (identical text) and a
> `.jsonl` (structured).
> 
> Provider-agnostic: typed
> against core.types.Policy, with a one-line
> truncation-event record so a rolling-window eviction is visible in the trace.

### class `Output`

Plain unicode only, so the file and the screen match.

- **`__init__(self, *, summary_chars: 'int', verbose: 'bool', log_path=None, jsonl_path=None)`**  
  Initialize self.  See help(type(self)) for accurate signature.

- **`_w(self, text: 'str' = '') -> 'None'`**  
  _(no docstring)_

- **`_record(self, obj: 'dict') -> 'None'`**  
  _(no docstring)_

- **`session_header(self, *, start: 'dict', model: 'str', thinking_level, run_label: 'str', seed=None, include_thoughts: 'bool' = False, context_budget=None, pricing_row=None, model_max_context=None, provenance: 'dict | None' = None) -> 'None'`**  
  _(no docstring)_

- **`turn_header(self, turn: 'int', state: 'dict') -> 'None'`**  
  _(no docstring)_

- **`_emit_reasoning(self, move: 'Move') -> 'None'`**  
  _(no docstring)_

- **`_emit_meta(self, move: 'Move') -> 'None'`**  
  _(no docstring)_

- **`turn_result(self, turn: 'int', step_index: 'int', before: 'dict', after: 'dict', move: 'Move', invalid: 'bool', cumulative: 'float') -> 'None'`**  
  _(no docstring)_

- **`turn_nudge(self, turn: 'int', before: 'dict', move: 'Move', reason: 'str', cumulative: 'float') -> 'None'`**  
  _(no docstring)_

- **`truncated(self, turn: 'int', detail: 'str') -> 'None'`**  
  A rolling-window truncation fired after this turn — record it so the
  trace shows the mechanism actually worked.

- **`error(self, detail) -> 'None'`**  
  _(no docstring)_

- **`warn(self, msg) -> 'None'`**  
  _(no docstring)_

- **`note(self, msg) -> 'None'`**  
  Neutral diagnostic line (e.g. the reasoning round-trip probe verdict) —
  not a warning.

- **`debrief(self, text: 'str | None') -> 'None'`**  
  _(no docstring)_

- **`summary(self, *, game: 'str', state: 'dict', stop_reason: 'str', turns: 'int', policy: 'Policy', wall_s: 'float', sid: 'str', debrief_text: 'str | None', server: 'str') -> 'None'`**  
  _(no docstring)_

- **`close(self) -> 'None'`**  
  _(no docstring)_

---

## `core.harness`

> The provider-agnostic game loop.
> 
> The turn loop — stop conditions,
> invalid/nudge caps, cost cap, Ctrl-C handling, debrief-once, saved run artifacts
> — plus a turn-aware rolling-window context-truncation step after each applied
> move. Drives any core.types.Policy; carries no game logic.

### Functions

#### `play(bench, policy: 'Policy', out: 'Output', start: 'dict', args, server: 'str', thinking_level, run_label: 'str', *, context_budget: 'int | None' = None, provenance: 'dict | None' = None) -> 'str'`

_(no docstring)_

---

## `core.cli`

> CLI + wiring: parse args, resolve credentials, build the provider policy + bench,
> resolve the context-truncation budget, run play().
> 
> Beyond standard arg parsing: the context-truncation flags
> (--context-proportion / --context-budget) and budget resolution. --provider selects the
> client (gemini | sglang | anthropic | openai); it has no default — omit it and you're
> prompted to pick one.
> 
> Credentials come from environment variables (or pass --api-key). Export them directly, or
> have a secrets manager of your choice inject them into the environment.
> 
> Run (from inside the repo root):
>     # Gemini
>     GEMINI_API_KEY=...  python -m core.cli P-1 --provider gemini --verbose
>     # SGLang open model (model auto-IDed from /v1/models, or picked if several are served)
>     python -m core.cli P-1 --provider sglang --base-url http://host:30000/v1 --verbose
>     # Anthropic on Bedrock (bearer token)
>     AWS_BEARER_TOKEN_BEDROCK=...  python -m core.cli P-1         --provider anthropic --model global.anthropic.claude-opus-5 --verbose
>     # OpenAI (Responses API)
>     OPENAI_API_KEY=...  python -m core.cli P-1 --provider openai --model gpt-5.5 --verbose

### Functions

#### `game_token() -> 'str'`

The bench API token, from DIGBENCH_API_TOKEN.

#### `gemini_key() -> 'str'`

_(no docstring)_

#### `bedrock_token() -> 'str'`

The AWS Bedrock bearer token (an API key, not SigV4 creds).

#### `openai_key() -> 'str'`

_(no docstring)_

#### `anthropic_key() -> 'str'`

_(no docstring)_

#### `resolve_budget(args, model_max_context: 'int | None', *, warn=None) -> 'int | None'`

The rolling-window budget in tokens, or None to DISABLE truncation.

Absolute --context-budget wins (ablation knob). Otherwise window minus a reserve big enough
for ~1.5 outputs (`max_tokens` counts toward the hard window, and reactive truncation lags one
turn, so the previous turn's output is carried before eviction catches up), falling back to the
flat (1 - proportion) buffer when that is larger. If the window is unknown and no absolute
budget is set, truncation is disabled (never crash) — warn and tell the user the knob.

#### `default_run_label(model: 'str', thinking_level: 'str | None') -> 'str'`

_(no docstring)_

#### `parse_args(argv=None)`

_(no docstring)_

#### `ensure_cost_cap_enforceable(args, policy) -> 'None'`

cost_usd stays 0.0 on unpriced models, so --max-cost-usd would silently never
trigger — an unbounded run the operator believes is capped. Refuse before starting.

#### `build_policy(args, run_dir: 'Path')`

Construct the provider's policy. For SGLang this also queries /v1/models
(model auto-ID + context window), so call it before reading policy.model.

#### `main() -> 'None'`

_(no docstring)_

---

## `core.provenance`

> Run provenance — everything a published trace needs to be reproducible.
> 
> Captured ONCE per run (core/cli.py) and recorded in the session JSONL row
> (core/output.py), so each trace is self-describing: the exact code version (git
> commit + dirty flag), interpreter and SDK versions, the full resolved CLI config
> (minus secrets), the endpoint actually hit, and a hash of the prompt contract —
> a silently edited prompt cannot masquerade as the published one.
> 
> Stdlib only, best-effort throughout: every field degrades to None (or is omitted)
> rather than ever failing a run.

### Functions

#### `prompt_contract_sha256() -> 'str'`

sha256 over the ENTIRE ``core/prompts.py`` source. Hashing selected constants would
miss model-facing literals inside functions (``seed_text``'s framing text,
``creative_mode_instruction``, serialization choices), which could then change without
changing the fingerprint — the whole module source is the prompt contract. Two runs with
the same hash saw byte-identical task framing (the per-game description/state is the
bench server's, recorded separately).

#### `collect(args, policy, server: 'str') -> 'dict'`

The provenance dict for the session JSONL row. `args` is the RESOLVED namespace
(after model auto-ID), `policy` the constructed provider policy.

---

## `clients.gemini_client`

> Gemini client — GeminiPolicy, implements core.types.Policy.
> 
> Append-only `contents`,
> the model's Content appended VERBATIM each turn (so the thought signature rides
> forward and Gemini keeps its reasoning), the forced `make_move` tool (mode=ANY),
> exact token/USD accounting, and the tool-free debrief.
> 
> The Policy truncation surface:
> - `last_prompt_tokens` — the server-reported prompt size, the truncation trigger;
> - `turns()` / `evict_oldest_turn()` — segment the flat
>   `contents` into evictable STEP-PAIR units. A unit = one model output
>   (functionCall + signature) plus the user functionResponse that follows it. The
>   head TASK DESCRIPTION (everything before the first model output) is PINNED and
>   never a unit; the latest unit is the ACTIVE unit and is never evicted.
> 
> Retry/error handling is the shared `core.clientutil` policy: transient errors
> (5xx/429-rate/network) retry with backoff; deterministic 4xx fail fast and dump the
> request. (A signature-validation 400 thus fails fast.)

### class `GeminiPolicy`

Owns the Gemini chat: the append-only `contents`, verbatim signature
refeed, the forced make_move tool, exact accounting, and the step-pair
truncation surface.

- **`__init__(self, *, api_key: 'str', model: 'str', thinking_level: 'str | None', timeout: 'int', max_retries: 'int', max_tokens: 'int' = 32000, pricing: 'dict | None' = None, include_thoughts: 'bool' = True, seed: 'int | None' = None, client=None)`**  
  Initialize self.  See help(type(self)) for accurate signature.

- **`_thinking_config(self, *, include_thoughts: 'bool') -> 'types.ThinkingConfig'`**  
  _(no docstring)_

- **`_build_turn_config(self) -> 'types.GenerateContentConfig'`**  
  _(no docstring)_

- **`_build_debrief_config(self) -> 'types.GenerateContentConfig'`**  
  _(no docstring)_

- **`start(self, description: 'str', state_slice: 'dict') -> 'None'`**  
  _(no docstring)_

- **`observe(self, result: 'dict') -> 'None'`**  
  _(no docstring)_

- **`add_nudge(self, text: 'str') -> 'None'`**  
  _(no docstring)_

- **`_generate(self, config: 'types.GenerateContentConfig', *, max_retries: 'int')`**  
  _(no docstring)_

- **`_account(self, metadata: 'Any') -> 'tuple[dict, float | None]'`**  
  _(no docstring)_

- **`_extract(resp, content) -> 'tuple[str | None, str, bool, str | None]'`** *(staticmethod)*  
  _(no docstring)_

- **`generate_move(self) -> 'Move'`**  
  _(no docstring)_

- **`debrief(self) -> 'str | None'`**  
  _(no docstring)_

- **`_model_indices(self) -> 'list[int]'`**  
  _(no docstring)_

- **`turns(self)`**  
  Segment `contents` into evictable step-pair units, oldest -> newest.
  Everything before the first model output is the PINNED head (not a unit).
  Each unit runs from a model output up to (not incl.) the next model output,
  so it captures the functionResponse (and any nudge) that follows. The last
  unit is the ACTIVE unit.

- **`evict_oldest_turn(self) -> 'None'`**  
  Drop the oldest completed step-pair: the contents from the first model
  output up to (not incl.) the second. The pinned head (before the first
  model output) and the active (latest) unit are never touched. No-op when
  only the head + the active unit remain.

---

## `clients.anthropic_client`

> Anthropic client — AnthropicPolicy, implements core.types.Policy.
> 
> Talks to Claude on AWS Bedrock via the official ``anthropic`` SDK's
> ``AnthropicBedrock`` client, authenticated with a Bedrock bearer token
> (``AWS_BEARER_TOKEN_BEDROCK`` / ``--api-key``). Closed-model sibling of the Gemini
> client: append-only ``messages``, the model's response content appended VERBATIM
> each turn (so the ``thinking`` block's ``signature`` rides forward and Claude keeps
> its reasoning), exact token/USD accounting, a tool-free debrief, and the step-pair
> truncation surface.
> 
> One Anthropic constraint shapes the thinking/move-channel mode: **forced
> tool_choice is INCOMPATIBLE with extended/adaptive thinking** (the API 400s), and
> on Sonnet 4.6/Opus 4.6/Haiku 4.5 the API runs WITHOUT thinking unless ``thinking``
> is sent. So the two are mutually exclusive and both map onto one knob — the
> thinking level:
> 
>   - thinking ON  (level low/medium/high/xhigh/max) -> ``thinking:{type:"adaptive"}``
>     + ``output_config.effort`` + ``tool_choice:{type:"auto"}`` (+ nudge fallback);
>     reasoning CARRIED (signature) — the default.
>   - thinking ON, pre-effort model (``ANTHROPIC_MANUAL_THINKING``, e.g. Haiku 4.5) ->
>     ``thinking:{type:"enabled", budget_tokens:N}`` (manual extended thinking; these models
>     400 on ``output_config.effort``) with N from the
>     SHARED level->budget table ``clientutil.THINKING_LEVEL_BUDGETS`` (same nominal budget per
>     level across providers), clamped to ``1024 <= N <= max_tokens - 1024``. Same auto-tool
>     + nudge channel (forced tool_choice is illegal with manual thinking too).
>   - thinking OFF (level "none") -> ``thinking:{type:"disabled"}`` +
>     ``tool_choice:{type:"tool"}`` (forced make_move); reasoning NOT carried — a clean
>     ablation / forced-tool mode.
> 
> The OFF branch sends ``disabled`` explicitly: on **Opus 5** and Sonnet 5 an omitted
> ``thinking`` means ADAPTIVE, not off. Forced tool_choice suppresses thinking on move
> turns, but the tool-free debrief has no forced tool to lean on, so mere omission
> would let the debrief silently think. Exception: on Fable 5 ``disabled``
> itself 400s (``ANTHROPIC_THINKING_ALWAYS_ON``), so there the OFF branch omits
> ``thinking``.
> 
> Reasoning carry is SELF-CHECKING like Gemini: re-feeding a modified thinking block
> 400s ("blocks in the latest assistant message cannot be modified"). Prior-turn
> thinking is auto-ignored by the API, so evicting whole old step-pairs is safe. Retry/
> error handling is the shared ``core.clientutil`` policy: such a validation 400 fails
> fast.

### class `AnthropicPolicy`

Owns the Claude-on-Bedrock chat: append-only `messages`, verbatim thinking+
tool_use refeed, the thinking-vs-forced-tool mode, exact accounting, the
tool-free debrief, and the step-pair truncation surface.

- **`__init__(self, *, api_key: 'str', model: 'str', aws_region: 'str', thinking_level: 'str | None', max_tokens: 'int', timeout: 'int', max_retries: 'int', pricing: 'dict | None' = None, include_thoughts: 'bool' = True, client=None)`**  
  Initialize self.  See help(type(self)) for accurate signature.

- **`start(self, description: 'str', state_slice: 'dict') -> 'None'`**  
  _(no docstring)_

- **`observe(self, result: 'dict') -> 'None'`**  
  _(no docstring)_

- **`add_nudge(self, text: 'str') -> 'None'`**  
  _(no docstring)_

- **`_append_user_text(self, text: 'str') -> 'None'`**  
  Append user text, MERGING into a trailing user message (e.g. a tool_result
  turn) so we never emit two consecutive user messages.

- **`_thinking_param(self) -> 'dict'`**  
  The `thinking` param for the thinking-ON regimes. Adaptive models take a display
  knob; manual (pre-effort) models take the token budget resolved at construction
  (shared level table, 1024 <= budget < max_tokens — feasibility already checked).

- **`_cache_messages(self) -> 'list'`**  
  `self.messages` with a cache_control breakpoint on the LAST block of the LAST message,
  so the append-only prefix is cached for the next turn (read ~0.1x). NEVER mutates the
  stored history (the verbatim reasoning-carry blocks) — the marker rides only on the
  request copy. Skips safely if the tail isn't a plain str/dict (e.g. SDK objects), in
  which case the static system+tools breakpoints still cache.

- **`_turn_kwargs(self) -> 'dict'`**  
  _(no docstring)_

- **`_debrief_kwargs(self) -> 'dict'`**  
  _(no docstring)_

- **`_create(self, build_kwargs, max_retries: 'int', timeout: 'int | None' = None)`**  
  _(no docstring)_

- **`_extract(resp) -> 'tuple[str | None, str | None, str, bool, str | None]'`** *(staticmethod)*  
  Return (action, tool_use_id, reasoning_summary, has_signature, stop_reason).

- **`_account(self, usage, *, has_thinking: 'bool' = False) -> 'tuple[dict, float | None]'`**  
  _(no docstring)_

- **`generate_move(self) -> 'Move'`**  
  _(no docstring)_

- **`debrief(self) -> 'str | None'`**  
  _(no docstring)_

- **`_assistant_indices(self) -> 'list[int]'`**  
  _(no docstring)_

- **`turns(self)`**  
  Step-pair units over `messages`, oldest -> newest. The head (the pinned
  user task before the first assistant) is NOT a unit. A unit = an assistant
  message + the tool_result/user message(s) up to the next assistant. The last
  unit is ACTIVE (its thinking blocks must stay intact -> never evicted).

- **`evict_oldest_turn(self) -> 'None'`**  
  Drop the oldest completed step-pair: from the first assistant message up to
  (not incl.) the second. Pinned head + active (latest) unit are never touched.
  No-op when only head + active remain. (Prior-turn thinking is auto-ignored by
  the API, so dropping whole old pairs needs no continuity bookkeeping.)

---

## `clients.openai_client`

> OpenAI client — OpenAIPolicy, implements core.types.Policy.
> 
> Talks to OpenAI reasoning models (gpt-5.5 / o-series) via the official ``openai`` SDK's
> **Responses API** (``client.responses.create``) — NOT Chat Completions (a different
> surface from the SGLang OpenAI-compatible Chat Completions client).
> Closed-reasoning-model sibling of the Anthropic client: append-only ``input`` (a list of
> typed Responses items), the model's ``response.output`` appended VERBATIM each turn (so the
> reasoning item's ``encrypted_content`` rides forward), exact token/USD accounting, a
> tool-free debrief, and the step-pair truncation surface.
> 
> Reasoning carry uses **stateless** mode — ``store=False`` +
> ``include=["reasoning.encrypted_content"]`` returns an encrypted reasoning item; pass the
> whole ``response.output`` (reasoning item + function_call) back UNCHANGED, then a
> ``function_call_output`` item. The reasoning item must accompany its function_call (the API
> 400s / degrades otherwise) → self-checking like Gemini/Anthropic; such a 400 fails fast under
> the shared ``core.clientutil`` policy. Unlike Anthropic, OpenAI reasoning models keep reasoning
> WITH forced tool calls, so the move channel is **forced make_move** (parity with Gemini/SGLang).
> ``encrypted_content`` is a dedicated field separate from the visible ``summary`` — the cleanest
> carry of the four. Seed: the Responses API has no seed parameter (warned no-op, like Anthropic).

### class `OpenAIPolicy`

Owns the Responses-API loop: append-only `input` items, verbatim reasoning+
function_call refeed, forced make_move, exact accounting, the tool-free debrief,
and the step-pair truncation surface.

- **`__init__(self, *, api_key: 'str', model: 'str', effort: 'str', max_tokens: 'int', timeout: 'int', max_retries: 'int', pricing: 'dict | None' = None, include_thoughts: 'bool' = True, base_url: 'str | None' = None, client=None)`**  
  Initialize self.  See help(type(self)) for accurate signature.

- **`start(self, description: 'str', state_slice: 'dict') -> 'None'`**  
  _(no docstring)_

- **`observe(self, result: 'dict') -> 'None'`**  
  _(no docstring)_

- **`add_nudge(self, text: 'str') -> 'None'`**  
  _(no docstring)_

- **`_reasoning_param(self) -> 'dict'`**  
  _(no docstring)_

- **`_turn_kwargs(self) -> 'dict'`**  
  _(no docstring)_

- **`_debrief_kwargs(self) -> 'dict'`**  
  _(no docstring)_

- **`_create(self, build_kwargs, max_retries: 'int', timeout: 'int | None' = None)`**  
  _(no docstring)_

- **`_extract(resp) -> 'tuple[str | None, str | None, str, bool, str | None]'`** *(staticmethod)*  
  Return (action, call_id, reasoning_summary, has_encrypted_reasoning, status).

- **`_account(self, usage) -> 'tuple[dict, float | None]'`**  
  _(no docstring)_

- **`generate_move(self) -> 'Move'`**  
  _(no docstring)_

- **`debrief(self) -> 'str | None'`**  
  _(no docstring)_

- **`_is_model_item(item) -> 'bool'`** *(staticmethod)*  
  A model-generated output item (reasoning / function_call / assistant message) —
  as opposed to a head/user message or a function_call_output (the result we add).

- **`_turn_start_indices(self) -> 'list[int]'`**  
  Indices where a model turn begins — a model-output item whose predecessor is not
  one (i.e. it follows the head, a function_call_output, or a user message).

- **`turns(self)`**  
  Step-pair units over `input`, oldest -> newest. The head (the pinned user task
  before the first model item) is NOT a unit. A unit = a model turn's items
  (reasoning + function_call) + the function_call_output that follows. The last unit
  is ACTIVE (its reasoning+call must stay paired/intact -> never evicted).

- **`evict_oldest_turn(self) -> 'None'`**  
  Drop the oldest completed unit (its reasoning + function_call + function_call_output
  together). Pinned head + active (latest) unit are never touched. No-op when only head +
  active remain. (The API harmlessly discards reasoning items that aren't relevant.)

---

## `clients.sglang_client`

> SGLang client — SglangPolicy, implements core.types.Policy.
> 
> Talks to an SGLang OpenAI-compatible Chat Completions endpoint (open models served
> on a server).
> Append-only `messages`,
> the forced/named make_move tool call (`--move-channel`; `guided-json` is selectable for builds
> whose grammar backend rejects forced tool decoding — a forced-tool grammar 400 fails loud with
> that remedy rather than switching channel mid-run), the
> `reasoning`-field round-trip (the open-model analog of Gemini thought signatures),
> 4xx fail-fast + dump. Transport is stdlib HTTP with SSE streaming, so the socket read
> timeout acts as a per-chunk IDLE timeout (`--stream-idle-timeout`): a silent/wedged server
> is caught and retried, while a slow-but-progressing generation runs to completion. The
> streamed deltas are reassembled into the same response shape a non-streamed call returns.
> 
> Parser-agnostic: the `--tool-call-parser` / `--reasoning-parser` are SGLang
> server-launch flags (the operator's job), NOT this client's. We only use stable
> framework features: `tool_choice` named fn, `response_format` json_schema,
> `reasoning_content`, SSE streaming (`stream` + `stream_options.include_usage`),
> `/v1/models`->`max_model_len`.
> 
> Behind the Policy truncation surface:
> - `model_max_context` from the server (`/v1/models` `max_model_len`) -> feeds the
>   shared truncation budget; no per-model table (unlike closed models).
> - step-pair truncation surface over `messages` (head = system+task pinned, unit =
>   assistant + its result, latest = active) — core/history.py reused unchanged.
> - a `/chat/completions` reasoning round-trip probe: detect a template that silently
>   strips carried reasoning so `has_continuity` is reported HONESTLY (not a no-op claim).

### Functions

#### `resolve_model_and_window(data: 'dict', model: 'str | None', *, select=None) -> 'tuple[str, int | None]'`

From a GET /v1/models response pick (model_id, max_model_len). If `model` is given,
match it; with exactly one served model auto-select it; with several, call `select(ids)`
to choose (or raise if no selector is provided).

### class `ChatError`

Unspecified run-time error.

- **`__init__(self, code: 'int | None', body: 'str')`**  
  Initialize self.  See help(type(self)) for accurate signature.

### class `SglangPolicy`

Owns the OpenAI-compatible chat: append-only `messages`, the move channel
(forced-tool by default; guided-json/auto-tool selectable per run), the reasoning round-trip,
exact token accounting (no USD), and the step-pair truncation surface.

- **`__init__(self, *, base_url: 'str', api_key: 'str', model: 'str | None', timeout: 'int' = 60, stream_idle_timeout: 'int' = 180, max_retries: 'int', max_tokens: 'int' = 32000, move_channel: 'str' = 'forced-tool', preserve_reasoning: 'bool' = True, seed: 'int | None' = None, debug_dir: 'str' = '.', select_model: 'Callable | None' = None, http_post: 'Callable' = <function _post_json>, http_get: 'Callable' = <function _get_json>)`**  
  Initialize self.  See help(type(self)) for accurate signature.

- **`start(self, description: 'str', state_slice: 'dict') -> 'None'`**  
  _(no docstring)_

- **`observe(self, result: 'dict') -> 'None'`**  
  _(no docstring)_

- **`add_nudge(self, text: 'str') -> 'None'`**  
  _(no docstring)_

- **`_turn_payload(self) -> 'dict'`**  
  _(no docstring)_

- **`_debrief_payload(self) -> 'dict'`**  
  _(no docstring)_

- **`_generate(self, build_payload, *, max_retries=None, timeout=None) -> 'dict | None'`**  
  _(no docstring)_

- **`_record_error(self, exc, attempt, retries, *, fatal: 'bool' = False, dump: 'str | None' = None) -> 'None'`**  
  _(no docstring)_

- **`_message(data: 'dict') -> 'dict'`** *(staticmethod)*  
  _(no docstring)_

- **`_extract_action(msg: 'dict') -> 'tuple[str | None, str | None]'`** *(staticmethod)*  
  Action + tool_call_id. Prefer the make_move tool call; else parse an
  {"action": ...} JSON object from message content (guided-JSON path).

- **`_account(self, usage: 'dict', has_reasoning: 'bool' = False) -> 'tuple[dict, None]'`**  
  _(no docstring)_

- **`generate_move(self) -> 'Move'`**  
  _(no docstring)_

- **`debrief(self) -> 'str | None'`**  
  _(no docstring)_

- **`probe_reasoning_roundtrip(self) -> 'None'`**  
  Does the server actually feed carried reasoning back into the prompt?
  
  Send two minimal /chat/completions requests that differ ONLY by a fat
  reasoning block (carried under both `reasoning_content` and `reasoning`, since
  builds vary in which key they inject) on a prior assistant message, and compare the
  server's reported `prompt_tokens`. If the one carrying reasoning isn't
  bigger, the server strips it on input => carry is a silent no-op =>
  `has_continuity` is reported False. Uses /chat/completions (the endpoint we
  know works), not a guessed /tokenize. Best-effort: on error -> unverified.
  
  This IS the 2-turn round-trip check, run once at startup.

- **`_assistant_indices(self) -> 'list[int]'`**  
  _(no docstring)_

- **`turns(self)`**  
  Step-pair units over `messages`, oldest -> newest. Head (everything before
  the first assistant: the system is separate + the pinned user task) is NOT a
  unit. A unit = an assistant message + the tool/user result(s) up to the next
  assistant. The last unit is ACTIVE.

- **`evict_oldest_turn(self) -> 'None'`**  
  Drop the oldest completed step-pair: from the first assistant message up to
  (not incl.) the second. The pinned head (before the first assistant) and the
  active (latest) unit are never touched. No-op when only head + active remain.
