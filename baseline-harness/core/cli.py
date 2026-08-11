"""CLI + wiring: parse args, resolve credentials, build the provider policy + bench,
resolve the context-truncation budget, run play().

Beyond standard arg parsing: the context-truncation flags
(--context-proportion / --context-budget) and budget resolution. --provider selects the
client (gemini | sglang | anthropic | openai); it has no default — omit it and you're
prompted to pick one.

Credentials come from environment variables (or pass --api-key). Export them directly, or
have a secrets manager of your choice inject them into the environment.

Run (from inside the repo root):
    # Gemini
    GEMINI_API_KEY=...  python -m core.cli P-1 --provider gemini --verbose
    # SGLang open model (model auto-IDed from /v1/models, or picked if several are served)
    python -m core.cli P-1 --provider sglang --base-url http://host:30000/v1 --verbose
    # Anthropic on Bedrock (bearer token)
    AWS_BEARER_TOKEN_BEDROCK=...  python -m core.cli P-1 \
        --provider anthropic --model global.anthropic.claude-opus-5 --verbose
    # OpenAI (Responses API)
    OPENAI_API_KEY=...  python -m core.cli P-1 --provider openai --model gpt-5.5 --verbose
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

from . import provenance
from .bench import Bench
from .output import Output
from .harness import play

SERVER = (os.environ.get("DIGBENCH_SERVER_URL") or "https://api.digbench.ai").rstrip("/")
SERVER = SERVER.removesuffix("/api/agent")  # tolerate the full base being pasted — it's appended below
BASE = SERVER + "/api/agent"
BENCH_TIMEOUT = 60


# ---- Credentials -------------------------------------------------------


def game_token() -> str:
    """The bench API token, from DIGBENCH_API_TOKEN."""
    return os.environ.get("DIGBENCH_API_TOKEN", "").strip()


def gemini_key() -> str:
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY"):
        value = os.environ.get(var, "").strip()
        if value:
            return value
    return ""


def bedrock_token() -> str:
    """The AWS Bedrock bearer token (an API key, not SigV4 creds)."""
    return os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip()


def openai_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "").strip()


def anthropic_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


# ---- Budget resolution ---------------------------------------


_BUDGET_SLACK = 2000  # headroom over the output reserve for prompt-size estimate / tokenizer drift
# Reserve ~1.5 outputs: this turn's output PLUS (part of) the previous turn's output, which is
# carried forward before the REACTIVE truncation (one turn behind) catches up. Evict-then-retry
# in the harness backstops the rare turn that still overshoots.
_OUTPUT_RESERVE_FACTOR = 1.5


def resolve_budget(args, model_max_context: int | None, *, warn=None) -> int | None:
    """The rolling-window budget in tokens, or None to DISABLE truncation.

    Absolute --context-budget wins (ablation knob). Otherwise window minus a reserve big enough
    for ~1.5 outputs (`max_tokens` counts toward the hard window, and reactive truncation lags one
    turn, so the previous turn's output is carried before eviction catches up), falling back to the
    flat (1 - proportion) buffer when that is larger. If the window is unknown and no absolute
    budget is set, truncation is disabled (never crash) — warn and tell the user the knob."""
    if args.context_budget and args.context_budget > 0:
        return args.context_budget
    if model_max_context:
        reserve = max(int(_OUTPUT_RESERVE_FACTOR * args.max_tokens) + _BUDGET_SLACK,
                      int((1 - args.context_proportion) * model_max_context))
        budget = model_max_context - reserve
        if budget <= 0:   # max_tokens (+slack) can't fit the window — no budget reserves a full output
            if warn:
                warn(f"--max-tokens ({args.max_tokens}) + reserve exceeds the model window "
                     f"({model_max_context}); lower --max-tokens or set --context-budget. "
                     "Proactive truncation off (the overflow evict-retry still bounds context, "
                     "less efficiently).")
            return None
        return budget   # a SMALL budget is safe (aggressive truncation); never floor it back UP
    if warn:
        warn(
            f"no context-window known for {args.model!r} and no --context-budget set: "
            "context truncation DISABLED (append-only). Pass --context-budget to enable it."
        )
    return None


# ---- CLI ---------------------------------------------------------------


def default_run_label(model: str, thinking_level: str | None) -> str:
    base = f"baseline-harness_{model}"
    return f"{base}-{thinking_level}" if thinking_level else base


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _select(prompt: str, options: list[str]) -> str:
    """Interactive numbered menu — the user picks by number, not by typing the value
    (avoids typos). Re-prompts on invalid input. Blocks on stdin (interactive use)."""
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        try:
            choice = input("> ").strip()
        except EOFError:  # non-tty / piped / CI stdin — can't prompt; fail cleanly with guidance
            sys.exit("no interactive stdin to choose from a menu — pass the value explicitly "
                     "(e.g. --provider / --model / the game name).")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print(f"  please enter a number 1-{len(options)}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="DiG-bench baseline harness — multi-provider (gemini, sglang, anthropic, openai).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("game", nargs="?", help="Game name. If omitted, lists games and prompts.")
    p.add_argument("--provider", default=None, choices=["gemini", "sglang", "anthropic", "openai"],
                   help="Which model provider. No default — if omitted you're prompted to pick one.")
    p.add_argument("--base-url", default="",
                   help="SGLang OpenAI-compatible base URL (…/v1). REQUIRED when --provider sglang.")
    p.add_argument("--api-key", default="",
                   help="API key for the provider. Overrides env; for gemini, env "
                        "(GEMINI_API_KEY/…) is the fallback; for sglang, SGLANG_API_KEY/OPENAI_API_KEY; "
                        "for anthropic/openai the fallback depends on the model id: Bedrock ids "
                        "(anthropic./openai. prefix, the defaults) use AWS_BEARER_TOKEN_BEDROCK; bare "
                        "ids use ANTHROPIC_API_KEY / OPENAI_API_KEY (direct API).")
    p.add_argument("--aws-region", default="",
                   help="AWS region for Bedrock model ids (anthropic & openai). Falls back to "
                        "AWS_REGION, then us-east-1 (anthropic) / us-east-2 (openai).")
    p.add_argument("--max-tokens", "--max-output-tokens", dest="max_tokens", type=int, default=32000,
                   help="Universal per-call output-token cap for ALL providers (reasoning/thinking "
                        "counts against it): anthropic max_tokens / openai & gemini max_output_tokens "
                        "/ sglang max_tokens. Set generously so it rarely truncates; the budget "
                        "reserves it (see --context-proportion).")
    p.add_argument("--model", default=None,
                   help="Model id. Gemini default gemini-3.1-pro-preview; SGLang auto-IDs from /v1/models if "
                        "omitted; anthropic default global.anthropic.claude-opus-5; openai default "
                        "openai.gpt-5.5 (Bedrock). Bare ids (claude-*/gpt-*) go direct to the vendor API.")
    p.add_argument("--move-channel", default="forced-tool",
                   choices=["forced-tool", "guided-json", "auto-tool"],
                   help="SGLang-only move channel (default forced-tool). On a grammar-backend tool 400 it "
                        "fails with a '--move-channel guided-json' hint (no silent fallback). "
                        "(gemini/anthropic/openai: ignored — they set their own channel.)")
    p.add_argument("--no-reasoning-carry", action="store_true",
                   help="SGLang-only: disable carrying reasoning across turns (ablation). Carry is ON by default.")
    p.add_argument("--thinking-level", default="high",
                   choices=["minimal", "low", "medium", "high", "xhigh", "max", "none"],
                   help="Reasoning depth. Gemini: thinking_level (2.5-family: mapped to thinking_budget). "
                        "Anthropic: output_config.effort (minimal->low) with adaptive thinking + auto-tool; "
                        "pre-effort models (Haiku 4.5/Sonnet 4.5) get manual thinking budget_tokens; `none` "
                        "sends thinking:disabled + forces make_move (unsupported on Fable 5, whose "
                        "thinking is always on). OpenAI: reasoning.effort (`none` = no reasoning). "
                        "`xhigh`/`max` 400 on models that lack them.")
    p.add_argument("--no-thought-summaries", action="store_true",
                   help="Disable the human-readable reasoning summary (include_thoughts=False). "
                        "Summaries are ON by default for inspectability; OFF is cheaper because the "
                        "summary text is re-fed as input every turn.")
    p.add_argument("--summary-chars", type=int, default=300, help="Terminal/txt truncation only; JSONL keeps full.")
    p.add_argument("--seed", type=int, default=None,
                   help="Fixed sampling seed for reproducibility (Gemini + SGLang). Anthropic has no "
                        "seed parameter, so --seed is ignored there (with a warning).")
    p.add_argument("--api-timeout-seconds", type=int, default=600,
                   help="Per-call API timeout (s) for gemini/anthropic/openai. SGLang ignores this "
                        "(it streams — see --stream-idle-timeout).")
    p.add_argument("--stream-idle-timeout", type=int, default=180,
                   help="SGLang-only: inter-chunk idle timeout (s) for the streaming response. A "
                        "silent (wedged) server is retried; a generation still emitting tokens is not. "
                        "Must exceed worst-case cold prefill time-to-first-token at your context size.")
    p.add_argument("--max-api-retries", type=int, default=10, help="Max retries per failed API/bench call.")
    p.add_argument("--max-steps", type=int, default=3000, help="Turn cap / cost guard.")
    p.add_argument("--max-cost-usd", type=float, default=0.0, help="0 = off; checked before each turn.")
    p.add_argument("--max-invalid-retries", type=int, default=5, help="Consecutive; beyond it the run stops 'blocked'.")
    p.add_argument("--context-proportion", type=float, default=1.0,
                   help="Optional EXTRA proportional buffer. The budget reserves ~1.5x --max-tokens "
                        "for output regardless; this only adds more when (1-this)*window exceeds that "
                        "reserve. Default 1.0 = reserve output only (the reserve is the same absolute "
                        "size for every model). Set <1.0 for an additional flat buffer.")
    p.add_argument("--context-budget", type=int, default=0,
                   help="Absolute context-truncation budget in tokens; >0 overrides --context-proportion.")
    p.add_argument("--no-save-run", action="store_true",
                   help="Skip writing .log + .jsonl artifacts. Artifacts are saved by default.")
    p.add_argument("--run-dir", default="runs", help="Directory for saved run artifacts.")
    p.add_argument("--run-label", default="", help="Default baseline-harness_<model>[-<thinking>]; -> session model_name.")
    p.add_argument("--verbose", action="store_true", help="finish_reason + the full untruncated reasoning summary.")
    args = p.parse_args(argv)
    # `--max-api-retries` bounds `range(max_retries)` in both the provider retry loop
    # (core.clientutil.run_request) and the bench client. 0 means zero attempts: the loop
    # never runs and the request silently yields no result and no error. Require >= 1.
    if args.max_api_retries < 1:
        p.error("--max-api-retries must be >= 1 (it is the attempt count, not extra retries)")
    # Central numeric validation: every remaining flag where a zero/negative/NaN value would
    # not error loudly but silently produce a meaningless run (0-token calls, instant stops,
    # never-firing timeouts, a NaN cap that disables every comparison).
    if not math.isfinite(args.max_cost_usd) or args.max_cost_usd < 0:
        p.error("--max-cost-usd must be a finite value >= 0 (0 disables the cap)")
    if not math.isfinite(args.context_proportion) or not 0.0 <= args.context_proportion <= 1.0:
        p.error("--context-proportion must be a finite value in [0, 1]")
    if args.max_tokens < 1:
        p.error("--max-tokens must be >= 1")
    if args.max_steps < 1:
        p.error("--max-steps must be >= 1")
    if args.max_invalid_retries < 0:
        p.error("--max-invalid-retries must be >= 0")
    if args.api_timeout_seconds < 1:
        p.error("--api-timeout-seconds must be >= 1")
    if args.stream_idle_timeout < 1:
        p.error("--stream-idle-timeout must be >= 1")
    if args.summary_chars < 0:
        p.error("--summary-chars must be >= 0")
    if args.context_budget < 0:
        p.error("--context-budget must be >= 0 (0 = derive from the model window)")
    return args


def ensure_cost_cap_enforceable(args, policy) -> None:
    """cost_usd stays 0.0 on unpriced models, so --max-cost-usd would silently never
    trigger — an unbounded run the operator believes is capped. Refuse before starting."""
    if args.max_cost_usd > 0 and not policy.has_pricing:
        sys.exit(f"--max-cost-usd set but {args.model!r} has no pricing row: the cap could "
                 "never trigger. Drop the flag (use --max-steps) or add a pricing row.")


def build_policy(args, run_dir: Path):
    """Construct the provider's policy. For SGLang this also queries /v1/models
    (model auto-ID + context window), so call it before reading policy.model."""
    if args.provider == "gemini":
        from clients.gemini_client import GeminiPolicy  # deferred: imports google.genai
        key = args.api_key or gemini_key()
        if not key:
            sys.exit("No Gemini key. Set GEMINI_API_KEY (or GOOGLE_API_KEY / GEMINI_KEY) or pass --api-key.")
        model = args.model or "gemini-3.1-pro-preview"
        thinking_level = None if args.thinking_level == "none" else args.thinking_level
        run_dir.mkdir(parents=True, exist_ok=True)  # 4xx dumps land here
        policy = GeminiPolicy(
            api_key=key, model=model, thinking_level=thinking_level,
            timeout=args.api_timeout_seconds, max_retries=args.max_api_retries,
            max_tokens=args.max_tokens,
            include_thoughts=not args.no_thought_summaries, seed=args.seed,
        )
        policy.debug_dir = str(run_dir)
        return policy
    if args.provider == "anthropic":
        from clients.anthropic_client import AnthropicPolicy  # deferred: imports the anthropic SDK
        model = args.model or "global.anthropic.claude-opus-5"
        if "anthropic." in model:  # Bedrock id (default)
            key = args.api_key or bedrock_token()
            if not key:
                sys.exit("No Bedrock bearer token. Set AWS_BEARER_TOKEN_BEDROCK or pass --api-key.")
        else:  # bare claude-* id → direct api.anthropic.com
            key = args.api_key or anthropic_key()
            if not key:
                sys.exit("No Anthropic key. Set ANTHROPIC_API_KEY or pass --api-key.")
        region = args.aws_region or os.environ.get("AWS_REGION", "").strip() or "us-east-1"
        if args.seed is not None:  # the Anthropic Messages API has no seed parameter
            print("  ⚠️  Anthropic has no seed parameter; --seed ignored (run is not reproducible).")
        thinking_level = None if args.thinking_level == "none" else args.thinking_level
        run_dir.mkdir(parents=True, exist_ok=True)  # 4xx dumps land here
        policy = AnthropicPolicy(
            api_key=key, model=model, aws_region=region, thinking_level=thinking_level,
            max_tokens=args.max_tokens, timeout=args.api_timeout_seconds, max_retries=args.max_api_retries,
            include_thoughts=not args.no_thought_summaries,
        )
        policy.debug_dir = str(run_dir)
        policy.resolved_region = region if "anthropic." in model else None  # provenance (direct API: n/a)
        return policy
    if args.provider == "openai":
        from clients.openai_client import OpenAIPolicy  # deferred: imports the openai SDK
        model = args.model or "openai.gpt-5.5"
        if model.startswith("openai."):  # Bedrock id (default)
            key = args.api_key or bedrock_token()
            if not key:
                sys.exit("No Bedrock bearer token. Set AWS_BEARER_TOKEN_BEDROCK or pass --api-key.")
            region = args.aws_region or os.environ.get("AWS_REGION", "").strip() or "us-east-2"  # GPT-5.5: Ohio
            base_url = f"https://bedrock-mantle.{region}.api.aws/openai/v1"
        else:  # bare id → direct api.openai.com
            key = args.api_key or openai_key()
            if not key:
                sys.exit("No OpenAI key. Set OPENAI_API_KEY or pass --api-key.")
            base_url = None
            region = None
        effort = args.thinking_level  # raw, incl "none"; maps to reasoning.effort ("max" 400s, fail-loud)
        if args.seed is not None:  # the Responses API has no seed parameter
            print("  ⚠️  OpenAI Responses API has no seed parameter; --seed ignored (run is not reproducible).")
        run_dir.mkdir(parents=True, exist_ok=True)  # 4xx dumps land here
        policy = OpenAIPolicy(
            api_key=key, model=model, effort=effort, max_tokens=args.max_tokens,
            timeout=args.api_timeout_seconds, max_retries=args.max_api_retries,
            include_thoughts=not args.no_thought_summaries, base_url=base_url,
        )
        policy.debug_dir = str(run_dir)
        policy.resolved_region = region  # provenance (None on the direct API)
        return policy
    # provider == sglang
    from clients.sglang_client import SglangPolicy
    if not args.base_url:
        sys.exit("--provider sglang requires --base-url (the SGLang …/v1 endpoint).")
    key = args.api_key or os.environ.get("SGLANG_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    run_dir.mkdir(parents=True, exist_ok=True)  # 4xx dumps land here
    return SglangPolicy(
        base_url=args.base_url, api_key=key, model=args.model,
        stream_idle_timeout=args.stream_idle_timeout, max_retries=args.max_api_retries,
        max_tokens=args.max_tokens,
        move_channel=args.move_channel, preserve_reasoning=not args.no_reasoning_carry,
        seed=args.seed, debug_dir=str(run_dir),
        select_model=lambda ids: _select("Multiple models served — choose one:", ids),
    )  # NB: --api-timeout-seconds intentionally not passed; SGLang streams (idle timeout). Discovery GET uses the 60s default.


def main() -> None:
    args = parse_args()
    if args.provider is None:  # no default — let the user pick
        args.provider = _select("Choose a provider:", ["gemini", "sglang", "anthropic", "openai"])
    # thinking applies to Gemini + Anthropic + OpenAI; SGLang reports thinking as none.
    if args.provider == "openai":
        thinking_level = args.thinking_level  # raw effort (incl "none") for the header
    elif args.provider in ("gemini", "anthropic"):
        thinking_level = None if args.thinking_level == "none" else args.thinking_level
    else:
        thinking_level = None

    token = game_token()
    if not token:
        sys.exit("No game token. export DIGBENCH_API_TOKEN=gtp_... "
                 "(mint at https://digbench.ai/account/tokens)")
    bench = Bench(BASE, token, timeout=BENCH_TIMEOUT, max_retries=args.max_api_retries)

    game = args.game
    if not game:
        print("Games:", ", ".join(bench.list_games()))
        try:
            game = input("game > ").strip()
        except EOFError:  # non-tty / piped / CI stdin
            sys.exit("no interactive stdin — pass the game name as an argument.")
        if not game:
            sys.exit("no game chosen")

    run_dir = Path(args.run_dir)
    policy = build_policy(args, run_dir)
    args.model = policy.model  # resolved (SGLang auto-ID); downstream uses this
    ensure_cost_cap_enforceable(args, policy)
    run_label = args.run_label or default_run_label(args.model, thinking_level)
    budget = resolve_budget(args, policy.model_max_context, warn=lambda m: print(f"  ⚠️  {m}"))
    prov = provenance.collect(args, policy, SERVER)  # recorded in the session JSONL row

    print(f"Playing {game} with {args.provider}:{args.model} (budget={budget or 'off'})\n")
    start = bench.start_session(game, run_label, args.model)
    sid = start["session_id"]

    save_run = not args.no_save_run
    log_path = jsonl_path = None
    if save_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        base_name = _safe(f"{run_label}_{time.strftime('%Y%m%d-%H%M%S')}_{sid[:8]}")
        log_path = run_dir / (base_name + ".log")
        jsonl_path = run_dir / (base_name + ".jsonl")

    out = Output(summary_chars=args.summary_chars, verbose=args.verbose, log_path=log_path, jsonl_path=jsonl_path)
    if args.provider == "sglang":  # report reasoning round-trip honestly before play
        policy.on_retry = out.warn
        policy.on_note = out.note   # probe verdict is a neutral diagnostic, not a warning
        policy.probe_reasoning_roundtrip()
    try:
        play(bench, policy, out, start, args, SERVER, thinking_level, run_label,
             context_budget=budget, provenance=prov)
    finally:
        out.close()
    if save_run:
        print(f"\nsaved: {log_path}  |  {jsonl_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
