#!/usr/bin/env python3
"""Isolation tests for core/cli.py — pure arg-parse + budget resolution.

    python tests/test_cli.py

Covers: defaults (proportion 1.0, budget off); output-reserve budget from the
model window; absolute --context-budget override; unknown-window fallback
(truncation disabled with a warning, never a crash).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from core import cli


def test_arg_defaults():
    args = cli.parse_args(["P-1"])
    assert args.game == "P-1"
    assert args.context_proportion == 1.0 and args.context_budget == 0
    assert args.provider is None and args.model is None  # no default provider -> prompted in main()
    assert args.thinking_level == "high" and args.move_channel == "forced-tool"
    # artifacts + thought summaries are ON by default; reasoning carry ON (opt-out flags)
    assert args.no_save_run is False and args.no_thought_summaries is False
    assert args.no_reasoning_carry is False


def test_negative_max_cost_rejected():
    # A negative cap would trip cost_cap on the very first turn (zero turns played); reject it.
    with pytest.raises(SystemExit):
        cli.parse_args(["G", "--max-cost-usd", "-1"])


def test_numeric_flags_validated_centrally():
    # Values that would not crash but silently produce a meaningless run must be rejected.
    for argv in (["G", "--max-tokens", "0"],
                 ["G", "--max-steps", "0"],
                 ["G", "--max-steps", "-5"],
                 ["G", "--max-invalid-retries", "-1"],
                 ["G", "--api-timeout-seconds", "0"],
                 ["G", "--stream-idle-timeout", "0"],
                 ["G", "--summary-chars", "-1"],
                 ["G", "--context-budget", "-1"],
                 ["G", "--context-proportion", "1.5"],
                 ["G", "--context-proportion", "-0.1"],
                 ["G", "--context-proportion", "nan"],
                 ["G", "--max-cost-usd", "nan"],
                 ["G", "--max-cost-usd", "inf"]):
        with pytest.raises(SystemExit):
            cli.parse_args(argv)
    # boundary values stay legal
    args = cli.parse_args(["G", "--max-invalid-retries", "0", "--context-proportion", "0",
                           "--summary-chars", "0", "--max-cost-usd", "0"])
    assert args.max_invalid_retries == 0 and args.context_proportion == 0.0


def test_cost_cap_on_unpriced_model_refused():
    # F1: an unpriced model accrues cost_usd 0.0 forever — a set cap must fail loud up front.
    from types import SimpleNamespace
    args = cli.parse_args(["G", "--max-cost-usd", "1.5"])
    args.model = "self-hosted-model"
    with pytest.raises(SystemExit, match="no pricing row"):
        cli.ensure_cost_cap_enforceable(args, SimpleNamespace(has_pricing=False))
    cli.ensure_cost_cap_enforceable(args, SimpleNamespace(has_pricing=True))   # priced: fine
    args_nocap = cli.parse_args(["G"])
    cli.ensure_cost_cap_enforceable(args_nocap, SimpleNamespace(has_pricing=False))  # no cap: fine


def test_provenance_collect_excludes_secrets_and_hashes_prompts():
    from types import SimpleNamespace
    from core import provenance
    args = cli.parse_args(["G", "--provider", "sglang", "--base-url", "http://h/v1",
                           "--api-key", "SECRET"])
    policy = SimpleNamespace(base_url="http://h/v1")
    prov = provenance.collect(args, policy, "https://bench.example")
    assert "api_key" not in prov["config"]                       # never dump credentials
    assert "SECRET" not in repr(prov)
    assert prov["config"]["model"] is None and prov["config"]["provider"] == "sglang"
    assert prov["endpoint"] == {"provider": "sglang", "server": "https://bench.example",
                                "base_url": "http://h/v1", "aws_region": None,
                                "api_timeout_seconds": 600, "stream_idle_timeout": 180}
    assert len(prov["prompt_contract_sha256"]) == 64
    # deterministic: same prompts -> same hash (the published-prompt fingerprint)
    assert prov["prompt_contract_sha256"] == provenance.prompt_contract_sha256()
    # this repo IS a git checkout, so the commit resolves here (None allowed elsewhere)
    assert prov["git"] is None or len(prov["git"]["commit"]) == 40
    # SGLang server capture rides along when the policy has it
    policy.server_info = {"version": "x"}
    policy.models_entry = {"id": "m"}
    prov = provenance.collect(args, policy, "s")
    assert prov["sglang_server"] == {"server_info": {"version": "x"}, "models_entry": {"id": "m"}}


def test_budget_reserves_output_large_window():
    # Default proportion 1.0 -> no extra flat buffer; budget = window - the ~1.5-output reserve
    # (same absolute reserve for every model).
    args = cli.parse_args(["G"])  # max_tokens 32000, proportion 1.0
    assert cli.resolve_budget(args, 1_048_576) == 1_048_576 - (int(1.5 * 32_000) + 2_000)


def test_context_proportion_adds_extra_buffer_when_set():
    # The knob works: a low proportion adds a bigger flat buffer than the output reserve.
    args = cli.parse_args(["G", "--context-proportion", "0.5"])  # (1-0.5)*1M = 524288 > 50000
    assert cli.resolve_budget(args, 1_048_576) == 1_048_576 - int(0.5 * 1_048_576)


def test_budget_reserves_for_max_tokens_small_window():
    # 200K window: the 5% buffer (10000) is SMALLER than the ~1.5-output reserve (1.5*32000 + 2000
    # = 50000), so the reserve grows to hold it -> budget = 200000 - 50000, so a
    # high-reasoning turn (this turn's + the carried previous output) can't push past the window.
    args = cli.parse_args(["G"])
    assert cli.resolve_budget(args, 200_000) == 200_000 - (int(1.5 * 32_000) + 2_000)


def test_budget_reserve_holds_for_midsize_window():
    # No floor may raise the budget back above (window - reserve). 80K window ->
    # reserve 50000 -> budget 30000; a `window//2` floor would WRONGLY raise it to
    # 40000 (re-introducing overflow). Assert the reserve wins.
    args = cli.parse_args(["G"])  # max_tokens 32000 -> reserve 50000
    b = cli.resolve_budget(args, 80_000)
    assert b == 80_000 - (int(1.5 * 32_000) + 2_000)    # 30000
    assert b < 80_000 // 2                               # below a window//2 floor (40000) — not raised
    assert b + args.max_tokens <= 80_000                # one full output still fits


def test_budget_none_when_max_tokens_exceeds_window():
    # A window too small for the output reserve has no safe budget -> disable + warn.
    args = cli.parse_args(["G"])  # max_tokens 32000 -> reserve 50000
    warns = []
    assert cli.resolve_budget(args, 16_000, warn=warns.append) is None
    assert warns and "max-tokens" in warns[0]


def test_budget_absolute_override_wins():
    args = cli.parse_args(["G", "--context-budget", "4096", "--context-proportion", "0.5"])
    assert cli.resolve_budget(args, 1_048_576) == 4096  # absolute beats proportion


def test_select_picks_by_number_and_reprompts():
    import builtins
    feed = iter(["9", "x", "2"])  # out-of-range, non-numeric, then a valid pick
    orig = builtins.input
    builtins.input = lambda *_: next(feed)
    try:
        assert cli._select("Choose:", ["a", "b", "c"]) == "b"
    finally:
        builtins.input = orig


def test_budget_unknown_window_disables_with_warning():
    args = cli.parse_args(["G"])
    warns = []
    assert cli.resolve_budget(args, None, warn=warns.append) is None
    assert warns and "DISABLED" in warns[0]


def test_seed_parses_and_anthropic_warns_and_ignores():
    import contextlib, io, tempfile
    pytest.importorskip("anthropic")  # build_policy constructs a REAL SDK client here
    pytest.importorskip("openai")
    assert cli.parse_args(["G", "--seed", "42"]).seed == 42
    # Anthropic has no seed param -> build_policy warns and the policy carries no seed
    # (so the neutral header renders "seed None"). Uses a bearer key; no network at init.
    args = cli.parse_args(["G", "--provider", "anthropic", "--seed", "7", "--api-key", "x"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        policy = cli.build_policy(args, pathlib.Path(tempfile.mkdtemp()))
    out = buf.getvalue()
    assert "no seed parameter" in out and "ignored" in out
    assert getattr(policy, "seed", None) is None
    # OpenAI Responses API also has no seed -> same warned no-op.
    oargs = cli.parse_args(["G", "--provider", "openai", "--seed", "7", "--api-key", "x"])
    obuf = io.StringIO()
    with contextlib.redirect_stdout(obuf):
        opolicy = cli.build_policy(oargs, pathlib.Path(tempfile.mkdtemp()))
    assert "no seed parameter" in obuf.getvalue() and getattr(opolicy, "seed", None) is None
    assert opolicy.model == "openai.gpt-5.5" and opolicy.move_channel == "forced-tool"


def test_provider_backend_routing_by_model_prefix(monkeypatch):
    """The model id picks the backend: Bedrock ids (anthropic./openai. prefix, the
    defaults) hit Bedrock; bare ids go direct to the vendor API with the vendor key."""
    import tempfile
    pytest.importorskip("anthropic")  # build_policy constructs REAL SDK clients here
    pytest.importorskip("openai")
    monkeypatch.delenv("AWS_REGION", raising=False)
    run = pathlib.Path(tempfile.mkdtemp())
    # openai default -> Bedrock: bedrock-mantle base_url (default region us-east-2), 272K window
    policy = cli.build_policy(cli.parse_args(["G", "--provider", "openai", "--api-key", "x"]), run)
    assert policy.model == "openai.gpt-5.5"
    assert str(policy.client.base_url).startswith("https://bedrock-mantle.us-east-2.api.aws/openai/v1")
    assert policy.model_max_context == 272_000 and policy.has_pricing  # Bedrock cap; substring pricing row
    # bare gpt id -> direct api.openai.com
    policy = cli.build_policy(
        cli.parse_args(["G", "--provider", "openai", "--model", "gpt-5.5", "--api-key", "x"]), run)
    assert "openai.com" in str(policy.client.base_url)
    assert policy.model_max_context == 1_050_000
    # anthropic default -> AnthropicBedrock; bare claude id -> direct Anthropic
    policy = cli.build_policy(cli.parse_args(["G", "--provider", "anthropic", "--api-key", "x"]), run)
    assert type(policy.client).__name__ == "AnthropicBedrock"
    policy = cli.build_policy(
        cli.parse_args(["G", "--provider", "anthropic", "--model", "claude-opus-4-8", "--api-key", "x"]), run)
    assert type(policy.client).__name__ == "Anthropic"
    # key fallback matches the backend: Bedrock ids want the bearer token, bare ids the vendor key
    for k in ("AWS_BEARER_TOKEN_BEDROCK", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit, match="AWS_BEARER_TOKEN_BEDROCK"):
        cli.build_policy(cli.parse_args(["G", "--provider", "openai"]), run)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        cli.build_policy(cli.parse_args(["G", "--provider", "openai", "--model", "gpt-5.5"]), run)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        cli.build_policy(cli.parse_args(["G", "--provider", "anthropic", "--model", "claude-opus-4-8"]), run)


def test_max_api_retries_must_be_positive():
    # >= 1 required: 0 gives `range(0)` -> zero attempts -> silent (None, None) from the
    # provider retry loop and a 0-attempt bench failure. argparse must reject it.
    assert cli.parse_args(["G"]).max_api_retries == 10          # default
    assert cli.parse_args(["G", "--max-api-retries", "1"]).max_api_retries == 1
    for bad in ("0", "-3"):
        try:
            cli.parse_args(["G", "--max-api-retries", bad])
        except SystemExit:
            pass  # argparse p.error -> SystemExit(2)
        else:
            raise AssertionError(f"--max-api-retries {bad} should have been rejected")


# ---- Runner ------------------------------------------------------------


def main():
    import inspect
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for t in tests:
        try:
            # Tests declaring a `monkeypatch` parameter get a real pytest.MonkeyPatch (pytest
            # is already a module import here), so the direct `python tests/test_cli.py` run
            # matches the pytest run instead of crashing on the missing fixture.
            if "monkeypatch" in inspect.signature(t).parameters:
                with pytest.MonkeyPatch.context() as mp:
                    t(monkeypatch=mp)
            else:
                t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
            failed += 1
        except BaseException as exc:  # pytest's Skipped (importorskip) is BaseException-derived
            if type(exc).__name__ != "Skipped":
                raise
            print(f"SKIP  {t.__name__}: {exc}")
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
