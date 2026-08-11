#!/usr/bin/env python3
"""Isolation tests for core/accounting.py — pure generic functions, no provider data.

    python tests/test_accounting.py

accounting.py holds no provider tables (those live in each client), so these
test the mechanism with a small local fake table: longest-substring matching (used
for both pricing and context windows), the cost formula, cached-cheaper billing,
and unpriced -> None.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import accounting as A

FAKE_PRICING = {
    "fake-pro": {"input_per_1m": 2.0, "cached_input_per_1m": 0.5, "output_per_1m": 12.0, "thoughts_per_1m": 12.0},
}
FAKE_WINDOWS = {"fake-pro-v1": 1_000_000, "fake": 256_000}


def test_match_model_substring_longest_and_unknown():
    assert A.match_model("fake-pro-v1", FAKE_WINDOWS) == 1_000_000   # exact
    assert A.match_model("fake-mini", FAKE_WINDOWS) == 256_000       # key "fake" is a substring
    assert A.match_model("other", FAKE_WINDOWS) is None              # no key is a substring
    # longest wins: "fake-pro-v1" beats "fake" for that model id
    assert A.match_model("fake-pro-v1-2026", FAKE_WINDOWS) == 1_000_000
    # substring (not prefix): a regional id prefix must still resolve — this is the
    # whole reason for one substring matcher (Bedrock/Vertex `us.`/`global.` ids).
    assert A.match_model("us.fake-pro-v1", FAKE_WINDOWS) == 1_000_000


def test_pricing_for_substring_and_unknown():
    assert A.pricing_for("fake-pro-002", FAKE_PRICING)["output_per_1m"] == 12.0
    assert A.pricing_for("unlisted", FAKE_PRICING) is None


def test_compute_cost_cached_is_cheaper():
    cost = A.compute_cost("fake-pro", 1000, 800, 20, 100, FAKE_PRICING)
    p = FAKE_PRICING["fake-pro"]
    expected = (200 * p["input_per_1m"] + 800 * p["cached_input_per_1m"]
                + 20 * p["output_per_1m"] + 100 * p["thoughts_per_1m"]) / 1_000_000
    assert abs(cost - round(expected, 8)) < 1e-12
    all_cached = A.compute_cost("fake-pro", 1000, 1000, 0, 0, FAKE_PRICING)
    none_cached = A.compute_cost("fake-pro", 1000, 0, 0, 0, FAKE_PRICING)
    assert all_cached < none_cached


def test_compute_cost_unpriced_is_none():
    assert A.compute_cost("unlisted", 1000, 0, 10, 10, FAKE_PRICING) is None


TIERED_PRICING = {
    "fake-pro": {"input_per_1m": 2.0, "cached_input_per_1m": 0.5, "output_per_1m": 12.0, "thoughts_per_1m": 12.0,
                 "long_context": {"threshold": 1000, "input_per_1m": 4.0, "cached_input_per_1m": 1.0,
                                  "output_per_1m": 18.0, "thoughts_per_1m": 18.0}},
}


def test_long_context_tier_applied_above_threshold_only():
    # At/below the threshold: base rates (existing behavior). Above: the sub-row's rates for the
    # WHOLE call (providers price the full request at the higher tier once input crosses it).
    at = A.compute_cost("fake-pro", 1000, 0, 10, 0, TIERED_PRICING)
    flat = A.compute_cost("fake-pro", 1000, 0, 10, 0, FAKE_PRICING)
    assert at == flat  # threshold is exclusive
    above = A.compute_cost("fake-pro", 1001, 100, 10, 5, TIERED_PRICING)
    t = TIERED_PRICING["fake-pro"]["long_context"]
    expected = (901 * t["input_per_1m"] + 100 * t["cached_input_per_1m"]
                + 10 * t["output_per_1m"] + 5 * t["thoughts_per_1m"]) / 1_000_000
    assert abs(above - round(expected, 8)) < 1e-12
    # rows WITHOUT the key stay flat however big the prompt (no accidental tiering)
    assert A.compute_cost("fake-pro", 10_000_000, 0, 0, 0, FAKE_PRICING) == \
        round(10_000_000 * 2.0 / 1_000_000, 8)


def test_real_provider_tier_rows_wired():
    # The tiers the docs promise are actually on the rows (gemini pro >200K, gpt-5.5 >272K),
    # and rows that must stay flat (flash, Bedrock-capped) carry no tier.
    from clients.gemini_client import GEMINI_PRICING
    from clients.openai_client import OPENAI_PRICING
    g = A.pricing_for("gemini-3.1-pro-preview", GEMINI_PRICING)
    assert g["long_context"]["threshold"] == 200_000 and g["long_context"]["input_per_1m"] == 2 * g["input_per_1m"]
    assert "long_context" not in A.pricing_for("gemini-3-flash", GEMINI_PRICING)
    o = A.pricing_for("gpt-5.5", OPENAI_PRICING)
    assert o["long_context"]["threshold"] == 272_000 and o["long_context"]["output_per_1m"] == 1.5 * o["output_per_1m"]


def test_real_provider_context_windows_resolve_correctly():
    # Lock the closed-model context windows (per provider docs) AND the
    # substring resolution on real ids. A wrong window silently mis-sizes the truncation budget
    # (a cross-model fairness bug), so guard the load-bearing cases here (always runs, no SDKs).
    from clients.gemini_client import GEMINI_MAX_CONTEXT
    from clients.anthropic_client import ANTHROPIC_MAX_CONTEXT
    from clients.openai_client import OPENAI_MAX_CONTEXT
    # Gemini: every model (incl. the flashes) is 1,048,576 — flashes match via "gemini-3"/"gemini-2.5".
    for m in ("gemini-3.1-pro-preview", "gemini-3-flash", "gemini-3.5-flash", "gemini-2.5-pro", "gemini-2.5-flash"):
        assert A.match_model(m, GEMINI_MAX_CONTEXT) == 1_048_576, m
    # Anthropic: the 1M family vs the 200K models — regional `us.`/`global.` ids must resolve right.
    for m in ("global.anthropic.claude-opus-5", "global.anthropic.claude-opus-4-8",
              "global.anthropic.claude-sonnet-5", "global.anthropic.claude-sonnet-4-6",
              "claude-fable-5"):
        assert A.match_model(m, ANTHROPIC_MAX_CONTEXT) == 1_000_000, m
    for m in ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "global.anthropic.claude-sonnet-4-5"):
        assert A.match_model(m, ANTHROPIC_MAX_CONTEXT) == 200_000, m
    # claude-sonnet-5 has its OWN 1M row + $3/$15 pricing — it must not fall through unpriced,
    # and its distinct id must not collide with the sonnet-4-5 / sonnet-4-6 rows.
    from clients.anthropic_client import ANTHROPIC_BEDROCK_PRICING
    s5 = A.pricing_for("global.anthropic.claude-sonnet-5", ANTHROPIC_BEDROCK_PRICING)
    assert s5 is not None and s5["input_per_1m"] == 3.0 and s5["output_per_1m"] == 15.0
    # Same trap one generation up: "claude-opus-4" is not a substring of an opus-5 id, so Opus 5
    # needs its own $5/$25 row or the default model runs unpriced.
    o5 = A.pricing_for("global.anthropic.claude-opus-5", ANTHROPIC_BEDROCK_PRICING)
    assert o5 is not None and o5["input_per_1m"] == 5.0 and o5["output_per_1m"] == 25.0
    # OpenAI: gpt-5.4-mini (400K) must NOT inherit gpt-5.4's 1.05M (longest-substring wins).
    assert A.match_model("gpt-5.4-mini", OPENAI_MAX_CONTEXT) == 400_000
    assert A.match_model("gpt-5.4", OPENAI_MAX_CONTEXT) == 1_050_000


# ---- Runner ------------------------------------------------------------


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
