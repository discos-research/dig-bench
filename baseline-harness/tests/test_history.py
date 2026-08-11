#!/usr/bin/env python3
"""Isolation tests for core/history.py — fake Policy, no provider, no network.

    python tests/test_history.py

Covers the rolling-window contract:
- under budget -> no eviction;
- first-turn bootstrap (last_prompt_tokens == 0) -> no eviction;
- a small overshoot evicts EXACTLY one unit (projection drops under budget);
- a large overshoot evicts ONLY as many as needed, not down to the floor;
- a hopeless overshoot stops at MIN_KEEP_TURNS and never touches the active unit;
- idempotent at the floor.

And the reactive overflow-recovery companion (evict_for_overflow):
- eviction sized to the deficit parsed from the server's 400 wording;
- doubling-batch fallback when the wording has no counts;
- stops at the active unit; returns 0 when eviction cannot shrink.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import history as H
from core.types import Turn


class FakePolicy:
    """A list of step-pair units (oldest -> newest); the last is the active unit.
    Each unit carries an est_tokens. evict_oldest_turn removes index 0 (the oldest
    non-active unit), refusing to drop the lone active unit."""

    def __init__(self, est_tokens_per_unit, last_prompt_tokens):
        self._units = list(est_tokens_per_unit)  # ints, oldest -> newest
        self._ids = list(range(len(self._units)))
        self.last_prompt_tokens = last_prompt_tokens
        self.evictions = 0

    def turns(self):
        n = len(self._units)
        return [
            Turn(payload=self._ids[k], is_active=(k == n - 1), est_tokens=self._units[k])
            for k in range(n)
        ]

    def evict_oldest_turn(self):
        if len(self._units) > 1:  # never drop the lone active unit
            self._units.pop(0)
            self._ids.pop(0)
            self.evictions += 1

    def active_id(self):
        return self._ids[-1]


def test_under_budget_no_eviction():
    p = FakePolicy([100, 100, 100], last_prompt_tokens=250)
    assert H.truncate_if_needed(p, budget=1000) == 0
    assert p.evictions == 0


def test_bootstrap_no_eviction():
    # huge would-be prompt but no server count yet -> never truncates on turn 1
    p = FakePolicy([500, 500, 500, 500], last_prompt_tokens=0)
    assert H.truncate_if_needed(p, budget=100) == 0
    assert p.evictions == 0


def test_disabled_budget_no_eviction():
    p = FakePolicy([500, 500, 500], last_prompt_tokens=9999)
    assert H.truncate_if_needed(p, budget=None) == 0
    assert H.truncate_if_needed(p, budget=0) == 0
    assert p.evictions == 0


def test_small_overshoot_evicts_exactly_one():
    # 5 units * 100 est; last reported 1050, budget 1000 -> drop one (proj 950) and stop.
    p = FakePolicy([100, 100, 100, 100, 100], last_prompt_tokens=1050)
    active = p.active_id()
    evicted = H.truncate_if_needed(p, budget=1000)
    assert evicted == 1
    assert len(p.turns()) == 4
    assert p.turns()[-1].payload == active  # active never moved


def test_large_overshoot_evicts_only_as_many_as_needed():
    # 8 units * 100; last 1450, budget 1000 -> need to shed 5 (1450-500=950). 8>MIN so room.
    p = FakePolicy([100] * 8, last_prompt_tokens=1450)
    evicted = H.truncate_if_needed(p, budget=1000)
    assert evicted == 5            # exactly enough, not the 6 the floor would allow
    assert len(p.turns()) == 3


def test_hopeless_overshoot_stops_at_floor_keeps_active():
    # tiny budget, can't get under it -> stop at MIN_KEEP_TURNS, active intact
    p = FakePolicy([100] * 5, last_prompt_tokens=10_000)
    active = p.active_id()
    evicted = H.truncate_if_needed(p, budget=50)
    assert len(p.turns()) == H.MIN_KEEP_TURNS
    assert evicted == 5 - H.MIN_KEEP_TURNS
    assert p.turns()[-1].payload == active and p.turns()[-1].is_active


def test_idempotent_at_floor():
    p = FakePolicy([100, 100], last_prompt_tokens=10_000)  # already at MIN_KEEP_TURNS
    assert H.truncate_if_needed(p, budget=50) == 0
    assert p.evictions == 0


def test_log_under_budget_says_le():
    # got under budget -> message uses "<=", no floor note
    p = FakePolicy([100, 100, 100, 100], last_prompt_tokens=1050)
    msgs = []
    H.truncate_if_needed(p, budget=1000, log=msgs.append)
    assert len(msgs) == 1 and "evicted 1 step-pair" in msgs[0]
    assert "<= budget 1000" in msgs[0] and "floor" not in msgs[0]


def _overflow_err(total, limit):
    return {"message": (f"litellm.BadRequestError: OpenAIException - Requested token count "
                        f"exceeds the model's maximum context length of {limit} tokens. "
                        f"You requested a total of {total} tokens.")}


def test_evict_for_overflow_sized_to_parsed_deficit():
    # Realistic overflow shape: deficit 264084-262144=1940 -> target 1940*1.25+1000=3425.
    # Oldest units est 500 each -> exactly 7 evictions (3500 >= 3425), not one, not all.
    p = FakePolicy([500] * 20, last_prompt_tokens=0)
    active = p.active_id()
    assert H.evict_for_overflow(p, _overflow_err(264084, 262144)) == 7
    assert p.evictions == 7
    assert p.turns()[-1].payload == active   # active never moved


def test_evict_for_overflow_fallback_doubles_per_round():
    # No counts in the message -> blind batch of 2**round_idx per round.
    p = FakePolicy([100] * 20, last_prompt_tokens=0)
    err = {"message": "Error 400: context_length_exceeded"}
    assert H.evict_for_overflow(p, err, round_idx=0) == 1
    assert H.evict_for_overflow(p, err, round_idx=1) == 2
    assert H.evict_for_overflow(p, err, round_idx=2) == 4
    assert p.evictions == 7


def test_evict_for_overflow_stops_at_active_and_reports_partial():
    # Deficit needs ~3425 est but only 2 evictable units exist (active pinned):
    # evict both, return 2 (caller retries; the server adjudicates), never touch active.
    p = FakePolicy([100, 100, 100], last_prompt_tokens=0)
    active = p.active_id()
    assert H.evict_for_overflow(p, _overflow_err(264084, 262144)) == 2
    assert len(p.turns()) == 1
    assert p.turns()[-1].payload == active and p.turns()[-1].is_active


def test_evict_for_overflow_cannot_shrink_returns_zero():
    # Lone active unit -> the client refuses; 0 tells the caller to give up (no spin).
    p = FakePolicy([100], last_prompt_tokens=0)
    assert H.evict_for_overflow(p, _overflow_err(264084, 262144)) == 0
    assert p.evictions == 0


def test_log_floor_capped_says_over_budget():
    # hit MIN_KEEP_TURNS floor still over budget -> message must say "> budget", not "<="
    p = FakePolicy([100] * 5, last_prompt_tokens=10_000)
    msgs = []
    H.truncate_if_needed(p, budget=50, log=msgs.append)
    assert len(msgs) == 1
    assert "> budget 50" in msgs[0] and "<=" not in msgs[0]
    assert "floor" in msgs[0]


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
