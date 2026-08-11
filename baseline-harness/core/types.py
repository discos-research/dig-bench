"""Shared dataclasses + the Policy interface the harness drives and the
rolling-window context truncation operates through. No provider or game logic here.

The truncation surface (turns / evict_oldest_turn) is the seam: core/history.py
uses ONLY these methods and never reconstructs a provider payload, so each
provider's verbatim re-append stays inside its own client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Move:
    """One LLM turn's outcome. `action` is None when the model made no move."""

    action: str | None
    reasoning_summary: str        # human-readable; for display/log only
    has_continuity: bool          # a continuity token rode forward this turn
    finish_reason: str | None
    usage: dict                   # prompt/cached/output/thoughts/total (provider-mapped)
    cost: float | None
    elapsed_s: float
    error: dict | None = None
    thoughts_basis: str = "none"  # how usage["thoughts"] was counted: exact | included | none
    # How has_continuity was established, three explicit states (None = nothing carried):
    #   verified   — the carry is API-validated (crypto token) or probe-proven round-tripped;
    #   unverified — reasoning re-sent, but the round-trip probe could not run (open models);
    #   stripped   — reasoning re-sent and the probe PROVED the server drops it on input.
    continuity: str | None = None


@dataclass
class Turn:
    """One evictable unit. For the forced-tool bench loop a unit is a completed
    step-pair (model functionCall+signature + its functionResponse).
    The payload is the provider-native chunk(s), kept VERBATIM — never rebuilt."""

    payload: object
    is_active: bool               # the latest (in-flight) unit — never evict
    est_tokens: int               # cheap per-unit size estimate, for MINIMAL eviction


@runtime_checkable
class Policy(Protocol):
    """What harness.play() drives and history.truncate_if_needed() truncates."""

    model: str
    model_max_context: int | None
    has_pricing: bool
    last_prompt_tokens: int       # server-reported prompt size of the last call (truncation trigger)
    # accounting counters: call_count, prompt_tokens, cached_tokens, output_tokens,
    #                      thoughts_tokens, total_tokens, cost_usd, elapsed_s

    def start(self, description: str, state_slice: dict) -> None: ...
    def generate_move(self) -> Move: ...      # appends the model's output VERBATIM
    def observe(self, result: dict) -> None: ...
    def add_nudge(self, text: str) -> None: ...
    def debrief(self) -> str | None: ...

    # truncation surface — history.py uses only these
    def turns(self) -> list[Turn]: ...        # evictable units, oldest -> newest (each carries is_active)
    def evict_oldest_turn(self) -> None: ...   # drop the oldest non-pinned, non-active unit
