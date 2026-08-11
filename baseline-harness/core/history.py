"""The shared turn-aware rolling-window context truncation.

This is the ONLY place the rolling window lives. It operates purely through the
Policy truncation surface (turns / evict_oldest_turn; the active unit is the one
Turn with is_active=True, never evicted) and never reconstructs a provider
payload, so it is identical for every provider. ``evict_for_overflow`` is the
reactive companion: deficit-sized eviction after a server overflow 4xx (used by
the harness's move retry and the SGLang debrief retry).

Note: this is *truncation* — drop the earliest whole step-pairs — NOT
consolidation/summarization. By design we never summarize evicted
turns: this is a minimal benchmark and must not help the model beat the games.

Invariants:
- evict only WHOLE, oldest, completed units (for the bench loop: step-pairs);
- NEVER touch the active unit; the client also pins the head task;
- keep >= MIN_KEEP_TURNS units;
- evict the FEWEST units needed to get just under budget (minimal eviction).

Note: the trigger, `policy.last_prompt_tokens`, is each provider's own server-reported,
cache-inclusive prompt size, and tokenizers differ — so "the same proportion of the window"
is consistent in intent but only approximate in absolute tokens ACROSS providers.

Fairness note (interpreting cross-model results): the output reserve (~1.5 × max_tokens, see
cli.resolve_budget) is an ABSOLUTE token count, not a proportion — a model retains
`window − reserve` of history, which is a LARGER fraction of a big window than a small one
(e.g. ~95% of a 1M window vs ~81% of 256K at the default 32K cap). This is intentional (output
needs an absolute reserve), but small-window models keep proportionally less context per turn;
lower `--max-tokens` to shrink the reserve, or set `--context-proportion < 1` for a proportional
buffer, if tighter per-window parity is wanted.
"""

from __future__ import annotations

from .clientutil import parse_overflow_tokens
from .types import Policy

MIN_KEEP_TURNS = 2  # the active unit + at least one prior, always retained

MAX_OVERFLOW_ROUNDS = 8  # evict-and-retry rounds after an overflow 4xx (move + debrief paths)
_OVERFLOW_MARGIN = 1.25  # est_tokens is a chars//4 proxy — overshoot the parsed deficit by 25%
_OVERFLOW_SLACK = 1000   # absolute headroom on top, for whatever the retry itself appends


def truncate_if_needed(policy: Policy, budget: int | None, *, log=None) -> int:
    """Reactively shrink the next request if the LAST call's server-reported
    prompt size exceeded `budget`. Returns the number of units evicted.

    The trigger is reactive: we only learn `prompt_tokens` after a call, so we
    evict before the next one. Eviction is minimal — we project the prompt size
    down by each evicted unit's `est_tokens` and stop as soon as the projection
    is under budget (never reflexively down to the floor). The next call's real
    `last_prompt_tokens` corrects any estimate drift.
    """
    if not budget or budget <= 0:        # truncation disabled / no resolvable window
        return 0
    last = policy.last_prompt_tokens
    if last <= 0:                        # first-turn bootstrap: no server count yet
        return 0
    if last <= budget:
        return 0

    projected = last
    evicted = 0
    while projected > budget:
        turns = policy.turns()
        if len(turns) <= MIN_KEEP_TURNS:  # at the floor — stop, don't under-shrink
            break
        oldest = turns[0]
        if oldest.is_active:              # only the active unit left to give — never evict it
            break
        before = len(turns)
        policy.evict_oldest_turn()
        if len(policy.turns()) >= before:  # client refused (head/active only) — stop
            break
        projected -= max(0, oldest.est_tokens)
        evicted += 1

    if evicted and log:
        kept = len(policy.turns())
        if projected <= budget:
            log(
                f"context truncated: evicted {evicted} step-pair(s), projected prompt "
                f"~{projected} <= budget {budget}, kept {kept} unit(s)"
            )
        else:  # stopped at the MIN_KEEP_TURNS floor, still over budget
            log(
                f"context truncated: evicted {evicted} step-pair(s), projected prompt "
                f"~{projected} > budget {budget} (floor MIN_KEEP_TURNS={MIN_KEEP_TURNS} "
                f"reached, still over budget), kept {kept} unit(s)"
            )
    return evicted


def evict_for_overflow(policy: Policy, err, *, round_idx: int = 0) -> int:
    """Reactive recovery for a context-overflow 4xx: evict enough oldest units to cover the
    deficit the server itself reported. Returns the number of units evicted; 0 means eviction
    cannot shrink further (only the pinned head + active unit remain) — the caller gives up.

    The overflow wording states the exact numbers ("requested a total of T tokens ...
    maximum context length of L"); evict oldest units until their summed ``est_tokens``
    covers ``(T - L) * _OVERFLOW_MARGIN + _OVERFLOW_SLACK``. Sizing to the deficit matters:
    the oldest units are the smallest (early turns carry little/no reasoning), so a fixed
    per-retry count starves against a multi-thousand-token overshoot. When the wording has
    no counts, fall back to a doubling batch (``2**round_idx``) so repeated rounds still
    converge. No MIN_KEEP_TURNS floor here — this mirrors the recovery path's pre-existing
    semantics (the client itself refuses to evict the pinned head / lone active unit)."""
    parsed = parse_overflow_tokens(err)
    target = None
    if parsed:
        total, limit = parsed
        target = int((total - limit) * _OVERFLOW_MARGIN) + _OVERFLOW_SLACK
    evicted = 0
    removed_est = 0
    while (removed_est < target) if target is not None else (evicted < 2 ** round_idx):
        turns = policy.turns()
        if not turns or turns[0].is_active:   # only the active unit left to give — never evict it
            break
        before = len(turns)
        policy.evict_oldest_turn()
        if len(policy.turns()) >= before:     # client refused (head/active only) — stop
            break
        removed_est += max(0, turns[0].est_tokens)
        evicted += 1
    return evicted
