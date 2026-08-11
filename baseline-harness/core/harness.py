"""The provider-agnostic game loop.

The turn loop — stop conditions,
invalid/nudge caps, cost cap, Ctrl-C handling, debrief-once, saved run artifacts
— plus a turn-aware rolling-window context-truncation step after each applied
move. Drives any core.types.Policy; carries no game logic.
"""

from __future__ import annotations

import time

from . import clientutil
from .bench import BenchError, state_for_model
from .history import MAX_OVERFLOW_ROUNDS, evict_for_overflow, truncate_if_needed
from .output import Output
from .prompts import NUDGE_TEXT, TRUNCATION_NUDGE_TEXT
from .types import Policy


def _generate_recovering_overflow(policy: Policy, out: Output):
    """``generate_move`` that recovers from a context-overflow 4xx by evicting oldest turn(s)
    sized to the deficit the server reported and retrying (bounded by ``MAX_OVERFLOW_ROUNDS``).
    A failed call returns ``Move(error)`` BEFORE mutating policy state, so re-calling after
    eviction is clean. Stops once eviction can no longer shrink (only the pinned head + active
    unit remain) — then the caller sees the error and ends."""
    move = policy.generate_move()
    rounds = 0
    while (move.error is not None and clientutil.is_context_overflow(move.error)
           and rounds < MAX_OVERFLOW_ROUNDS):
        evicted = evict_for_overflow(policy, move.error, round_idx=rounds)
        if not evicted:                       # nothing evicted — cannot recover
            break
        rounds += 1
        out.warn(f"context overflow — evicted {evicted} oldest turn(s), retrying ({rounds})")
        move = policy.generate_move()
    return move


def play(
    bench, policy: Policy, out: Output, start: dict, args, server: str,
    thinking_level, run_label: str, *, context_budget: int | None = None,
    provenance: dict | None = None,
) -> str:
    policy.on_retry = out.warn  # log failed LLM attempts / continuity recoveries
    bench.on_retry = out.warn   # log transient bench retries
    sid = start["session_id"]
    game = start.get("game")
    state = start["state"]
    idx = start.get("step_index", 0)

    out.session_header(
        start=start, model=args.model, thinking_level=thinking_level, run_label=run_label,
        seed=getattr(policy, "seed", None), include_thoughts=getattr(policy, "include_thoughts", False),
        context_budget=context_budget, pricing_row=getattr(policy, "pricing_row", None),
        model_max_context=getattr(policy, "model_max_context", None), provenance=provenance,
    )
    policy.start(start.get("description", ""), state_for_model(state))

    wall_start = time.time()
    turn = 0
    consecutive_invalid = 0
    stop_reason: str | None = None

    try:
        while not state.get("done") and turn < args.max_steps:
            if args.max_cost_usd and policy.cost_usd >= args.max_cost_usd:
                stop_reason = "cost_cap"
                break
            turn += 1
            out.turn_header(turn, state)

            move = _generate_recovering_overflow(policy, out)
            if move.error is not None:
                out.error(move.error)
                stop_reason = "api_failure"
                break

            if move.action is None:  # no make_move call (or no candidate)
                # A cap cutoff (output truncated before make_move) gets a concise-and-call nudge +
                # a warning, so it's recorded as a truncation, not silently a plain invalid move.
                fr = (move.finish_reason or "").lower()
                if "length" in fr or "max_tok" in fr or fr == "incomplete":
                    out.warn("output truncated on the token cap before make_move — nudging to be concise")
                    policy.add_nudge(TRUNCATION_NUDGE_TEXT)
                else:
                    policy.add_nudge(NUDGE_TEXT)
                out.turn_nudge(turn, state, move, "no make_move call", policy.cost_usd)
                # Roll the window on the nudge turn too — it grew provider history and updated
                # last_prompt_tokens; skipping it lets a burst of nudges accrue past the budget.
                truncate_if_needed(policy, context_budget, log=lambda m: out.truncated(turn, m))
                consecutive_invalid += 1
                if consecutive_invalid > args.max_invalid_retries:
                    stop_reason = "blocked"
                    break
                continue

            # Anti-cheat guard: never send an action the model was not offered. The state
            # slice the model saw carries its `legal_actions` (= state["actions"]); enforce
            # that membership locally so a hidden/undocumented server action can't be played
            # to win, and an obviously illegal move is charged as invalid without a server
            # round-trip. Only enforced when we actually have a populated legal set — an
            # empty/absent one means "let the server adjudicate" (preserves prior behavior).
            legal = state.get("actions")
            if isinstance(legal, list) and legal and move.action not in legal:
                result = state_for_model(state)  # unchanged state: an illegal move advances nothing
                result.pop("transition", None)   # ...so a prior clear/fail transition is stale here
                result["invalid_action"] = True
                result["note"] = f"{move.action!r} is not legal — pick one of legal_actions."
                policy.observe(result)
                out.turn_result(turn, idx, state, state, move, True, policy.cost_usd)
                truncate_if_needed(policy, context_budget, log=lambda m: out.truncated(turn, m))
                consecutive_invalid += 1
                if consecutive_invalid > args.max_invalid_retries:
                    stop_reason = "blocked"
                    break
                continue

            before = state
            try:
                step = bench.step(sid, idx + 1, move.action)
            except BenchError as exc:
                out.error(exc)
                stop_reason = "bench_failure"
                break
            new_idx = step.get("step_index")
            invalid = bool(step.get("invalid_action"))
            # Index-progression contract: a valid move advances the index by exactly one; an
            # invalid move does not advance (the state is unchanged). Anything else — missing,
            # stale, or over-advanced — would silently replay or skip steps on the next request.
            expected = idx if invalid else idx + 1
            if new_idx != expected:
                out.error({"server_protocol": f"step_index {new_idx!r} != expected {expected}"})
                stop_reason = "server_protocol"
                break
            idx = new_idx
            state = step["state"]
            result = state_for_model(state)
            result["invalid_action"] = invalid
            if invalid:
                result["note"] = f"{move.action!r} is not legal — pick one of legal_actions."
            policy.observe(result)
            out.turn_result(turn, idx, before, state, move, invalid, policy.cost_usd)

            # Roll the window (truncate) if the last call's prompt exceeded the budget.
            truncate_if_needed(policy, context_budget, log=lambda m: out.truncated(turn, m))

            if invalid:
                consecutive_invalid += 1
                if consecutive_invalid > args.max_invalid_retries:
                    stop_reason = "blocked"
                    break
            else:
                consecutive_invalid = 0
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        out.error("interrupted by user (Ctrl-C)")

    if stop_reason is None:
        stop_reason = "done" if state.get("done") else "max_steps"

    # Debrief on every normal stop; skip it when the user bailed (cost cap or Ctrl-C) so we
    # don't spend/wait against their intent. RE-CHECK the cap here too: it is tested before
    # each turn, so a run whose final move crossed it stops with a normal reason — the debrief
    # (a whole-transcript call, often the most expensive of the run) must not overspend it.
    debrief_text = None
    over_cap = bool(args.max_cost_usd) and policy.cost_usd >= args.max_cost_usd
    if over_cap and stop_reason not in ("cost_cap", "interrupted"):
        out.warn(f"cost cap ${args.max_cost_usd} reached (${policy.cost_usd:.4f}) — skipping debrief")
    if stop_reason not in ("cost_cap", "interrupted") and not over_cap:
        try:
            debrief_text = policy.debrief()
            if debrief_text is None:
                out.error({"debrief_failed_after_retries": getattr(policy, "_last_error", None)})
        except KeyboardInterrupt:
            out.error("debrief interrupted")

    out.debrief(debrief_text)
    out.summary(
        game=game, state=state, stop_reason=stop_reason, turns=turn, policy=policy,
        wall_s=time.time() - wall_start, sid=sid, debrief_text=debrief_text, server=server,
    )
    return stop_reason
