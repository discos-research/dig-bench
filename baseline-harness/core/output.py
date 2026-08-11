"""Output — streams one turn at a time to the terminal and, when run artifacts are
saved (the default; disable with --no-save-run), to a `.log` (identical text) and a
`.jsonl` (structured).

Provider-agnostic: typed
against core.types.Policy, with a one-line
truncation-event record so a rolling-window eviction is visible in the trace.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from .bench import fmt_level, levels_beaten, state_for_model, terminal_banner
from .types import Move, Policy


def _fmt_beaten(b: int | None) -> str:
    return "n/a" if b is None else str(b)


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _fmt_dur(s: float) -> str:
    return f"{int(s // 60)}m{int(s % 60):02d}s" if s >= 60 else f"{s:.1f}s"


def _tok_seg(output: int, thoughts: int, basis, fmt) -> str:
    """The `out … think …` segment. When the provider can't separate thinking from
    output (basis "included" — e.g. SGLang without reasoning_tokens), `output` already
    contains the thinking, so show it merged as `out + think N` rather than a fake split."""
    if basis == "included":
        return f"out + think {fmt(output)}"
    return f"out {fmt(output)} think {fmt(thoughts)}"


class Output:
    """Plain unicode only, so the file and the screen match."""

    def __init__(self, *, summary_chars: int, verbose: bool, log_path=None, jsonl_path=None):
        self.summary_chars = summary_chars
        self.verbose = verbose
        self._log = open(log_path, "a", encoding="utf-8") if log_path else None
        try:
            self._jsonl = open(jsonl_path, "a", encoding="utf-8") if jsonl_path else None
        except OSError:
            if self._log:  # don't leak the first handle when the second open fails
                self._log.close()
            raise

    def _w(self, text: str = "") -> None:
        print(text, flush=True)
        if self._log:
            self._log.write(text + "\n")
            self._log.flush()

    def _record(self, obj: dict) -> None:
        if self._jsonl:
            self._jsonl.write(json.dumps(obj) + "\n")
            self._jsonl.flush()

    def session_header(
        self, *, start: dict, model: str, thinking_level, run_label: str,
        seed=None, include_thoughts: bool = False, context_budget=None, pricing_row=None,
        model_max_context=None, provenance: dict | None = None,
    ) -> None:
        sid = start["session_id"]
        self._w(f"session {sid}  game {start.get('game')}  game_seed {start.get('seed')}")
        self._w(f"model {model}  thinking {thinking_level or 'none'}  label {run_label}")
        # Log the resolved window alongside the budget — a wrong window (e.g. a 200K id misread as
        # 1M) scales the whole truncation budget, so it must be auditable in the run record.
        self._w(
            f"seed {seed}  thought_summaries {'on' if include_thoughts else 'off'}"
            f"  context_window {model_max_context if model_max_context else 'unknown'}"
            f"  context_budget {context_budget if context_budget else 'off'}"
        )
        self._w(f"task: {start.get('description', '')}")
        self._record({
            "type": "session",
            "session_id": sid,
            "game": start.get("game"),
            "game_seed": start.get("seed"),
            "framework_version": start.get("framework_version"),
            "model": model,
            "model_version": model,
            "run_label": run_label,
            "thinking_level": thinking_level,
            "seed": seed,
            "include_thoughts": include_thoughts,
            "context_window": model_max_context,
            "context_budget": context_budget,
            "pricing": pricing_row,
            # TZ-aware (offset-carrying) timestamp — a naive local time is ambiguous in a
            # published artifact.
            "start_time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "description": start.get("description", ""),
            # Environment/config provenance (core/provenance.py): git commit, interpreter +
            # SDK versions, resolved CLI config (secrets excluded), endpoint, prompt hash.
            "provenance": provenance,
        })

    def turn_header(self, turn: int, state: dict) -> None:
        bar = "─" * 58
        seg = f" step {turn}  ·  level {fmt_level(state)}"
        if state.get("mode") is not None:
            seg += f"  ·  {state['mode']}"
        self._w(bar)
        self._w(seg)
        self._w(bar)
        self._w(state.get("observation", "") or "(empty state)")
        self._w("")
        self._w(
            f"  lives {state.get('lives_left')} · "
            f"steps_left {state.get('steps_remaining')} · status {state.get('status')}"
        )
        self._w(f"  legal: {', '.join(map(str, state.get('actions', [])))}")

    def _emit_reasoning(self, move: Move) -> None:
        summary = move.reasoning_summary or "(no summary)"
        if not self.verbose and len(summary) > self.summary_chars:
            summary = summary[: self.summary_chars] + "…"
        self._w(f"  🧠 {summary}")

    def _emit_meta(self, move: Move) -> None:
        u = move.usage
        turn_cost = "n/a" if move.cost is None else f"${move.cost:.5f}"
        total_cost = f"${self._cumulative:.4f}" if self._cumulative is not None else "n/a"
        # core stays neutral — the provider sets move.thoughts_basis, we just render it.
        seg = _tok_seg(u["output"], u["thoughts"], getattr(move, "thoughts_basis", None), str)
        self._w(
            f"  ⏱ {move.elapsed_s:.1f}s · tokens in {u['prompt']} (cached {u['cached']}) "
            f"{seg} · {turn_cost} (total {total_cost})"
        )
        if self.verbose and move.finish_reason:
            self._w(f"     finish_reason={move.finish_reason}")

    def turn_result(
        self, turn: int, step_index: int, before: dict, after: dict, move: Move, invalid: bool, cumulative: float
    ) -> None:
        self._cumulative = cumulative if move.cost is not None else None
        self._emit_reasoning(move)
        continuity = getattr(move, "continuity", None)
        if move.has_continuity:
            # "verified" (API-validated / probe-proven) renders bare; an unverified open-model
            # carry says so out loud — never a positive claim on an unproven round-trip.
            sig = "🔑 reasoning carried" + (" (unverified)" if continuity == "unverified" else "")
        elif continuity == "stripped":
            sig = "· reasoning re-sent, server STRIPS it (not carried)"
        elif move.reasoning_summary:
            sig = "· reasoning shown, not carried"   # model thought, but it isn't fed forward
        else:
            sig = "· no reasoning this turn"
        self._w(f"  ➡️  move: {move.action}        {sig}")
        if invalid:
            self._w("  → illegal move — state unchanged")
        else:
            self._w(
                f"  → effect: levels beaten "
                f"{_fmt_beaten(levels_beaten(before))}→{_fmt_beaten(levels_beaten(after))} · "
                f"level {fmt_level(after)} · "
                f"lives {after.get('lives_left')} · "
                f"status {after.get('status')} · done {str(after.get('done')).lower()}"
            )
        if after.get("transition"):  # explicit non-terminal level event
            self._w(f"  ★ {after['transition']}")
        if after.get("done"):  # terminal: bench omits transition, so synthesize a banner
            self._w(f"  ★ {terminal_banner(after)}")
        self._emit_meta(move)
        self._w("")
        self._record({
            "type": "turn",
            "turn": turn,
            "step_index": step_index,
            "elapsed_s": round(move.elapsed_s, 3),
            "input": state_for_model(before),
            "output": {
                "action": move.action,
                "reasoning_summary": move.reasoning_summary,
                "has_continuity": move.has_continuity,
                "continuity": getattr(move, "continuity", None),  # verified|unverified|stripped|None
                "thoughts_basis": getattr(move, "thoughts_basis", None),
                "finish_reason": move.finish_reason,
                "invalid_action": invalid,
            },
            "transition": after.get("transition"),
            "levels_beaten": levels_beaten(after),  # derived from level; display/analysis only
            "usage": move.usage,
            "cost": {"turn": move.cost, "cumulative": round(cumulative, 6)},
        })

    def turn_nudge(self, turn: int, before: dict, move: Move, reason: str, cumulative: float) -> None:
        self._cumulative = cumulative if move.cost is not None else None
        self._emit_reasoning(move)
        self._w(f"  · no make_move call — {reason}; nudging")
        self._emit_meta(move)
        self._w("")
        self._record({
            "type": "nudge",
            "turn": turn,
            "step_index": None,
            "elapsed_s": round(move.elapsed_s, 3),
            "input": state_for_model(before),
            "output": {
                "action": None,
                "reasoning_summary": move.reasoning_summary,
                "has_continuity": move.has_continuity,
                "continuity": getattr(move, "continuity", None),  # verified|unverified|stripped|None
                "thoughts_basis": getattr(move, "thoughts_basis", None),
                "finish_reason": move.finish_reason,
                "nudged": True,
            },
            "usage": move.usage,
            "cost": {"turn": move.cost, "cumulative": round(cumulative, 6)},
        })

    def truncated(self, turn: int, detail: str) -> None:
        """A rolling-window truncation fired after this turn — record it so the
        trace shows the mechanism actually worked."""
        self._w(f"  ↺ {detail}")
        self._record({"type": "truncation", "turn": turn, "detail": detail})

    def error(self, detail) -> None:
        self._w(f"  ✗ error: {detail}")
        self._record({"type": "error", "detail": detail if isinstance(detail, dict) else str(detail)})

    def warn(self, msg) -> None:
        self._w(f"  ⚠️  {msg}")
        self._record({"type": "warn", "message": str(msg)})

    def note(self, msg) -> None:
        """Neutral diagnostic line (e.g. the reasoning round-trip probe verdict) —
        not a warning."""
        self._w(f"  🔎 {msg}")

    def debrief(self, text: str | None) -> None:
        if not text:
            return
        bar = "─" * 58
        self._w("")
        self._w(bar)
        self._w(" Debrief")
        self._w(bar)
        self._w(text)

    def summary(
        self, *, game: str, state: dict, stop_reason: str, turns: int, policy: Policy,
        wall_s: float, sid: str, debrief_text: str | None, server: str,
    ) -> None:
        result = state.get("status") if stop_reason == "done" else stop_reason
        playback = f"{server}/agent-runs/{sid}/playback"
        cost = f"${policy.cost_usd:.4f}" if policy.has_pricing else "n/a (no pricing for model)"
        # How the `think` count was obtained (provider sets it; core stays neutral):
        # exact (counted by the API) / included (folded into output, not separable) / none.
        thoughts_basis = getattr(policy, "thoughts_basis", None)
        seg = _tok_seg(policy.output_tokens, policy.thoughts_tokens, thoughts_basis, _fmt_count)
        self._w("")
        self._w("════════════════════  SUMMARY  ════════════════════")
        self._w(
            f"game {game} · result {result} · "
            f"levels beaten {_fmt_beaten(levels_beaten(state))} · "
            f"level {fmt_level(state)} · turns {turns}"
        )
        self._w(
            f"LLM calls {policy.call_count} · tokens in {_fmt_count(policy.prompt_tokens)} "
            f"(cached {_fmt_count(policy.cached_tokens)}) {seg}"
        )
        self._w(f"cost {cost} total · wall {_fmt_dur(wall_s)} (llm {_fmt_dur(policy.elapsed_s)})")
        self._w(f"playback {playback}")
        self._record({
            "type": "summary",
            "result": result,
            "stop_reason": stop_reason,
            # The model id the SERVER reported on responses (e.g. an immutable snapshot id) —
            # pins the exact revision behind a mutable alias. None until a call succeeded.
            "reported_model": getattr(policy, "reported_model", None),
            "levels_beaten": levels_beaten(state),  # derived from level; the headline metric
            "level": state.get("level"),
            "max_level": state.get("max_level"),
            "turns": turns,
            "llm_calls": policy.call_count,
            "tokens": {
                "prompt": policy.prompt_tokens,
                "cached": policy.cached_tokens,
                "output": policy.output_tokens,
                "thoughts": policy.thoughts_tokens,
                "total": policy.total_tokens,
            },
            "cost_usd": round(policy.cost_usd, 6) if policy.has_pricing else None,
            "thoughts_basis": thoughts_basis,  # exact | included | none — how `think` was counted
            "wall_s": round(wall_s, 1),
            "llm_s": round(policy.elapsed_s, 1),
            "playback": playback,
            # provenance (open-model fields; None for closed providers via getattr)
            "move_channel": getattr(policy, "move_channel", None),
            "reasoning_roundtrip": getattr(policy, "_reasoning_roundtrips", None),
            "debrief": debrief_text,
        })

    def close(self) -> None:
        if self._log:
            self._log.close()
        if self._jsonl:
            self._jsonl.close()

    _cumulative: float | None = None
