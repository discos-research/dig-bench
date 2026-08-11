"""Generic token + USD accounting mechanism: longest-substring model matching and
the per-call cost formula.

This module holds **no provider data**. Each client owns its own pricing table,
context-window table, and any provider-specific tier thresholds
(clients/<provider>.py) and passes its pricing table into compute_cost here — so
adding a provider touches only that client, never this file.
"""

from __future__ import annotations


def match_model(model: str, table: dict):
    """Value under the LONGEST table key that is a substring of `model`; None if no
    key matches. An exact id wins (it is its own longest substring). The one matcher
    for both pricing rows and context-window lookups across every provider — substring
    (not prefix) so it tolerates regional id prefixes like `us.`/`global.` on
    Bedrock/Vertex model ids (e.g. `us.anthropic.claude-...` still matches `claude-...`)."""
    best = None
    for key, value in table.items():
        if key in model and (best is None or len(key) > len(best[0])):
            best = (key, value)
    return best[1] if best else None


def pricing_for(model: str, table: dict) -> dict | None:
    return match_model(model, table)


def compute_cost(
    model: str, prompt: int, cached: int, output: int, thoughts: int, table: dict
) -> float | None:
    """Per-call USD. Cached tokens are billed at the cheaper cached rate; the
    rest of the prompt at full input rate. Returns None if the model is unpriced.

    A pricing row may carry an optional ``"long_context"`` sub-row
    (``{"threshold": N, <same rate keys>}``): calls whose prompt exceeds the threshold are
    billed at the sub-row's rates instead (per call — providers price the whole request at
    the higher tier once the input crosses it). Rows without the key stay flat."""
    pricing = pricing_for(model, table)
    if pricing is None:
        return None
    tier = pricing.get("long_context")
    if tier and prompt > tier["threshold"]:
        pricing = {**pricing, **{k: v for k, v in tier.items() if k != "threshold"}}
    uncached = max(0, prompt - cached)
    cost = (
        uncached * pricing["input_per_1m"]
        + cached * pricing.get("cached_input_per_1m", pricing["input_per_1m"])
        + output * pricing["output_per_1m"]
        + thoughts * pricing.get("thoughts_per_1m", pricing["output_per_1m"])
    ) / 1_000_000
    return round(cost, 8)
