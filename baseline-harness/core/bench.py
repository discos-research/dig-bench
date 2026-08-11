"""Agent Benchmark API HTTP client (stdlib only) + the game-state slice helpers.

This is the sole game
interface and its contract (idempotent stepping off the server's returned
``step_index``) is identical across every provider, so it lives in the
provider-agnostic core. Carries no game-solving intelligence.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request

from . import clientutil


# Identify the client to the bench server. Required: the public endpoint sits behind
# Cloudflare, which rejects Python-urllib's default User-Agent outright (error 1010,
# "browser signature banned") — without a product UA no request ever reaches the API.
USER_AGENT = "digbench-baseline-harness/1.0 (+https://digbench.ai)"


class BenchError(RuntimeError):
    """A bench-API call failed (after retries, or a non-retryable 4xx)."""


class Bench:
    """Thin HTTP client for the Agent Benchmark API (stdlib only).

    Retries transient (5xx / network) errors with backoff; 4xx are raised
    immediately. Stepping is idempotent: callers send the server's returned
    step_index + 1, and resending an applied index returns the cached response.
    """

    def __init__(self, base: str, token: str, *, timeout: int, max_retries: int):
        self.base = base
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries
        self.on_retry = None  # optional callable(msg) to log transient retries

    def _call(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        last = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                self.base + path,
                data=data,
                method=method,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
            )
            try:
                with clientutil.urlopen_no_redirect(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                # 408/429 are transient (timeout / rate-limit) — retry; other 4xx are deterministic.
                if exc.code < 500 and exc.code not in (408, 429):
                    raise BenchError(f"{method} {path} -> HTTP {exc.code}: {body}")
                last = f"HTTP {exc.code}: {body}"
            except urllib.error.URLError as exc:
                last = f"network: {exc.reason}"
            except (OSError, http.client.HTTPException, ValueError) as exc:
                # read()/decode/json.loads failures (IncompleteRead, timeout, reset, bad JSON) are
                # transient — retry, then raise BenchError so the run never crashes past the harness.
                last = f"read/parse: {exc}"
            if attempt < self.max_retries - 1:
                if self.on_retry:
                    self.on_retry(
                        f"bench {method} {path} attempt {attempt + 1}/{self.max_retries} "
                        f"failed: {last}; retrying"
                    )
                time.sleep(min(2 ** attempt, 30))
        raise BenchError(f"{method} {path} failed after {self.max_retries} attempts: {last}")

    def list_games(self) -> list[str]:
        return self._call("GET", "/games").get("games", [])

    def start_session(self, game: str, model_name: str, model_version: str) -> dict:
        return self._call(
            "POST", "/sessions",
            {"game": game, "model_name": model_name, "model_version": model_version},
        )

    def step(self, sid: str, step_index: int, action: str) -> dict:
        return self._call(
            "POST", f"/sessions/{sid}/step", {"step_index": step_index, "action": action}
        )


def state_for_model(state: dict) -> dict:
    """The slice of game state handed to the model (seed text / function result)."""
    out = {
        "observation": state.get("observation", ""),
        "level": state.get("level"),
        "max_level": state.get("max_level"),
        "lives_left": state.get("lives_left"),
        "steps_remaining": state.get("steps_remaining"),
        "status": state.get("status"),
        "done": state.get("done"),
        "legal_actions": state.get("actions", []),
    }
    if state.get("mode") is not None:  # creative-mode games only
        out["mode"] = state["mode"]
        out["creative_toggle"] = state.get("creative_toggle")
    # Explicit per-level event from the bench, e.g. "Level 1 cleared — advancing to
    # level 2." / "Level 2 failed — restarting level 2." `status` stays
    # "in_progress" through these (only flips at terminal game_over/completed), so
    # this is the model's only direct clear/fail signal. Null on normal steps ->
    # omitted.
    if state.get("transition") is not None:
        out["transition"] = state["transition"]
    return out


def fmt_level(state: dict) -> str:
    level, mx = state.get("level"), state.get("max_level")
    return f"{level}/{mx}" if mx is not None else str(level)


def levels_beaten(state: dict) -> int | None:
    """Levels cleared so far — the run's headline performance metric, DERIVED from
    `level` (the bench carries no per-turn score): mid-game you have cleared
    `level - 1`, and a fully `completed` game has beaten every level (`max_level`).
    Returns None if `level` is absent (the bench always sends it; this is purely
    defensive).

    Display/analysis ONLY: recorded in the trace and shown in the operator terminal,
    but NEVER part of `state_for_model` — the model tracks progress via
    `level`/`transition` exactly as a human does (parity), and is handed no score.
    """
    level = state.get("level")
    if not isinstance(level, int):
        return None
    if state.get("status") == "completed":
        mx = state.get("max_level")
        if isinstance(mx, int):
            return mx
    return max(0, level - 1)


def terminal_banner(state: dict) -> str:
    # Symmetric with the per-level `transition` line: at the terminal step the bench
    # carries the outcome in `status`/`done` (transition is None there), so synthesize
    # a matching banner for the log/terminal.
    status = state.get("status")
    if status == "completed":
        return "Game completed!"
    if status == "game_over":
        return "Game over — out of lives" if state.get("lives_left") == 0 else "Game over"
    return f"Game ended ({status})"
