#!/usr/bin/env python3
"""Isolation tests for core/bench.py — fakes only, no network.

Run from the repo root:
    python tests/test_bench.py

Covers the load-bearing transport contract:
- step() sends the caller's step_index verbatim and returns the server dict
  (so an invalid-action no-op surfaces unchanged);
- a transient 5xx is retried with backoff, then a later success returns;
- a 4xx (client error) is raised immediately, never retried;
- the state-slice helpers (state_for_model / fmt_level / terminal_banner).
"""

from __future__ import annotations

import io
import pathlib
import sys
import urllib.error

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core import bench as B


# ---- Fake urllib transport ---------------------------------------------


class FakeResp:
    """Context-manager response with .read() -> bytes, like urlopen()."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


def http_error(code: int, body: str = "boom") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(body.encode()))


def patched_bench(script, *, max_retries=3):
    """A Bench whose transport pops the next behavior from `script` (a callable or
    a value to return/raise), recording every Request. time.sleep is neutralized."""
    bench = B.Bench("http://x/api/agent", "tok", timeout=1, max_retries=max_retries)
    calls = []
    queue = list(script)

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResp(item)

    B.clientutil.urlopen_no_redirect = fake_urlopen
    B.time.sleep = lambda *_: None
    return bench, calls


# ---- Tests -------------------------------------------------------------


def test_step_sends_index_and_returns_server_dict():
    bench, calls = patched_bench([b'{"state": {"observation": "X"}, "step_index": 5, "invalid_action": true}'])
    out = bench.step("sid", 5, "north")
    # the server dict is returned verbatim — an invalid-action no-op surfaces unchanged
    assert out["invalid_action"] is True and out["step_index"] == 5
    req = calls[0]
    assert req.full_url == "http://x/api/agent/sessions/sid/step"
    import json
    assert json.loads(req.data.decode()) == {"step_index": 5, "action": "north"}
    assert req.headers["Authorization"] == "Bearer tok"
    # A product User-Agent is REQUIRED: the public endpoint's Cloudflare rejects
    # Python-urllib's default UA outright (error 1010), so its absence bricks the client.
    assert req.headers["User-agent"].startswith("digbench-baseline-harness/")


def test_3xx_redirect_raised_immediately_never_followed():
    # The transport refuses redirects (urlopen_no_redirect raises the 3xx), and bench treats it
    # as deterministic: one request, no retry — the bearer token is never replayed anywhere.
    bench, calls = patched_bench([http_error(302), b'{"games": ["A"]}'])
    try:
        bench.list_games()
        assert False, "expected BenchError on a refused redirect"
    except B.BenchError as e:
        assert "302" in str(e)
    assert len(calls) == 1  # never followed, never retried


def test_5xx_retried_then_success():
    bench, calls = patched_bench([http_error(503), http_error(500), b'{"games": ["A"]}'])
    out = bench.list_games()
    assert out == ["A"]
    assert len(calls) == 3  # two failures + one success


def test_429_retried_then_success():
    # 429 (rate-limit) is transient, not a fatal 4xx.
    bench, calls = patched_bench([http_error(429), b'{"games": ["A"]}'])
    assert bench.list_games() == ["A"]
    assert len(calls) == 2


def test_read_and_parse_errors_retry_then_bencherror():
    # IncompleteRead / socket timeout / bad JSON must be caught, retried, and surfaced as
    # BenchError — never a raw crash past the harness.
    bench, calls = patched_bench(
        [TimeoutError("t"), B.http.client.IncompleteRead(b""), b"not-json{"], max_retries=3)
    try:
        bench.list_games()
        assert False, "expected BenchError, not a raw crash"
    except B.BenchError:
        pass
    assert len(calls) == 3


def test_read_error_recovers_on_retry():
    bench, calls = patched_bench([TimeoutError("t"), b'{"games": ["A"]}'])
    assert bench.list_games() == ["A"]
    assert len(calls) == 2


def test_4xx_raised_immediately():
    bench, calls = patched_bench([http_error(409, "wrong index")])
    raised = False
    try:
        bench.step("sid", 99, "x")
    except B.BenchError as exc:
        raised = True
        assert "409" in str(exc) and "wrong index" in str(exc)
    assert raised and len(calls) == 1  # never retried


def test_retries_exhausted_raises():
    bench, calls = patched_bench([http_error(500), http_error(500), http_error(500)], max_retries=3)
    raised = False
    try:
        bench.list_games()
    except B.BenchError as exc:
        raised = True
        assert "after 3 attempts" in str(exc)
    assert raised and len(calls) == 3


def test_state_helpers():
    state = {"observation": "obs", "score": 7, "level": 2, "max_level": 9, "lives_left": 3,
             "steps_remaining": 4, "status": "in_progress", "done": False, "actions": ["1", "2"],
             "mode": "creative", "creative_toggle": "~", "transition": "Level 1 cleared"}
    sliced = B.state_for_model(state)
    assert sliced["legal_actions"] == ["1", "2"]
    # `score` (levels beaten) is run metadata the bench carries OUTSIDE state; the
    # agent slice must never expose it (parity: the model tracks progress via level).
    assert "score" not in sliced
    assert sliced["mode"] == "creative" and sliced["creative_toggle"] == "~"
    assert sliced["transition"] == "Level 1 cleared"
    # null mode/transition omitted
    bare = B.state_for_model({"observation": "o", "actions": []})
    assert "mode" not in bare and "creative_toggle" not in bare and "transition" not in bare
    assert B.fmt_level(state) == "2/9"
    assert B.fmt_level({"level": 3, "max_level": None}) == "3"
    assert B.terminal_banner({"status": "completed"}) == "Game completed!"
    assert B.terminal_banner({"status": "game_over", "lives_left": 0}) == "Game over — out of lives"


def test_levels_beaten():
    # Derived purely from `level` (the bench does not send a per-turn `score`):
    # mid-game you have cleared level - 1, regardless of any `score` in the dict.
    on_lvl2 = {"score": 7, "level": 2, "max_level": 9, "status": "in_progress"}
    assert B.levels_beaten(on_lvl2) == 1  # level 2, in_progress -> 1 cleared (NOT score=7)
    assert B.levels_beaten({"level": 1, "status": "in_progress"}) == 0  # on level 1, none cleared
    # a `completed` game has beaten every level (max_level), even if `level` stalls at max
    assert B.levels_beaten({"level": 9, "max_level": 9, "status": "completed"}) == 9
    assert B.levels_beaten({"level": 7, "max_level": 7, "status": "completed"}) == 7
    # game_over uses the plain level - 1 (you kept whatever you had cleared)
    assert B.levels_beaten({"level": 4, "max_level": 9, "status": "game_over"}) == 3
    # defensive: clamp to >= 0, and None when level is absent/non-int
    assert B.levels_beaten({"level": 0}) == 0
    assert B.levels_beaten({"observation": "o"}) is None
    # never exposed to the model (parity) — neither the derived metric nor a raw score
    sliced = B.state_for_model(on_lvl2)
    assert "levels_beaten" not in sliced and "score" not in sliced


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
