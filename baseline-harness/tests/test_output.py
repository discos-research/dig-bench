#!/usr/bin/env python3
"""Isolation tests for core/output.py JSONL records.

    python tests/test_output.py

Covers the observability contract: error()/warn() reach the .jsonl (not just the .log),
carrying an explicit "type" so a consumer can filter failures out of the turn stream;
the session row carries the provenance dict and a TZ-aware start_time.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core.output import Output


def test_error_and_warn_emit_typed_jsonl():
    with tempfile.TemporaryDirectory() as d:
        jp = pathlib.Path(d) / "r.jsonl"
        out = Output(summary_chars=80, verbose=False, jsonl_path=jp)
        with redirect_stdout(io.StringIO()):
            out.error({"server_protocol": "missing step_index"})   # dict detail preserved
            out.error(ValueError("boom"))                          # non-dict -> str()
            out.warn("context overflow — evicted oldest turn")
        out.close()
        rows = [json.loads(line) for line in jp.read_text().splitlines()]
        assert {"type": "error", "detail": {"server_protocol": "missing step_index"}} in rows
        assert {"type": "error", "detail": "boom"} in rows
        assert {"type": "warn", "message": "context overflow — evicted oldest turn"} in rows


def test_continuity_states_rendered_and_recorded():
    from core.types import Move
    usage = {"prompt": 10, "cached": 0, "output": 5, "thoughts": 0, "total": 15}
    state = {"observation": "S", "actions": ["1"], "level": 1, "max_level": 2}

    def mv(has, cont):
        return Move("1", "thought about it", has, "stop", usage, None, 0.1, continuity=cont)

    with tempfile.TemporaryDirectory() as d:
        jp = pathlib.Path(d) / "r.jsonl"
        out = Output(summary_chars=80, verbose=False, jsonl_path=jp)
        buf = io.StringIO()
        with redirect_stdout(buf):
            out.turn_result(1, 1, state, state, mv(True, "verified"), False, 0.0)
            out.turn_result(2, 2, state, state, mv(True, "unverified"), False, 0.0)
            out.turn_result(3, 3, state, state, mv(False, "stripped"), False, 0.0)
        out.close()
        text = buf.getvalue()
        carried = [l for l in text.splitlines() if "reasoning carried" in l]
        assert len(carried) == 2
        assert "(unverified)" not in carried[0]    # verified renders bare
        assert "(unverified)" in carried[1]        # unverified says so out loud
        assert "server STRIPS it" in text          # stripped is explicit, not silent
        rows = [json.loads(line) for line in jp.read_text().splitlines()]
        assert [r["output"]["continuity"] for r in rows] == ["verified", "unverified", "stripped"]


def test_session_row_carries_provenance_and_tz_aware_start_time():
    from datetime import datetime
    prov = {"git": {"commit": "abc123", "dirty": False}, "python": "3.12.0",
            "config": {"model": "m"}, "prompt_contract_sha256": "f" * 64}
    with tempfile.TemporaryDirectory() as d:
        jp = pathlib.Path(d) / "r.jsonl"
        out = Output(summary_chars=80, verbose=False, jsonl_path=jp)
        with redirect_stdout(io.StringIO()):
            out.session_header(start={"session_id": "s1", "game": "G", "seed": 7},
                               model="m", thinking_level="high", run_label="lbl",
                               provenance=prov)
        out.close()
        row = json.loads(jp.read_text().splitlines()[0])
        assert row["type"] == "session" and row["provenance"] == prov
        # start_time must parse AND carry a UTC offset (naive local time is ambiguous)
        assert datetime.fromisoformat(row["start_time"]).utcoffset() is not None


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
