#!/usr/bin/env python3
"""Isolation tests for core/prompts.py — pure prompt construction, no dependencies.

    python tests/test_prompts.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core.prompts import creative_mode_instruction, seed_text


def test_creative_mode_instruction_names_exact_toggle():
    instruction = creative_mode_instruction("/")
    assert instruction == (
        "Important: creative mode\n"
        'At nearly any time, you can use a button to switch into "creative mode", '
        "where you can experiment safely without losing steps or lives.\n"
        "It may be necessary to use creative mode in order to discover the rules "
        "of the game without running out of steps.\n\n"
        'Call `make_move` with action "/" to enter creative mode.\n'
        'Call `make_move` with action "/" again to return to survival mode.\n'
        'Only submit "/" when it appears in `legal_actions`.'
    )


def test_seed_text_uses_game_specific_toggle():
    text = seed_text("desc", {"creative_toggle": "~", "legal_actions": ["~"]})
    assert 'Call `make_move` with action "~" to enter creative mode.' in text
    assert 'Call `make_move` with action "~" again to return to survival mode.' in text
    assert 'Only submit "~" when it appears in `legal_actions`.' in text
    assert 'action "/"' not in text


def test_seed_text_omits_creative_guidance_without_toggle():
    text = seed_text("desc", {"legal_actions": ["1"]})
    assert "Important: creative mode" not in text


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
