"""Shared, provider-neutral prompt text and the make_move tool contract.

Every provider uses the SAME wording (fairness) and builds its own
native tool from one neutral spec. Deliberately NO `google.genai` import here —
each client translates MAKE_MOVE_SPEC into its own tool object (Gemini
FunctionDeclaration, OpenAI/SGLang function tool, Anthropic tool), keeping this
module dependency-free.
"""

from __future__ import annotations

import json

SYSTEM_INSTRUCTION = (
    "We are not going to tell you the rules of this game — you have to figure "
    "them out for yourself.\n\n"
    "Levels, lives and steps:\n"
    "- The aim is to reach as high a level as possible. For each level you reach "
    "you will be awarded a bonus.\n"
    "- You advance levels by reaching certain states within the game. You will "
    "have to figure out what these are.\n"
    "- Within each level, you have a limited number of steps. If you run out of "
    "steps, you lose a life. If you lose all your lives, the game is over.\n"
    "- It is also possible to lose a life by reaching certain states within the "
    "game.\n\n"
    "How you play:\n"
    "- A short TASK DESCRIPTION at the start gives the objective and any special "
    "actions (not the rules). Each turn you then receive the current state: a "
    "text `observation` (the rendered screen), `level`, `max_level`, "
    "`lives_left`, `steps_remaining`, the list of `legal_actions`, plus `mode`, "
    "`creative_toggle`, and a `transition` message when those apply.\n"
    "- Infer what each action does from how the state changes, and build on what "
    "you learn across turns. Reason carefully, then call `make_move` with EXACTLY "
    "ONE action from `legal_actions`."
)

DEBRIEF_SYSTEM_INSTRUCTION = (
    "You are debriefing after playing a text game. Explain the discovered "
    "mechanics, objective, useful strategies, and remaining uncertainties for "
    "human collaborators. Do not choose another action."
)

DEBRIEF_PROMPT = "The game has ended. Write your debrief now."

NUDGE_TEXT = (
    "You did not call make_move. Call make_move with exactly one action from the "
    "current legal_actions."
)

TRUNCATION_NUDGE_TEXT = (
    "Your previous response was cut off at the output-token limit before you called make_move. "
    "Be concise: reason briefly, then call make_move with exactly one action from the current "
    "legal_actions."
)


def creative_mode_instruction(toggle: str | None) -> str:
    """Human-facing creative-mode guidance plus the game's exact action token."""
    if not toggle:
        return ""
    action = json.dumps(toggle)
    return (
        "Important: creative mode\n"
        'At nearly any time, you can use a button to switch into "creative mode", '
        "where you can experiment safely without losing steps or lives.\n"
        "It may be necessary to use creative mode in order to discover the rules "
        "of the game without running out of steps.\n\n"
        f"Call `make_move` with action {action} to enter creative mode.\n"
        f"Call `make_move` with action {action} again to return to survival mode.\n"
        f"Only submit {action} when it appears in `legal_actions`."
    )


def seed_text(description: str, state_slice: dict) -> str:
    """The first user turn: the task description (objective + special actions; NOT the rules) + the
    initial state. IDENTICAL wording for every provider (fairness)."""
    creative_instruction = creative_mode_instruction(state_slice.get("creative_toggle"))
    return (
        "You are now playing this game.\n\n"
        + (creative_instruction + "\n\n" if creative_instruction else "")
        + "TASK DESCRIPTION (objective + any special actions, NOT the rules):\n"
        + (description or "(none provided)")
        + "\n\nINITIAL STATE:\n"
        + json.dumps(state_slice, indent=2)
        + "\n\nReason about it, then call make_move."
    )


# Neutral move-tool spec. `parameters` is a JSON Schema; each client builds its
# own native tool from this (one legal-action token per turn).
MAKE_MOVE_SPEC = {
    "name": "make_move",
    "description": (
        "Submit your move for this turn. `action` must be EXACTLY one of the "
        "legal_actions from the latest game state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "One legal action token (e.g. '1', '2', 'w', '/').",
            }
        },
        "required": ["action"],
    },
}
