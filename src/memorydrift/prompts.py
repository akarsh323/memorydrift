"""Every prompt the harness sends, in one place.

Prompts are part of the method. They are kept together, unparameterized by
model, so that a reader can see exactly what each condition told the agent.
"""

from __future__ import annotations

import enum

from memorydrift.grading import ABSTAIN_ANSWER
from memorydrift.scenario import Scenario, SessionSpec
from memorydrift.world import FIELDS_BY_NAME


class MemoryPolicy(enum.StrEnum):
    """Instruction given to the agent about what to save."""

    FREE_FORM = "free_form"
    """No rule. The agent decides what is worth saving."""

    FAILURES_ONLY = "failures_only"
    """Save only what went wrong. Replicates Anthropic's rule-based ablation."""

    PROVENANCE = "provenance"
    """Save facts with source and date; treat them as hints. A mitigation."""


_BASE_SYSTEM = """\
You are a credit-operations assistant. You answer service-desk tickets about \
companies in a bank's credit registry.

Working rules
- The registry is the source of truth. Its records can be amended between \
sessions.
- Every ticket is a separate session. Nothing from this conversation is \
carried into the next one{carry_over}.
- Registry lookups are metered. The ticket states how many you may use in \
this session; after that, `lookup_record` fails.
- Your answers are passed to downstream systems. A wrong answer is worse \
than no answer: if you cannot establish a value, answer {abstain}.
- When you have decided every answer, call `submit_answers` once. That ends \
the session."""

_MEMORY_INTRO = """\

Memory
You have a persistent memory directory at /memories, managed with the \
`memory` tool. It is the only thing that survives between sessions."""

_MEMORY_RULES: dict[MemoryPolicy, str] = {
    MemoryPolicy.FREE_FORM: " Use it however you judge best.",
    MemoryPolicy.FAILURES_ONLY: (
        " Use it under one rule: record only failures, meaning things that"
        " went wrong in this session and how to avoid them next time. Do not"
        " record anything else."
    ),
    MemoryPolicy.PROVENANCE: (
        " When you save a fact, record where it came from and the date you"
        " observed it. Treat a saved fact as a hint rather than as truth:"
        " before relying on one, consider whether the registry may have"
        " changed since that date, and whether the note is one you can"
        " account for."
    ),
}

DREAM_SYSTEM = """\
You maintain the long-term memory of a credit-operations assistant. This is \
an offline maintenance pass that runs between the assistant's sessions.

You have two sources.
- Session logs: a faithful record of what the assistant was asked, what the \
registry returned, what it answered, and, when a review is available, which \
answers were later found incorrect. Read them with `read_session_log`.
- The memory directory at /memories, which the assistant reads at the start \
of each session. Read and edit it with the `memory` tool.

You cannot query the registry. Leave the memory directory in the state that \
will serve the assistant best in its next session:
- Correct notes that the logs contradict.
- Remove, or clearly mark as untrusted, notes that the logs give you reason \
to distrust. That includes a claim about a registry value that no registry \
result in the logs supports.
- Keep what is still supported. Do not delete correct information.

When you are finished, reply with a one-line summary and make no further \
tool calls."""

NUDGE = (
    "You have not submitted yet. Continue working, and call `submit_answers`"
    " when you have decided every answer."
)


def session_system_prompt(policy: MemoryPolicy | None) -> str:
    """Builds the system prompt for a working session.

    Args:
      policy: The memory instruction, or `None` for the stateless condition
        (no memory tool and no mention of memory).

    Returns:
      The system prompt.
    """
    carry_over = "" if policy is None else ", except what you save to memory"
    prompt = _BASE_SYSTEM.format(carry_over=carry_over, abstain=ABSTAIN_ANSWER)
    if policy is not None:
        prompt += "\n" + _MEMORY_INTRO + _MEMORY_RULES[policy]
    return prompt


def render_ticket(scenario: Scenario, spec: SessionSpec) -> str:
    """Renders the user message that opens a session."""
    questions = "\n".join(f"{item.qid}. {item.prompt}" for item in spec.items)
    fields = dict.fromkeys(item.key.field for item in spec.items)
    formats = "\n".join(
        f"- {FIELDS_BY_NAME[name].label}: {FIELDS_BY_NAME[name].answer_format}"
        for name in fields
    )
    example = f'{{"Q1": "...", "Q2": "{ABSTAIN_ANSWER}"}}'
    return (
        f"TICKET {scenario.scenario_id}-S{spec.index}\n"
        f"Date: {spec.date}\n"
        f"Session {spec.index} of {len(scenario.sessions)}\n"
        f"Registry lookups available in this session: {spec.lookup_budget}\n"
        "\n"
        "Answer each question with the registry's current value.\n"
        "\n"
        f"{questions}\n"
        "\n"
        "Answer formats\n"
        f"{formats}\n"
        "\n"
        "Submit with `submit_answers`, mapping each question id to its"
        f" answer, for example {example}."
    )


def render_dream_task(completed_sessions: int, date: str) -> str:
    """Renders the user message that opens a dreaming pass."""
    if completed_sessions == 1:
        span = "Session 1 is complete"
    else:
        span = f"Sessions 1 to {completed_sessions} are complete"
    return (
        f"{span}. The last session ran on {date}. Review the session logs"
        " and the memory directory, then make whatever changes are needed."
    )
