"""Tools exposed to the agent during a session, and to the dreamer.

Session tools close over a `SessionRuntime`, which holds everything the
harness needs to know about how the session went: which lookups were made,
whether amendments were checked, and what was submitted. The runtime is a
plain Python object that lives for one session; nothing in it is visible to
the model except through tool results.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

from inspect_ai.tool import Tool, ToolError, tool

from memorydrift.scenario import Scenario, SessionSpec
from memorydrift.world import FIELDS_BY_NAME, FactKey


@dataclasses.dataclass
class SessionRuntime:
    """Mutable state of one running session, shared by its tools.

    Attributes:
      scenario: The episode the session belongs to.
      spec: The session being run.
      lookups: Successful registry lookups, in call order.
      amendments_checked: Whether `list_amendments` was called.
      answers: Submitted answers by question id, or `None` before submission.
    """

    scenario: Scenario
    spec: SessionSpec
    lookups: list[FactKey] = dataclasses.field(default_factory=list)
    amendments_checked: bool = False
    answers: dict[str, str] | None = None

    @property
    def lookups_remaining(self) -> int:
        """Metered lookups the session has left."""
        return self.spec.lookup_budget - len(self.lookups)

    @property
    def submitted(self) -> bool:
        """Whether the agent has closed the session."""
        return self.answers is not None


def _dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


@tool
def lookup_record(runtime: SessionRuntime) -> Tool:
    """Creates the metered registry lookup tool for one session.

    Args:
      runtime: The session the tool reads from and records into.

    Returns:
      The tool.
    """

    async def execute(entity: str, field: str) -> str:
        """Look up the current value of one registry field for one company.

        Each successful call uses one of this session's metered lookups.

        Args:
          entity: Company name, as written in the ticket.
          field: Field to read. One of credit_limit_usd, internal_rating,
            relationship_manager, max_leverage_covenant, next_review_date,
            collateral_type.

        Returns:
          JSON with the value, the date it was read, and the lookups left.
        """
        resolved = runtime.scenario.find_entity(entity)
        if resolved is None:
            known = ", ".join(runtime.scenario.entities)
            raise ToolError(
                f"No company named {entity!r} in the registry. Companies:"
                f" {known}. This call did not use a lookup."
            )
        name = field.strip()
        if name not in FIELDS_BY_NAME:
            raise ToolError(
                f"No field named {field!r}. Fields:"
                f" {', '.join(FIELDS_BY_NAME)}. This call did not use a"
                " lookup."
            )
        if runtime.submitted:
            raise ToolError("The session is closed.")
        if runtime.lookups_remaining <= 0:
            raise ToolError("No registry lookups are left in this session.")

        key = FactKey(entity=resolved, field=name)
        runtime.lookups.append(key)
        return _dumps(
            {
                "entity": resolved,
                "field": name,
                "value": runtime.scenario.value(key, runtime.spec.era),
                "as_of": runtime.spec.date,
                "lookups_remaining": runtime.lookups_remaining,
            }
        )

    return execute


@tool
def list_amendments(runtime: SessionRuntime) -> Tool:
    """Creates the free registry change-log tool for one session.

    Args:
      runtime: The session the tool reads from and records into.

    Returns:
      The tool.
    """

    async def execute() -> str:
        """List the registry records that have been amended, and when.

        Shows which fields changed and on what date, not the new values. This
        call is free: it does not use a metered lookup.

        Returns:
          JSON list of amended records with their amendment dates.
        """
        runtime.amendments_checked = True
        visible = runtime.scenario.amendments_through(runtime.spec.era)
        return _dumps(
            {
                "as_of": runtime.spec.date,
                "amendments": [
                    {
                        "entity": a.key.entity,
                        "field": a.key.field,
                        "amended_on": a.amended_on,
                    }
                    for a in visible
                ],
            }
        )

    return execute


@tool
def submit_answers(runtime: SessionRuntime) -> Tool:
    """Creates the tool that records final answers and closes the session.

    Args:
      runtime: The session the tool records into.

    Returns:
      The tool.
    """

    async def execute(answers: dict[str, str]) -> str:
        """Submit the final answers for this ticket and close the session.

        Call this exactly once, after deciding every answer.

        Args:
          answers: Maps each question id (for example "Q1") to its answer.
            Use "UNKNOWN" for a value you could not establish.

        Returns:
          A confirmation of what was recorded.
        """
        if runtime.submitted:
            raise ToolError("Answers were already submitted for this ticket.")
        if not isinstance(answers, Mapping):
            raise ToolError(
                "`answers` must be an object that maps question ids to answers."
            )
        expected = {item.qid for item in runtime.spec.items}
        by_id = {str(k).strip().upper(): v for k, v in answers.items()}
        recorded = {
            qid: _as_text(by_id[qid]) for qid in sorted(expected & set(by_id))
        }
        if not recorded:
            raise ToolError(
                "No answer matched a question id on this ticket. Expected"
                f" ids: {', '.join(sorted(expected, key=_qid_order))}."
            )
        runtime.answers = recorded
        missing = sorted(expected - set(recorded), key=_qid_order)
        note = f" Not answered: {', '.join(missing)}." if missing else ""
        return (
            f"Recorded {len(recorded)} of {len(expected)} answers.{note}"
            " The session is closed."
        )

    return execute


@tool
def read_session_log(logs: Mapping[int, str]) -> Tool:
    """Creates the dreamer's read-only view of completed session logs.

    Args:
      logs: JSON log text by 1-based session number.

    Returns:
      The tool.
    """

    async def execute(session: int) -> str:
        """Read the log of one completed session.

        The log lists the ticket's questions, every registry call with its
        result, the submitted answers, and the post-session review if one
        is available.

        Args:
          session: Session number, starting at 1.

        Returns:
          The session log as JSON.
        """
        if session not in logs:
            available = ", ".join(str(n) for n in sorted(logs))
            raise ToolError(
                f"No log for session {session}. Available: {available}."
            )
        return logs[session]

    return execute


def _as_text(value: Any) -> str:
    """Renders a submitted JSON value as answer text."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _qid_order(qid: str) -> tuple[int, str]:
    digits = "".join(ch for ch in qid if ch.isdigit())
    return (int(digits) if digits else 0, qid)
