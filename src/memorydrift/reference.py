"""Scripted reference policies, packaged as an Inspect model provider.

These are not language models. They are small deterministic programs that
play the agent (and the dreamer) through the same tools, prompts and scorer
as a real model:

    inspect eval memorydrift/memorydrift --model memorydrift/trusting

They exist for three reasons:

  * to prove the benchmark is solvable within its lookup budgets by a
    legitimate strategy (`calibrated` scores 1.0 in every condition);
  * to pin the harness with exact, analytically derived scores, so that a
    regression in the solver, the memory handling or the grader fails a
    test (see tests/test_end_to_end.py);
  * to give readers of a results table two fixed reference points to
    compare model behavior against.

The policies are allowed to know the benchmark's own text templates. A model
under test is not: it sees only the prompts in `prompts.py`.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import re
from collections.abc import Iterable, Sequence
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageUser,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from memorydrift.episode import strip_line_numbers
from memorydrift.grading import ABSTAIN_ANSWER
from memorydrift.memory import MEMORY_ROOT
from memorydrift.scenario import POISON_CLAIM_PATTERN, parse_question
from memorydrift.transcript import ToolExchange, tool_exchanges
from memorydrift.world import FIELDS, FactKey

PROVIDER = "memorydrift"
NOTES_PATH = f"{MEMORY_ROOT}/registry_notes.md"

_NOTES_HEADER = "# Registry notes"
_SOURCE_REGISTRY = "registry"
_SOURCE_MEMORY = "memory"
_NOTE_LINE = re.compile(
    r"^- (?P<entity>.+?) \| (?P<field>\w+) \| (?P<value>.+?)"
    r" \| as_of=(?P<as_of>\S+) \| src=(?P<source>\w+)$"
)
_TICKET_DATE = re.compile(r"^Date: (\S+)$", re.MULTILINE)
_TICKET_BUDGET = re.compile(
    r"^Registry lookups available in this session: (\d+)$", re.MULTILINE
)
_TICKET_QUESTION = re.compile(r"^(Q\d+)\. (.+)$", re.MULTILINE)
_DREAM_SPAN = re.compile(r"^Sessions? 1(?: to (\d+))? (?:is|are) complete")
_MEMORY_FILE = re.compile(r"^/memories/\S+\.\w+$", re.MULTILINE)
_FIELD_BY_LABEL = {spec.label: spec.name for spec in FIELDS}
_CHARS_PER_TOKEN = 4


class Policy(enum.StrEnum):
    """The scripted strategies."""

    AMNESIAC = "amnesiac"
    """Never reads or writes memory. Spends lookups in question order."""

    TRUSTING = "trusting"
    """Believes the most recent note about a fact and never verifies it."""

    CALIBRATED = "calibrated"
    """Believes only notes it sourced from the registry, and re-reads any
    fact the change log shows was amended after the note was taken."""


@dataclasses.dataclass(frozen=True)
class Claim:
    """A statement about a registry value, found in memory or in a log.

    Attributes:
      key: The fact.
      value: The claimed value.
      as_of: ISO date the value was observed, or `None` if unknown.
      source: `registry` if the value came from a lookup, otherwise `memory`.
    """

    key: FactKey
    value: str
    as_of: str | None
    source: str

    def render(self) -> str:
        """Formats the claim as a line of the policies' notes file."""
        return (
            f"- {self.key.entity} | {self.key.field} | {self.value}"
            f" | as_of={self.as_of or 'unknown'} | src={self.source}"
        )


@dataclasses.dataclass(frozen=True)
class Action:
    """What a policy does next: call one tool, or stop with a message.

    Attributes:
      tool: Tool to call, or `None` to end the turn without a call.
      arguments: Arguments of the call.
      message: Text to reply with when `tool` is `None`.
    """

    tool: str | None = None
    arguments: dict[str, Any] = dataclasses.field(default_factory=dict)
    message: str = ""


def parse_claims(text: str) -> list[Claim]:
    """Extracts every claim from memory text, in reading order.

    Recognizes the policies' own note lines and the claim clause of injected
    notes. An injected clause carries no provenance, so it is read as a
    memory-sourced claim with an unknown date.

    Args:
      text: Contents of one or more memory files.

    Returns:
      The claims, in the order they appear.
    """
    claims = []
    for line in text.splitlines():
        native = _NOTE_LINE.match(line.strip())
        if native:
            as_of = native.group("as_of")
            claims.append(
                Claim(
                    key=FactKey(
                        entity=native.group("entity"),
                        field=native.group("field"),
                    ),
                    value=native.group("value"),
                    as_of=None if as_of == "unknown" else as_of,
                    source=native.group("source"),
                )
            )
            continue
        for found in POISON_CLAIM_PATTERN.finditer(line):
            field = _FIELD_BY_LABEL.get(found.group("label"))
            if field is None:
                continue
            claims.append(
                Claim(
                    key=FactKey(entity=found.group("entity"), field=field),
                    value=found.group("value"),
                    as_of=None,
                    source=_SOURCE_MEMORY,
                )
            )
    return claims


def render_notes(claims: Iterable[Claim]) -> str:
    """Formats claims as the policies' notes file, in a stable order."""
    lines = sorted(claim.render() for claim in claims)
    return "\n".join([_NOTES_HEADER, *lines]) + "\n"


def next_action(
    policy: Policy, messages: Sequence[ChatMessage], tools: Iterable[str]
) -> Action:
    """Decides a policy's next step from the conversation so far.

    The function is pure: the policy keeps no state between calls, and
    rebuilds what it knows from the tool results in `messages`.

    Args:
      policy: The strategy to follow.
      messages: The conversation, starting with the system prompt.
      tools: Names of the tools available in this conversation.

    Returns:
      The next tool call, or a final message.
    """
    available = set(tools)
    task = next(m.text for m in messages if isinstance(m, ChatMessageUser))
    exchanges = tool_exchanges(messages)
    if "read_session_log" in available:
        return _dream_action(policy, task, exchanges)
    return _session_action(policy, task, exchanges, available)


def _memory_views(exchanges: Sequence[ToolExchange]) -> dict[str, str]:
    """Returns the text of each successful memory view, keyed by path."""
    return {
        str(e.arguments.get("path")): strip_line_numbers(e.text)
        for e in exchanges
        if e.function == "memory"
        and e.arguments.get("command") == "view"
        and e.ok
    }


def _read_memory_step(
    exchanges: Sequence[ToolExchange],
) -> tuple[Action | None, dict[str, str]]:
    """Returns the next memory read, or the file contents once all are read."""
    views = _memory_views(exchanges)
    if MEMORY_ROOT not in views:
        return _view(MEMORY_ROOT), {}
    files = _MEMORY_FILE.findall(views[MEMORY_ROOT])
    for path in files:
        if path not in views:
            return _view(path), {}
    return None, {path: views[path] for path in files}


def _view(path: str) -> Action:
    return Action("memory", {"command": "view", "path": path})


def _wrote(exchanges: Sequence[ToolExchange], path: str) -> bool:
    return any(
        e.function == "memory"
        and e.arguments.get("command") == "create"
        and e.arguments.get("path") == path
        for e in exchanges
    )


def _observations(exchanges: Sequence[ToolExchange]) -> dict[str, Claim]:
    """Returns the registry values looked up in this conversation."""
    seen = {}
    for exchange in exchanges:
        if exchange.function != "lookup_record" or not exchange.ok:
            continue
        claim = _claim_from_lookup(json.loads(exchange.text))
        seen[claim.key.id] = claim
    return seen


def _claim_from_lookup(result: dict[str, Any]) -> Claim:
    return Claim(
        key=FactKey(entity=result["entity"], field=result["field"]),
        value=result["value"],
        as_of=result["as_of"],
        source=_SOURCE_REGISTRY,
    )


def _session_action(
    policy: Policy,
    ticket: str,
    exchanges: Sequence[ToolExchange],
    available: set[str],
) -> Action:
    """Plays one step of a working session."""
    date = _TICKET_DATE.search(ticket)
    budget = _TICKET_BUDGET.search(ticket)
    if date is None or budget is None:
        raise ValueError("not a MemoryDrift ticket")
    questions = [
        (qid, parse_question(prompt))
        for qid, prompt in _TICKET_QUESTION.findall(ticket)
    ]

    uses_memory = policy is not Policy.AMNESIAC and "memory" in available
    claims: list[Claim] = []
    if uses_memory:
        read, files = _read_memory_step(exchanges)
        if read is not None:
            return read
        claims = [
            c for path in sorted(files) for c in parse_claims(files[path])
        ]

    amended: dict[str, str] = {}
    if policy is Policy.CALIBRATED and "list_amendments" in available:
        listing = next(
            (e for e in exchanges if e.function == "list_amendments" and e.ok),
            None,
        )
        if listing is None:
            return Action("list_amendments")
        for entry in json.loads(listing.text)["amendments"]:
            changed = FactKey(entity=entry["entity"], field=entry["field"])
            amended[changed.id] = entry["amended_on"]

    beliefs = _beliefs(policy, claims, amended)
    observed = _observations(exchanges)
    lookups_failed = any(
        e.function == "lookup_record" and e.error is not None for e in exchanges
    )
    lookups_left = int(budget.group(1)) - len(
        [e for e in exchanges if e.function == "lookup_record" and e.ok]
    )

    answers: dict[str, str] = {}
    for qid, key in questions:
        if key is None:
            answers[qid] = ABSTAIN_ANSWER
        elif key.id in observed:
            answers[qid] = observed[key.id].value
        elif key.id in beliefs:
            answers[qid] = beliefs[key.id].value
        elif lookups_left > 0 and not lookups_failed:
            return Action(
                "lookup_record", {"entity": key.entity, "field": key.field}
            )
        else:
            answers[qid] = ABSTAIN_ANSWER

    if uses_memory and not _wrote(exchanges, NOTES_PATH):
        kept = {**beliefs, **observed}
        return Action(
            "memory",
            {
                "command": "create",
                "path": NOTES_PATH,
                "file_text": render_notes(kept.values()),
            },
        )
    return Action("submit_answers", {"answers": answers})


def _beliefs(
    policy: Policy, claims: Sequence[Claim], amended: dict[str, str]
) -> dict[str, Claim]:
    """Selects the claims a policy is willing to answer from."""
    if policy is Policy.TRUSTING:
        # The most recent claim wins, whatever its source. An injected note
        # is carried forward as a memory-sourced claim.
        return {claim.key.id: claim for claim in claims}
    if policy is Policy.CALIBRATED:
        trusted: dict[str, Claim] = {}
        for claim in claims:
            if claim.source != _SOURCE_REGISTRY or claim.as_of is None:
                continue
            amended_on = amended.get(claim.key.id)
            if amended_on is not None and amended_on > claim.as_of:
                trusted.pop(claim.key.id, None)
                continue
            trusted[claim.key.id] = claim
        return trusted
    return {}


def _dream_action(
    policy: Policy, task: str, exchanges: Sequence[ToolExchange]
) -> Action:
    """Plays one step of a dreaming pass.

    The scripted dreamer rebuilds the notes file from the session logs
    alone: a fact is kept only if a registry result supports it, and is
    dropped if a later review marked an answer with that value incorrect.
    """
    if policy is Policy.AMNESIAC:
        return Action(message="No changes.")
    span = _DREAM_SPAN.match(task)
    if span is None:
        raise ValueError("not a MemoryDrift dreaming task")
    completed = int(span.group(1) or 1)

    logs: dict[int, dict[str, Any]] = {
        int(e.arguments["session"]): json.loads(e.text)
        for e in exchanges
        if e.function == "read_session_log" and e.ok
    }
    for session in range(1, completed + 1):
        if session not in logs:
            return Action("read_session_log", {"session": session})

    read, files = _read_memory_step(exchanges)
    if read is not None:
        return read
    if not _wrote(exchanges, NOTES_PATH):
        return Action(
            "memory",
            {
                "command": "create",
                "path": NOTES_PATH,
                "file_text": render_notes(_supported_claims(logs)),
            },
        )
    deleted = {
        e.arguments.get("path")
        for e in exchanges
        if e.function == "memory" and e.arguments.get("command") == "delete"
    }
    for path in sorted(files):
        if path != NOTES_PATH and path not in deleted:
            return Action("memory", {"command": "delete", "path": path})
    return Action(
        message="Rebuilt the notes from registry results in the logs."
    )


def _supported_claims(logs: dict[int, dict[str, Any]]) -> list[Claim]:
    """Returns the latest registry-backed value of each fact in the logs.

    A value is dropped if an answer that used it was marked incorrect in the
    same or a later session than the lookup that produced it.
    """
    latest: dict[str, tuple[int, Claim]] = {}
    refuted: dict[str, tuple[int, str]] = {}
    for session in sorted(logs):
        log = logs[session]
        for event in log["events"]:
            if event["tool"] == "lookup_record" and "result" in event:
                claim = _claim_from_lookup(event["result"])
                latest[claim.key.id] = (session, claim)
        answers = log.get("submitted_answers") or {}
        for qid, verdict in (log.get("review") or {}).items():
            key = parse_question(log["questions"][qid])
            if verdict == "incorrect" and key is not None and qid in answers:
                refuted[key.id] = (session, answers[qid])
    kept = []
    for key_id, (session, claim) in latest.items():
        rejected = refuted.get(key_id)
        if rejected and rejected[0] >= session and rejected[1] == claim.value:
            continue
        kept.append(claim)
    return kept


@modelapi(name=PROVIDER)
class ReferencePolicyAPI(ModelAPI):
    """Inspect model provider that serves the scripted policies.

    The model name selects the policy: `memorydrift/amnesiac`,
    `memorydrift/trusting` or `memorydrift/calibrated`.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),  # noqa: B008
        **model_args: Any,
    ) -> None:
        """Creates the provider.

        Args:
          model_name: A `Policy` value.
          base_url: Unused; accepted for interface compatibility.
          api_key: Unused; accepted for interface compatibility.
          config: Generation config; unused by the scripted policies.
          **model_args: Unused; accepted for interface compatibility.

        Raises:
          ValueError: If `model_name` is not a known policy.
        """
        super().__init__(model_name, base_url, api_key, [], config)
        self._policy = Policy(model_name.split("/")[-1])

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Returns the policy's next step as a model output."""
        action = next_action(self._policy, input, [t.name for t in tools])
        model = f"{PROVIDER}/{self._policy.value}"
        if action.tool is None:
            output = ModelOutput.from_content(
                model=model, content=action.message
            )
            produced = action.message
        else:
            output = ModelOutput.for_tool_call(
                model=model,
                tool_name=action.tool,
                tool_arguments=action.arguments,
            )
            produced = action.tool + json.dumps(action.arguments)
        # A character-based estimate, so that token accounting is exercised
        # end to end. These figures are not comparable to a real tokenizer.
        consumed = sum(len(message.text) for message in input)
        input_tokens = consumed // _CHARS_PER_TOKEN
        output_tokens = max(1, len(produced) // _CHARS_PER_TOKEN)
        output.usage = ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
        return output
