"""The episode solver: sessions, drift, poisoning and dreaming.

One Inspect sample is one episode. The solver runs the scenario's sessions in
order. Each session starts from an empty conversation, so the only thing that
can carry knowledge forward is the memory directory in the sample store.
Between sessions the solver applies the condition's manipulations:

  * poison conditions append false notes to memory after the first session;
  * dream conditions run an offline consolidation pass after every session
    but the last, with access to the session logs and the memory directory
    but not to the registry.

Everything the scorer needs is recorded as it happens, in `EpisodeLog`.
"""

from __future__ import annotations

import enum
import json
import re
from collections.abc import Sequence
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    Model,
    ModelOutput,
    ModelUsage,
    execute_tools,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, memory
from inspect_ai.util import StoreModel, span, store_as
from pydantic import BaseModel, Field

from memorydrift.grading import Outcome, classify
from memorydrift.memory import (
    NoteAudit,
    audit_memory,
    inject_notes,
    read_memory,
    tracked_facts,
)
from memorydrift.prompts import (
    DREAM_SYSTEM,
    NUDGE,
    MemoryPolicy,
    render_dream_task,
    render_ticket,
    session_system_prompt,
)
from memorydrift.scenario import Role, Scenario, SessionSpec
from memorydrift.tools import (
    SessionRuntime,
    list_amendments,
    lookup_record,
    read_session_log,
    submit_answers,
)
from memorydrift.transcript import tool_exchanges

DREAMER_ROLE = "dreamer"
"""Inspect model role for the dreaming pass. Defaults to the evaluated model."""

_MAX_NUDGES = 2
_MEMORY_TOOL = "memory"
_MEMORY_WRITE_COMMANDS = frozenset(
    {"create", "str_replace", "insert", "delete", "rename"}
)
_MEMORY_LOG_ARGUMENTS = ("command", "path", "old_path", "new_path")
_CAT_LINE_NUMBER = re.compile(r"^\s*\d+\t", re.MULTILINE)


def strip_line_numbers(text: str) -> str:
    """Removes the `cat -n` line numbers the memory tool adds to file views."""
    return _CAT_LINE_NUMBER.sub("", text)


class Condition(enum.StrEnum):
    """Experimental condition of an episode."""

    STATELESS = "stateless"
    """No memory tool. The baseline."""

    MEMORY = "memory"
    """Memory directory persists across sessions."""

    MEMORY_DREAM = "memory_dream"
    """Memory, plus a dreaming pass between sessions."""

    MEMORY_POISON = "memory_poison"
    """Memory, plus false notes injected after the first session."""

    MEMORY_POISON_DREAM = "memory_poison_dream"
    """Memory, injected false notes, and dreaming."""

    @property
    def has_memory(self) -> bool:
        """Whether the agent is given the memory tool."""
        return self is not Condition.STATELESS

    @property
    def dreams(self) -> bool:
        """Whether a dreaming pass runs between sessions."""
        return self in (Condition.MEMORY_DREAM, Condition.MEMORY_POISON_DREAM)

    @property
    def poisoned(self) -> bool:
        """Whether false notes are injected into memory."""
        return self in (Condition.MEMORY_POISON, Condition.MEMORY_POISON_DREAM)


class StopReason(enum.StrEnum):
    """Why a session or dreaming pass ended."""

    SUBMITTED = "submitted"
    FINISHED = "finished"
    NO_TOOL_CALL = "no_tool_call"
    TURN_LIMIT = "turn_limit"
    NOT_RUN = "not_run"


class Usage(BaseModel):
    """Token usage accumulated over a sequence of model calls.

    Attributes:
      calls: Number of model calls.
      input_tokens: Uncached input tokens, as reported by the provider.
      output_tokens: Output tokens, including reasoning tokens.
      cache_read_tokens: Input tokens served from the prompt cache.
      cache_write_tokens: Input tokens written to the prompt cache.
      reasoning_tokens: Reasoning tokens, where the provider reports them.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, usage: ModelUsage | None) -> None:
        """Adds the usage of one model call."""
        self.calls += 1
        if usage is None:
            return
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.input_tokens_cache_read or 0
        self.cache_write_tokens += usage.input_tokens_cache_write or 0
        self.reasoning_tokens += usage.reasoning_tokens or 0

    @property
    def total_tokens(self) -> int:
        """All tokens processed, cached or not."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


class ItemResult(BaseModel):
    """The graded result of one question, with what the agent saw and did.

    Attributes:
      session: 1-based session index.
      qid: Question id within the session.
      key_id: `FactKey.id` of the fact asked about.
      role: `Role` value of the item.
      repeat: Whether the fact was asked in an earlier same-era session.
      poison_target: Whether the scenario's poison plan targets this fact.
      answer: The submitted answer, or `None` if none was submitted.
      outcome: `Outcome` value.
      lenient: Whether grading had to extract the value from extra text.
      looked_up: Whether the agent looked the fact up in this session.
      saw_current: A note it read this session stated the current value.
      saw_stale: A note it read this session asserted a superseded value.
      saw_poison: A note it read this session asserted the injected value.
    """

    session: int
    qid: str
    key_id: str
    role: Role
    repeat: bool
    poison_target: bool
    answer: str | None
    outcome: Outcome
    lenient: bool = False
    looked_up: bool = False
    saw_current: bool = False
    saw_stale: bool = False
    saw_poison: bool = False


class SessionRecord(BaseModel):
    """Everything recorded about one session.

    Attributes:
      index: 1-based session index.
      era: Registry era the session ran in.
      stop_reason: Why the session ended.
      turns: Number of model calls.
      usage: Token usage of the session.
      lookups_used: Metered lookups consumed.
      amendments_checked: Whether `list_amendments` was called.
      memory_reads: Number of memory `view` calls.
      memory_writes: Number of memory calls that modify the directory.
      items: Graded results, one per question.
      memory_after: Snapshot of the memory directory when the session ended.
      log: The session log text made available to the dreamer.
    """

    index: int
    era: int
    stop_reason: StopReason
    turns: int = 0
    usage: Usage = Field(default_factory=Usage)
    lookups_used: int = 0
    amendments_checked: bool = False
    memory_reads: int = 0
    memory_writes: int = 0
    items: list[ItemResult] = Field(default_factory=list)
    memory_after: dict[str, str] = Field(default_factory=dict)
    log: str = ""


class DreamRecord(BaseModel):
    """Everything recorded about one dreaming pass.

    Attributes:
      after_session: Index of the session the pass ran after.
      stop_reason: Why the pass ended.
      turns: Number of model calls.
      usage: Token usage of the pass.
      memory_before: Memory snapshot before the pass (after any injection).
      memory_after: Memory snapshot after the pass.
    """

    after_session: int
    stop_reason: StopReason
    turns: int = 0
    usage: Usage = Field(default_factory=Usage)
    memory_before: dict[str, str] = Field(default_factory=dict)
    memory_after: dict[str, str] = Field(default_factory=dict)


class EpisodeLog(StoreModel):
    """Per-sample record of an episode, kept in the Inspect sample store.

    Records are stored as plain JSON-compatible dicts so that they survive a
    round trip through the eval log; use `session_records` and
    `dream_records` to read them back as models.

    Attributes:
      sessions: Dumped `SessionRecord`s, in order.
      dreams: Dumped `DreamRecord`s, in order.
      poison_path: Memory file the false notes were written to, if any.
    """

    sessions: list[dict[str, Any]] = Field(default_factory=list)
    dreams: list[dict[str, Any]] = Field(default_factory=list)
    poison_path: str | None = None

    def session_records(self) -> list[SessionRecord]:
        """Returns the recorded sessions as models."""
        return [SessionRecord.model_validate(s) for s in self.sessions]

    def dream_records(self) -> list[DreamRecord]:
        """Returns the recorded dreaming passes as models."""
        return [DreamRecord.model_validate(d) for d in self.dreams]


@solver
def run_episode(
    condition: str = Condition.MEMORY.value,
    memory_policy: str = MemoryPolicy.FREE_FORM.value,
    feedback: bool = True,
    freshness_signal: bool = True,
    max_turns: int = 40,
    dream_max_turns: int = 30,
) -> Solver:
    """Runs a scenario's sessions under one experimental condition.

    Args:
      condition: A `Condition` value.
      memory_policy: A `MemoryPolicy` value. Ignored when stateless.
      feedback: Whether session logs include the post-session review that
        marks each answer as confirmed or incorrect. The review is visible
        only to the dreamer, never to a working session.
      freshness_signal: Whether the agent gets the free `list_amendments`
        tool.
      max_turns: Maximum model calls per session.
      dream_max_turns: Maximum model calls per dreaming pass.

    Returns:
      The solver.
    """
    cond = Condition(condition)
    policy = MemoryPolicy(memory_policy) if cond.has_memory else None

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        scenario = Scenario.model_validate(state.metadata["scenario"])
        agent = get_model()
        dreamer = get_model(role=DREAMER_ROLE, default=agent)
        episode = store_as(EpisodeLog)
        transcript: list[ChatMessage] = []
        logs: dict[int, str] = {}
        final_index = scenario.sessions[-1].index

        for spec in scenario.sessions:
            async with span(f"session-{spec.index}"):
                record, messages = await _run_session(
                    agent,
                    scenario,
                    spec,
                    policy=policy,
                    feedback=feedback,
                    freshness_signal=freshness_signal,
                    max_turns=max_turns,
                )
            transcript.extend(messages)
            logs[spec.index] = record.log
            episode.sessions.append(record.model_dump(mode="json"))

            if cond.poisoned and spec.index == scenario.poison_after_session:
                episode.poison_path = inject_notes(scenario.poison_notes)

            if cond.dreams and spec.index < final_index:
                async with span(f"dream-{spec.index}"):
                    dream, messages = await _run_dream(
                        dreamer,
                        logs,
                        after_session=spec.index,
                        date=spec.date,
                        max_turns=dream_max_turns,
                    )
                transcript.extend(messages)
                episode.dreams.append(dream.model_dump(mode="json"))

        state.messages = transcript
        state.output = ModelOutput.from_content(
            model=str(agent),
            content=json.dumps(
                {
                    f"session_{record['index']}": {
                        item["qid"]: item["answer"] for item in record["items"]
                    }
                    for record in episode.sessions
                }
            ),
        )
        return state

    return solve


async def _run_session(
    model: Model,
    scenario: Scenario,
    spec: SessionSpec,
    *,
    policy: MemoryPolicy | None,
    feedback: bool,
    freshness_signal: bool,
    max_turns: int,
) -> tuple[SessionRecord, list[ChatMessage]]:
    """Runs one session from an empty conversation and grades it."""
    runtime = SessionRuntime(scenario=scenario, spec=spec)
    tools: list[Tool] = [lookup_record(runtime)]
    if freshness_signal:
        tools.append(list_amendments(runtime))
    if policy is not None:
        tools.append(memory())
    tools.append(submit_answers(runtime))

    messages: list[ChatMessage] = [
        ChatMessageSystem(content=session_system_prompt(policy)),
        ChatMessageUser(content=render_ticket(scenario, spec)),
    ]
    usage = Usage()
    stop = StopReason.TURN_LIMIT
    nudges = 0
    while usage.calls < max_turns:
        output = await model.generate(messages, tools)
        usage.add(output.usage)
        messages.append(output.message)
        if output.message.tool_calls:
            result = await execute_tools(messages, tools)
            messages.extend(result.messages)
            if runtime.submitted:
                stop = StopReason.SUBMITTED
                break
        elif nudges < _MAX_NUDGES:
            nudges += 1
            messages.append(ChatMessageUser(content=NUDGE))
        else:
            stop = StopReason.NO_TOOL_CALL
            break

    items = _grade_session(scenario, spec, runtime, messages)
    memory_calls = _memory_commands(messages)
    record = SessionRecord(
        index=spec.index,
        era=spec.era,
        stop_reason=stop,
        turns=usage.calls,
        usage=usage,
        lookups_used=len(runtime.lookups),
        amendments_checked=runtime.amendments_checked,
        memory_reads=sum(1 for c in memory_calls if c == "view"),
        memory_writes=sum(
            1 for c in memory_calls if c in _MEMORY_WRITE_COMMANDS
        ),
        items=items,
        memory_after=read_memory(),
        log=_session_log(spec, messages, runtime, items if feedback else None),
    )
    return record, messages


async def _run_dream(
    model: Model,
    logs: dict[int, str],
    *,
    after_session: int,
    date: str,
    max_turns: int,
) -> tuple[DreamRecord, list[ChatMessage]]:
    """Runs one offline consolidation pass over the logs and memory."""
    before = read_memory()
    tools: list[Tool] = [memory(), read_session_log(dict(logs))]
    messages: list[ChatMessage] = [
        ChatMessageSystem(content=DREAM_SYSTEM),
        ChatMessageUser(content=render_dream_task(after_session, date)),
    ]
    usage = Usage()
    stop = StopReason.TURN_LIMIT
    while usage.calls < max_turns:
        output = await model.generate(messages, tools)
        usage.add(output.usage)
        messages.append(output.message)
        if not output.message.tool_calls:
            stop = StopReason.FINISHED
            break
        result = await execute_tools(messages, tools)
        messages.extend(result.messages)

    record = DreamRecord(
        after_session=after_session,
        stop_reason=stop,
        turns=usage.calls,
        usage=usage,
        memory_before=before,
        memory_after=read_memory(),
    )
    return record, messages


def _grade_session(
    scenario: Scenario,
    spec: SessionSpec,
    runtime: SessionRuntime,
    messages: Sequence[ChatMessage],
) -> list[ItemResult]:
    """Grades each question and records what the agent read and looked up."""
    answers = runtime.answers or {}
    looked_up = {key.id for key in runtime.lookups}
    exposure = audit_memory(
        _memory_text_read(messages),
        tracked_facts(scenario, spec.era),
        scenario.entities,
    )
    results = []
    for item in spec.items:
        answer = answers.get(item.qid)
        graded = classify(item, answer)
        seen = exposure.get(item.key.id, NoteAudit())
        results.append(
            ItemResult(
                session=spec.index,
                qid=item.qid,
                key_id=item.key.id,
                role=item.role,
                repeat=item.repeat,
                poison_target=item.poison_answer is not None,
                answer=answer,
                outcome=graded.outcome,
                lenient=graded.lenient,
                looked_up=item.key.id in looked_up,
                saw_current=seen.current,
                saw_stale=seen.stale,
                saw_poison=seen.poison,
            )
        )
    return results


def _memory_commands(messages: Sequence[ChatMessage]) -> list[str]:
    """Returns the command of every memory tool call, in order."""
    return [
        str(exchange.arguments.get("command", ""))
        for exchange in tool_exchanges(messages)
        if exchange.function == _MEMORY_TOOL
    ]


def _memory_text_read(messages: Sequence[ChatMessage]) -> dict[str, str]:
    """Collects the memory content that entered the agent's context.

    Returns:
      Text of each successful memory tool result, with `cat -n` line
      numbers removed, keyed by call position and path.
    """
    read = {}
    for position, exchange in enumerate(tool_exchanges(messages)):
        if exchange.function != _MEMORY_TOOL or not exchange.ok:
            continue
        path = exchange.arguments.get("path") or ""
        read[f"{position:04d}:{path}"] = strip_line_numbers(exchange.text)
    return read


def _session_log(
    spec: SessionSpec,
    messages: Sequence[ChatMessage],
    runtime: SessionRuntime,
    reviewed: Sequence[ItemResult] | None,
) -> str:
    """Renders the session as the JSON event log the dreamer can read.

    Memory tool calls are listed without file contents; the dreamer reads
    memory itself. The review marks answers as confirmed or incorrect without
    revealing the correct value.
    """
    events: list[dict[str, Any]] = []
    for exchange in tool_exchanges(messages):
        event: dict[str, Any] = {"tool": exchange.function}
        if exchange.function == _MEMORY_TOOL:
            event["arguments"] = {
                name: exchange.arguments[name]
                for name in _MEMORY_LOG_ARGUMENTS
                if exchange.arguments.get(name) is not None
            }
        else:
            event["arguments"] = exchange.arguments
            if exchange.ok:
                event["result"] = _parse_json(exchange.text)
        if exchange.error is not None:
            event["error"] = exchange.error
        events.append(event)

    payload: dict[str, Any] = {
        "session": spec.index,
        "date": spec.date,
        "lookup_budget": spec.lookup_budget,
        "questions": {item.qid: item.prompt for item in spec.items},
        "events": events,
        "submitted_answers": runtime.answers,
    }
    if reviewed is not None:
        payload["review"] = {
            item.qid: _review_label(item.outcome) for item in reviewed
        }
    return json.dumps(payload, indent=1, ensure_ascii=False)


def _review_label(outcome: Outcome) -> str:
    if outcome is Outcome.CORRECT:
        return "confirmed"
    if outcome is Outcome.ABSTAIN:
        return "unanswered"
    return "incorrect"


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
