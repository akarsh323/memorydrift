"""Tests for the scripted reference policies' pure logic."""

import json

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall

from memorydrift.prompts import render_dream_task, render_ticket
from memorydrift.reference import (
    NOTES_PATH,
    Claim,
    Policy,
    next_action,
    parse_claims,
    render_notes,
)
from memorydrift.scenario import generate_scenario
from memorydrift.world import FactKey

SCENARIO = generate_scenario(0)
SESSION_TOOLS = ["lookup_record", "list_amendments", "memory", "submit_answers"]


def conversation(task: str, *steps: tuple[str, dict, str]):
    """Builds a conversation from (tool, arguments, result) steps."""
    messages = [
        ChatMessageSystem(content="system"),
        ChatMessageUser(content=task),
    ]
    for number, (tool, arguments, result) in enumerate(steps):
        call = ToolCall(id=f"call-{number}", function=tool, arguments=arguments)
        messages.append(ChatMessageAssistant(content="", tool_calls=[call]))
        messages.append(
            ChatMessageTool(content=result, tool_call_id=call.id, function=tool)
        )
    return messages


def test_notes_round_trip():
    claims = [
        Claim(
            FactKey(entity="A Co", field="internal_rating"),
            "BB",
            "2026-01-02",
            "registry",
        ),
        Claim(
            FactKey(entity="B Co", field="credit_limit_usd"),
            "1500000",
            None,
            "memory",
        ),
    ]
    assert parse_claims(render_notes(claims)) == claims


def test_injected_notes_are_read_as_memory_sourced_claims():
    note = SCENARIO.poison_notes[0]
    (claim,) = parse_claims(note.text)
    assert claim.key == note.key
    assert claim.value == note.false_value
    assert claim.source == "memory"
    assert claim.as_of is None


def test_amnesiac_looks_up_in_question_order_and_ignores_memory():
    ticket = render_ticket(SCENARIO, SCENARIO.sessions[1])
    action = next_action(Policy.AMNESIAC, conversation(ticket), SESSION_TOOLS)
    first = SCENARIO.sessions[1].items[0].key
    assert action.tool == "lookup_record"
    assert action.arguments == {"entity": first.entity, "field": first.field}


def test_memory_policies_start_by_reading_memory():
    ticket = render_ticket(SCENARIO, SCENARIO.sessions[1])
    for policy in (Policy.TRUSTING, Policy.CALIBRATED):
        action = next_action(policy, conversation(ticket), SESSION_TOOLS)
        assert action.tool == "memory"
        assert action.arguments == {"command": "view", "path": "/memories"}


def test_calibrated_checks_the_change_log_after_reading_memory():
    ticket = render_ticket(SCENARIO, SCENARIO.sessions[1])
    listing = (
        "memory",
        {"command": "view", "path": "/memories"},
        "Directory /memories is empty.\n",
    )
    messages = conversation(ticket, listing)
    assert (
        next_action(Policy.CALIBRATED, messages, SESSION_TOOLS).tool
        == "list_amendments"
    )
    assert (
        next_action(Policy.TRUSTING, messages, SESSION_TOOLS).tool
        == "lookup_record"
    )


def test_policy_abstains_when_the_budget_is_spent():
    session = SCENARIO.sessions[2]
    ticket = render_ticket(SCENARIO, session)
    steps = [
        (
            "lookup_record",
            {"entity": item.key.entity, "field": item.key.field},
            json.dumps(
                {
                    "entity": item.key.entity,
                    "field": item.key.field,
                    "value": item.answer,
                    "as_of": session.date,
                    "lookups_remaining": 0,
                }
            ),
        )
        for item in session.items[: session.lookup_budget]
    ]
    action = next_action(
        Policy.AMNESIAC, conversation(ticket, *steps), SESSION_TOOLS
    )
    assert action.tool == "submit_answers"
    answers = action.arguments["answers"]
    assert (
        list(answers.values()).count("UNKNOWN")
        == len(session.items) - session.lookup_budget
    )
    for item in session.items[: session.lookup_budget]:
        assert answers[item.qid] == item.answer


def test_dreamer_reads_every_log_first_and_amnesiac_does_nothing():
    task = render_dream_task(2, "2026-06-01")
    tools = ["memory", "read_session_log"]
    action = next_action(Policy.TRUSTING, conversation(task), tools)
    assert (action.tool, action.arguments) == (
        "read_session_log",
        {"session": 1},
    )
    assert next_action(Policy.AMNESIAC, conversation(task), tools).tool is None


def test_notes_path_is_inside_the_memory_directory():
    assert NOTES_PATH.startswith("/memories/")
