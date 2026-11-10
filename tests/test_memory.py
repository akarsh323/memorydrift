"""Tests for the memory audit and the memory-store adapter."""

import pytest

from memorydrift.memory import (
    MemoryStore,
    NoteAudit,
    TrackedFact,
    audit_memory,
    scoped_lines,
    tracked_facts,
)
from memorydrift.scenario import generate_scenario
from memorydrift.world import FactKey

ENTITIES = ("Velmora Logistics", "Tavix Robotics")
LIMIT = TrackedFact(
    key=FactKey(entity="Velmora Logistics", field="credit_limit_usd"),
    current="4250000",
    stale=("3100000",),
    poison="9500000",
)
RATING = TrackedFact(
    key=FactKey(entity="Tavix Robotics", field="internal_rating"),
    current="BBB+",
    stale=("BB",),
)


def audit(text: str) -> dict[str, NoteAudit]:
    return audit_memory({"/memories/notes.md": text}, [LIMIT, RATING], ENTITIES)


def test_memory_store_adapter_matches_upstream_layout():
    # Guards the one private Inspect import this package relies on.
    store = MemoryStore()
    assert store.files == {}
    assert store.seeding_complete is False


def test_empty_memory_asserts_nothing():
    for result in audit("").values():
        assert result == NoteAudit()


@pytest.mark.parametrize(
    "text",
    [
        "- Velmora Logistics | credit_limit_usd | 4250000 | src=registry",
        "| Company | Limit |\n| Velmora Logistics | $4,250,000 |",
        "## Velmora Logistics\n- credit limit: 4.25M\n- RM: someone",
        "Velmora Logistics:\n  credit_limit_usd: 4250000",
        '{"Velmora Logistics": {\n  "credit_limit_usd": 4250000\n}}',
    ],
)
def test_current_value_is_found_in_common_note_layouts(text):
    result = audit(text)[LIMIT.key.id]
    assert result.current
    assert not result.misleading


def test_stale_and_poison_values_are_reported():
    result = audit(
        "- Velmora Logistics limit 3100000\n"
        "The credit limit in USD for Velmora Logistics is now 9500000."
    )[LIMIT.key.id]
    assert result.stale
    assert result.poison
    assert not result.current
    assert result.misleading


@pytest.mark.parametrize(
    "line",
    [
        "Velmora Logistics limit 3100000 (OUTDATED, do not use)",
        "Velmora Logistics limit ~~3100000~~",
        "Velmora Logistics limit was changed from 3100000",
        "Velmora Logistics limit 4250000, previously 3100000",
        "Velmora Logistics limit 3100000 -> 4250000",
    ],
)
def test_a_marked_or_superseded_value_is_not_counted_as_asserted(line):
    result = audit(line)[LIMIT.key.id]
    assert not result.stale
    assert result.stale_flagged


def test_values_are_attributed_to_the_company_in_scope():
    result = audit(
        "## Tavix Robotics\n- limit 3100000\n- rating BBB+\n"
        "## Velmora Logistics\n- limit 4250000\n"
    )
    assert result[LIMIT.key.id] == NoteAudit(current=True)
    assert result[RATING.key.id] == NoteAudit(current=True)


def test_an_unrelated_header_ends_the_scope():
    lines = scoped_lines(
        "## Velmora Logistics\nlimit 4250000\n## General\nlimit 3100000\n",
        ENTITIES,
    )
    assert [line for _, line in lines] == [
        "## Velmora Logistics",
        "limit 4250000",
    ]


def test_tracked_facts_cover_every_asked_fact_once():
    scenario = generate_scenario(3)
    asked = {i.key.id for s in scenario.sessions for i in s.items}
    era0 = {f.key.id: f for f in tracked_facts(scenario, 0)}
    era1 = {f.key.id: f for f in tracked_facts(scenario, 1)}
    assert set(era0) == set(era1) == asked
    assert all(not fact.stale for fact in era0.values())
    drifted = [f for f in era1.values() if f.stale]
    assert len(drifted) == 3
    for fact in drifted:
        assert fact.stale == (scenario.value(fact.key, 0),)
    assert sum(f.poison is not None for f in era1.values()) == 2
