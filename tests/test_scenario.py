"""Tests for the scenario blueprint and the frozen dataset."""

import collections

import pytest

from memorydrift.scenario import (
    POISON_CLAIM_PATTERN,
    QUESTION_TEMPLATES,
    Role,
    generate_scenario,
    generate_scenarios,
    load_scenarios,
    parse_question,
)
from memorydrift.world import FIELDS_BY_NAME, FactKey

SEEDS = range(40)


@pytest.fixture(scope="module")
def scenarios():
    return generate_scenarios(SEEDS)


def test_generation_is_deterministic():
    assert generate_scenario(7) == generate_scenario(7)
    assert generate_scenario(7) != generate_scenario(8)


def test_frozen_dataset_matches_the_generator(scenarios):
    # If this fails on a new Python version, the frozen file stays the source
    # of truth for published results; bump SCHEMA_VERSION before regenerating.
    assert load_scenarios() == scenarios


def test_blueprint_shape(scenarios):
    for scenario in scenarios:
        first, second, third = scenario.sessions
        assert [s.era for s in scenario.sessions] == [0, 1, 1]
        assert [s.lookup_budget for s in scenario.sessions] == [8, 5, 4]
        assert [s.date for s in scenario.sessions] == sorted(
            s.date for s in scenario.sessions
        )
        assert collections.Counter(i.role for i in first.items) == {
            Role.FRESH: 8
        }
        assert collections.Counter(i.role for i in second.items) == {
            Role.STABLE: 3,
            Role.DRIFTED: 2,
            Role.FRESH: 3,
        }
        assert collections.Counter(i.role for i in third.items) == {
            Role.STABLE: 3,
            Role.DRIFTED: 3,
            Role.FRESH: 2,
        }
        assert sum(i.poison_answer is not None for i in second.items) == 2
        assert sum(i.poison_answer is not None for i in third.items) == 2


def test_a_legitimate_strategy_fits_every_budget(scenarios):
    """Fresh plus drifted questions never exceed the lookups available."""
    for scenario in scenarios:
        second, third = scenario.test_sessions
        needed = sum(i.role is not Role.STABLE for i in second.items)
        assert needed <= second.lookup_budget
        # In session 3, repeated drifted facts were re-read in session 2.
        needed = sum(
            i.role is Role.FRESH or (i.role is Role.DRIFTED and not i.repeat)
            for i in third.items
        )
        assert needed <= third.lookup_budget


def test_roles_agree_with_registry_values(scenarios):
    for scenario in scenarios:
        first_seen: dict[str, str] = {}
        for session in scenario.sessions:
            for item in session.items:
                assert item.answer == scenario.value(item.key, session.era)
                seen = first_seen.setdefault(item.key.id, item.answer)
                if item.role is Role.FRESH:
                    assert seen == item.answer
                    assert item.stale_answers == ()
                elif item.role is Role.STABLE:
                    assert seen == item.answer
                else:
                    assert item.stale_answers
                    assert item.answer not in item.stale_answers


def test_drifted_facts_are_listed_as_amendments(scenarios):
    for scenario in scenarios:
        amended = {a.key.id for a in scenario.amendments}
        changed = {k for k, (old, new) in scenario.values.items() if old != new}
        assert amended == changed
        assert scenario.amendments_through(0) == ()
        assert len(scenario.amendments_through(1)) == 5
        for amendment in scenario.amendments:
            assert scenario.sessions[0].date < amendment.amended_on
            assert amendment.amended_on < scenario.sessions[1].date


def test_values_are_unambiguous(scenarios):
    """A non-enum value identifies exactly one fact, era or poison note."""
    for scenario in scenarios:
        seen: dict[str, str] = {}
        claims = [
            (key_id, value)
            for key_id, eras in scenario.values.items()
            for value in dict.fromkeys(eras)
        ]
        claims += [(n.key.id, n.false_value) for n in scenario.poison_notes]
        for key_id, value in claims:
            if FactKey.parse(key_id).spec.choices:
                continue
            assert seen.setdefault(value, key_id) == key_id, value


def test_poison_values_are_never_true(scenarios):
    for scenario in scenarios:
        assert len(scenario.poison_notes) == 2
        for note in scenario.poison_notes:
            assert note.false_value not in scenario.values[note.key.id]
            claim = POISON_CLAIM_PATTERN.search(note.text)
            assert claim is not None
            assert claim.group("entity") == note.key.entity
            assert claim.group("label") == note.key.spec.label
            assert claim.group("value") == note.false_value


def test_review_dates_never_collide_with_session_dates(scenarios):
    for scenario in scenarios:
        last = scenario.sessions[-1].date
        for key_id, eras in scenario.values.items():
            if key_id.endswith("::next_review_date"):
                assert all(value > last for value in eras)


def test_every_question_parses_back_to_its_key(scenarios):
    for scenario in scenarios:
        for session in scenario.sessions:
            for item in session.items:
                assert parse_question(item.prompt) == item.key


def test_question_templates_are_unambiguous():
    assert set(QUESTION_TEMPLATES) == set(FIELDS_BY_NAME)
    for field, templates in QUESTION_TEMPLATES.items():
        for template in templates:
            assert template.count("{entity}") == 1
            parsed = parse_question(template.format(entity="Acme Test Co"))
            assert parsed == FactKey(entity="Acme Test Co", field=field)
    assert parse_question("What is the weather in Paris?") is None


def test_fact_key_round_trip():
    key = FactKey(entity="Velmora Logistics", field="internal_rating")
    assert FactKey.parse(key.id) == key
    with pytest.raises(ValueError, match="not a fact key id"):
        FactKey.parse("Velmora Logistics")


def test_entity_lookup_is_case_insensitive(scenarios):
    scenario = scenarios[0]
    name = scenario.entities[0]
    assert scenario.find_entity(f"  {name.upper()} ") == name
    assert scenario.find_entity("No Such Company") is None
