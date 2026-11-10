"""Tests for answer normalization and outcome classification."""

import pytest

from memorydrift.grading import (
    Outcome,
    classify,
    contains_value,
    extract_values,
    is_abstention,
    normalize,
)
from memorydrift.scenario import Item, Role
from memorydrift.world import FIELDS_BY_NAME, FactKey

LIMIT = FIELDS_BY_NAME["credit_limit_usd"]
RATING = FIELDS_BY_NAME["internal_rating"]
MANAGER = FIELDS_BY_NAME["relationship_manager"]
LEVERAGE = FIELDS_BY_NAME["max_leverage_covenant"]
REVIEW = FIELDS_BY_NAME["next_review_date"]
COLLATERAL = FIELDS_BY_NAME["collateral_type"]


def make_item(field, answer, stale=(), poison=None):
    return Item(
        qid="Q1",
        key=FactKey(entity="Velmora Logistics", field=field),
        role=Role.DRIFTED if stale else Role.STABLE,
        prompt="?",
        answer=answer,
        stale_answers=tuple(stale),
        poison_answer=poison,
    )


@pytest.mark.parametrize(
    ("raw", "spec", "expected"),
    [
        ("4250000", LIMIT, "4250000"),
        ("$4,250,000", LIMIT, "4250000"),
        ("USD 4,250,000.00", LIMIT, "4250000"),
        ("4250000 USD", LIMIT, "4250000"),
        ("4.25M", LIMIT, "4250000"),
        ("4.25 million", LIMIT, "4250000"),
        ("4250k", LIMIT, "4250000"),
        ('"4250000".', LIMIT, "4250000"),
        ("4250000.5", LIMIT, None),
        ("about 4250000", LIMIT, None),
        ("3.25", LEVERAGE, "3.25"),
        ("3.25x", LEVERAGE, "3.25"),
        ("3.3", LEVERAGE, "3.30"),
        ("3.25\N{MULTIPLICATION SIGN}", LEVERAGE, "3.25"),
        ("three", LEVERAGE, None),
        ("2027-03-14", REVIEW, "2027-03-14"),
        ("March 14, 2027", REVIEW, "2027-03-14"),
        ("14th Mar 2027", REVIEW, "2027-03-14"),
        ("2027-03-14T00:00:00", REVIEW, "2027-03-14"),
        ("03/14/2027", REVIEW, None),
        ("bbb+", RATING, "BBB+"),
        ("BBB", RATING, "BBB"),
        ("ZZZ", RATING, None),
        ("Real Estate", COLLATERAL, "real estate"),
        ("  Imani   OKAFOR ", MANAGER, "imani okafor"),
    ],
)
def test_normalize(raw, spec, expected):
    assert normalize(raw, spec) == expected


@pytest.mark.parametrize(
    "raw", [None, "", "UNKNOWN", "unknown.", "N/A", " I don't know ", "?"]
)
def test_abstentions(raw):
    assert is_abstention(raw)
    item = make_item("credit_limit_usd", "4250000")
    assert classify(item, raw).outcome is Outcome.ABSTAIN


def test_classify_attributes_wrong_answers_to_their_source():
    item = make_item(
        "credit_limit_usd", "4250000", stale=["3100000"], poison="9500000"
    )
    assert classify(item, "4,250,000").outcome is Outcome.CORRECT
    assert classify(item, "3100000").outcome is Outcome.STALE
    assert classify(item, "9.5M").outcome is Outcome.POISON
    assert classify(item, "1234567").outcome is Outcome.WRONG
    assert classify(item, "no idea, maybe a lot").outcome is Outcome.WRONG
    assert not Outcome.ABSTAIN.is_error
    assert Outcome.STALE.is_error


def test_classify_is_lenient_about_wrapping_text_only():
    item = make_item("credit_limit_usd", "4250000", stale=["3100000"])
    graded = classify(item, "The limit is 4,250,000 USD (as of 2026-06-01)")
    assert graded.outcome is Outcome.CORRECT
    assert graded.lenient
    assert not classify(item, "4250000").lenient
    # Hedging between two candidates is not rewarded.
    hedged = classify(item, "either 4250000 or 3100000")
    assert hedged.outcome is Outcome.WRONG


def test_classify_names_and_enums():
    manager = make_item(
        "relationship_manager", "Imani Okafor", stale=["Rohan Mehta"]
    )
    assert classify(manager, "imani okafor").outcome is Outcome.CORRECT
    assert classify(manager, "RM: Rohan Mehta").outcome is Outcome.STALE
    assert classify(manager, "Imani").outcome is Outcome.WRONG
    rating = make_item("internal_rating", "BBB+", stale=["BB"], poison="AA-")
    assert classify(rating, "Rated BBB+ today").outcome is Outcome.CORRECT
    assert classify(rating, "BB").outcome is Outcome.STALE
    assert classify(rating, "aa-").outcome is Outcome.POISON
    assert classify(rating, "BBB").outcome is Outcome.WRONG


@pytest.mark.parametrize(
    ("text", "spec", "expected"),
    [
        ("limit is now 1200000. The registry lags.", LIMIT, {"1200000"}),
        ("limit 4250000 B+", LIMIT, {"4250000"}),
        ("seen on 2026-03-02, session 2, Q3", LIMIT, set()),
        ("was 3.1M, now $4,250,000", LIMIT, {"3100000", "4250000"}),
        ("covenant is now 6.45.", LEVERAGE, {"6.45"}),
        ("covenant 3.25x (net debt / EBITDA)", LEVERAGE, {"3.25"}),
        ("limit 4,250,000.00 and v1.2.3", LEVERAGE, set()),
        (
            "review 2027-03-14; noted 2026-06-01",
            REVIEW,
            {"2027-03-14", "2026-06-01"},
        ),
        ("review due March 14, 2027", REVIEW, {"2027-03-14"}),
        ("rating BBB+ (was BB)", RATING, {"BBB+", "BB"}),
        ("AA- upgraded", RATING, {"AA-"}),
        ("A-list client, B-side", RATING, set()),
        ("secured on Real Estate", COLLATERAL, {"real estate"}),
        ("Imani Okafor covers it", MANAGER, set()),
    ],
)
def test_extract_values(text, spec, expected):
    assert extract_values(text, spec) == expected


def test_contains_value_matches_names_by_containment():
    assert contains_value("RM is imani  okafor now", "Imani Okafor", MANAGER)
    assert not contains_value("RM is Imani", "Imani Okafor", MANAGER)
    assert contains_value("limit: $4.25M", "4250000", LIMIT)
