"""The synthetic credit registry that MemoryDrift agents work against.

The world is a small table of facts about fictional companies. Every fact is
addressed by a `FactKey` (entity, field) and holds one canonical string value
per *era*. Drift is a change of value between eras.

The world is synthetic on purpose. A model cannot know any value from
pre-training, so a correct answer can only come from a registry lookup in the
current session or from a note written in an earlier one. That isolates the
effect of memory from the effect of prior knowledge.

This module is pure: no I/O, no Inspect imports, and all randomness flows
through an explicit `random.Random`.
"""

from __future__ import annotations

import datetime
import enum
import random
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict


class FieldKind(enum.StrEnum):
    """Value type of a registry field. Drives sampling and grading."""

    INTEGER = "integer"
    DECIMAL = "decimal"
    DATE = "date"
    ENUM = "enum"
    NAME = "name"


class FieldSpec(BaseModel):
    """Static description of one registry field.

    Attributes:
      name: Machine name used in tool calls, for example `credit_limit_usd`.
      label: Human label used in questions and notes.
      kind: Value type.
      answer_format: One-line format hint shown to the agent.
      choices: Allowed values when `kind` is `ENUM`, otherwise empty.
      keywords: Lowercase phrases that identify this field in free text. Used
        by the memory audit and the scripted reference policies.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    label: str
    kind: FieldKind
    answer_format: str
    choices: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()


# Single-letter symbols ("A", "B") are left out on purpose: they would be
# indistinguishable from ordinary words when notes are scanned for values.
RATINGS: tuple[str, ...] = (
    "AA+",
    "AA",
    "AA-",
    "A+",
    "A-",
    "BBB+",
    "BBB",
    "BBB-",
    "BB+",
    "BB",
    "BB-",
    "B+",
    "B-",
    "CCC+",
)

COLLATERAL_TYPES: tuple[str, ...] = (
    "receivables",
    "inventory",
    "real estate",
    "equipment",
    "cash deposit",
    "listed securities",
    "unsecured",
)

FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="credit_limit_usd",
        label="credit limit in USD",
        kind=FieldKind.INTEGER,
        answer_format="whole number of US dollars, digits only (e.g. 4250000)",
        keywords=("credit_limit_usd", "credit limit", "limit"),
    ),
    FieldSpec(
        name="internal_rating",
        label="internal credit rating",
        kind=FieldKind.ENUM,
        answer_format="rating symbol exactly as the registry shows it (BBB+)",
        choices=RATINGS,
        keywords=("internal_rating", "internal credit rating", "rating"),
    ),
    FieldSpec(
        name="relationship_manager",
        label="relationship manager",
        kind=FieldKind.NAME,
        answer_format="full name (e.g. Imani Okafor)",
        keywords=("relationship_manager", "relationship manager"),
    ),
    FieldSpec(
        name="max_leverage_covenant",
        label="maximum leverage covenant (net debt / EBITDA)",
        kind=FieldKind.DECIMAL,
        answer_format="number with two decimals (e.g. 3.25)",
        keywords=(
            "max_leverage_covenant",
            "maximum leverage covenant",
            "leverage covenant",
            "leverage",
            "covenant",
        ),
    ),
    FieldSpec(
        name="next_review_date",
        label="next credit review date",
        kind=FieldKind.DATE,
        answer_format="ISO date (e.g. 2027-03-14)",
        keywords=(
            "next_review_date",
            "next credit review date",
            "next review date",
            "review date",
            "review",
        ),
    ),
    FieldSpec(
        name="collateral_type",
        label="primary collateral type",
        kind=FieldKind.ENUM,
        answer_format="one of: " + ", ".join(COLLATERAL_TYPES),
        choices=COLLATERAL_TYPES,
        keywords=(
            "collateral_type",
            "primary collateral type",
            "collateral type",
            "collateral",
        ),
    ),
)

FIELDS_BY_NAME: dict[str, FieldSpec] = {f.name: f for f in FIELDS}

# Invented company names. Any resemblance to a real company is coincidental.
ENTITY_NAMES: tuple[str, ...] = (
    "Velmora Logistics",
    "Quendry Foods",
    "Tavix Robotics",
    "Ostrava Kiln Works",
    "Halcyon Drayage",
    "Pellam Optics",
    "Sorrelwick Dairy",
    "Juniper Vale Textiles",
    "Brannoch Marine",
    "Cindervale Glass",
    "Marrowby Tooling",
    "Thistledown Pharma",
    "Kestrelby Aviation",
    "Lowmoor Packaging",
    "Aldervane Energy",
    "Wexcombe Timber",
    "Yarrowgate Seeds",
    "Fennimore Cold Chain",
    "Dunmarrow Castings",
    "Ravelstone Quarries",
    "Northfold Bearings",
    "Saltmarsh Fisheries",
    "Umberleigh Paper",
    "Garrowby Cement",
    "Islewood Biologics",
    "Embercroft Plastics",
    "Zennor Circuits",
    "Hollinbeck Brewing",
    "Carrowmere Freight",
    "Winterbourne Tannery",
)

_FIRST_NAMES: tuple[str, ...] = (
    "Imani",
    "Rohan",
    "Mateo",
    "Aiko",
    "Freya",
    "Tunde",
    "Leila",
    "Dmitri",
    "Sana",
    "Caleb",
    "Yara",
    "Bao",
    "Ingrid",
    "Kwame",
    "Noor",
    "Elias",
    "Priya",
    "Santiago",
    "Mei",
    "Oisin",
    "Zofia",
    "Hamza",
    "Thandi",
    "Lucas",
    "Anneke",
    "Farid",
    "Chiara",
    "Joon",
    "Marisol",
    "Tavita",
)

_LAST_NAMES: tuple[str, ...] = (
    "Okafor",
    "Mehta",
    "Alvarez",
    "Tanaka",
    "Lindqvist",
    "Adeyemi",
    "Haddad",
    "Volkov",
    "Qureshi",
    "Whitlock",
    "Mansour",
    "Nguyen",
    "Halvorsen",
    "Boateng",
    "Rahimi",
    "Brandt",
    "Raman",
    "Castillo",
    "Zhou",
    "Gallagher",
    "Nowak",
    "Siddiqui",
    "Dlamini",
    "Moreau",
    "Visser",
    "Karimi",
    "Ricci",
    "Park",
    "Ibarra",
    "Faleolo",
)

_LIMIT_STEP_USD = 50_000
_LIMIT_MIN_STEPS = 20  # 1,000,000
_LIMIT_MAX_STEPS = 400  # 20,000,000
_LEVERAGE_TICK = 0.05
_LEVERAGE_MIN_TICKS = 40  # 2.00
_LEVERAGE_MAX_TICKS = 130  # 6.50
_REVIEW_WINDOW_DAYS = 540


class FactKey(BaseModel):
    """Address of one fact in the registry.

    Attributes:
      entity: Company name, exactly as listed in the registry.
      field: Field machine name, one of `FIELDS_BY_NAME`.
    """

    model_config = ConfigDict(frozen=True)

    entity: str
    field: str

    @property
    def id(self) -> str:
        """Stable string form, usable as a JSON object key."""
        return f"{self.entity}::{self.field}"

    @property
    def spec(self) -> FieldSpec:
        """The static description of this key's field."""
        return FIELDS_BY_NAME[self.field]

    @classmethod
    def parse(cls, key_id: str) -> FactKey:
        """Inverse of `id`.

        Args:
          key_id: A string produced by `FactKey.id`.

        Returns:
          The parsed key.

        Raises:
          ValueError: If `key_id` is not of the form `entity::field`.
        """
        entity, sep, field = key_id.partition("::")
        if not sep or not entity or field not in FIELDS_BY_NAME:
            raise ValueError(f"not a fact key id: {key_id!r}")
        return cls(entity=entity, field=field)


class ValueSampler:
    """Draws canonical field values, never repeating one within a scenario.

    Values are unique across every field, era and poison note of a scenario
    (enum values are instead unique per key, since the enum is small). That
    makes outcome classification unambiguous: a submitted answer that equals a
    superseded value can only be a stale answer, and one that equals an injected
    value can only come from the poison note.
    """

    def __init__(
        self, rng: random.Random, review_anchor: datetime.date
    ) -> None:
        """Creates a sampler.

        Args:
          rng: Source of randomness. The sampler owns no other state that
            affects reproducibility.
          review_anchor: Review dates are drawn strictly after this date, so
            they can never collide with a session or amendment date.
        """
        self._rng = rng
        self._review_anchor = review_anchor
        self._used: set[str] = set()

    def sample(self, key: FactKey, avoid: Sequence[str] = ()) -> str:
        """Draws a fresh value for `key`.

        Args:
          key: The fact to draw a value for.
          avoid: Extra values that must not be returned, typically the earlier
            values of the same fact.

        Returns:
          A canonical value string that this sampler has not returned before
          for a non-enum field, and that is not in `avoid`.

        Raises:
          RuntimeError: If no unused value can be found.
        """
        blocked = set(avoid)
        for _ in range(10_000):
            value = self._draw(key.spec)
            if value in blocked:
                continue
            if key.spec.kind is FieldKind.ENUM:
                return value
            if value not in self._used:
                self._used.add(value)
                return value
        raise RuntimeError(f"could not sample a fresh value for {key.id}")

    def _draw(self, spec: FieldSpec) -> str:
        """Draws one candidate value for a field, possibly a repeat."""
        match spec.kind:
            case FieldKind.INTEGER:
                steps = self._rng.randint(_LIMIT_MIN_STEPS, _LIMIT_MAX_STEPS)
                return str(steps * _LIMIT_STEP_USD)
            case FieldKind.DECIMAL:
                ticks = self._rng.randint(
                    _LEVERAGE_MIN_TICKS, _LEVERAGE_MAX_TICKS
                )
                return f"{ticks * _LEVERAGE_TICK:.2f}"
            case FieldKind.DATE:
                offset = self._rng.randint(1, _REVIEW_WINDOW_DAYS)
                day = self._review_anchor + datetime.timedelta(days=offset)
                return day.isoformat()
            case FieldKind.ENUM:
                return self._rng.choice(spec.choices)
            case FieldKind.NAME:
                first = self._rng.choice(_FIRST_NAMES)
                last = self._rng.choice(_LAST_NAMES)
                return f"{first} {last}"
        raise AssertionError(f"unhandled field kind: {spec.kind}")


def all_keys(entities: Sequence[str]) -> list[FactKey]:
    """Returns every (entity, field) key for `entities`, in a stable order."""
    return [
        FactKey(entity=entity, field=spec.name)
        for entity in entities
        for spec in FIELDS
    ]
