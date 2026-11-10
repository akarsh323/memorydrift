"""Renders an analysis `Report` as Markdown tables and as JSON."""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from memorydrift.analysis import (
    CONTRAST_METRICS,
    Contrast,
    Estimate,
    GroupKey,
    GroupReport,
    Report,
)
from memorydrift.reference import PROVIDER

_REFERENCE_PREFIX = f"{PROVIDER}/"


def model_label(model: str) -> str:
    """Returns a display name that marks scripted policies as not models."""
    if model.startswith(_REFERENCE_PREFIX):
        name = model.removeprefix(_REFERENCE_PREFIX)
        return f"reference policy: {name} (scripted, not an LLM)"
    return model


def _setting_label(setting: tuple[str, str | None, bool, bool]) -> str:
    model, policy, feedback, signal = setting
    parts = [model_label(model)]
    if policy and policy != "free_form":
        parts.append(f"memory policy `{policy}`")
    if not feedback:
        parts.append("no post-session review")
    if not signal:
        parts.append("no change-log tool")
    return ", ".join(parts)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(" --- " for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _pct(estimate: Estimate) -> str:
    return estimate.format(percent=True)


def _plain(estimate: Estimate, digits: int = 0) -> str:
    if not estimate.defined:
        return "n/a"
    return f"{estimate.value:,.{digits}f}"


def _signed(estimate: Estimate) -> str:
    """Formats a difference in rates as signed percentage points."""
    if not estimate.defined:
        return "n/a"
    text = f"{estimate.value * 100:+.1f}"
    if math.isnan(estimate.low):
        return text
    return f"{text} [{estimate.low * 100:+.1f}, {estimate.high * 100:+.1f}]"


def _headline(groups: Sequence[GroupReport]) -> str:
    k = groups[0].k
    priced = any("usd_per_correct_answer" in g.rates for g in groups)
    header = [
        "Condition",
        "Episodes",
        "Accuracy",
        "Net score",
        "Stale-trust rate",
        "Poison-follow rate",
        f"Solved^{k}",
        f"Clean^{k}",
        "Tokens per correct answer",
    ]
    if priced:
        header.append("USD per correct answer")
    rows = []
    for g in groups:
        row = [
            f"`{g.group.condition}`",
            str(g.episodes),
            _pct(g.rates["accuracy"]),
            g.rates["net_score"].format(digits=3),
            _pct(g.rates["stale_trust_rate"]),
            _pct(g.rates["poison_follow_rate"]),
            _pct(g.solved_k),
            _pct(g.clean_k),
            _plain(g.rates["tokens_per_correct_answer"]),
        ]
        if priced:
            cost = g.rates.get("usd_per_correct_answer")
            row.append(_plain(cost, 4) if cost else "n/a")
        rows.append(row)
    return _table(header, rows)


def _by_role(groups: Sequence[GroupReport]) -> str:
    header = [
        "Condition",
        "Stable (a good note answers it)",
        "Drifted (the note is now wrong)",
        "Fresh (never seen)",
        "Poison-targeted",
    ]
    rows = [
        [
            f"`{g.group.condition}`",
            _pct(g.rates["accuracy_stable"]),
            _pct(g.rates["accuracy_drifted"]),
            _pct(g.rates["accuracy_fresh"]),
            _pct(g.rates["accuracy_poison_target"]),
        ]
        for g in groups
    ]
    return _table(header, rows)


def _behavior(groups: Sequence[GroupReport]) -> str:
    header = [
        "Condition",
        "Sessions that checked the change log",
        "Remembered values re-read from the registry",
        "Stale answer, given a stale note was read",
        "Poison answer, given the poison note was read",
        "Abstained",
    ]
    rows = [
        [
            f"`{g.group.condition}`",
            _pct(g.rates["amendments_check_rate"]),
            _pct(g.rates["verified_given_note"]),
            _pct(g.rates["stale_trust_given_seen"]),
            _pct(g.rates["poison_follow_given_seen"]),
            _pct(g.rates["abstain_rate"]),
        ]
        for g in groups
    ]
    return _table(header, rows)


def _dreaming(groups: Sequence[GroupReport]) -> str | None:
    dreamers = [g for g in groups if g.rates["dream_token_share"].value > 0]
    if not dreamers:
        return None
    header = [
        "Condition",
        "Stale notes repaired",
        "Poison notes repaired",
        "Correct notes lost",
        "Share of tokens spent dreaming",
        "Stale notes left at the end (per episode)",
        "Poison notes left at the end (per episode)",
    ]
    rows = [
        [
            f"`{g.group.condition}`",
            _pct(g.rates["dream_stale_repair_rate"]),
            _pct(g.rates["dream_poison_repair_rate"]),
            _pct(g.rates["dream_correct_loss_rate"]),
            _pct(g.rates["dream_token_share"]),
            _plain(g.rates["final_stale_notes"], 2),
            _plain(g.rates["final_poison_notes"], 2),
        ]
        for g in dreamers
    ]
    return _table(header, rows)


def _effects(contrasts: Sequence[Contrast]) -> str:
    header = [
        "Comparison",
        "Accuracy",
        "Net score",
        "Stale-trust rate",
        "Poison-follow rate",
        "Scenarios",
    ]
    rows = [
        [
            f"{c.label} (`{c.treated}` - `{c.control}`)",
            _signed(c.effects["accuracy"]),
            _signed(c.effects["net_score"]),
            _signed(c.effects["stale_trust_rate"]),
            _signed(c.effects["poison_follow_rate"]),
            f"{c.effects['accuracy'].denominator:.0f}",
        ]
        for c in contrasts
    ]
    return _table(header, rows)


def _health(groups: Sequence[GroupReport]) -> str:
    header = [
        "Condition",
        "Scenarios",
        "Epochs",
        "Episodes scored",
        "Excluded (errors)",
        "Episodes with a session that did not submit",
        "Answers graded leniently",
    ]
    rows = [
        [
            f"`{g.group.condition}`",
            str(g.scenarios),
            str(g.epochs),
            str(g.episodes),
            str(g.excluded),
            _pct(g.rates["incomplete_rate"]),
            _pct(g.rates["lenient_grading_rate"]),
        ]
        for g in groups
    ]
    return _table(header, rows)


def render_markdown(report: Report, *, heading_level: int = 2) -> str:
    """Renders a report as Markdown.

    Args:
      report: The analysis to render.
      heading_level: Markdown level of the per-setting headings.

    Returns:
      Markdown text, one section per (model, policy) setting.
    """
    h = "#" * heading_level
    by_setting: dict[tuple[str, str | None, bool, bool], list[GroupReport]] = {}
    stateless: dict[tuple[str, bool, bool], GroupReport] = {}
    for group in report.groups:
        key = group.group
        if key.condition == "stateless":
            stateless[(key.model, key.feedback, key.freshness_signal)] = group
        else:
            by_setting.setdefault(key.setting, []).append(group)
    # A model that was only run stateless still gets a section.
    for (model, feedback, signal), group in stateless.items():
        if not any(s[0] == model for s in by_setting):
            by_setting[(model, None, feedback, signal)] = [group]

    sections = []
    for setting, groups in by_setting.items():
        model, _, feedback, signal = setting
        baseline = stateless.get((model, feedback, signal))
        if baseline is not None and baseline not in groups:
            groups = [baseline, *groups]
        parts = [f"{h} {_setting_label(setting)}", _headline(groups)]
        parts += [
            f"{h}# Accuracy by what the agent could know",
            _by_role(groups),
        ]
        effects = [c for c in report.contrasts if c.setting == setting]
        if effects:
            parts += [
                f"{h}# Paired effects (percentage points)",
                _effects(effects),
            ]
        parts += [f"{h}# Behavior", _behavior(groups)]
        dreaming = _dreaming(groups)
        if dreaming:
            parts += [f"{h}# What dreaming did to memory", dreaming]
        parts += [f"{h}# Run health", _health(groups)]
        sections.append("\n\n".join(parts))

    if report.exclusions:
        reasons = Counter(e.reason for e in report.exclusions)
        listing = "\n".join(
            f"- {count} x {reason}" for reason, count in reasons.most_common()
        )
        sections.append(
            f"{h} Excluded samples\n\n"
            "These samples ended in an error and are not counted as"
            f" failures.\n\n{listing}"
        )
    return "\n\n".join(sections) + "\n"


def _estimate_json(estimate: Estimate) -> dict[str, float | None]:
    return {
        name: (None if math.isnan(value) else value)
        for name, value in dataclasses.asdict(estimate).items()
    }


def _group_json(key: GroupKey) -> dict[str, Any]:
    return dataclasses.asdict(key)


def report_to_json(report: Report) -> dict[str, Any]:
    """Converts a report to a JSON-serializable dict.

    Args:
      report: The analysis to convert.

    Returns:
      Every estimate with its interval, numerator and denominator.
    """
    return {
        "groups": [
            {
                **_group_json(g.group),
                "scenarios": g.scenarios,
                "episodes": g.episodes,
                "epochs": g.epochs,
                "excluded": g.excluded,
                "k": g.k,
                "solved_k": _estimate_json(g.solved_k),
                "clean_k": _estimate_json(g.clean_k),
                "rates": {n: _estimate_json(e) for n, e in g.rates.items()},
            }
            for g in report.groups
        ],
        "contrasts": [
            {
                "model": c.setting[0],
                "memory_policy": c.setting[1],
                "feedback": c.setting[2],
                "freshness_signal": c.setting[3],
                "treated": c.treated,
                "control": c.control,
                "label": c.label,
                "effects": {
                    metric: _estimate_json(c.effects[metric])
                    for metric in CONTRAST_METRICS
                },
            }
            for c in report.contrasts
        ],
        "exclusions": [
            {
                **_group_json(e.group),
                "scenario_id": e.scenario_id,
                "epoch": e.epoch,
                "reason": e.reason,
            }
            for e in report.exclusions
        ],
    }
