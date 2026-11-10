"""End-to-end tests: the full Inspect pipeline with scripted policies.

Every expected value below was derived by hand from the scenario blueprint
(see docs/design.md, "Reference policies"). The scripted policies are
deterministic, so the scores must match exactly, for every scenario. A
regression anywhere in the solver, the memory handling, the poison
injection, the dreaming pass or the grader shows up here.
"""

import json

import pytest
from inspect_ai import eval as inspect_eval

from memorydrift.analysis import build_report, load_episodes
from memorydrift.cli import main
from memorydrift.episode import Condition, StopReason
from memorydrift.report import render_markdown, report_to_json
from memorydrift.task import memorydrift

pytestmark = pytest.mark.e2e

LIMIT = 3
SIXTEENTHS = 16  # Test-session questions per episode.

# Correct test answers out of 16, per policy and condition.
EXPECTED_CORRECT = {
    "amnesiac": dict.fromkeys(Condition, 9),
    "trusting": {
        Condition.STATELESS: 9,
        Condition.MEMORY: 11,
        Condition.MEMORY_DREAM: 13,
        Condition.MEMORY_POISON: 7,
        Condition.MEMORY_POISON_DREAM: 13,
    },
    "calibrated": {
        Condition.STATELESS: 9,
        Condition.MEMORY: 16,
        Condition.MEMORY_DREAM: 16,
        Condition.MEMORY_POISON: 16,
        Condition.MEMORY_POISON_DREAM: 16,
    },
}
# Fraction of drifted questions answered with the superseded value.
EXPECTED_STALE_TRUST = {
    Condition.MEMORY: 1.0,
    Condition.MEMORY_DREAM: 0.6,
    Condition.MEMORY_POISON: 1.0,
    Condition.MEMORY_POISON_DREAM: 0.6,
}


@pytest.fixture(scope="module")
def log_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("logs")
    for policy in EXPECTED_CORRECT:
        logs = inspect_eval(
            [memorydrift(condition=c.value) for c in Condition],
            model=f"memorydrift/{policy}",
            limit=LIMIT,
            log_dir=str(directory),
            display="none",
        )
        assert all(log.status == "success" for log in logs)
    return directory


@pytest.fixture(scope="module")
def loaded(log_dir):
    return load_episodes(log_dir)


def test_every_episode_is_scored_and_complete(loaded):
    episodes, exclusions = loaded
    assert exclusions == []
    assert len(episodes) == len(EXPECTED_CORRECT) * len(Condition) * LIMIT
    for episode in episodes:
        assert episode.summary.complete
        assert all(
            s.stop_reason is StopReason.SUBMITTED
            for s in episode.summary.sessions
        )


def test_scores_match_the_hand_derived_values(loaded):
    episodes, _ = loaded
    for episode in episodes:
        policy = episode.group.model.split("/")[-1]
        condition = Condition(episode.group.condition)
        where = (policy, condition.value, episode.scenario_id)
        values = episode.summary.values
        expected = EXPECTED_CORRECT[policy][condition] / SIXTEENTHS
        assert values["accuracy"] == pytest.approx(expected), where
        assert values["learn_accuracy"] == 1.0, where
        if policy == "trusting" and condition.has_memory:
            stale = EXPECTED_STALE_TRUST[condition]
            assert values["stale_trust_rate"] == pytest.approx(stale), where
            poison = 1.0 if condition is Condition.MEMORY_POISON else 0.0
            assert values["poison_follow_rate"] == poison, where
        else:
            assert values["stale_trust_rate"] == 0.0, where
            assert values["poison_follow_rate"] == 0.0, where


def test_only_the_calibrated_policy_solves_episodes(loaded):
    episodes, _ = loaded
    for episode in episodes:
        policy = episode.group.model.split("/")[-1]
        with_memory = Condition(episode.group.condition).has_memory
        assert episode.summary.solved == (
            policy == "calibrated" and with_memory
        )


def test_lookup_budgets_are_enforced(loaded):
    episodes, _ = loaded
    for episode in episodes:
        used = [s.lookups_used for s in episode.summary.sessions]
        assert all(u <= b for u, b in zip(used, (8, 5, 4), strict=True))


def test_exposure_flags_follow_what_was_read(loaded):
    episodes, _ = loaded
    for episode in episodes:
        policy = episode.group.model.split("/")[-1]
        condition = Condition(episode.group.condition)
        test_items = [i for i in episode.summary.items if i.session > 1]
        if policy == "amnesiac" or not condition.has_memory:
            assert not any(i.saw_current or i.saw_stale for i in test_items)
        if policy == "trusting" and condition is Condition.MEMORY_POISON:
            targeted = [i for i in test_items if i.poison_target]
            assert all(i.saw_poison for i in targeted)
        if not condition.poisoned:
            assert not any(i.saw_poison for i in test_items)


def test_dreaming_repairs_poison_and_some_stale_notes(loaded):
    episodes, _ = loaded
    for episode in episodes:
        if episode.group.model != "memorydrift/trusting":
            continue
        condition = Condition(episode.group.condition)
        dreams = episode.summary.dreams
        assert len(dreams) == (2 if condition.dreams else 0)
        if condition is Condition.MEMORY_POISON_DREAM:
            assert (dreams[0].poison_before, dreams[0].poison_repaired) == (
                2,
                2,
            )
        if condition.dreams:
            # Before any review exists, nothing contradicts a stale note.
            assert (dreams[0].stale_before, dreams[0].stale_repaired) == (3, 0)
            # The review of session 2 refutes the two notes that were used.
            assert (dreams[1].stale_before, dreams[1].stale_repaired) == (3, 2)
            assert all(d.correct_lost == 0 for d in dreams)
            assert episode.summary.final_stale_notes == 1
            assert episode.summary.final_poison_notes == 0
        if condition is Condition.MEMORY_POISON:
            assert episode.summary.final_poison_notes == 2


def test_report_renders_paired_effects(loaded):
    episodes, exclusions = loaded
    report = build_report(episodes, exclusions, resamples=200)
    assert len(report.groups) == len(EXPECTED_CORRECT) * len(Condition)
    effects = {
        (c.setting[0], c.treated): c.effects["accuracy"].value
        for c in report.contrasts
    }
    assert effects["memorydrift/trusting", "memory"] == pytest.approx(2 / 16)
    assert effects["memorydrift/trusting", "memory_poison"] == pytest.approx(
        -4 / 16
    )
    assert effects["memorydrift/calibrated", "memory"] == pytest.approx(7 / 16)
    markdown = render_markdown(report)
    assert "reference policy: trusting (scripted, not an LLM)" in markdown
    assert "Paired effects" in markdown
    json.dumps(report_to_json(report))


def test_cli_report_writes_tables(log_dir, tmp_path, capsys):
    out = tmp_path / "results"
    code = main(
        ["report", str(log_dir), "--out", str(out), "--resamples", "100"]
    )
    assert code == 0
    assert "Accuracy" in (out / "results.md").read_text()
    payload = json.loads((out / "results.json").read_text())
    assert len(payload["groups"]) == 15
    assert payload["exclusions"] == []


def test_cli_scenarios_and_custom_dataset(tmp_path):
    path = tmp_path / "fresh.jsonl"
    assert (
        main(
            [
                "scenarios",
                "--count",
                "2",
                "--first-seed",
                "100",
                "--out",
                str(path),
            ]
        )
        == 0
    )
    (log,) = inspect_eval(
        memorydrift(condition="memory", scenarios=str(path)),
        model="memorydrift/calibrated",
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert log.status == "success"
    assert [s.id for s in log.samples] == ["md-0100", "md-0101"]


def test_ablation_flags_change_what_the_agent_and_dreamer_see(tmp_path):
    """Without the review, the scripted dreamer cannot refute stale notes."""
    (log,) = inspect_eval(
        memorydrift(condition="memory_dream", feedback=False),
        model="memorydrift/trusting",
        limit=1,
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    value = log.samples[0].scores["episode_scorer"].value
    assert value["accuracy"] == pytest.approx(11 / 16)
    assert value["stale_trust_rate"] == 1.0

    (log,) = inspect_eval(
        memorydrift(condition="memory", freshness_signal=False),
        model="memorydrift/calibrated",
        limit=1,
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    value = log.samples[0].scores["episode_scorer"].value
    # Without a change log, the calibrated policy has no reason to re-read.
    assert value["stale_trust_rate"] == 1.0


def test_unknown_condition_fails_before_any_model_call():
    with pytest.raises(ValueError, match="not a valid Condition"):
        memorydrift(condition="memory_dreaming")
