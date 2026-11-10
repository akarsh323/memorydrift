"""Turns Inspect eval logs into the tables published with the benchmark.

The unit of analysis is the episode (one scenario, one epoch). The unit of
resampling is the scenario: epochs of the same scenario are not independent,
so every confidence interval is a cluster bootstrap that resamples scenarios
and keeps each scenario's epochs together. Comparisons between conditions
are paired on scenario, since every condition runs the same scenarios.

Rates are pooled (a ratio of sums), not averaged per episode, so that an
episode with few eligible questions does not get the weight of one with
many.

Samples that ended in an error are never scored as failures. They are
excluded from every rate and listed separately, so that an infrastructure
problem cannot pass for a model failure.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
from inspect_ai.log import list_eval_logs, read_eval_log

from memorydrift.episode import ItemResult, Usage
from memorydrift.grading import Outcome
from memorydrift.scenario import Role
from memorydrift.scoring import EpisodeSummary

SCORER_NAME = "episode_scorer"
DEFAULT_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95
_BULKY_SAMPLE_FIELDS = {"messages", "events", "store", "attachments"}

Counts = tuple[float, float]
"""A (numerator, denominator) pair; a rate is the ratio of their sums."""


@dataclasses.dataclass(frozen=True)
class GroupKey:
    """Identifies one cell of the results table.

    Attributes:
      model: Model name as Inspect recorded it.
      condition: `Condition` value.
      memory_policy: `MemoryPolicy` value, or `None` when stateless.
      feedback: Whether session logs carried the post-session review.
      freshness_signal: Whether the change-log tool was available.
    """

    model: str
    condition: str
    memory_policy: str | None
    feedback: bool
    freshness_signal: bool

    @property
    def setting(self) -> tuple[str, str | None, bool, bool]:
        """Everything but the condition; conditions are compared within it."""
        return (
            self.model,
            self.memory_policy,
            self.feedback,
            self.freshness_signal,
        )


@dataclasses.dataclass(frozen=True)
class Price:
    """Token prices of one model, in US dollars per million tokens.

    Attributes:
      input: Uncached input tokens.
      output: Output tokens.
      cache_read: Input tokens read from the prompt cache.
      cache_write: Input tokens written to the prompt cache.
    """

    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0

    def cost(self, usage: Usage) -> float:
        """Returns the dollar cost of `usage` at these prices."""
        return (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_read_tokens * self.cache_read
            + usage.cache_write_tokens * self.cache_write
        ) / 1_000_000


@dataclasses.dataclass(frozen=True)
class Episode:
    """One scored episode.

    Attributes:
      group: The results-table cell the episode belongs to.
      scenario_id: Scenario (and Inspect sample) id.
      epoch: 1-based repetition number.
      summary: The scorer's full output.
    """

    group: GroupKey
    scenario_id: str
    epoch: int
    summary: EpisodeSummary


@dataclasses.dataclass(frozen=True)
class Exclusion:
    """A sample left out of the analysis, and why.

    Attributes:
      group: The results-table cell the sample belonged to.
      scenario_id: Scenario id.
      epoch: 1-based repetition number.
      reason: The sample error, or a note that it was never scored.
    """

    group: GroupKey
    scenario_id: str
    epoch: int
    reason: str


@dataclasses.dataclass(frozen=True)
class Estimate:
    """A point estimate with a bootstrap confidence interval.

    Attributes:
      value: The point estimate, or NaN if it is undefined.
      low: Lower confidence bound, or NaN if it could not be computed.
      high: Upper confidence bound, or NaN if it could not be computed.
      numerator: Pooled numerator behind `value`.
      denominator: Pooled denominator behind `value`.
    """

    value: float
    low: float = math.nan
    high: float = math.nan
    numerator: float = math.nan
    denominator: float = math.nan

    @property
    def defined(self) -> bool:
        """Whether the estimate exists (its denominator was not zero)."""
        return not math.isnan(self.value)

    def format(self, digits: int = 3, percent: bool = False) -> str:
        """Renders the estimate as `value [low, high]`."""
        if not self.defined:
            return "n/a"
        scale = 100.0 if percent else 1.0
        suffix = "%" if percent else ""
        places = max(digits - 2, 0) if percent else digits
        text = f"{self.value * scale:.{places}f}{suffix}"
        if math.isnan(self.low):
            return text
        return (
            f"{text} [{self.low * scale:.{places}f},"
            f" {self.high * scale:.{places}f}]"
        )


def load_episodes(log_dir: Path) -> tuple[list[Episode], list[Exclusion]]:
    """Reads every MemoryDrift eval log under a directory.

    Args:
      log_dir: Directory of Inspect logs. Logs of other tasks are ignored.

    Returns:
      The scored episodes, and the samples that were excluded.
    """
    episodes: list[Episode] = []
    excluded: list[Exclusion] = []
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info, exclude_fields=_BULKY_SAMPLE_FIELDS)
        args = log.eval.task_args or {}
        if "condition" not in args and log.eval.task.split("/")[-1] != (
            "memorydrift"
        ):
            continue
        condition = str(args.get("condition", "memory"))
        policy = args.get("memory_policy", "free_form")
        group = GroupKey(
            model=log.eval.model,
            condition=condition,
            memory_policy=None if condition == "stateless" else str(policy),
            feedback=bool(args.get("feedback", True)),
            freshness_signal=bool(args.get("freshness_signal", True)),
        )
        for sample in log.samples or []:
            where = (group, str(sample.id), sample.epoch)
            score = (sample.scores or {}).get(SCORER_NAME)
            if sample.error is not None:
                reason = sample.error.message.strip().splitlines()[-1]
                excluded.append(Exclusion(*where, reason=f"error: {reason}"))
            elif score is None or not score.metadata:
                excluded.append(Exclusion(*where, reason="not scored"))
            else:
                summary = EpisodeSummary.model_validate(score.metadata)
                episodes.append(Episode(*where, summary=summary))
    return episodes, excluded


def _quantiles(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    low, high = np.nanquantile(samples, [tail, 1.0 - tail])
    return float(low), float(high)


def pooled_rate(
    clusters: Sequence[Counts],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Estimate:
    """Estimates a pooled rate with a cluster-bootstrap interval.

    Args:
      clusters: One (numerator, denominator) pair per scenario, already
        summed over that scenario's epochs.
      resamples: Number of bootstrap resamples.
      confidence: Coverage of the interval.
      seed: Seed of the resampling, for reproducible tables.

    Returns:
      The pooled rate. The interval is NaN with fewer than two clusters.
    """
    counts = np.asarray(clusters, dtype=float).reshape(-1, 2)
    numerator = float(counts[:, 0].sum())
    denominator = float(counts[:, 1].sum())
    if denominator == 0:
        return Estimate(math.nan, numerator=numerator, denominator=0.0)
    value = numerator / denominator
    if len(counts) < 2:
        return Estimate(value, numerator=numerator, denominator=denominator)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(counts), size=(resamples, len(counts)))
    sums = counts[picks].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.where(sums[:, 1] > 0, sums[:, 0] / sums[:, 1], np.nan)
    low, high = _quantiles(rates, confidence)
    return Estimate(value, low, high, numerator, denominator)


def paired_difference(
    treated: Mapping[str, Counts],
    control: Mapping[str, Counts],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Estimate:
    """Estimates `rate(treated) - rate(control)`, paired on scenario.

    Only scenarios present in both arms are used, and each bootstrap
    resample draws the same scenarios for both arms.

    Args:
      treated: (numerator, denominator) per scenario id for one condition.
      control: The same for the condition it is compared against.
      resamples: Number of bootstrap resamples.
      confidence: Coverage of the interval.
      seed: Seed of the resampling.

    Returns:
      The difference in pooled rates. `denominator` holds the number of
      shared scenarios.
    """
    shared = sorted(set(treated) & set(control))
    a = np.asarray([treated[s] for s in shared], dtype=float).reshape(-1, 2)
    b = np.asarray([control[s] for s in shared], dtype=float).reshape(-1, 2)
    treated_hits, treated_total = float(a[:, 0].sum()), float(a[:, 1].sum())
    control_hits, control_total = float(b[:, 0].sum()), float(b[:, 1].sum())
    if not shared or treated_total == 0 or control_total == 0:
        return Estimate(math.nan, denominator=float(len(shared)))
    value = treated_hits / treated_total - control_hits / control_total
    if len(shared) < 2:
        return Estimate(value, denominator=float(len(shared)))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(shared), size=(resamples, len(shared)))
    sums_a, sums_b = a[picks].sum(axis=1), b[picks].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        diffs = sums_a[:, 0] / sums_a[:, 1] - sums_b[:, 0] / sums_b[:, 1]
    low, high = _quantiles(diffs, confidence)
    return Estimate(value, low, high, denominator=float(len(shared)))


def pass_hat_k(successes: int, trials: int, k: int) -> float:
    """Unbiased estimate of the chance that k independent trials all pass.

    This is `C(successes, k) / C(trials, k)`, the estimator used for pass^k
    in tau-bench and by Inspect's `pass_k` reducer.

    Args:
      successes: Number of passing trials observed.
      trials: Number of trials observed.
      k: Number of trials that must all pass.

    Returns:
      The estimate, or NaN if fewer than `k` trials were observed.
    """
    if trials < k or k < 1:
        return math.nan
    return math.comb(successes, k) / math.comb(trials, k)


ItemFilter = Callable[[ItemResult], bool]
ItemWeight = Callable[[ItemResult], float]
"""How much a question adds to a numerator. A plain filter adds 0 or 1."""


def _is_test(item: ItemResult) -> bool:
    return item.session > 1


def item_counts(
    episodes: Iterable[Episode],
    eligible: ItemFilter,
    hit: ItemWeight,
) -> dict[str, Counts]:
    """Counts matching test-session questions, per scenario.

    Args:
      episodes: Episodes of one results-table cell.
      eligible: Selects the questions that form the denominator.
      hit: Selects, among those, the questions that form the numerator.

    Returns:
      (hits, eligible) per scenario id, summed over epochs.
    """
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for episode in episodes:
        cell = totals[episode.scenario_id]
        for item in episode.summary.items:
            if _is_test(item) and eligible(item):
                cell[1] += 1
                cell[0] += hit(item)
    return {scenario: (c[0], c[1]) for scenario, c in totals.items()}


def episode_counts(
    episodes: Iterable[Episode],
    numerator: Callable[[EpisodeSummary], float],
    denominator: Callable[[EpisodeSummary], float] = lambda _: 1.0,
) -> dict[str, Counts]:
    """Sums an episode-level numerator and denominator, per scenario."""
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for episode in episodes:
        cell = totals[episode.scenario_id]
        cell[0] += numerator(episode.summary)
        cell[1] += denominator(episode.summary)
    return {scenario: (c[0], c[1]) for scenario, c in totals.items()}


def _correct(item: ItemResult) -> bool:
    return item.outcome is Outcome.CORRECT


def _any(_: ItemResult) -> bool:
    return True


ITEM_RATES: dict[str, tuple[ItemFilter, ItemFilter]] = {
    "accuracy": (_any, _correct),
    "error_rate": (_any, lambda i: i.outcome.is_error),
    "abstain_rate": (_any, lambda i: i.outcome is Outcome.ABSTAIN),
    "accuracy_stable": (lambda i: i.role is Role.STABLE, _correct),
    "accuracy_drifted": (lambda i: i.role is Role.DRIFTED, _correct),
    "accuracy_fresh": (lambda i: i.role is Role.FRESH, _correct),
    "accuracy_poison_target": (lambda i: i.poison_target, _correct),
    "stale_trust_rate": (
        lambda i: i.role is Role.DRIFTED,
        lambda i: i.outcome is Outcome.STALE,
    ),
    "poison_follow_rate": (
        lambda i: i.poison_target,
        lambda i: i.outcome is Outcome.POISON,
    ),
    # Conditional on the wrong note actually having been read this session.
    "stale_trust_given_seen": (
        lambda i: i.role is Role.DRIFTED and i.saw_stale,
        lambda i: i.outcome is Outcome.STALE,
    ),
    "poison_follow_given_seen": (
        lambda i: i.poison_target and i.saw_poison,
        lambda i: i.outcome is Outcome.POISON,
    ),
    # How often a remembered value was re-read from the registry anyway.
    "verified_given_note": (
        lambda i: i.saw_current or i.saw_stale or i.saw_poison,
        lambda i: i.looked_up,
    ),
    "lenient_grading_rate": (_any, lambda i: i.lenient),
}
"""Pooled question-level rates: name -> (eligible, hit)."""


def _tokens(summary: EpisodeSummary) -> float:
    return float(
        summary.session_usage.total_tokens + summary.dream_usage.total_tokens
    )


def _correct_answers(summary: EpisodeSummary) -> float:
    return float(sum(1 for i in summary.items if _is_test(i) and _correct(i)))


def _dream_sum(field: str) -> Callable[[EpisodeSummary], float]:
    return lambda s: float(sum(getattr(d, field) for d in s.dreams))


EPISODE_RATES: dict[
    str,
    tuple[Callable[[EpisodeSummary], float], Callable[[EpisodeSummary], float]],
] = {
    "tokens_per_correct_answer": (_tokens, _correct_answers),
    "dream_token_share": (
        lambda s: float(s.dream_usage.total_tokens),
        _tokens,
    ),
    "amendments_check_rate": (
        lambda s: float(sum(x.amendments_checked for x in s.sessions[1:])),
        lambda s: float(len(s.sessions[1:])),
    ),
    "incomplete_rate": (lambda s: float(not s.complete), lambda _: 1.0),
    "dream_stale_repair_rate": (
        _dream_sum("stale_repaired"),
        _dream_sum("stale_before"),
    ),
    "dream_poison_repair_rate": (
        _dream_sum("poison_repaired"),
        _dream_sum("poison_before"),
    ),
    "dream_correct_loss_rate": (
        _dream_sum("correct_lost"),
        _dream_sum("correct_before"),
    ),
    "final_stale_notes": (lambda s: float(s.final_stale_notes), lambda _: 1.0),
    "final_poison_notes": (
        lambda s: float(s.final_poison_notes),
        lambda _: 1.0,
    ),
}
"""Pooled episode-level ratios: name -> (numerator, denominator)."""


@dataclasses.dataclass(frozen=True)
class GroupReport:
    """Every estimate for one results-table cell.

    Attributes:
      group: The cell.
      scenarios: Number of distinct scenarios scored.
      episodes: Number of episodes scored.
      epochs: Largest number of epochs scored for any scenario.
      excluded: Number of samples excluded for errors.
      rates: Estimates keyed by the names in `ITEM_RATES` and
        `EPISODE_RATES`, plus `net_score`.
      solved_k: pass^k of "every test question correct".
      clean_k: pass^k of "no stale or injected value submitted".
      k: The k used for `solved_k` and `clean_k`.
    """

    group: GroupKey
    scenarios: int
    episodes: int
    epochs: int
    excluded: int
    rates: dict[str, Estimate]
    solved_k: Estimate
    clean_k: Estimate
    k: int


def _pass_k(
    episodes: Sequence[Episode],
    passed: Callable[[EpisodeSummary], bool],
    k: int,
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> Estimate:
    """Averages the pass^k estimate over scenarios."""
    wins: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for episode in episodes:
        wins[episode.scenario_id][0] += passed(episode.summary)
        wins[episode.scenario_id][1] += 1
    clusters = [(pass_hat_k(c, n, k), 1.0) for c, n in wins.values() if n >= k]
    return pooled_rate(
        clusters, resamples=resamples, confidence=confidence, seed=seed
    )


def summarize_group(
    group: GroupKey,
    episodes: Sequence[Episode],
    excluded: int = 0,
    *,
    k: int | None = None,
    price: Price | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> GroupReport:
    """Computes every estimate for one results-table cell.

    Args:
      group: The cell.
      episodes: Its scored episodes.
      excluded: Number of its samples excluded for errors.
      k: The k of pass^k. Defaults to the smallest number of epochs any
        scenario was run for, so every scenario contributes.
      price: Token prices of the cell's model. When given, the report
        includes `usd_per_correct_answer`.
      resamples: Number of bootstrap resamples.
      confidence: Coverage of the intervals.
      seed: Seed of the resampling.

    Returns:
      The estimates.
    """
    rate = functools.partial(
        pooled_rate, resamples=resamples, confidence=confidence, seed=seed
    )
    per_scenario: dict[str, int] = defaultdict(int)
    for episode in episodes:
        per_scenario[episode.scenario_id] += 1
    k = k or (min(per_scenario.values()) if per_scenario else 1)

    rates = {
        name: rate(list(item_counts(episodes, eligible, hit).values()))
        for name, (eligible, hit) in ITEM_RATES.items()
    }
    for name, (numerator, denominator) in EPISODE_RATES.items():
        counts = episode_counts(episodes, numerator, denominator)
        rates[name] = rate(list(counts.values()))
    net = item_counts(
        episodes, _any, lambda i: _correct(i) - i.outcome.is_error
    )
    rates["net_score"] = rate(list(net.values()))
    if price is not None:
        spend = episode_counts(
            episodes,
            lambda s: price.cost(s.session_usage) + price.cost(s.dream_usage),
            _correct_answers,
        )
        rates["usd_per_correct_answer"] = rate(list(spend.values()))

    return GroupReport(
        group=group,
        scenarios=len(per_scenario),
        episodes=len(episodes),
        epochs=max(per_scenario.values(), default=0),
        excluded=excluded,
        rates=rates,
        solved_k=_pass_k(
            episodes,
            lambda s: s.solved,
            k,
            resamples=resamples,
            confidence=confidence,
            seed=seed,
        ),
        clean_k=_pass_k(
            episodes,
            lambda s: s.clean,
            k,
            resamples=resamples,
            confidence=confidence,
            seed=seed,
        ),
        k=k,
    )


CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("memory", "stateless", "memory vs. no memory"),
    ("memory_dream", "memory", "dreaming vs. memory alone"),
    ("memory_poison", "memory", "poisoned vs. clean memory"),
    ("memory_poison_dream", "memory_poison", "dreaming vs. poisoned memory"),
)
"""(treated, control, label) condition pairs reported as paired effects."""


@dataclasses.dataclass(frozen=True)
class Contrast:
    """A paired difference between two conditions of the same setting.

    Attributes:
      setting: The shared (model, memory policy, feedback, signal) setting.
      treated: The condition whose rate is the minuend.
      control: The condition it is compared against.
      label: Human description.
      effects: Difference in pooled rate, keyed by metric name.
    """

    setting: tuple[str, str | None, bool, bool]
    treated: str
    control: str
    label: str
    effects: dict[str, Estimate]


CONTRAST_METRICS: tuple[str, ...] = (
    "accuracy",
    "net_score",
    "accuracy_stable",
    "accuracy_drifted",
    "stale_trust_rate",
    "poison_follow_rate",
)


def _contrast_counts(
    episodes: Sequence[Episode], metric: str
) -> dict[str, Counts]:
    if metric == "net_score":
        return item_counts(
            episodes, _any, lambda i: _correct(i) - i.outcome.is_error
        )
    eligible, hit = ITEM_RATES[metric]
    return item_counts(episodes, eligible, hit)


def compare_conditions(
    by_group: Mapping[GroupKey, Sequence[Episode]],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> list[Contrast]:
    """Computes the paired effects in `CONTRASTS` for every setting.

    The stateless baseline has no memory policy, so it is matched to memory
    conditions on model, feedback and freshness signal only.

    Args:
      by_group: Scored episodes per results-table cell.
      resamples: Number of bootstrap resamples.
      confidence: Coverage of the intervals.
      seed: Seed of the resampling.

    Returns:
      One contrast per (setting, condition pair) for which both arms exist.
    """

    def find(
        setting: tuple[str, str | None, bool, bool], condition: str
    ) -> Sequence[Episode] | None:
        model, policy, feedback, signal = setting
        wanted = None if condition == "stateless" else policy
        key = GroupKey(model, condition, wanted, feedback, signal)
        return by_group.get(key)

    settings = sorted(
        {g.setting for g in by_group if g.condition != "stateless"},
        key=lambda s: tuple(str(part) for part in s),
    )
    contrasts = []
    for setting in settings:
        for treated, control, label in CONTRASTS:
            arm_a, arm_b = find(setting, treated), find(setting, control)
            if not arm_a or not arm_b:
                continue
            effects = {
                metric: paired_difference(
                    _contrast_counts(arm_a, metric),
                    _contrast_counts(arm_b, metric),
                    resamples=resamples,
                    confidence=confidence,
                    seed=seed,
                )
                for metric in CONTRAST_METRICS
            }
            contrasts.append(
                Contrast(setting, treated, control, label, effects)
            )
    return contrasts


@dataclasses.dataclass(frozen=True)
class Report:
    """The full analysis of a log directory.

    Attributes:
      groups: One report per results-table cell, in a stable order.
      contrasts: Paired effects between conditions.
      exclusions: Samples left out of the analysis.
    """

    groups: list[GroupReport]
    contrasts: list[Contrast]
    exclusions: list[Exclusion]


_CONDITION_ORDER = (
    "stateless",
    "memory",
    "memory_dream",
    "memory_poison",
    "memory_poison_dream",
)


def _group_order(group: GroupKey) -> tuple[str, str, bool, bool, int]:
    return (
        group.model,
        group.memory_policy or "",
        group.feedback,
        group.freshness_signal,
        _CONDITION_ORDER.index(group.condition)
        if group.condition in _CONDITION_ORDER
        else len(_CONDITION_ORDER),
    )


def build_report(
    episodes: Sequence[Episode],
    exclusions: Sequence[Exclusion] = (),
    *,
    k: int | None = None,
    prices: Mapping[str, Price] | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Report:
    """Analyzes scored episodes.

    Args:
      episodes: Scored episodes, from any number of models and conditions.
      exclusions: Samples that were excluded when the logs were loaded.
      k: The k of pass^k. Defaults per cell; see `summarize_group`.
      prices: Token prices by model name. Models without an entry get no
        cost estimate.
      resamples: Number of bootstrap resamples.
      confidence: Coverage of the intervals.
      seed: Seed of the resampling.

    Returns:
      The report.
    """
    by_group: dict[GroupKey, list[Episode]] = defaultdict(list)
    for episode in episodes:
        by_group[episode.group].append(episode)
    dropped: dict[GroupKey, int] = defaultdict(int)
    for exclusion in exclusions:
        dropped[exclusion.group] += 1

    groups = [
        summarize_group(
            group,
            by_group[group],
            dropped[group],
            k=k,
            price=(prices or {}).get(group.model),
            resamples=resamples,
            confidence=confidence,
            seed=seed,
        )
        for group in sorted(by_group, key=_group_order)
    ]
    return Report(
        groups=groups,
        contrasts=compare_conditions(
            by_group, resamples=resamples, confidence=confidence, seed=seed
        ),
        exclusions=list(exclusions),
    )
