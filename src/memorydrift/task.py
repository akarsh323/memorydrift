"""The MemoryDrift Inspect task.

Run one condition:

    inspect eval memorydrift/memorydrift -T condition=memory_dream --epochs 5

Each sample is a complete three-session episode. The five conditions share
the same scenarios, so results are paired by sample id.
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from memorydrift.episode import Condition, run_episode
from memorydrift.prompts import MemoryPolicy
from memorydrift.scenario import SCHEMA_VERSION, Scenario, load_scenarios
from memorydrift.scoring import episode_scorer

TASK_VERSION = f"{SCHEMA_VERSION}.0.0"
"""Changes whenever scenarios, prompts or scoring change comparably."""


def scenario_sample(scenario: Scenario) -> Sample:
    """Wraps a scenario as an Inspect sample.

    The sample input is only a label. The solver builds every prompt itself,
    and neither the target nor the metadata is ever shown to the model.

    Args:
      scenario: The episode to wrap.

    Returns:
      The sample, with the full scenario in its metadata.
    """
    target = {
        f"session_{spec.index}": {item.qid: item.answer for item in spec.items}
        for spec in scenario.sessions
    }
    return Sample(
        id=scenario.scenario_id,
        input=(
            f"Episode {scenario.scenario_id}: {len(scenario.sessions)}"
            " service-desk sessions against a drifting credit registry."
        ),
        target=json.dumps(target),
        metadata={"scenario": scenario.model_dump(mode="json")},
    )


@task
def memorydrift(
    condition: str = Condition.MEMORY.value,
    memory_policy: str = MemoryPolicy.FREE_FORM.value,
    feedback: bool = True,
    freshness_signal: bool = True,
    scenarios: str | None = None,
    max_turns: int = 40,
    dream_max_turns: int = 30,
) -> Task:
    """Agent memory under staleness and poisoning, with a dreaming pass.

    Args:
      condition: One of `stateless`, `memory`, `memory_dream`,
        `memory_poison`, `memory_poison_dream`.
      memory_policy: What the agent is told to save: `free_form`,
        `failures_only` or `provenance`. Ignored when stateless.
      feedback: Whether session logs carry the post-session review of each
        answer. Only the dreaming pass can read it.
      freshness_signal: Whether the agent gets the free `list_amendments`
        change-log tool.
      scenarios: Path to a scenario JSON Lines file. Defaults to the frozen
        v1 dataset bundled with the package.
      max_turns: Maximum model calls per session.
      dream_max_turns: Maximum model calls per dreaming pass.

    Returns:
      The task.
    """
    # Validate eagerly so that a typo fails before any model call.
    cond = Condition(condition)
    policy = MemoryPolicy(memory_policy)
    data = load_scenarios(Path(scenarios) if scenarios else None)
    return Task(
        dataset=MemoryDataset(
            [scenario_sample(scenario) for scenario in data],
            name="memorydrift-v1"
            if scenarios is None
            else Path(scenarios).stem,
        ),
        solver=run_episode(
            condition=cond.value,
            memory_policy=policy.value,
            feedback=feedback,
            freshness_signal=freshness_signal,
            max_turns=max_turns,
            dream_max_turns=dream_max_turns,
        ),
        scorer=episode_scorer(),
        version=TASK_VERSION,
        metadata={
            "condition": cond.value,
            "memory_policy": policy.value if cond.has_memory else None,
            "feedback": feedback,
            "freshness_signal": freshness_signal,
        },
    )
