"""Inspect registry hook.

Listed under the `inspect_ai` entry-point group in pyproject.toml. Importing
this module registers the task and the reference-policy model provider, so
that both resolve by name once the package is installed.
"""

from memorydrift.reference import ReferencePolicyAPI
from memorydrift.task import memorydrift

__all__ = ["ReferencePolicyAPI", "memorydrift"]
