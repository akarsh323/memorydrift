"""Tests that packaging metadata stays consistent."""

import tomllib
from pathlib import Path

import memorydrift
from memorydrift import _registry
from memorydrift.tools import lookup_record
from memorydrift.world import FIELDS_BY_NAME

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def requirements(name: str) -> set[str]:
    lines = (ROOT / name).read_text().splitlines()
    return {
        line.strip()
        for line in lines
        if line.strip() and not line.startswith(("#", "-r"))
    }


def test_requirements_match_pyproject():
    project = PYPROJECT["project"]
    assert requirements("requirements.txt") == set(project["dependencies"])
    assert requirements("requirements-dev.txt") == set(
        project["optional-dependencies"]["dev"]
    )


def test_version_is_single_sourced():
    assert PYPROJECT["project"]["version"] == memorydrift.__version__


def test_inspect_entry_point_exposes_task_and_provider():
    entry_points = PYPROJECT["project"]["entry-points"]["inspect_ai"]
    assert entry_points == {"memorydrift": "memorydrift._registry"}
    assert set(_registry.__all__) == {"ReferencePolicyAPI", "memorydrift"}


def test_lookup_tool_documents_every_field():
    # The tool description is a static docstring; keep it complete.
    description = lookup_record.__wrapped__.__doc__ or ""
    source = Path(memorydrift.__file__).with_name("tools.py").read_text()
    assert description is not None
    for field in FIELDS_BY_NAME:
        assert field in source
