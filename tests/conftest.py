"""Shared fixtures. No Spark, no warehouse: gold tables live in an in-memory DuckDB (see test_assistant.gold)."""

from __future__ import annotations

import pytest

from smogcast.app.config import load_config

# The same small scale as the pipeline repo's tests: two cities.
TEST_OVERRIDES = {"run": {"cities": ["krakow", "gdansk"]}}


@pytest.fixture
def cfg():
    return load_config(TEST_OVERRIDES)
