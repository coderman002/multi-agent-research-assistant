"""
tests/test_config.py
---------------------
Unit tests for the Config class validation logic.
"""

import os
import sys
import importlib
from contextlib import contextmanager

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@contextmanager
def _override_env(**env_vars):
    """
    Context manager: temporarily override env vars, reimport Config,
    yield the fresh class, then restore originals on exit — even if
    the test body raises an exception.

    Using a context manager (not a bare generator) guarantees the
    cleanup block runs via the `with` statement's __exit__.
    """
    original = {k: os.environ.get(k) for k in env_vars}
    for k, v in env_vars.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    # Force reimport of config module to pick up new env
    import config as cfg_module
    importlib.reload(cfg_module)
    try:
        yield cfg_module.Config
    finally:
        # Restore original env
        for k, orig_v in original.items():
            if orig_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = orig_v
        importlib.reload(cfg_module)


class TestConfigValidation:

    def test_validate_passes_with_openai_and_tavily(self):
        with _override_env(
            OPENAI_API_KEY="sk-test",
            TAVILY_API_KEY="tvly-test",
            USE_AZURE="false",
            AZURE_OPENAI_API_KEY="",
            AZURE_OPENAI_ENDPOINT="",
        ) as cls:
            cls.validate()  # Should NOT raise

    def test_validate_fails_missing_openai_key(self):
        with _override_env(
            OPENAI_API_KEY="",
            TAVILY_API_KEY="tvly-test",
            USE_AZURE="false",
        ) as cls:
            with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
                cls.validate()

    def test_validate_fails_missing_tavily_key(self):
        with _override_env(
            OPENAI_API_KEY="sk-test",
            TAVILY_API_KEY="",
            USE_AZURE="false",
        ) as cls:
            with pytest.raises(EnvironmentError, match="TAVILY_API_KEY"):
                cls.validate()

    def test_validate_azure_requires_azure_key_and_endpoint(self):
        with _override_env(
            USE_AZURE="true",
            AZURE_OPENAI_API_KEY="",
            AZURE_OPENAI_ENDPOINT="",
            TAVILY_API_KEY="tvly-test",
            OPENAI_API_KEY="",
        ) as cls:
            with pytest.raises(EnvironmentError) as exc_info:
                cls.validate()
            assert "AZURE_OPENAI_API_KEY" in str(exc_info.value)
            assert "AZURE_OPENAI_ENDPOINT" in str(exc_info.value)

    def test_use_azure_flag_parsing(self):
        with _override_env(USE_AZURE="true") as cls:
            assert cls.USE_AZURE is True

        with _override_env(USE_AZURE="false") as cls:
            assert cls.USE_AZURE is False

    def test_max_search_results_default(self):
        with _override_env(MAX_SEARCH_RESULTS="7") as cls:
            assert cls.MAX_SEARCH_RESULTS == 7

    def test_max_research_iterations_default(self):
        with _override_env(MAX_RESEARCH_ITERATIONS="2") as cls:
            assert cls.MAX_RESEARCH_ITERATIONS == 2
