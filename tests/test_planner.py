"""
tests/test_planner.py
----------------------
Unit tests for the Planner node's JSON parsing and fallback logic.

These test the pure parsing functions in isolation without calling
any LLM or external APIs — fast, deterministic, and dependency-free.
"""

import json
import re
import sys
import os
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Helpers replicated from agents.py for isolated testing ────────────────────

def _parse_planner_output(raw_output: str) -> list[str]:
    """Mirror of the parsing logic in planner_node for unit testing."""
    try:
        clean = re.sub(r"^```(?:json)?\s*", "", raw_output.strip())
        clean = re.sub(r"\s*```$", "", clean).strip()
        queries = json.loads(clean)
        if not isinstance(queries, list):
            raise ValueError("Expected a JSON array of strings")
        return queries
    except (json.JSONDecodeError, ValueError):
        return [
            line.strip('- "').strip()
            for line in raw_output.splitlines()
            if line.strip()
        ]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPlannerJsonParsing:

    def test_clean_json_array(self):
        raw = '["query one", "query two", "query three"]'
        result = _parse_planner_output(raw)
        assert result == ["query one", "query two", "query three"]

    def test_json_with_markdown_fence(self):
        raw = '```json\n["query one", "query two"]\n```'
        result = _parse_planner_output(raw)
        assert result == ["query one", "query two"]

    def test_json_with_plain_fence(self):
        raw = '```\n["query one", "query two"]\n```'
        result = _parse_planner_output(raw)
        assert result == ["query one", "query two"]

    def test_json_preserves_special_chars(self):
        """Regression test: old str.strip() would corrupt JSON with j/s/o/n chars."""
        raw = '["json format explained", "JSON vs XML comparison"]'
        result = _parse_planner_output(raw)
        assert result == ["json format explained", "JSON vs XML comparison"]

    def test_fallback_on_invalid_json(self):
        raw = "- query about AI\n- query about LLMs\n- query about agents"
        result = _parse_planner_output(raw)
        assert len(result) == 3
        assert "query about AI" in result

    def test_fallback_filters_empty_lines(self):
        raw = "- query one\n\n- query two\n\n"
        result = _parse_planner_output(raw)
        assert len(result) == 2

    def test_json_non_list_falls_back(self):
        """If LLM returns a JSON object instead of array, use fallback."""
        raw = '{"queries": ["q1", "q2"]}'
        result = _parse_planner_output(raw)
        # Fallback treats the line as a bullet
        assert isinstance(result, list)

    def test_empty_string_returns_empty_list(self):
        result = _parse_planner_output("")
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        result = _parse_planner_output("   \n  \n  ")
        assert result == []
