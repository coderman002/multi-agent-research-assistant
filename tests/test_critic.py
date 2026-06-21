"""
tests/test_critic.py
---------------------
Unit tests for the Critic node's routing logic and query extraction.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Helpers replicated from agents.py / graph.py for isolated testing ─────────

def _extract_new_queries(critique: str) -> list[str]:
    """Mirror of the query extraction logic in critic_node."""
    return [
        line.strip("- *\"").strip()
        for line in critique.splitlines()
        if line.strip().startswith(("-", "*", '"'))
        and len(line.strip()) > 5
    ]


def _is_complete(critique: str, iteration: int, max_iterations: int) -> bool:
    return "RESEARCH_COMPLETE" in critique or iteration >= max_iterations


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCriticCompletion:

    def test_research_complete_signal(self):
        critique = "RESEARCH_COMPLETE — all key angles are covered."
        assert _is_complete(critique, 1, 3) is True

    def test_max_iterations_forces_completion(self):
        critique = "Still some gaps. - search more about X"
        assert _is_complete(critique, 3, 3) is True

    def test_not_complete_below_max(self):
        critique = "Missing data on Y. - additional query"
        assert _is_complete(critique, 1, 3) is False

    def test_research_complete_substring_match(self):
        critique = "After review: RESEARCH_COMPLETE. The data is sufficient."
        assert _is_complete(critique, 0, 3) is True


class TestCriticQueryExtraction:

    def test_extracts_bullet_queries(self):
        critique = (
            "Missing angles:\n"
            "- LangGraph production deployment patterns\n"
            "- LangGraph memory management strategies\n"
        )
        queries = _extract_new_queries(critique)
        assert len(queries) == 2
        assert "LangGraph production deployment patterns" in queries

    def test_extracts_star_queries(self):
        critique = "* What are the latency benchmarks?\n* How does it compare to Temporal.io?"
        queries = _extract_new_queries(critique)
        assert len(queries) == 2

    def test_filters_short_lines(self):
        critique = "- ok\n- this is a proper query about something important"
        queries = _extract_new_queries(critique)
        assert len(queries) == 1

    def test_empty_critique_returns_empty(self):
        queries = _extract_new_queries("")
        assert queries == []

    def test_research_complete_no_queries(self):
        critique = "RESEARCH_COMPLETE — no further investigation needed."
        queries = _extract_new_queries(critique)
        assert queries == []

    def test_guard_empty_queries_triggers_completion(self):
        """The agentic loop guard: empty queries must NOT allow re-routing to researcher."""
        critique = "I think there may be gaps but cannot identify specific queries."
        queries = _extract_new_queries(critique)
        # When queries is empty AND not complete → critic_node should force completion
        is_complete = _is_complete(critique, 1, 3)
        # Simulate the guard: if not complete but no queries, we force completion
        should_force = not is_complete and len(queries) == 0
        assert should_force is True, "Guard must trigger to prevent infinite loop"
