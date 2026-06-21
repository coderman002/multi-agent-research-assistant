"""
tests/test_tools.py
--------------------
Unit tests for the SearchResult class and TavilySearchTool formatting logic.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch env to avoid TavilySearchTool instantiation failing in tests
os.environ.setdefault("TAVILY_API_KEY", "test-key-placeholder")
os.environ.setdefault("OPENAI_API_KEY", "test-key-placeholder")

from tools import SearchResult


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSearchResult:

    def test_to_text_format(self):
        result = SearchResult(
            title="LangGraph Docs",
            url="https://langchain-ai.github.io/langgraph/",
            content="LangGraph is a library for building stateful, multi-actor apps.",
            score=0.92,
        )
        text = result.to_text()
        assert "LangGraph Docs" in text
        assert "https://langchain-ai.github.io/langgraph/" in text
        assert "LangGraph is a library" in text

    def test_default_score(self):
        result = SearchResult(title="Test", url="http://example.com", content="Content")
        assert result.score == 0.0

    def test_repr(self):
        result = SearchResult(title="Test", url="http://x.com", content="...", score=0.75)
        r = repr(result)
        assert "Test" in r
        assert "0.75" in r

    def test_to_text_contains_all_fields(self):
        result = SearchResult(
            title="My Source",
            url="https://my.source.com",
            content="Important finding here.",
            score=0.88,
        )
        text = result.to_text()
        assert "My Source" in text
        assert "https://my.source.com" in text
        assert "Important finding here." in text


class TestFormatResultsForLLM:

    def _make_tool(self):
        """Construct a TavilySearchTool with a mocked client to avoid real API calls."""
        from unittest.mock import MagicMock, patch
        with patch("tools.TavilyClient"):
            from tools import TavilySearchTool
            tool = TavilySearchTool.__new__(TavilySearchTool)
            tool.client = MagicMock()
            tool.max_results = 5
            return tool

    def test_empty_results_returns_no_results_message(self):
        tool = self._make_tool()
        output = tool.format_results_for_llm([])
        assert "No search results found." in output

    def test_single_result_numbering(self):
        tool = self._make_tool()
        results = [SearchResult("Title A", "http://a.com", "Content A", 0.9)]
        output = tool.format_results_for_llm(results)
        assert "[1]" in output
        assert "Title A" in output

    def test_multiple_results_sequential_numbering(self):
        tool = self._make_tool()
        results = [
            SearchResult("A", "http://a.com", "Content A", 0.9),
            SearchResult("B", "http://b.com", "Content B", 0.8),
            SearchResult("C", "http://c.com", "Content C", 0.7),
        ]
        output = tool.format_results_for_llm(results)
        assert "[1]" in output
        assert "[2]" in output
        assert "[3]" in output

    def test_results_sorted_by_score(self):
        tool = self._make_tool()
        results = [
            SearchResult("Low", "http://low.com", "Low score", 0.2),
            SearchResult("High", "http://high.com", "High score", 0.95),
        ]
        # Sort descending (as search() would do)
        results.sort(key=lambda r: r.score, reverse=True)
        output = tool.format_results_for_llm(results)
        # High-scored result should appear first
        assert output.index("High") < output.index("Low")
