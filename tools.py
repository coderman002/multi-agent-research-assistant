"""
tools.py
--------
External tool integrations for the Multi-Agent Research Assistant.
Currently wraps the Tavily web search API with structured output parsing.

Tavily is purpose-built for LLM agents — it returns clean, pre-processed
web content rather than raw HTML, making it ideal for direct LLM ingestion.

To add new tools (e.g. ArXiv, Wikipedia, NewsAPI), follow the same pattern:
    1. Create a wrapper class
    2. Implement a .search(query) method returning List[SearchResult]
    3. Implement .format_results_for_llm(results) -> str
    4. Register the tool in agents.py

Docs: https://docs.tavily.com
"""

import logging
from typing import Any

from tavily import TavilyClient

from config import config

logger = logging.getLogger(__name__)


class SearchResult:
    """
    Standardised container for a single search result.
    Decouples the rest of the app from Tavily's raw response format,
    making it easy to swap search providers without touching agent code.
    """

    def __init__(self, title: str, url: str, content: str, score: float = 0.0):
        self.title = title
        self.url = url
        self.content = content
        self.score = score  # Relevance score from Tavily (0.0 to 1.0)

    def to_text(self) -> str:
        """Format result as readable text block for LLM consumption."""
        return (
            f"Source: {self.title}\n"
            f"URL: {self.url}\n"
            f"Content: {self.content}\n"
        )

    def __repr__(self) -> str:
        return f"SearchResult(title={self.title!r}, score={self.score:.2f})"


class TavilySearchTool:
    """
    Wrapper around the Tavily Search API.

    Uses 'advanced' search depth for higher quality results.
    Strips raw HTML and returns clean content suitable for LLM prompts.
    Results are sorted by relevance score (highest first).
    """

    def __init__(self):
        if not config.TAVILY_API_KEY:
            raise EnvironmentError(
                "TAVILY_API_KEY is not set. "
                "Get a free key at https://tavily.com and add it to your .env file."
            )
        self.client = TavilyClient(api_key=config.TAVILY_API_KEY)
        self.max_results = config.MAX_SEARCH_RESULTS
        logger.info("TavilySearchTool initialised (max_results=%d)", self.max_results)

    def search(self, query: str) -> list[SearchResult]:
        """
        Execute a web search and return structured, ranked results.

        Args:
            query: The search query string.

        Returns:
            List of SearchResult objects sorted by relevance score (desc).

        Raises:
            RuntimeError: If the Tavily API call fails.
        """
        logger.info("Searching Tavily: %r", query)
        try:
            response: dict[str, Any] = self.client.search(
                query=query,
                search_depth="advanced",    # Higher quality, slightly slower
                max_results=self.max_results,
                include_answer=False,        # We synthesise our own answer
                include_raw_content=False,   # Clean content is sufficient
            )
            results = [
                SearchResult(
                    title=r.get("title", "Untitled"),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    score=r.get("score", 0.0),
                )
                for r in response.get("results", [])
            ]
            # Sort by score descending — best results first
            results.sort(key=lambda r: r.score, reverse=True)
            logger.info("Found %d results for: %r", len(results), query)
            return results

        except Exception as e:
            logger.error("Tavily search failed for %r: %s", query, e)
            raise RuntimeError(f"Search failed for query '{query}': {e}") from e

    def format_results_for_llm(self, results: list[SearchResult]) -> str:
        """
        Concatenate search results into a single numbered string for LLM input.

        Args:
            results: List of SearchResult objects.

        Returns:
            Formatted string with all results numbered and separated by blank lines.
        """
        if not results:
            return "No search results found."
        return "\n\n".join(
            f"[{i + 1}] {r.to_text()}" for i, r in enumerate(results)
        )


# ── Module-level singleton ─────────────────────────────────────────────────────
# Reuse the same client instance across the app to avoid re-initialising
# the HTTP session on every search call.
#
# NOTE: We always instantiate TavilySearchTool here.
# If TAVILY_API_KEY is missing, __init__ raises a clear EnvironmentError
# at import time — far better than exposing None and crashing with an
# AttributeError inside an agent node call.
search_tool = TavilySearchTool()
