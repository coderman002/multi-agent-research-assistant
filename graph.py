"""
graph.py
--------
Constructs and compiles the LangGraph StateGraph for the research pipeline.

Separating graph wiring from node logic keeps agents.py focused on
*what* each agent does, while this module owns *how* they connect.

Graph topology:
    planner ──► researcher ──► critic ──┬──► synthesiser ──► END
                    ▲                   │ (gap found)
                    └───────────────────┘
                        (agentic loop)

Usage:
    from graph import build_graph
    app = build_graph()
    result = app.invoke({"topic": "...", "search_queries": [], ...})
"""

import logging

from langgraph.graph import END, StateGraph

from agents import (
    critic_node,
    planner_node,
    researcher_node,
    synthesiser_node,
)
from state import ResearchState

logger = logging.getLogger(__name__)


# ── Routing Function ──────────────────────────────────────────────────────────

def _route_after_critic(state: ResearchState) -> str:
    """
    Conditional edge function: decides what happens after the Critic runs.

    Returns:
        "synthesiser"  — if the Critic determined research is complete.
        "researcher"   — if the Critic found gaps and wants more research.

    IMPORTANT: We route based on the critique text and status, NOT on
    search_queries. Because search_queries uses Annotated[list, operator.add],
    it accumulates ALL queries from ALL iterations — it's never empty after
    the Planner runs. Checking it would always route back to researcher.

    The Critic node writes:
      - "RESEARCH_COMPLETE" in critique  → done
      - "✅" in status                   → done (forced completion / no queries)
      - Otherwise                         → loop back for gap-filling
    """
    critique = state.get("critique", "") or ""
    status = state.get("status", "") or ""

    if "RESEARCH_COMPLETE" in critique or "✅" in status:
        logger.info("[Router] Research complete → synthesiser")
        return "synthesiser"

    # The Critic found gaps and returned new queries → loop back
    logger.info("[Router] Critic found gaps → researcher")
    return "researcher"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Build and compile the multi-agent research StateGraph.

    Returns:
        A compiled LangGraph app ready to be invoked.
    """
    workflow = StateGraph(ResearchState)

    # Register nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("synthesiser", synthesiser_node)

    # Linear edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "critic")

    # Conditional edge: the agentic loop decision point
    workflow.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "researcher": "researcher",
            "synthesiser": "synthesiser",
        },
    )

    # Terminal edge
    workflow.add_edge("synthesiser", END)

    logger.info("Research graph compiled successfully.")
    return workflow.compile()
