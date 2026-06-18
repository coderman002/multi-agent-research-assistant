"""
agents.py
---------
Defines all four agent node functions for the LangGraph research pipeline.

Each function is a LangGraph node:
  - planner_node     : Decomposes topic into targeted search queries
  - researcher_node  : Executes searches and extracts structured findings
  - critic_node      : Evaluates completeness and decides whether to loop
  - synthesiser_node : Writes the final polished research report

All nodes accept ResearchState and return a partial dict update.
LangGraph merges these updates into the running state automatically,
respecting the Annotated[list, operator.add] fields for accumulation.

LLM Support:
  - Standard OpenAI (default)
  - Azure OpenAI (set USE_AZURE=true in .env)
"""

import json
import logging
from typing import Any

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import config
from prompts import Prompts
from state import ResearchState
from tools import search_tool

logger = logging.getLogger(__name__)


# ── LLM Factory ───────────────────────────────────────────────────────────────

def _get_llm() -> Any:
    """
    Return the correct LLM client based on environment config.
    Supports both standard OpenAI and Azure OpenAI seamlessly.
    Called fresh each invocation to respect dynamic config changes.
    """
    if config.USE_AZURE:
        logger.info("Using Azure OpenAI: %s", config.AZURE_OPENAI_DEPLOYMENT)
        return AzureChatOpenAI(
            azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
            temperature=0.3,  # Low temperature for factual, consistent output
        )
    else:
        logger.info("Using OpenAI: %s", config.OPENAI_MODEL)
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.3,
        )


def _invoke_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Invoke the LLM with a system + user message pair and return the text response.

    Args:
        system_prompt : Sets the LLM's role, persona, and output format.
        user_prompt   : The specific task and input data for this call.

    Returns:
        Stripped string content of the LLM response.
    """
    llm = _get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


# ── Node 1: Planner ───────────────────────────────────────────────────────────

def planner_node(state: ResearchState) -> dict:
    """
    PLANNER AGENT
    -------------
    Receives the research topic and decomposes it into 3-5 distinct,
    targeted search queries that together cover the topic comprehensively.

    Design note: Using JSON output format makes parsing deterministic.
    A regex fallback handles cases where the LLM wraps output in markdown.

    State read  : topic
    State write : search_queries, status
    """
    logger.info("[Planner] Generating queries for: %r", state["topic"])

    raw_output = _invoke_llm(
        system_prompt=Prompts.PLANNER_SYSTEM,
        user_prompt=Prompts.PLANNER_USER.format(topic=state["topic"]),
    )

    # Parse JSON array — strip markdown fences if present
    try:
        clean = raw_output.strip().strip("```json").strip("```").strip()
        queries: list[str] = json.loads(clean)
        if not isinstance(queries, list):
            raise ValueError("Expected a JSON array of strings")
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("[Planner] JSON parse failed (%s), using line fallback", e)
        # Fallback: treat each non-empty line as a query
        queries = [
            line.strip('- "').strip()
            for line in raw_output.splitlines()
            if line.strip()
        ]

    logger.info("[Planner] Generated %d queries: %s", len(queries), queries)
    return {
        "search_queries": queries,
        "status": f"📋 Planned {len(queries)} research queries",
    }


# ── Node 2: Researcher ────────────────────────────────────────────────────────

def researcher_node(state: ResearchState) -> dict:
    """
    RESEARCHER AGENT
    ----------------
    Iterates over all pending search queries, executes each via Tavily,
    then uses the LLM to extract and structure key findings from the results.

    In the agentic loop, this node may run multiple times — on the first
    pass with the Planner's queries, and on subsequent passes with any
    additional queries suggested by the Critic.

    Design note: We process queries sequentially to respect API rate limits
    and maintain clear per-query attribution in the research notes.

    State read  : search_queries
    State write : search_results, research_notes, status
    """
    queries = state["search_queries"]
    logger.info("[Researcher] Processing %d queries", len(queries))

    all_raw_results: list[str] = []
    all_notes: list[str] = []

    for i, query in enumerate(queries, 1):
        logger.info("[Researcher] Query %d/%d: %r", i, len(queries), query)

        # Step 1: Execute web search via Tavily
        try:
            results = search_tool.search(query)
            formatted = search_tool.format_results_for_llm(results)
            all_raw_results.append(f"Query: {query}\n{formatted}")
        except RuntimeError as e:
            logger.error("[Researcher] Search failed: %s", e)
            formatted = "Search failed — no results available for this query."

        # Step 2: LLM extracts and structures key findings
        notes = _invoke_llm(
            system_prompt=Prompts.RESEARCHER_SYSTEM,
            user_prompt=Prompts.RESEARCHER_USER.format(
                query=query,
                results=formatted,
            ),
        )
        all_notes.append(f"### Findings: {query}\n{notes}")

    return {
        "search_results": all_raw_results,
        "research_notes": all_notes,
        "status": f"🔍 Researched {len(queries)} queries, extracted key findings",
    }


# ── Node 3: Critic ────────────────────────────────────────────────────────────

def critic_node(state: ResearchState) -> dict:
    """
    CRITIC AGENT
    ------------
    Reviews accumulated research notes and decides whether the research is
    comprehensive enough to write a report, or whether additional queries
    are needed to fill identified gaps.

    This node is the decision point of the agentic loop:
    - Returns 'RESEARCH_COMPLETE' in critique → graph routes to Synthesiser
    - Returns gap analysis with new queries → graph routes back to Researcher
    - If MAX_RESEARCH_ITERATIONS reached → forces completion regardless

    State read  : topic, research_notes, iteration
    State write : critique, search_queries (if gaps), iteration, status
    """
    combined_notes = "\n\n".join(state["research_notes"])
    iteration = state.get("iteration", 0)
    logger.info("[Critic] Reviewing research (iteration %d)", iteration)

    critique = _invoke_llm(
        system_prompt=Prompts.CRITIC_SYSTEM,
        user_prompt=Prompts.CRITIC_USER.format(
            topic=state["topic"],
            research=combined_notes,
        ),
    )
    logger.info("[Critic] Result: %s", critique[:150])

    # Force completion if max iterations reached
    is_complete = (
        "RESEARCH_COMPLETE" in critique
        or iteration >= config.MAX_RESEARCH_ITERATIONS
    )

    if is_complete:
        return {
            "critique": critique,
            "iteration": iteration + 1,
            "status": "✅ Research complete — synthesising report",
        }
    else:
        # Extract new queries the Critic suggested (bullet-point lines)
        new_queries = [
            line.strip("- *\"").strip()
            for line in critique.splitlines()
            if line.strip().startswith(("-", "*", '"'))
            and len(line.strip()) > 5
        ]
        logger.info("[Critic] Requesting %d additional queries", len(new_queries))
        return {
            "critique": critique,
            "search_queries": new_queries if new_queries else [],
            "iteration": iteration + 1,
            "status": f"🔄 Iteration {iteration + 1}: filling research gaps",
        }


# ── Node 4: Synthesiser ───────────────────────────────────────────────────────

def synthesiser_node(state: ResearchState) -> dict:
    """
    SYNTHESISER AGENT
    -----------------
    Combines all accumulated research notes into a polished, structured
    Markdown report. This is the terminal node — after this the graph ends.

    The report follows a fixed structure (Executive Summary → Key Findings
    → Analysis → Conclusion → Sources) defined in the system prompt, ensuring
    consistent, professional output regardless of topic.

    State read  : topic, research_notes
    State write : final_report, status
    """
    combined_notes = "\n\n".join(state["research_notes"])
    logger.info("[Synthesiser] Writing report for: %r", state["topic"])

    report = _invoke_llm(
        system_prompt=Prompts.SYNTHESISER_SYSTEM.format(
            language=config.REPORT_LANGUAGE
        ),
        user_prompt=Prompts.SYNTHESISER_USER.format(
            topic=state["topic"],
            research=combined_notes,
        ),
    )

    logger.info("[Synthesiser] Report ready (%d chars)", len(report))
    return {
        "final_report": report,
        "status": "📄 Report ready!",
    }
