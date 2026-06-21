"""
app.py
------
Streamlit frontend for the Multi-Agent Research Assistant.

Provides a premium dark-mode UI with:
  - Sidebar configuration panel (API keys, model selection, settings)
  - Topic input with example topics
  - Live animated progress updates as agents run
  - Rich Markdown report rendering with syntax highlighting
  - One-click report download as .md file
  - Full error handling with user-friendly messages

Run with:
    streamlit run app.py
"""

import logging
import time
from datetime import datetime

import streamlit as st

# ── Page Config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/coderman002/multi-agent-research-assistant",
        "Report a bug": "https://github.com/coderman002/multi-agent-research-assistant/issues",
        "About": "### Multi-Agent Research Assistant\nPowered by LangGraph · Tavily · GPT-4",
    },
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App background ── */
    .stApp {
        background: linear-gradient(135deg, #0d0f1a 0%, #111827 50%, #0d1117 100%);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0d0f1a 100%);
        border-right: 1px solid rgba(83, 74, 183, 0.3);
    }

    /* ── Cards / containers ── */
    .agent-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(83,74,183,0.25);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    .agent-card:hover {
        border-color: rgba(83,74,183,0.6);
        background: rgba(255,255,255,0.07);
    }
    .agent-card.active {
        border-color: #534AB7;
        background: rgba(83,74,183,0.12);
        box-shadow: 0 0 20px rgba(83,74,183,0.2);
    }
    .agent-card.done {
        border-color: rgba(16,185,129,0.4);
        background: rgba(16,185,129,0.07);
    }

    /* ── Score badge ── */
    .score-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, #534AB7, #7C3AED);
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }

    /* ── Report section ── */
    .report-container {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(83,74,183,0.2);
        border-radius: 16px;
        padding: 28px 32px;
    }

    /* ── Topic input pill ── */
    .example-topic {
        display: inline-block;
        background: rgba(83,74,183,0.15);
        border: 1px solid rgba(83,74,183,0.35);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        color: #a78bfa;
        margin: 3px;
        cursor: pointer;
    }

    /* ── Status pill ── */
    .status-pill {
        background: rgba(83,74,183,0.2);
        border: 1px solid rgba(83,74,183,0.4);
        border-radius: 8px;
        padding: 8px 14px;
        font-size: 0.88rem;
        color: #c4b5fd;
        font-family: 'Fira Code', monospace;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(83,74,183,0.2);
        border-radius: 10px;
        padding: 12px;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #534AB7 0%, #7C3AED 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.55rem 2rem;
        transition: all 0.2s ease;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(83,74,183,0.4);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Headings ── */
    h1 { color: #f8fafc; font-weight: 700; }
    h2, h3 { color: #e2e8f0; font-weight: 600; }

    /* ── Divider ── */
    hr { border-color: rgba(83,74,183,0.2); }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(83,74,183,0.4); border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 20px 0 10px;">
            <div style="font-size: 2.5rem;">🤖</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Research Assistant</div>
            <div style="font-size: 0.75rem; color: #6b7280; margin-top: 4px;">
                LangGraph · Tavily · GPT-4
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### ⚙️ Configuration")

    # LLM Provider
    provider = st.selectbox(
        "LLM Provider",
        ["OpenAI", "Azure OpenAI"],
        help="Select the LLM backend to use.",
    )

    # Initialise all provider-specific variables so they are always defined
    # regardless of which branch runs. Prevents NameError when the Azure
    # branch variables are referenced later under the OpenAI provider.
    azure_endpoint = ""
    azure_deployment = "gpt-4o"

    if provider == "OpenAI":
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Get your key at platform.openai.com",
        )
        model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])
    else:
        openai_key = st.text_input("Azure OpenAI API Key", type="password")
        azure_endpoint = st.text_input("Azure Endpoint", placeholder="https://your-resource.openai.azure.com/")
        azure_deployment = st.text_input("Deployment Name", value="gpt-4o")
        model = azure_deployment

    st.divider()

    tavily_key = st.text_input(
        "Tavily API Key",
        type="password",
        placeholder="tvly-...",
        help="Free key at tavily.com",
    )

    st.divider()

    st.markdown("### 🎛️ Agent Settings")

    max_results = st.slider(
        "Max Search Results per Query",
        min_value=3,
        max_value=10,
        value=5,
        help="More results = richer context but slower",
    )

    max_iterations = st.slider(
        "Max Research Iterations",
        min_value=1,
        max_value=5,
        value=3,
        help="How many times the Critic can loop back for more research",
    )

    report_language = st.selectbox(
        "Report Language",
        ["English", "Spanish", "French", "German", "Portuguese", "Hindi"],
    )

    st.divider()

    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #6b7280; text-align:center;">
            <a href="https://github.com/coderman002/multi-agent-research-assistant"
               style="color: #7C3AED; text-decoration: none;">
                ⭐ Star on GitHub
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Main Content ──────────────────────────────────────────────────────────────

# Header
st.markdown(
    """
    <div style="padding: 20px 0 10px;">
        <h1 style="margin: 0; font-size: 2rem;">
            🔬 Multi-Agent Research Assistant
        </h1>
        <p style="color: #9ca3af; margin-top: 8px; font-size: 1rem;">
            An agentic AI pipeline that autonomously plans, searches, critiques,
            and synthesises research reports — powered by LangGraph.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Agent pipeline visual
st.markdown("#### 🔄 Agent Pipeline")
cols = st.columns(4)
pipeline_agents = [
    ("📋", "Planner", "Decomposes topic into 3–5 search queries"),
    ("🔍", "Researcher", "Executes searches & extracts key findings"),
    ("🧐", "Critic", "Reviews gaps, loops back if needed"),
    ("📄", "Synthesiser", "Writes the polished research report"),
]

for col, (icon, name, desc) in zip(cols, pipeline_agents):
    with col:
        st.markdown(
            f"""
            <div class="agent-card">
                <div style="font-size:1.4rem;">{icon}</div>
                <div style="font-weight:600; color:#e2e8f0; margin-top:6px;">{name}</div>
                <div style="font-size:0.78rem; color:#9ca3af; margin-top:4px;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# Topic input
st.markdown("#### 🎯 Research Topic")

example_topics = [
    "The future of agentic AI systems in enterprise",
    "LangGraph vs LangChain: when to use each",
    "Retrieval-Augmented Generation best practices 2025",
    "Model Context Protocol (MCP) explained",
    "GPT-4o vs Claude 3.5: technical comparison",
]

st.markdown(
    "<div style='margin-bottom:10px; font-size:0.82rem; color:#9ca3af;'>💡 Try an example:</div>",
    unsafe_allow_html=True,
)

# Example topic pills
example_cols = st.columns(len(example_topics))
for i, (col, topic) in enumerate(zip(example_cols, example_topics)):
    with col:
        if st.button(f"#{i+1}", key=f"example_{i}", help=topic, use_container_width=True):
            st.session_state["topic_input"] = topic

topic = st.text_area(
    "Enter your research topic",
    value=st.session_state.get("topic_input", ""),
    placeholder="e.g. The future of agentic AI systems and their impact on enterprise software development",
    height=80,
    key="topic_area",
    label_visibility="collapsed",
)

# Validation
can_run = bool(topic.strip()) and bool(tavily_key.strip()) and bool(openai_key.strip() if provider == "OpenAI" else True)

if not can_run and st.session_state.get("run_attempted"):
    if not topic.strip():
        st.error("⚠️ Please enter a research topic.")
    elif not tavily_key.strip():
        st.error("⚠️ Tavily API key is required. Get one free at tavily.com")
    elif not openai_key.strip() and provider == "OpenAI":
        st.error("⚠️ OpenAI API key is required.")

run_col, _ = st.columns([1, 4])
with run_col:
    run_button = st.button("🚀 Start Research", use_container_width=True, type="primary")

# ── Run Pipeline ──────────────────────────────────────────────────────────────

if run_button:
    st.session_state["run_attempted"] = True

    if not topic.strip():
        st.error("⚠️ Please enter a research topic.")
        st.stop()
    if not tavily_key.strip():
        st.error("⚠️ Please enter your Tavily API key in the sidebar.")
        st.stop()
    if provider == "OpenAI" and not openai_key.strip():
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
        st.stop()

    # Inject config into environment before importing dependent modules
    import os
    os.environ["TAVILY_API_KEY"] = tavily_key
    os.environ["MAX_SEARCH_RESULTS"] = str(max_results)
    os.environ["MAX_RESEARCH_ITERATIONS"] = str(max_iterations)
    os.environ["REPORT_LANGUAGE"] = report_language

    if provider == "OpenAI":
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["OPENAI_MODEL"] = model
        os.environ["USE_AZURE"] = "false"
    else:
        os.environ["AZURE_OPENAI_API_KEY"] = openai_key
        os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
        os.environ["AZURE_OPENAI_DEPLOYMENT"] = azure_deployment
        os.environ["USE_AZURE"] = "true"

    # Lazy import after env is set
    try:
        from graph import build_graph
        from config import config
        config.validate()
    except EnvironmentError as e:
        st.error(f"❌ Configuration error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to initialise: {e}")
        st.stop()

    st.divider()
    st.markdown("### 🔄 Research in Progress...")

    # Live status display
    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    steps = [
        (0.1,  "📋 Planner is decomposing your topic into search queries..."),
        (0.35, "🔍 Researcher is executing web searches..."),
        (0.6,  "🧐 Critic is reviewing research quality..."),
        (0.85, "📄 Synthesiser is writing your report..."),
        (1.0,  "✅ Report ready!"),
    ]

    def update_status(msg: str, progress: float):
        status_placeholder.markdown(
            f'<div class="status-pill">⚡ {msg}</div>',
            unsafe_allow_html=True,
        )
        progress_bar.progress(progress)

    update_status(steps[0][1], steps[0][0])

    start_time = time.time()

    try:
        app = build_graph()

        initial_state = {
            "topic": topic.strip(),
            "search_queries": [],
            "search_results": [],
            "research_notes": [],
            "iteration": 0,
        }

        # Stream events for live status
        result = None
        step_idx = 0

        for event in app.stream(initial_state, {"recursion_limit": 50}):
            node_name = list(event.keys())[0]
            node_map = {
                "planner": (1, steps[1]),
                "researcher": (2, steps[2]),
                "critic": (3, steps[3]),
                "synthesiser": (4, steps[4]),
            }
            if node_name in node_map:
                idx, (prog, msg) = node_map[node_name]
                update_status(msg, prog)

            # Capture final result
            if "synthesiser" in event:
                result = event["synthesiser"]

        elapsed = time.time() - start_time
        progress_bar.progress(1.0)
        status_placeholder.empty()

    except Exception as e:
        st.error(f"❌ Research pipeline failed: {e}")
        logging.exception("Pipeline error")
        st.stop()

    # ── Report Display ─────────────────────────────────────────────────────────

    st.divider()

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("⏱️ Time Taken", f"{elapsed:.1f}s")
    with m2:
        # Show actual number of search queries executed, not a formula estimate
        actual_searches = len(result.get("search_results", [])) if result else 0
        st.metric("🔍 Searches Run", str(actual_searches) if actual_searches else "—")
    with m3:
        actual_iterations = result.get("iteration", 0) if result else 0
        st.metric("🔄 Iterations", str(actual_iterations))
    with m4:
        st.metric("📄 Report Length", f"~{len(result.get('final_report', '')) // 5} words" if result else "—")

    st.markdown("### 📄 Research Report")

    if result and result.get("final_report"):
        report = result["final_report"]

        # Render the report as proper Markdown (not inside a raw HTML div,
        # which would prevent Markdown headings/bold/lists from rendering).
        st.markdown(report)

        st.divider()

        # Download button
        filename = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        st.download_button(
            label="⬇️ Download Report (.md)",
            data=report,
            file_name=filename,
            mime="text/markdown",
            use_container_width=False,
        )
    else:
        st.warning("⚠️ The research pipeline completed but no report was generated. Check your API keys and try again.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    """
    <div style="text-align:center; color:#4b5563; font-size:0.78rem; padding: 10px 0;">
        Built by <a href="https://github.com/coderman002" style="color:#7C3AED;">Manish Kumar Shaw</a>
        · Powered by <b>LangGraph</b>, <b>Tavily</b>, <b>GPT-4</b>
        · <a href="https://github.com/coderman002/multi-agent-research-assistant" style="color:#7C3AED;">View on GitHub ⭐</a>
    </div>
    """,
    unsafe_allow_html=True,
)
