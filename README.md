# 🤖 Multi-Agent Research Assistant

<div align="center">

[![CI](https://github.com/coderman002/multi-agent-research-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/coderman002/multi-agent-research-assistant/actions)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.28-534AB7?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

**An agentic AI pipeline that autonomously plans, searches, critiques, and synthesises research reports.**

Built with [LangGraph](https://github.com/langchain-ai/langgraph) · [Tavily](https://tavily.com) · [GPT-4o](https://openai.com) · [Streamlit](https://streamlit.io)

[🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#️-architecture) · [✨ Features](#-features) · [⚙️ Configuration](#️-configuration)

</div>

---

## ✨ Features

- 🧠 **4-Agent Agentic Loop** — Planner → Researcher → Critic → Synthesiser
- 🔄 **Self-Improving Research** — Critic loops back to fill identified gaps
- 🌐 **Real-Time Web Search** — Powered by Tavily's AI-optimised search API
- 📄 **Polished Markdown Reports** — Executive summary, findings, analysis, sources
- 🎛️ **Streamlit UI** — Dark-mode frontend with live status updates & download
- ☁️ **Dual LLM Support** — Standard OpenAI or Azure OpenAI (toggle via `.env`)
- 🔒 **Secure by Design** — All credentials via `.env`, never in code
- ✅ **Fully Tested** — Unit test suite with coverage reporting

---

## 🏗️ Architecture

```
User Topic
    │
    ▼
┌─────────────┐
│   PLANNER   │  Decomposes topic into 3–5 targeted search queries
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ RESEARCHER  │  Executes searches via Tavily, extracts structured findings
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   CRITIC    │  Reviews completeness — loops back if gaps found
└──────┬──────┘
       │ RESEARCH_COMPLETE
       ▼
┌─────────────┐
│ SYNTHESISER │  Writes polished Markdown report (600–1000 words)
└──────┬──────┘
       │
       ▼
  Final Report
```

### State Flow

LangGraph's `StateGraph` passes a shared `ResearchState` TypedDict between nodes. Lists (`search_queries`, `search_results`, `research_notes`) are annotated with `operator.add` so they **accumulate** across iterations rather than being replaced.

---

## 📁 Project Structure

```
multi-agent-research-assistant/
│
├── app.py              # Streamlit frontend (run this!)
├── graph.py            # LangGraph StateGraph construction & routing
├── agents.py           # 4 agent node functions (Planner, Researcher, Critic, Synthesiser)
├── state.py            # ResearchState TypedDict definition
├── prompts.py          # All LLM prompt templates (centralised)
├── config.py           # Environment-based configuration
├── tools.py            # Tavily web search wrapper
│
├── tests/
│   ├── test_planner.py   # JSON parsing & fallback logic tests
│   ├── test_critic.py    # Routing & query extraction tests
│   ├── test_tools.py     # SearchResult & formatter tests
│   └── test_config.py    # Config validation tests
│
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions CI (lint + test across 3 Python versions)
│
├── requirements.txt       # Pinned production dependencies
├── requirements-dev.txt   # Dev / testing dependencies
├── .env.example           # Environment variable template
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/coderman002/multi-agent-research-assistant.git
cd multi-agent-research-assistant
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required keys in `.env`:
```env
OPENAI_API_KEY=sk-...          # From platform.openai.com
TAVILY_API_KEY=tvly-...        # Free key from tavily.com
```

### 3. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` → enter a topic → click **🚀 Start Research** → download your report.

---

## ⚙️ Configuration

All settings are controlled via `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model to use |
| `TAVILY_API_KEY` | — | Tavily search API key |
| `MAX_SEARCH_RESULTS` | `5` | Results per query (3–10) |
| `MAX_RESEARCH_ITERATIONS` | `3` | Max critic loop iterations |
| `REPORT_LANGUAGE` | `English` | Output report language |
| `USE_AZURE` | `false` | Set `true` for Azure OpenAI |

### Azure OpenAI

```env
USE_AZURE=true
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2025-01-01-preview
```

---

## 🧪 Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## 🔑 Getting API Keys

| Service | Link | Cost |
|---|---|---|
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) | Pay-per-use |
| Tavily | [app.tavily.com](https://app.tavily.com) | 1,000 free searches/month |

---

## 🛠️ Extending the Pipeline

### Adding a new search tool

Follow the pattern in `tools.py`:
1. Create a wrapper class with `.search(query)` → `list[SearchResult]`
2. Implement `.format_results_for_llm(results)` → `str`
3. Register it in `agents.py`

### Adding a new agent node

1. Define a `def my_node(state: ResearchState) -> dict:` function in `agents.py`
2. Wire it into the graph in `graph.py`

---

## 👨‍💻 Author

**Manish Kumar Shaw** — AI & Data Science Engineer II @ Deloitte USI

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/mks113)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/coderman002)

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
