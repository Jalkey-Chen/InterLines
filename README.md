<div align="center">

<img src="docs/assets/logo.png" alt="InterLines Logo" width="200" height="auto" />

# InterLines

**Turn Complex Papers into Accessible Public Briefs.**

The AI-native editorial pipeline that reads, thinks, verifies, and refines.

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)](https://github.com/Jalkey-Chen/InterLines)
[![Dependency Manager](https://img.shields.io/badge/uv-managed-purple)](https://github.com/astral-sh/uv)

[Features](#-key-features) • [Philosophy](#-design-philosophy) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Roadmap](#-roadmap)

</div>

---

## 📖 Introduction

**InterLines** is a multi-agent system designed to bridge the gap between technical complexity and public understanding. It doesn't just "summarize" text; it orchestrates a team of specialized AI agents to parse, interpret, fact-check, and rewrite complex documents (Research Papers, Policy Actions, Technical Specs) into clear, engaging public briefs.

Unlike standard chatbots, InterLines features a **Self-Correction Loop**. If the output is too dense or inaccurate, the system automatically replans and refines the content before delivering the final report.

![InterLines CLI Demo](docs/assets/demo.gif)

## 🧠 Design Philosophy

Why build a complex multi-agent system instead of a simple prompt?

### 1. The "Context Amnesia" Problem
Single-pass LLMs often struggle with long-context coherence (64k+ tokens). They hallucinate details or lose the "forest for the trees."
> **InterLines Solution:** We use a **Blackboard Architecture**. Agents (Parser, Explainer, Historian) work independently on specific tasks and write structured artifacts to a shared memory state, ensuring no context is lost.

### 2. The "Jack of All Trades" Fallacy
Asking one model to be a Scholar, a Journalist, and an Editor simultaneously often results in a muddled tone.
> **InterLines Solution:** **Role Specialization.**
> * **The Explainer** digs deep into technical proofs.
> * **The Citizen Agent** translates concepts for the general public.
> * **The Planner** acts as the Editor-in-Chief, dynamically routing tasks based on document type.

### 3. From "Black Box" to "Glass Box"
Chatbots give you an answer, but not the *process*.
> **InterLines Solution:** **Flight Recorder.** Every thought, decision, and draft is serialized into a Trace Log. You can replay, inspect, and debug the entire reasoning chain offline.

---

## 🌟 Key Features

- **🤖 Semantic Routing**: The **Planner Agent** scans the document header and dynamically builds a DAG (Directed Acyclic Graph) execution plan. It knows when to skip the "Timeline" step for technical manuals or enforce "Deep Translation" for academic papers.
- **🔄 Refinement Loop (M5)**: The **Editor Agent** scores drafts for Clarity and Factuality. If the score is low, the Planner triggers a *Replan*, forcing the writers to revise their work automatically.
- **🛡️ Robustness & Fallback**: Built-in guardrails handle LLM hallucinations (e.g., malformed JSON, markdown leakage) with graceful degradation strategies.
- **📝 Structured Artifacts**: Outputs aren't just text strings; they are strict Pydantic objects (`ExplanationCard`, `TimelineEvent`, `RelevanceNote`), ready for API consumption.
- **⚡ Async & Parallel**: Built on FastAPI and Python's `asyncio` for high-performance concurrent processing.

---

## 🏗 Architecture

InterLines follows a **Controller-Agent-Blackboard** pattern.

```mermaid
graph TD
    User[Input Document] --> Parser
    Parser --> BB[(Blackboard)]
    
    subgraph "Phase 1: Planning"
        BB --> Planner
        Planner --"PlanSpec (DAG)"--> Orchestrator
    end
    
    subgraph "Phase 2: Execution"
        Orchestrator --> Explainer[Explainer Agent]
        Orchestrator --> History[History Agent]
        Orchestrator --> Jargon[Jargon Agent]
        Explainer & History & Jargon --> BB
    end
    
    subgraph "Phase 3: Refinement Loop"
        BB --> Editor[Editor Agent]
        Editor --"ReviewReport"--> Planner
        Planner --"Replan Decision"--> Orchestrator
        Orchestrator --"Refine Steps"--> Citizen[Citizen Agent]
        Citizen --> BB
    end
    
    subgraph "Phase 4: Synthesis"
        BB --> BriefBuilder[Brief Builder]
        BriefBuilder --> Report[Markdown Report]
    end
````

### 📂 Project Structure

```plaintext
INTERLINES/
├── artifacts/          # Generated reports, images, and trace logs
├── docs/               # Documentation (Architecture, Roadmap, Prompts)
├── examples/           # Sample PDF documents
├── schemas/            # JSON Schemas for data contracts
├── src/
│   └── interlines/
│       ├── agents/     # Specialized AI Agents (Planner, Explainer, Citizen...)
│       ├── api/        # FastAPI backend implementation
│       ├── contracts/  # Pydantic data models (The "Language" of the system)
│       ├── core/       # Core logic: Blackboard, Planner DAG, Strategies
│       ├── llm/        # Model registry and client wrappers
│       └── pipelines/  # Orchestration logic (Public Translation Pipeline)
├── tests/              # E2E and Unit tests
├── pyproject.toml      # Project configuration and dependencies
└── uv.lock             # Lockfile
```

-----

## 🚀 Quick Start

### Prerequisites

  - **Python 3.11** or higher
  - [uv](https://github.com/astral-sh/uv) (Recommended) or pip
  - API Keys for **OpenAI**, **Google Gemini**, or **DeepSeek**.

### Installation

```bash
# 1. Clone the repository
git clone [https://github.com/your-username/interlines.git](https://github.com/your-username/interlines.git)
cd interlines

# 2. Install dependencies via uv (fast!)
uv sync

# 3. Configure Environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY / GOOGLE_API_KEY
```

### Usage (CLI)

InterLines comes with a beautiful CLI powered by `Typer` and `Rich`.

**1. Interpret a Document**
Run the pipeline on a local PDF. The system will parse, plan, and generate a brief.

```bash
uv run interlines interpret samples/Native_Sparse_Attention.pdf
```

**2. Replay a Trace (Zero Cost)**
Want to debug a run without spending API tokens? Replay a saved trace file.

```bash
uv run interlines replay artifacts/runs/20251209_xxxx_paper.json
```

-----

## 🛠️ Advanced: API Server

To integrate InterLines into your own frontend (React/Vue), start the FastAPI server:

```bash
uv run uvicorn interlines.api.server:app --reload
```

  - **Swagger UI**: `http://localhost:8000/docs`
  - **Submit Job**: `POST /interpret`
  - **Check Status**: `GET /jobs/{id}`

-----

## 📂 Artifacts Example

We separate content into layers for different audiences:

| Artifact | Audience | Purpose |
| :--- | :--- | :--- |
| **ExplanationCard** | Experts | Deep dive into claims, evidence, and rationale. |
| **RelevanceNote** | Public | "Why this matters" (e.g. for policymakers, students). |
| **TimelineEvent** | Historians | Chronological evolution of the topic. |
| **TermCard** | Learners | Plain-language glossary of jargon. |

-----

## 🗺️ Roadmap

  - [x] **M5: Refinement Loop** (Self-correcting Editor-Planner cycle)
  - [x] **M6: Trace Replay & CLI** (Flight recorder for debugging)
  - [ ] **M7: Web Dashboard** (React-based visualization of the Blackboard)
  - [ ] **M8: Human-in-the-loop** (Allow users to approve/reject Plans)
  - [ ] **M9: Multi-modal Support** (Reading charts and diagrams from PDFs)

-----

## 🤝 Contributing

Contributions are welcome\! Please read our [CONTRIBUTING.md](https://www.google.com/search?q=docs/CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

1.  Fork the repo
2.  Create your feature branch (`git checkout -b feature/amazing-feature`)
3.  Commit your changes (`git commit -m 'feat: Add amazing feature'`)
4.  Push to the branch (`git push origin feature/amazing-feature`)
5.  Open a Pull Request

## 📄 License

Distributed under the **GNU General Public License v3 (GPLv3)**. See `LICENSE` for more information.