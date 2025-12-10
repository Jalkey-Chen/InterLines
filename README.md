<div align="center">

<!--<img src="docs/assets/logo.png" alt="InterLines Logo" width="200" height="auto" />-->

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

### Demo
<img src="docs/assets/screenshot.png" alt="InterLines CLI screenshot" width="auto" height="auto"

-----

## 📂 Included Examples

InterLines comes with two fully processed examples — a technical paper and a public policy plan — so you can explore the outputs without running the pipeline yourself.

### 📝 Sample Public Briefs (Markdown)

- [Americas AI Action Plan](examples/briefs/Americas%20AI%20Action%20Plan.md)  
  *A public-policy oriented brief summarizing the strategic goals and implications of the Americas AI Action Plan.*

- [Natively Sparse Attention](examples/briefs/Natively%20Sparse%20Attention.md)  
  *A technical brief explaining NSA, a hierarchical sparse attention mechanism for LLMs.*

You can view these directly on GitHub or download them as Markdown/PDF.

### 📜 Execution Traces (JSON)

Each run also includes a full trace containing planner decisions, agent outputs, and intermediate artifacts:

- [Trace: Native Sparse Attention](examples/trace/20251209_225432_Native_Sparse_Attention.json)
- [Trace: Americas AI Action Plan](examples/trace/20251209_230043_Americas-AI-Action-Plan.json)

These trace files are useful for:

- Debugging agent behavior  
- Understanding planner decisions  
- Research on multi-agent interpretability  
- Reproducing full execution states  

### 📜 Execution Traces (for debugging & research)

For each brief, InterLines also stores a full trace of the multi-agent run:

- `examples/trace/20251209_225432_Native_Sparse_Attention.json`
- `examples/trace/20251209_230043_Americas-AI-Action-Plan.json`

Each trace JSON contains:

- Planner decisions (strategy, phases, re-plans)
- All intermediate cards (Explanation, Jargon, Citizen, History, Review)
- Timing information and model metadata

These traces are useful if you want to:

- Inspect how the system arrived at a particular explanation
- Compare different prompt/model settings
- Build evaluation pipelines or research on multi-agent LLM systems

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

### ✅ **M5: Refinement Loop** — *Self-correcting Editor–Planner cycle*

The system gains the ability to reflect on its own outputs using an Editor–Planner feedback loop.

* Multi-pass revisions of explanations, timelines, and term cards
* Automatic detection of missing provenance, weak chains of reasoning, and stylistic inconsistency
* Adaptive planning: Planner rewrites the task graph based on Editor’s critiques
* Produces significantly more reliable Public Briefs

### ✅ **M6: Trace Replay & CLI** — *“Flight recorder” for debugging & reproducibility*

A full snapshot of every agent’s reasoning is recorded and can be replayed.

* Deterministic re-generation of outputs for audits
* Step-by-step view of agent decisions and intermediate cards
* CLI-based time-travel debugging (`interlines replay <run_id>`)
* Essential for research settings and model evaluation


### 🛠️ **M7: Web Dashboard** — *Visual interface for the Blackboard ecosystem*

A React/Vite-based dashboard that visualizes the internal machinery of InterLines.

* Live DAG view of the Planner (nodes = agents, edges = dependencies)
* Blackboard visualization showing evolving artifacts
* Timeline animations for HistoryAgent outputs
* Compare multiple runs side-by-side
* User-friendly portal for demos and educational use


### 🧩 **M8: Human-in-the-loop** — *Collaborative sensemaking workflow*

Allows researchers, journalists, and policy analysts to participate in the reasoning loop.

* Editable cards: Users can rewrite Explanation or Timeline cards
* Approval/rejection of agent plans
* Optional constraints (“don’t oversimplify”, “cite more empirical evidence”)
* Human corrections propagate through DAG → downstream agents re-compute


### 🖼️ **M9: Multi-modal Support** — *Understanding diagrams, charts, and structured figures*

InterLines becomes capable of interpreting visual information from academic PDFs.

* Chart recognition (axes, labels, slope direction, statistical claims)
* Diagram understanding: pipelines, flow charts, conceptual figures
* Extraction of quantitative data from tables and plots
* Enables the system to handle real scientific papers end-to-end


### 🎛️ **M10: Multi-Document Synthesis** — *Building a unified narrative from many sources*

InterLines learns to read *sets* of papers/policies and generate integrated briefs.

* Cross-document evidence clustering
* Contradiction detection and argument alignment
* Automatically generated “literature map” + timeline of field evolution
* Perfect for research surveys, policy syntheses, and course materials


### 🧱 **M11: Knowledge Graph Integration** — *Stable, persistent memory for concepts & citations*

A structured knowledge layer complements the agent system.

* Terminology normalization across papers
* Auto-citation linking to original sources
* Concept–event–claim graph for scientific fields
* Ensures consistency: definitions stay stable across runs


### 🖼️✨ **M12: PosterBuilder — Automated Academic Poster Generation**

*(Your ultimate vision — system-generated conference posters)*
InterLines becomes capable of producing full academic posters (like NeurIPS/ICLR sessions).

* Converts a Public Brief into a poster layout
* Dynamic templates:

  * 3-column research posters
  * Data-rich policy posters
  * Minimalist design styles (IEEE/ACM formats)
* Auto-selects figures, tables, and timelines
* Generates vector graphics (SVG/PDF)
* Supports theme customization: color palettes, institution branding
* Optional agent: **VisualNarrator**

  * Designs schematics
  * Generates illustrative diagrams
  * Summarizes methods visually (flowcharts, box diagrams)

This milestone turns InterLines into a full **scientific communication engine**, not just a text-based explainer.

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