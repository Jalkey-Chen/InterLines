# -----------------------------------------------------------------------------
# This module defines a tiny, in-process model registry used by the LLM client.
# It now supports multiple providers (OpenAI / Google Gemini / xAI / Zhipu /
# Moonshot / DeepSeek) while still exposing a simple alias → config mapping.
#
# The registry gives us a single place to:
#   - declare human-friendly aliases (e.g. "fast", "balanced", "planner")
#   - pin them to concrete provider model IDs
#   - keep default sampling parameters (temperature, max_tokens)
#   - attach provider-specific base URLs when needed
#
# The implementation is deliberately lightweight and pure-Python so that it can
# be imported anywhere (CLI, API handlers, agents) without side effects.
# -----------------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for a single LLM model.

    This data structure is intentionally minimal: it stores enough information
    for the HTTP client to talk to the provider while keeping the higher-level
    application code provider-agnostic.

    Parameters
    ----------
    name:
        Provider-specific model identifier, e.g. ``"gpt-5.1"`` or
        ``"gemini-2.5-pro"``.
    provider:
        Logical provider name. This drives authentication and endpoint
        selection inside the client. Expected values include:
        ``"openai"``, ``"google"``, ``"xai"``, ``"zhipu"``, ``"moonshot"``,
        ``"deepseek"``.
    base_url:
        Base URL for the API endpoint. For OpenAI-style chat-completions this
        usually looks like ``"https://api.openai.com/v1"``. For Gemini, this
        is typically ``"https://generativelanguage.googleapis.com/v1beta"``.
        Callers may override this via environment variables on a per-provider
        basis, but the registry carries a sensible default.
    max_tokens:
        Soft default for the maximum number of tokens to generate. Callers
        may override this per request via :func:`generate`.
    temperature:
        Default sampling temperature for the model. Callers may override this
        per request. A lower value yields more deterministic outputs.
    """

    name: str
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    max_tokens: int = 2048
    temperature: float = 0.5


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

#: A small, opinionated default registry of logical aliases → model configs.
#:
#: These aliases are intentionally generic and may be remapped as hosting
#: providers evolve. Application code (agents, API handlers, CLI) should
#: prefer these aliases rather than hard-coding provider model IDs.
#:
#: We keep a mix of:
#:   - coarse-grained aliases ("fast", "balanced", "research") for generic use
#:   - fine-grained, agent-scoped aliases ("planner", "parser", "explainer", …)
#:     that encode the multi-provider design of the PKI / InterLines pipeline.
MODEL_REGISTRY: dict[str, ModelConfig] = {
    # --------------------------------------------------------------------- #
    # Generic aliases (safe defaults for quick scripts, tests, and tooling)
    # --------------------------------------------------------------------- #
    # Lower-latency, cheaper model suitable for quick iterations and tools.
    "fast": ModelConfig(
        name="gpt-4o-mini",
        provider="openai",
        base_url="https://api.openai.com/v1",
        max_tokens=1024,
        temperature=0.4,
    ),
    # Balanced cost/quality default for most explanations.
    "balanced": ModelConfig(
        name="gpt-5.1",
        provider="openai",
        base_url="https://api.openai.com/v1",
        max_tokens=4096,
        temperature=0.5,
    ),
    # Higher-quality / more careful reasoning profile (for ad-hoc research).
    "research": ModelConfig(
        name="o3-mini",
        provider="openai",
        base_url="https://api.openai.com/v1",
        max_tokens=4096,
        temperature=0.3,
    ),
    # --------------------------------------------------------------------- #
    # Agent-scoped aliases (PKI / InterLines multi-agent architecture)
    # --------------------------------------------------------------------- #
    # Planner — global brain that builds DAGs and chooses strategies.
    # Uses xAI Grok 4.1 Fast Reasoning for strong multi-step reasoning.
    "planner": ModelConfig(
        name="grok-4-1-fast-reasoning",
        provider="xai",
        base_url="https://api.x.ai/v1",
        max_tokens=64000,
        temperature=0.3,
    ),
    # Parser — structure + segmentation (JSON output, schema fidelity).
    # Uses Zhipu GLM-4.6, which is highly compatible with JSON schema tasks.
    "parser": ModelConfig(
        name="glm-4.6",
        provider="zhipu",
        base_url="https://api.glm.ai/v1",
        max_tokens=65536,
        temperature=0.2,
    ),
    # Explainer — heavy-weight multi-evidence synthesis and deep explanation.
    # Uses OpenAI GPT-5.1 as the main "thinker".
    "explainer": ModelConfig(
        name="gpt-5.1",
        provider="openai",
        base_url="https://api.openai.com/v1",
        max_tokens=64000,
        temperature=0.4,
    ),
    # Jargon — term definition and cross-lingual clarification.
    # Uses Moonshot Kimi K2 with strong CN/EN semantics.
    "jargon": ModelConfig(
        name="kimi-k2-0905-preview",
        provider="moonshot",
        base_url="https://api.moonshot.cn/v1",
        max_tokens=64000,
        temperature=0.4,
    ),
    # Citizen — public-facing voice, style & persona control.
    # Uses DeepSeek V3.2 for accessible, conversational tone.
    "citizen": ModelConfig(
        name="deepseek-reasoner",
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
        max_tokens=64000,
        temperature=0.6,
    ),
    # History — temporal reasoning, timelines, and concept evolution.
    # Uses Google Gemini 2.5 Pro for long-context multi-needle tasks.
    "history": ModelConfig(
        name="gemini-2.5-pro",
        provider="google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        max_tokens=64000,
        temperature=0.3,
    ),
    # Editor — factuality, hallucination checks, and consistency validation.
    # Uses Google Gemini 2.0 Flash (fast, low hallucination).
    "editor": ModelConfig(
        name="gemini-2.0-flash",
        provider="google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        max_tokens=64000,
        temperature=0.2,
    ),
    # Visual — diagram / Mermaid / SVG / spec-style code generation.
    # Uses Google Gemini 3 for structured, code-like output.
    "visual": ModelConfig(
        name="gemini-3",
        provider="google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        max_tokens=4096,
        temperature=0.4,
    ),
    # Brief builder — final assembly into Markdown / brief formats.
    # Uses OpenAI GPT-4o-mini for low-cost, fast formatting and stitching.
    "brief_builder": ModelConfig(
        name="gpt-4o",
        provider="openai",
        base_url="https://api.openai.com/v1",
        max_tokens=16384,
        temperature=0.4,
    ),
}

#: Default logical alias used when callers do not explicitly choose a model.
DEFAULT_ALIAS: str = "balanced"


def get_model(alias_or_name: str) -> ModelConfig:
    """Return a :class:`ModelConfig` for the given alias or model name.

    Resolution rules
    ----------------
    1. If ``alias_or_name`` exists as a key in :data:`MODEL_REGISTRY`,
       the corresponding config is returned.
    2. Otherwise, we treat the argument as a *concrete* provider model ID and
       construct a new :class:`ModelConfig` on the fly, assuming an OpenAI-
       compatible endpoint with sensible default parameters.

    This behaviour allows application code to either:
    - stick to high-level aliases (preferred), or
    - directly request a specific provider model ID when necessary.

    Parameters
    ----------
    alias_or_name:
        Either a logical alias, such as ``"explainer"`` or ``"fast"``, or a
        concrete provider model ID such as ``"gpt-5.1"``. The function is
        forgiving and will always return a :class:`ModelConfig`.
    """
    if alias_or_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[alias_or_name]
    # Fallback: treat input as the provider model ID itself (OpenAI by default).
    return ModelConfig(name=alias_or_name)


def all_models() -> Mapping[str, ModelConfig]:
    """Return a read-only snapshot of the current registry.

    This is mainly useful for diagnostics (e.g., a future CLI command like
    ``interlines llm list``) and for tests that want to assert on the presence
    of aliases without mutating global state.

    The returned mapping is a shallow copy so callers cannot accidentally
    mutate :data:`MODEL_REGISTRY` directly.
    """
    return dict(MODEL_REGISTRY)


__all__ = ["ModelConfig", "MODEL_REGISTRY", "DEFAULT_ALIAS", "get_model", "all_models"]
