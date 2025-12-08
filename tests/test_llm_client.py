from __future__ import annotations

from typing import Any

from interlines.llm.client import LLMClient
from interlines.llm.models import DEFAULT_ALIAS, get_model


def test_from_env_reads_openai_api_key_and_base_url(monkeypatch: Any) -> None:
    """LLMClient.from_env() should honour OPENAI_API_KEY and OPENAI_BASE_URL."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")

    client = LLMClient.from_env()

    assert client.api_key == "test-openai-key"
    assert client.base_url == "https://example.com/v1"
    assert client.default_model_alias == DEFAULT_ALIAS


def test_generate_openai_compatible_uses_registry_and_returns_text(
    monkeypatch: Any,
) -> None:
    """For OpenAI-compatible providers, `generate()` uses Chat Completions."""
    # We create a client with a dummy default API key. For provider="openai",
    # the helper will fall back to this if the env var is missing.
    client = LLMClient(
        api_key="dummy-openai-key",
        base_url="https://api.example.com/v1",
        default_model_alias="explainer",  # alias defined in the registry
    )

    captured: dict[str, Any] = {}

    def fake_post(
        self: LLMClient,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Fake HTTP call that records the request and returns a stub response."""
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        # Minimal OpenAI-compatible response.
        return {
            "choices": [
                {
                    "message": {
                        "content": "Hello from fake explainer.",
                    }
                }
            ]
        }

    # Patch the internal network call at the class level (slots-safe).
    monkeypatch.setattr(LLMClient, "_post", fake_post)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello."},
    ]

    text = client.generate(messages)

    # We should get the stub content back.
    assert text == "Hello from fake explainer."

    # URL should point to a Chat Completions endpoint.
    assert captured["url"].endswith("/chat/completions")

    # Payload should include the explainer model from the registry.
    model_cfg = get_model("explainer")
    assert captured["payload"]["model"] == model_cfg.name
    assert isinstance(captured["payload"]["messages"], list)


def test_generate_gemini_uses_google_key_and_extracts_content(
    monkeypatch: Any,
) -> None:
    """For provider='google', `generate()` should call the Gemini helper."""
    # Ensure GOOGLE_API_KEY is present so _generate_gemini does not fail.
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    # Use a short custom base URL so we can assert on it.
    monkeypatch.setenv("GOOGLE_API_BASE_URL", "https://gemini.example.com/v1beta")

    # The default alias is "balanced" (OpenAI), so here we explicitly select
    # the "history" alias from the registry, which uses provider="google".
    client = LLMClient(
        api_key="",  # not used for google
        base_url="https://api.openai.com/v1",
        default_model_alias="history",
    )

    captured: dict[str, Any] = {}

    def fake_post(
        self: LLMClient,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Fake Gemini call that records request and returns a stub response."""
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        # Minimal Gemini `generateContent`-style response.
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Part A. "},
                            {"text": "Part B."},
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(LLMClient, "_post", fake_post)

    messages: list[dict[str, str]] = [
        {"role": "user", "content": "Explain the history of a concept briefly."},
    ]

    text = client.generate(messages, model="history")

    # The concatenated parts should be returned.
    assert text == "Part A. Part B."

    # URL should target the Gemini `generateContent` endpoint.
    assert captured["url"].startswith("https://gemini.example.com/v1beta/models/")
    assert captured["url"].endswith(":generateContent")

    # Headers should carry the Google API key via x-goog-api-key.
    assert captured["headers"]["x-goog-api-key"] == "test-google-key"

    # Payload should contain `contents` with text parts.
    contents = captured["payload"]["contents"]
    assert isinstance(contents, list)
    assert len(contents) == 1
    assert contents[0]["parts"][0]["text"].startswith("Explain the history")
