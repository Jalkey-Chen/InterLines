# -----------------------------------------------------------------------------
# This module provides a small, synchronous LLM client abstraction that:
#   - reads provider API keys / base URLs from environment variables
#   - uses the model registry to resolve logical aliases → concrete model IDs
#   - exposes a single `generate()` method that returns a text completion
#
# The implementation uses only the Python standard library (`urllib.request`)
# so that it does not introduce additional dependencies. Unit tests are
# expected to *mock* the internal `_post()` method so that no real HTTP calls
# are made during CI.
#
# Provider support
# ----------------
# The client implements two families of protocols:
#
# 1. OpenAI-compatible Chat Completions:
#    - OpenAI      (provider="openai")
#    - DeepSeek    (provider="deepseek")
#    - Moonshot    (provider="moonshot" / Kimi)
#    - Zhipu GLM   (provider="zhipu")
#    - xAI Grok    (provider="xai")
#
#    These all use POST /chat/completions and the same request/response shape.
#
# 2. Google Gemini "generateContent":
#    - Google Gemini (provider="google")
#
#    These use POST /models/{model}:generateContent with a slightly different
#    payload and response format. We map our simple chat-style messages into
#    Gemini's `contents` structure and convert the response back to a string.
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from .models import DEFAULT_ALIAS, ModelConfig, get_model


@dataclass(slots=True)
class LLMClient:
    """Universal, multi-provider LLM client with a simple `generate()` API.

    The client is intentionally synchronous and minimal. It focuses on the
    common case of "given a small list of chat messages, return one string",
    while hiding the differences between providers and HTTP endpoints.

    Parameters
    ----------
    api_key:
        Optional default API key used for authentication with OpenAI-compatible
        providers. When :meth:`from_env` is used, this value is populated from
        ``OPENAI_API_KEY`` but may be left empty if you only use other
        providers in a given environment.
    base_url:
        Default base URL for OpenAI-compatible endpoints. For OpenAI itself,
        this is usually ``"https://api.openai.com/v1"`` but can be overridden
        via ``OPENAI_BASE_URL``. Other providers (DeepSeek, Moonshot, Zhipu,
        xAI, etc.) will use provider-specific defaults from the registry and
        may be overridden via their own environment variables.
    default_model_alias:
        Logical alias used to look up a :class:`ModelConfig` in the model
        registry when callers do not explicitly specify a model, e.g.
        ``"balanced"`` or ``"explainer"``.
    timeout_seconds:
        Network timeout for the underlying HTTP requests in seconds.
    """

    api_key: str
    base_url: str
    default_model_alias: str = DEFAULT_ALIAS
    timeout_seconds: float = 30.0

    # --------------------------------------------------------------------- #
    # Constructors
    # --------------------------------------------------------------------- #
    @classmethod
    def from_env(cls, default_model_alias: str = DEFAULT_ALIAS) -> LLMClient:
        """Construct a client by reading configuration from environment vars.

        Environment variables
        ---------------------
        The following variables are *optionally* read. Only the variables for
        the providers you actually use need to be defined:

        - ``OPENAI_API_KEY``             (used for provider="openai")
        - ``OPENAI_BASE_URL``            (default: ``https://api.openai.com/v1``)
        - ``GOOGLE_API_KEY``             (used for provider="google")
        - ``GOOGLE_API_BASE_URL``        (default: Gemini v1beta endpoint)
        - ``MOONSHOT_API_KEY``           (used for provider="moonshot")
        - ``MOONSHOT_API_BASE_URL``      (optional override for Kimi)
        - ``DEEPSEEK_API_KEY``           (used for provider="deepseek")
        - ``DEEPSEEK_BASE_URL``          (optional override, OpenAI-compatible)
        - ``ZHIPU_API_KEY``              (used for provider="zhipu")
        - ``ZHIPU_BASE_URL``             (optional override)
        - ``XAI_API_KEY``                (used for provider="xai")
        - ``XAI_BASE_URL``               (optional override)

        Only the OpenAI-related variables are stored directly on the client as
        defaults because they are the most commonly used. Other providers look
        up their keys lazily inside :meth:`generate`.

        Returns
        -------
        LLMClient
            A configured client instance ready to make requests for any
            provider whose API key is present in the environment.
        """
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return cls(
            api_key=api_key,
            base_url=base_url,
            default_model_alias=default_model_alias,
        )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a single text completion from the given chat messages.

        Parameters
        ----------
        messages:
            A sequence of chat-style messages, each with at least
            ``{"role": "...", "content": "..."}``. The structure is deliberately
            kept simple so that both agents and tests can construct it without
            extra helper classes.
        model:
            Optional logical alias or concrete provider model ID. If omitted,
            :attr:`default_model_alias` is used to look up a :class:`ModelConfig`
            in the registry.
        temperature:
            Optional override of the model's default sampling temperature.
        max_tokens:
            Optional override of the model's default ``max_tokens`` limit.

        Returns
        -------
        str
            The text content of the first candidate / choice in the response.

        Raises
        ------
        RuntimeError
            If the HTTP request fails, if a required API key is missing, or if
            the response payload cannot be parsed into a text string.
        """
        config: ModelConfig = get_model(model or self.default_model_alias)

        # Resolve effective sampling parameters with simple overrides.
        effective_temperature = float(
            temperature if temperature is not None else config.temperature
        )
        effective_max_tokens = int(max_tokens if max_tokens is not None else config.max_tokens)

        provider = config.provider.lower().strip()

        if provider == "google":
            # Gemini "generateContent" protocol.
            response = self._generate_gemini(
                config=config,
                messages=messages,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
            )
            return self._extract_content_gemini(response)

        # All other providers are assumed to be OpenAI-compatible.
        response = self._generate_openai_compatible(
            config=config,
            messages=messages,
            temperature=effective_temperature,
            max_tokens=effective_max_tokens,
        )
        return self._extract_content_openai(response)

    # --------------------------------------------------------------------- #
    # Provider-specific helpers
    # --------------------------------------------------------------------- #
    def _generate_openai_compatible(
        self,
        *,
        config: ModelConfig,
        messages: Sequence[Mapping[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Call an OpenAI-compatible Chat Completions endpoint.

        This helper supports providers that expose a ``/chat/completions``
        endpoint using the standard OpenAI request/response format, including
        OpenAI itself, DeepSeek, Moonshot (Kimi), Zhipu GLM, and xAI Grok.

        Authentication and base URL selection
        -------------------------------------
        For each provider, the helper looks up the API key and base URL using
        the following precedence:

        1. Provider-specific environment variable (e.g. ``DEEPSEEK_API_KEY``),
        2. Registry default (:class:`ModelConfig.base_url`),
        3. Client defaults (:attr:`api_key`, :attr:`base_url`) for OpenAI.

        If no API key can be found for a provider that is actually used, a
        :class:`RuntimeError` is raised.
        """
        provider = config.provider.lower().strip()

        # Map providers to their preferred env var names.
        api_key_env_map: dict[str, str] = {
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "moonshot": "MOONSHOT_API_KEY",
            "zhipu": "ZHIPU_API_KEY",
            "xai": "XAI_API_KEY",
        }
        base_url_env_map: dict[str, str] = {
            "openai": "OPENAI_BASE_URL",
            "deepseek": "DEEPSEEK_BASE_URL",
            "moonshot": "MOONSHOT_API_BASE_URL",
            "zhipu": "ZHIPU_BASE_URL",
            "xai": "XAI_BASE_URL",
        }

        api_key_env = api_key_env_map.get(provider, "OPENAI_API_KEY")
        base_url_env = base_url_env_map.get(provider, "OPENAI_BASE_URL")

        # Resolve API key with provider-specific override first.
        api_key = os.getenv(api_key_env, "")
        if not api_key and provider == "openai":
            # Fall back to the client-level default for OpenAI.
            api_key = self.api_key

        if not api_key:
            raise RuntimeError(
                f"Missing API key for provider '{provider}'. "
                f"Expected environment variable '{api_key_env}' to be set."
            )

        # Resolve base URL: env override → registry default → client default.
        base_url = os.getenv(base_url_env) or config.base_url or self.base_url
        base_url = base_url.rstrip("/")

        url = base_url + "/chat/completions"

        payload: MutableMapping[str, Any] = {
            "model": config.name,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        return self._post(url=url, headers=headers, payload=payload)

    def _generate_gemini(
        self,
        *,
        config: ModelConfig,
        messages: Sequence[Mapping[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Call a Google Gemini `generateContent` endpoint.

        This helper performs the minimal mapping between our chat-style
        messages and the Gemini ``contents`` format, and then calls:

            POST {base_url}/models/{model}:generateContent

        using an API key supplied via ``GOOGLE_API_KEY``.

        Notes
        -----
        - We only support simple text parts at this layer; more advanced
          multimodal features (images, audio, tools) can be added later as
          needed.
        """
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY; cannot call Google Gemini models.")

        base_url = (
            os.getenv("GOOGLE_API_BASE_URL")
            or config.base_url
            or "https://generativelanguage.googleapis.com/v1beta"
        )
        base_url = base_url.rstrip("/")

        url = f"{base_url}/models/{config.name}:generateContent"

        contents = [
            {
                "role": m["role"],
                "parts": [{"text": m["content"]}],
            }
            for m in messages
        ]

        payload: MutableMapping[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

        return self._post(url=url, headers=headers, payload=payload)

    # --------------------------------------------------------------------- #
    # Internal helpers (test seams)
    # --------------------------------------------------------------------- #
    def _post(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Perform an HTTP POST request and decode the JSON response.

        This method is intentionally kept small and synchronous, and serves as
        the main seam for unit tests: tests can monkeypatch or patch-object
        :meth:`_post` to return a stubbed response without performing any real
        network I/O.

        Parameters
        ----------
        url:
            Fully-qualified URL including scheme, host, and path.
        headers:
            HTTP headers to attach to the request, including any
            provider-specific authentication headers.
        payload:
            JSON-serialisable request body.

        Returns
        -------
        dict
            Parsed JSON response body.

        Raises
        ------
        RuntimeError
            If the HTTP request fails for any reason, or if the response body
            cannot be decoded as JSON.
        """
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body,
            headers=dict(headers),
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            # Best-effort extraction of provider error message for easier debugging.
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {exc.reason}; body={detail!r}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM network error: {exc}") from exc

        try:
            decoded: dict[str, Any] = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Failed to decode LLM response as JSON") from exc

        return decoded

    # --------------------------------------------------------------------- #
    # Response extraction helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _extract_content_openai(response: Mapping[str, Any]) -> str:
        """Extract ``choices[0].message.content`` from a Chat Completions payload.

        This follows the standard OpenAI-compatible response format, which is
        also used by DeepSeek, Moonshot (Kimi), Zhipu GLM, xAI Grok, and other
        providers that emulate the OpenAI API.

        Parameters
        ----------
        response:
            A dictionary representing the JSON body returned by the provider.

        Returns
        -------
        str
            The first completion's content string.

        Raises
        ------
        RuntimeError
            If the expected fields are missing or malformed.
        """
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM response has no choices; cannot extract content.")

        first = choices[0]
        message = first.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("LLM response choice[0].message is missing or invalid.")

        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise RuntimeError("LLM response choice[0].message.content is empty.")

        return content

    @staticmethod
    def _extract_content_gemini(response: Mapping[str, Any]) -> str:
        """Extract text from a Gemini `generateContent` response.

        We follow the documented Gemini response shape:

        .. code-block:: json

            {
              "candidates": [
                {
                  "content": {
                    "parts": [
                      { "text": "..." }
                    ]
                  }
                }
              ]
            }

        Parameters
        ----------
        response:
            A dictionary representing the JSON body returned by the Gemini
            ``generateContent`` endpoint.

        Returns
        -------
        str
            The concatenated text from the first candidate's content parts.

        Raises
        ------
        RuntimeError
            If the expected fields are missing or malformed.
        """
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError("Gemini response has no candidates; cannot extract content.")

        first = candidates[0]
        content = first.get("content")
        if not isinstance(content, Mapping):
            raise RuntimeError("Gemini response candidates[0].content is missing or invalid.")

        parts = content.get("parts")
        if not isinstance(parts, list) or not parts:
            raise RuntimeError("Gemini response candidates[0].content.parts is empty.")

        texts = []
        for part in parts:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                texts.append(part["text"])

        if not texts:
            raise RuntimeError(
                "Gemini response parts contain no text fields; cannot extract content."
            )

        return "".join(texts)


__all__ = ["LLMClient"]
