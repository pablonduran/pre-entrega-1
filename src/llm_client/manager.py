"""AsyncLLMManager: carga dinámicamente el proveedor según configuración."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from typing import ClassVar

from llm_client.base import BaseLLMClient
from llm_client.providers.anthropic_client import AnthropicClient
from llm_client.providers.gemini_client import GeminiClient
from llm_client.providers.openai_client import OpenAIClient
from llm_client.schemas import ChatMessage, ModelConfig, ModelResponse

# Modelos por defecto de cada proveedor.
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "gemini": "gemini-3.6-flash",
}

# Variables de entorno donde se espera la API key de cada proveedor.
API_KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class AsyncLLMManager:
    """Fachada unificada sobre los clientes de LLM.

    Se instancia con el cliente del proveedor elegido y expone la misma
    interfaz para modo normal y streaming, sin importar el proveedor.
    """

    _registry: ClassVar[dict[str, type[BaseLLMClient]]] = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "gemini": GeminiClient,
    }

    def __init__(self, client: BaseLLMClient) -> None:
        self._client = client

    @property
    def provider_name(self) -> str:
        return self._client.provider_name

    @classmethod
    def from_env(cls, provider: str | None = None) -> "AsyncLLMManager":
        """Construye el manager leyendo la configuración de variables de entorno.

        Variables utilizadas:

        - ``LLM_PROVIDER``: ``openai`` | ``anthropic`` | ``gemini`` (default: ``openai``).
        - ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` / ``GEMINI_API_KEY``: clave del proveedor activo.
        - ``LLM_MODEL``: modelo a usar (default según proveedor).
        - ``LLM_TEMPERATURE``, ``LLM_MAX_TOKENS``, ``LLM_TOP_P``: parámetros opcionales.
        """
        provider = (provider or os.getenv("LLM_PROVIDER") or "openai").strip().lower()

        client_cls = cls._registry.get(provider)
        if client_cls is None:
            disponibles = ", ".join(sorted(cls._registry))
            raise ValueError(
                f"Proveedor '{provider}' no soportado. Opciones: {disponibles}"
            )

        env_var = API_KEY_ENV_VARS[provider]
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(
                f"Falta la variable de entorno {env_var} para el proveedor "
                f"'{provider}'. Definila en tu archivo .env."
            )

        config = ModelConfig(
            model=os.getenv("LLM_MODEL", DEFAULT_MODELS[provider]),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            top_p=float(os.getenv("LLM_TOP_P", "1.0")),
        )
        return cls(client=client_cls(api_key=api_key, config=config))

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Delega la generación completa en el cliente activo."""
        return await self._client.generate(messages)

    def generate_stream(
        self, messages: Sequence[ChatMessage]
    ) -> AsyncIterator[str]:
        """Delega el streaming en el cliente activo (generador asíncrono)."""
        return self._client.generate_stream(messages)
