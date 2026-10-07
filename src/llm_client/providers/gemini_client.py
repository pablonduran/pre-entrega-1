"""Cliente asíncrono para Google Gemini.

Gemini expone un endpoint compatible con la API de OpenAI, por lo que se
reutiliza ``AsyncOpenAI`` apuntando a la base URL de Google.

Documentación: https://ai.google.dev/gemini-api/docs/openai
"""

from __future__ import annotations

from llm_client.providers.openai_client import OpenAIClient
from llm_client.schemas import ModelConfig

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiClient(OpenAIClient):
    """Implementación de :class:`BaseLLMClient` para Google Gemini.

    Hereda toda la lógica de :class:`OpenAIClient` (reintentos, streaming y
    manejo de errores) y solo cambia el endpoint y el nombre del proveedor.
    """

    def __init__(self, api_key: str, config: ModelConfig) -> None:
        super().__init__(api_key, config)
        # Reemplaza el cliente apuntando al endpoint compatible de Gemini.
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

    @property
    def provider_name(self) -> str:
        return "gemini"
