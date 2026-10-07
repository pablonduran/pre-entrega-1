"""Clase base abstracta para clientes de LLM asíncronos."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

from llm_client.schemas import ChatMessage, ModelConfig, ModelResponse


class BaseLLMClient(ABC):
    """Interfaz común e intercambiable para cualquier proveedor de LLM.

    Todas las llamadas son no bloqueantes (async/await) y el streaming se
    expone como un generador asíncrono de fragmentos de texto.
    """

    def __init__(self, api_key: str, config: ModelConfig) -> None:
        self._api_key = api_key
        self.config = config

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nombre legible del proveedor (p. ej. ``openai``)."""

    @abstractmethod
    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa (modo normal).

        Nunca lanza excepciones de red o de proveedor hacia el llamador:
        devuelve un ``ModelResponse`` con ``status=ERROR`` en su lugar.
        """

    @abstractmethod
    def generate_stream(
        self, messages: Sequence[ChatMessage]
    ) -> AsyncIterator[str]:
        """Devuelve un generador asíncrono que emite tokens conforme llegan."""

    @staticmethod
    def split_system(
        messages: Sequence[ChatMessage],
    ) -> tuple[str | None, list[ChatMessage]]:
        """Separa el mensaje ``system`` (si existe) del resto de la conversación.

        Útil para proveedores como Anthropic que reciben el system prompt
        como parámetro independiente.
        """
        from llm_client.schemas import MessageRole

        system_parts = [m.content for m in messages if m.role is MessageRole.SYSTEM]
        conversation = [m for m in messages if m.role is not MessageRole.SYSTEM]
        system = "\n\n".join(system_parts) if system_parts else None
        return system, conversation
