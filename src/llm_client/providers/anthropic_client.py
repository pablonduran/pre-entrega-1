"""Cliente asíncrono para Anthropic basado en ``AsyncAnthropic``."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import anthropic
from anthropic import AsyncAnthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_client.base import BaseLLMClient
from llm_client.schemas import ChatMessage, ModelConfig, ModelResponse, ResponseStatus

# Errores transitorios sobre los que tiene sentido reintentar con backoff.
_RETRYABLE = (
    anthropic.RateLimitError,       # límite de tasa / cuota
    anthropic.APIConnectionError,   # fallos de red
    anthropic.APITimeoutError,      # timeouts
    anthropic.InternalServerError,  # 5xx del lado del proveedor
)


class AnthropicClient(BaseLLMClient):
    """Implementación de :class:`BaseLLMClient` para Anthropic (Claude)."""

    def __init__(self, api_key: str, config: ModelConfig) -> None:
        super().__init__(api_key, config)
        self._client = AsyncAnthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def generate(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa con reintentos ante errores transitorios."""
        try:
            return await self._generate_with_retry(messages)
        except Exception as exc:  # noqa: BLE001 — el error se devuelve estructurado
            return ModelResponse.from_exception(
                exc, provider=self.provider_name, model=self.config.model
            )

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    async def _generate_with_retry(
        self, messages: Sequence[ChatMessage]
    ) -> ModelResponse:
        system, conversation = self.split_system(messages)
        kwargs: dict = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "messages": [
                {"role": m.role.value, "content": m.content} for m in conversation
            ],
        }
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        content = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return ModelResponse(
            status=ResponseStatus.SUCCESS,
            content=content,
            model=response.model,
            provider=self.provider_name,
            finish_reason=response.stop_reason,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

    async def generate_stream(
        self, messages: Sequence[ChatMessage]
    ) -> AsyncIterator[str]:
        """Generador asíncrono: emite cada fragmento de texto conforme llega.

        Ante un error durante el stream, emite un mensaje de error legible
        en lugar de romper el loop del consumidor.
        """
        system, conversation = self.split_system(messages)
        kwargs: dict = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "messages": [
                {"role": m.role.value, "content": m.content} for m in conversation
            ],
        }
        if system:
            kwargs["system"] = system

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as exc:  # noqa: BLE001 — error controlado, sin crash
            yield f"\n[ERROR {type(exc).__name__}: {exc}]"
