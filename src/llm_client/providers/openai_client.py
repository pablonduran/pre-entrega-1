"""Cliente asíncrono para OpenAI basado en ``AsyncOpenAI``."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import openai
from openai import AsyncOpenAI
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
    openai.RateLimitError,       # límite de tasa / cuota
    openai.APIConnectionError,   # fallos de red
    openai.APITimeoutError,      # timeouts
    openai.InternalServerError,  # 5xx del lado del proveedor
)


class OpenAIClient(BaseLLMClient):
    """Implementación de :class:`BaseLLMClient` para OpenAI."""

    def __init__(self, api_key: str, config: ModelConfig) -> None:
        super().__init__(api_key, config)
        self._client = AsyncOpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

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
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": m.role.value, "content": m.content} for m in messages
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
        )
        choice = response.choices[0]
        usage = response.usage
        return ModelResponse(
            status=ResponseStatus.SUCCESS,
            content=choice.message.content or "",
            model=response.model,
            provider=self.provider_name,
            finish_reason=choice.finish_reason,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )

    async def generate_stream(
        self, messages: Sequence[ChatMessage]
    ) -> AsyncIterator[str]:
        """Generador asíncrono: emite cada fragmento de texto conforme llega.

        Ante un error durante el stream, emite un mensaje de error legible
        en lugar de romper el loop del consumidor.
        """
        try:
            stream = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": m.role.value, "content": m.content} for m in messages
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:  # noqa: BLE001 — error controlado, sin crash
            yield f"\n[ERROR {type(exc).__name__}: {exc}]"
