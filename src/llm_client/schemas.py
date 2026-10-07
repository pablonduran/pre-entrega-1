"""Esquemas de datos y validación de entrada con Pydantic.

Definir estos modelos primero evita el clásico "error de diccionarios
anidados": todos los mensajes y configuraciones pasan por validación
antes de tocar la API del proveedor.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MessageRole(StrEnum):
    """Roles válidos para un mensaje de chat."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ResponseStatus(StrEnum):
    """Estado de una respuesta del modelo."""

    SUCCESS = "success"
    ERROR = "error"


class ChatMessage(BaseModel):
    """Mensaje individual de una conversación."""

    model_config = ConfigDict(frozen=True)

    role: MessageRole
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("el contenido del mensaje no puede estar vacío")
        return value


class ModelConfig(BaseModel):
    """Configuración validada de los parámetros de generación.

    - ``temperature``: rango 0 a 2.
    - ``max_tokens``: tope de tokens de salida, mayor a 0.
    - ``model``: identificador del modelo.
    """

    model: str = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0, le=128_000)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_retries: int = Field(default=3, ge=0, le=10)

    @field_validator("model")
    @classmethod
    def model_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("el nombre del modelo no puede estar vacío")
        return value.strip()


class ModelResponse(BaseModel):
    """Respuesta estructurada del modelo, incluyendo errores controlados.

    Un error de API nunca rompe el flujo: se devuelve un ``ModelResponse``
    con ``status=ResponseStatus.ERROR`` y el detalle en ``error``.
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    content: str = ""
    model: str = ""
    provider: str = ""
    finish_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    error: str | None = None

    @model_validator(mode="after")
    def error_requires_message(self) -> Self:
        if self.status is ResponseStatus.ERROR and not self.error:
            raise ValueError("una respuesta con error debe incluir el detalle")
        return self

    @classmethod
    def from_exception(cls, exc: Exception, provider: str, model: str) -> "ModelResponse":
        """Construye una respuesta de error controlada a partir de una excepción."""
        return cls(
            status=ResponseStatus.ERROR,
            provider=provider,
            model=model,
            error=f"{type(exc).__name__}: {exc}",
        )
