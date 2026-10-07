"""Script de validación del Unified Async LLM Client.

Carga el archivo .env, instancia el proveedor configurado (OpenAI por
defecto) y realiza una pregunta corta en modo normal y en modo streaming.

Uso:
    python main.py
"""

import asyncio
import sys

from dotenv import load_dotenv

from llm_client import AsyncLLMManager, ChatMessage, MessageRole, ResponseStatus

QUESTION = "¿Qué es la entropía?"


async def main() -> None:
    load_dotenv()

    try:
        manager = AsyncLLMManager.from_env()
    except ValueError as exc:
        print(f"[CONFIG ERROR] {exc}")
        sys.exit(1)

    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="Sos un asistente conciso. Respondé en español en 3 oraciones máximo en texto plano, sin asteriscos ni negritas.",
        ),
        ChatMessage(role=MessageRole.USER, content=QUESTION),
    ]

    # ------------------------------------------------------------------
    # 1. Modo normal (respuesta completa)
    # ------------------------------------------------------------------
    print(f"Proveedor: {manager.provider_name}")
    print(f"\nModo normal\nPregunta: {QUESTION}\n")

    response = await manager.generate(messages)

    if response.status is ResponseStatus.ERROR:
        print(f"[ERROR controlado] {response.error}")
    else:
        print(f"Respuesta: {response.content}")
        print(
            f"\n(modelo={response.model}, finish={response.finish_reason}, "
            f"tokens={response.prompt_tokens}+{response.completion_tokens})"
        )

    # ------------------------------------------------------------------
    # 2. Modo streaming (tokens en tiempo real)
    # ------------------------------------------------------------------
    print(f"\nModo streaming\nPregunta: {QUESTION}\n")

    async for token in manager.generate_stream(messages):
        print(token, end="", flush=True)
    print()

    # ------------------------------------------------------------------
    # 3. Prueba de robustez: varias llamadas concurrentes sin bloqueo
    # ------------------------------------------------------------------
    print("\nConcurrencia\n")

    preguntas = [
        "¿Qué es una API REST?",
        "¿Qué es un generador en Python?",
        "¿Qué es el event loop de asyncio?",
    ]
    resultados = await asyncio.gather(
        *(
            manager.generate([ChatMessage(role=MessageRole.USER, content=p)])
            for p in preguntas
        )
    )
    for pregunta, resultado in zip(preguntas, resultados, strict=True):
        if resultado.status is ResponseStatus.ERROR:
            print(f"• {pregunta}\n  [ERROR controlado] {resultado.error}")
        else:
            resumen = resultado.content.replace("\n", " ")[:120]
            print(f"• {pregunta}\n  {resumen}...")


if __name__ == "__main__":
    asyncio.run(main())
