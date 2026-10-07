# Pre-entrega 1 — Cliente de LLM robusto y asincrono

Cliente de LLM en Python 3.12. Permite usar
OpenAI o Anthropic detrás de una interfaz común, con validación de
entrada mediante Pydantic, streaming de tokens en tiempo real y manejo
controlado de errores de red y rate limiting (con reintentos y backoff
exponencial vía `tenacity`).

## Requisitos

- Python **3.12**
- Una API key de OpenAI y/o Anthropic

## Instalación

```powershell
# 1. Crear y activar el entorno virtual
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # Linux / macOS

# 2. Instalar dependencias y el paquete
pip install -r requirements.txt
pip install -e .

# 3. Configurar variables de entorno
Copy-Item .env.example .env       # Windows
# cp .env.example .env            # Linux / macOS
# Editar .env y completar el API key
```

## Variables de entorno

| Variable | Obligatoria | Descripción |
| -------- | ----------- | ----------- |
| `LLM_PROVIDER` | No (default `openai`) | Proveedor activo: `openai`, `anthropic` o `gemini`. |
| `OPENAI_API_KEY` | Si el proveedor es `openai` | API key de OpenAI. |
| `ANTHROPIC_API_KEY` | Si el proveedor es `anthropic` | API key de Anthropic. |
| `GEMINI_API_KEY` | Si el proveedor es `gemini` | API key de Google Gemini (**gratis**, ver abajo). |
| `LLM_MODEL` | No | Modelo a usar. Defaults: `gpt-4o-mini` (OpenAI), `claude-3-5-haiku-latest` (Anthropic), `gemini-3.6-flash` (Gemini). |
| `LLM_TEMPERATURE` | No (default `0.7`) | Temperatura de generación, validada entre 0 y 2. |
| `LLM_MAX_TOKENS` | No (default `1024`) | Máximo de tokens de salida. |
| `LLM_TOP_P` | No (default `1.0`) | Nucleus sampling, rango (0, 1]. |

### API key gratuita con Google Gemini

Si no tenés clave de OpenAI/Anthropic, Gemini ofrece una API key **gratuita**:

1. Entrá a [Google AI Studio](https://aistudio.google.com).
2. Click en **Get API key** → **Create API key**.
3. En tu `.env` configurá:
   ```
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=tu-clave-generada
   ```

## Ejecución

```powershell
python main.py
```

El script realiza tres pruebas sobre la pregunta *"¿Qué es la entropía?"*:

1. **Modo normal** — respuesta completa con metadatos (modelo, tokens, finish reason).
2. **Modo streaming** — tokens impresos en tiempo real mediante un generador asíncrono (`async for` + `yield`).
3. **Concurrencia** — 3 preguntas en paralelo con `asyncio.gather`, demostrando que el event loop no se bloquea.

## Decisiones de diseño

- **Intercambiabilidad:** `OpenAIClient` y `AnthropicClient` heredan de la clase
  abstracta `BaseLLMClient`; `AsyncLLMManager` selecciona la implementación
  según `LLM_PROVIDER` mediante un registro de clases.
- **Validación:** `ModelConfig` valida `temperature` (0–2), `max_tokens` (> 0) y
  `top_p` (0–1] antes de tocar la API. Los mensajes usan el modelo `ChatMessage`
  en lugar de diccionarios anidados.
- **Resiliencia:** los errores transitorios (rate limit, red, timeout, 5xx) se
  reintentan hasta 3 veces con backoff exponencial (`tenacity`). Si el error
  persiste o no es reintentable (p. ej. API key inválida), se devuelve un
  `ModelResponse` con `status=ERROR` en lugar de lanzar una excepción: el
  programa nunca crashea.
- **Streaming:** `generate_stream()` es un generador asíncrono que hace `yield`
  de cada fragmento de texto. Un error durante el stream se emite como mensaje
  legible sin romper el loop del consumidor.
- **Anthropic y system prompt:** `BaseLLMClient.split_system()` separa el mensaje
  `system` para enviarlo como parámetro independiente, como exige la API de Anthropic.
