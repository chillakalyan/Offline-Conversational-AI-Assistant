"""
llm.py — Local LLM access via the Ollama Python SDK.

Stage 1 fixes: CR-5, C-2, C-3, C-7, D-6, M-5
Phase D additions:
  - System prompt for LockIN student voice assistant
  - Conversation history injection (multi-turn)
  - generate_with_history() for Gradio interface
  - Streaming support via generate_stream()
"""

import logging
from typing import Generator, List, Optional

import ollama

import config
from languages import language_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — LockIN student voice assistant persona
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are LockIN, a helpful and friendly local voice assistant designed to "
    "support students. You answer questions clearly and concisely. You help with "
    "study topics, scheduling, deadlines, course information, and general student "
    "life queries. You run entirely offline on the student's computer — no internet "
    "or cloud services are used. Keep responses brief and natural for text-to-speech "
    "output: avoid markdown, bullet points, and long lists unless explicitly asked. "
    "Speak directly to the student in a warm, encouraging tone."
)


class OllamaConnectionError(RuntimeError):
    """Raised when the Ollama daemon cannot be reached."""


class OllamaModelError(RuntimeError):
    """Raised when the requested model is not available in Ollama."""


class OllamaResponseError(RuntimeError):
    """Raised when the Ollama response has an unexpected structure."""


# Type alias for Ollama message dicts
Message = dict  # {"role": str, "content": str}


class LLM:
    """Thin wrapper around the Ollama Python SDK for offline LLM inference."""

    def __init__(self, model: str = config.OLLAMA_MODEL) -> None:
        self.model = model
        logger.info("LLM configured: model=%s, timeout=%ds",
                    self.model, config.OLLAMA_TIMEOUT)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def check_connection(self) -> None:
        """
        Verify that Ollama is reachable and the configured model exists.

        Raises OllamaConnectionError or OllamaModelError with a clear,
        actionable message.  Called once at startup.
        """
        logger.info("Checking Ollama connection…")
        try:
            models = ollama.list()
        except Exception as exc:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {config.OLLAMA_HOST}.\n"
                "Make sure Ollama is installed and running:\n"
                "  https://ollama.com\n"
                f"Original error: {exc}"
            ) from exc

        available = [m.model for m in models.models]
        if self.model not in available:
            raise OllamaModelError(
                f"Model '{self.model}' is not available in Ollama.\n"
                f"Available models: {available or ['(none)']}\n"
                f"Pull it with:  ollama pull {self.model}"
            )
        logger.info("Ollama OK — model '%s' is available.", self.model)

    def generate(self, prompt: str) -> str:
        """
        Single-turn generation (console app compatibility).

        Wraps generate_with_history with no prior history.
        """
        return self.generate_with_history(prompt, history=[])

    def generate_with_history(
        self,
        prompt: str,
        history: Optional[List[Message]] = None,
        system_prompt: str = SYSTEM_PROMPT,
        response_language: Optional[str] = None,
    ) -> str:
        """
        Multi-turn generation with conversation history and system prompt.

        Args:
            prompt:        The current user message.
            history:       List of prior {"role", "content"} dicts.
            system_prompt: System-level instruction (defaults to SYSTEM_PROMPT).

        Returns:
            The assistant's response text.
        """
        messages = _build_messages(prompt, history or [], _language_prompt(system_prompt, response_language))
        logger.debug("Sending %d-message conversation to LLM…", len(messages))

        try:
            response = ollama.chat(model=self.model, messages=messages)
        except ollama.ResponseError as exc:
            if "model" in str(exc).lower() and "not found" in str(exc).lower():
                raise OllamaModelError(
                    f"Model '{self.model}' was not found by Ollama.\n"
                    f"Pull it with:  ollama pull {self.model}"
                ) from exc
            raise OllamaConnectionError(
                f"Ollama returned an error: {exc}\n"
                "Make sure the Ollama daemon is running."
            ) from exc
        except Exception as exc:
            raise OllamaConnectionError(
                f"Could not connect to Ollama: {exc}\n"
                "Make sure Ollama is installed and running:\n"
                "  https://ollama.com"
            ) from exc

        content = _extract_content(response)
        logger.debug("LLM response received (%d chars).", len(content))
        return content

    def generate_stream(
        self,
        prompt: str,
        history: Optional[List[Message]] = None,
        system_prompt: str = SYSTEM_PROMPT,
        response_language: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Streamed multi-turn generation — yields text chunks as they arrive.

        Each yielded value is the full accumulated text so far (suitable for
        Gradio's streaming textbox update pattern).
        """
        messages = _build_messages(prompt, history or [], _language_prompt(system_prompt, response_language))
        logger.debug("Starting streamed LLM response…")

        try:
            stream = ollama.chat(model=self.model, messages=messages, stream=True)
        except Exception as exc:
            raise OllamaConnectionError(
                f"Could not start streamed response: {exc}"
            ) from exc

        accumulated = ""
        try:
            for chunk in stream:
                delta = ""
                try:
                    delta = chunk.message.content or ""
                except AttributeError:
                    try:
                        delta = chunk.get("message", {}).get("content", "")
                    except Exception:
                        pass
                accumulated += delta
                yield accumulated
        except Exception as exc:
            if accumulated:
                # Partial response already received — surface it but also log
                logger.warning("Stream interrupted after %d chars: %s", len(accumulated), exc)
                return
            raise OllamaConnectionError(
                f"Stream failed before any content was received: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_messages(
    prompt: str,
    history: List[Message],
    system_prompt: str,
) -> List[Message]:
    """Construct the full messages list with system prompt and history."""
    messages: List[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    return messages


def _language_prompt(system_prompt: str, language: Optional[str]) -> str:
    extra = (
        " Use the user's latest message exactly as given — do not translate, "
        "correct, or rewrite their transcript."
    )
    if not language or language == "auto":
        return (
            system_prompt
            + " Reply in the same language as the user's latest message."
            + extra
        )
    name = language_name(language)
    return system_prompt + f" Reply in {name}." + extra


def _extract_content(response) -> str:
    """Defensively extract content from an Ollama response object."""
    try:
        content: Optional[str] = response.message.content
    except AttributeError:
        try:
            content = response["message"]["content"]  # type: ignore[index]
        except (KeyError, TypeError) as exc:
            raise OllamaResponseError(
                f"Unexpected Ollama response structure: {response!r}"
            ) from exc

    if not content:
        raise OllamaResponseError(
            "Ollama returned an empty response. "
            "The model may be loading or out of memory."
        )
    return content
