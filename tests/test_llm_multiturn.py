"""
tests/test_llm_multiturn.py — Automated tests for multi-turn LLM and streaming.

Tests:
  - SYSTEM_PROMPT inclusion
  - Conversation history message construction
  - generate_with_history()
  - generate_stream()
  - Graceful fallback on connection errors
"""

from unittest.mock import MagicMock, patch
import pytest

from llm import (
    LLM,
    OllamaConnectionError,
    OllamaModelError,
    SYSTEM_PROMPT,
    _build_messages,
)


def test_build_messages_structure():
    """_build_messages includes system prompt, prior history, and user prompt in order."""
    history = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]
    messages = _build_messages("Tell me more.", history, SYSTEM_PROMPT)

    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"] == "What is Python?"
    assert messages[2]["content"] == "Python is a programming language."
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "Tell me more."


def test_generate_with_history_success():
    """generate_with_history invokes ollama.chat and returns response content."""
    llm = LLM()
    mock_response = MagicMock()
    mock_response.message.content = "Sure, I can help with your schedule."

    with patch("llm.ollama.chat", return_value=mock_response) as mock_chat:
        history = [{"role": "user", "content": "Hi"}]
        reply = llm.generate_with_history("Can you help me?", history=history)

        assert reply == "Sure, I can help with your schedule."
        mock_chat.assert_called_once()
        args, kwargs = mock_chat.call_args
        assert kwargs["model"] == llm.model
        sent_messages = kwargs["messages"]
        assert len(sent_messages) == 3


def test_generate_stream_yields_accumulated():
    """generate_stream yields progressive accumulated text chunks."""
    llm = LLM()

    chunk1 = MagicMock()
    chunk1.message.content = "Hello "
    chunk2 = MagicMock()
    chunk2.message.content = "student!"

    with patch("llm.ollama.chat", return_value=[chunk1, chunk2]):
        chunks = list(llm.generate_stream("Hi"))
        assert chunks == ["Hello ", "Hello student!"]


def test_generate_stream_raises_connection_error_on_crash():
    """generate_stream raises OllamaConnectionError if initial stream creation fails."""
    llm = LLM()

    with patch("llm.ollama.chat", side_effect=Exception("Connection refused")):
        with pytest.raises(OllamaConnectionError, match="Could not start streamed response"):
            list(llm.generate_stream("Hi"))
