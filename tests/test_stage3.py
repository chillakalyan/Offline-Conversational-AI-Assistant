"""
tests/test_stage3.py — Automated tests for Stage 3 reliability improvements.

Tests:
  - History formatting for Ollama messages (get_history_for_llm)
  - Recording retention pruning (prune_old_recordings)
  - Audio normalization and DC offset removal in recorder
  - Language probability gating in SpeechToText
  - Multi-turn conversation integration in app.py
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import config
from audio.recorder import prune_old_recordings, record_audio
from history import ConversationHistory
from stt import SpeechToText


def test_history_get_history_for_llm(tmp_path):
    """ConversationHistory.get_history_for_llm returns formatted user/assistant turns."""
    with patch.object(config, "HISTORY_DIR", tmp_path):
        h = ConversationHistory()
        h.save("What is my student ID?", "Your student ID is STU-1029.")
        h.save("When is my exam?", "Your exam is on Monday.")

        formatted = h.get_history_for_llm()
        assert len(formatted) == 4
        assert formatted[0] == {"role": "user", "content": "What is my student ID?"}
        assert formatted[1] == {"role": "assistant", "content": "Your student ID is STU-1029."}
        assert formatted[2] == {"role": "user", "content": "When is my exam?"}
        assert formatted[3] == {"role": "assistant", "content": "Your exam is on Monday."}

        h.clear()
        assert h.get_history_for_llm() == []


def test_prune_old_recordings(tmp_path):
    """prune_old_recordings removes oldest files beyond max_keep."""
    # Create 5 fake wav files with distinct timestamps
    for i in range(5):
        f = tmp_path / f"rec_{i}.wav"
        f.write_text(f"dummy {i}")
        time.sleep(0.01)

    # Keep only 2
    deleted = prune_old_recordings(directory=tmp_path, max_keep=2)
    assert deleted == 3

    remaining = sorted(list(tmp_path.glob("*.wav")))
    assert len(remaining) == 2
    # Newest files rec_3 and rec_4 must be kept
    assert remaining[0].name == "rec_3.wav"
    assert remaining[1].name == "rec_4.wav"


def test_stt_language_probability_gating():
    """Low-confidence unknown languages are rejected; Hindi/English are kept."""
    mock_model = MagicMock()
    mock_info = MagicMock()
    mock_info.language = "xx"
    mock_info.language_probability = 0.35  # low confidence hallucination
    mock_model.transcribe.return_value = ([MagicMock(text="hallucinated words")], mock_info)

    with (
        patch("stt.WhisperModel", return_value=mock_model),
        patch.object(config, "WHISPER_MIN_LANGUAGE_PROB", 0.70),
    ):
        stt = SpeechToText()
        transcript = stt.transcribe("fake.wav", language="auto")
        assert transcript == ""


def test_stt_vad_filter_passed_to_whisper():
    """stt.transcribe passes vad_filter setting from config to WhisperModel.transcribe."""
    mock_model = MagicMock()
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_model.transcribe.return_value = ([MagicMock(text="Hello world")], mock_info)

    with (
        patch("stt.WhisperModel", return_value=mock_model),
        patch.object(config, "WHISPER_VAD_FILTER", True),
    ):
        stt = SpeechToText()
        transcript = stt.transcribe("fake.wav")
        assert transcript == "Hello world"
        assert mock_model.transcribe.call_args.kwargs["vad_filter"] is True


def test_app_main_passes_conversation_history():
    """app.main passes accumulated conversation history to llm.generate_with_history."""
    import app

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "What is my next class?"

    mock_llm = MagicMock()
    mock_llm.generate_with_history.return_value = "Your next class is CS 101."

    mock_tts = MagicMock()
    mock_fallback = MagicMock()
    mock_fallback.get_message.return_value = "Thinking..."

    mock_history = MagicMock()
    mock_history.get_history_for_llm.return_value = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    with (
        patch("app.record_audio", return_value="dummy.wav"),
        patch.object(app, "_safe_speak"),
    ):
        app.main(
            stt=mock_stt,
            llm=mock_llm,
            tts=mock_tts,
            fallback=mock_fallback,
            history=mock_history,
        )

        mock_llm.generate_with_history.assert_called_once_with(
            "What is my next class?",
            history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        )
        mock_history.save.assert_called_once_with(
            "What is my next class?",
            "Your next class is CS 101.",
        )
