"""
tests/test_gradio.py — Automated tests for Gradio UI and browser audio processing.

All tests run without physical microphones, browser instances, or live models.
Dependencies are isolated using unittest.mock.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from audio.processing import (
    AudioProcessingError,
    browser_audio_to_wav,
    resample_if_needed,
    to_mono_float32,
    validate_audio,
)
import gradio_app


# ===========================================================================
# Browser Audio Processing Tests (audio/processing.py)
# ===========================================================================

def test_validate_audio_valid():
    """Valid audio array passes without exception."""
    audio = np.zeros(16000, dtype=np.float32)
    validate_audio(16000, audio)


def test_validate_audio_empty():
    """Empty or None audio raises AudioProcessingError."""
    with pytest.raises(AudioProcessingError, match="No audio"):
        validate_audio(16000, np.array([]))


def test_validate_audio_invalid_rate():
    """Negative or zero sample rate raises AudioProcessingError."""
    with pytest.raises(AudioProcessingError, match="Invalid sample rate"):
        validate_audio(0, np.zeros(1000))


def test_validate_audio_too_short():
    """Recordings shorter than 0.1s raise AudioProcessingError."""
    short_audio = np.zeros(500, dtype=np.float32)  # ~0.03s at 16kHz
    with pytest.raises(AudioProcessingError, match="Recording too short"):
        validate_audio(16000, short_audio)


def test_to_mono_float32_stereo():
    """Stereo 2D array is averaged to mono 1D array in float32."""
    stereo = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    mono = to_mono_float32(stereo)
    assert mono.ndim == 1
    assert len(mono) == 2
    assert mono[0] == 0.5
    assert mono[1] == 0.5


def test_to_mono_float32_int16():
    """int16 audio is normalized to [-1.0, 1.0] float32 range."""
    int_audio = np.array([32767, -32768, 0], dtype=np.int16)
    float_audio = to_mono_float32(int_audio)
    assert float_audio.dtype == np.float32
    assert np.isclose(float_audio[0], 1.0, atol=1e-4)
    assert np.isclose(float_audio[1], -1.0, atol=1e-4)
    assert float_audio[2] == 0.0


def test_resample_if_needed_same_rate():
    """Matching sample rate returns identical array without modification."""
    audio = np.ones(1000, dtype=np.float32)
    resampled = resample_if_needed(audio, 16000, 16000)
    assert resampled is audio


def test_resample_if_needed_resamples():
    """Different sample rate correctly resamples length."""
    audio = np.ones(48000, dtype=np.float32)  # 1s at 48kHz
    resampled = resample_if_needed(audio, 48000, 16000)
    assert len(resampled) == 16000


def test_browser_audio_to_wav_roundtrip():
    """browser_audio_to_wav creates a readable WAV file and cleans up."""
    audio = np.zeros(16000, dtype=np.float32)
    wav_path = browser_audio_to_wav((16000, audio))
    try:
        assert Path(wav_path).exists()
        assert Path(wav_path).stat().st_size > 0
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ===========================================================================
# Gradio Handler Tests (gradio_app.py)
# ===========================================================================

def test_load_components_records_errors():
    """_load_components catches startup exceptions into _startup_errors dictionary."""
    with (
        patch("gradio_app.SpeechToText", side_effect=RuntimeError("Whisper missing")),
        patch("gradio_app.LLM", side_effect=RuntimeError("Ollama down")),
        patch("gradio_app.TextToSpeech", side_effect=RuntimeError("Piper missing")),
    ):
        gradio_app._startup_errors.clear()
        gradio_app._load_components()
        assert "stt" in gradio_app._startup_errors
        assert "llm" in gradio_app._startup_errors
        assert "tts" in gradio_app._startup_errors


def test_handle_audio_input_no_audio():
    """handle_audio_input with None audio yields a warning without crashing."""
    gradio_app._startup_errors.clear()
    generator = gradio_app.handle_audio_input(
        audio=None,
        language="en",
        chat_history=[],
        conversation_messages=[],
    )
    results = list(generator)
    assert len(results) == 1
    assert len(results[0]) == 10
    chatbot, state_chat, conv_msgs, transcript, audio_out, status, timing, asr, llm, tts = results[0]
    assert "No audio detected" in status


def test_handle_audio_input_full_pipeline_mocked():
    """handle_audio_input runs through ASR, streaming LLM, and TTS with mocks."""
    gradio_app._startup_errors.clear()

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "What is computer science?"

    mock_llm = MagicMock()
    mock_llm.generate_stream.return_value = iter(["CS is ", "the study of computation."])

    fake_audio = (16000, np.zeros(16000, dtype=np.float32))

    with (
        patch.object(gradio_app, "_stt", mock_stt),
        patch.object(gradio_app, "_llm", mock_llm),
        patch.object(gradio_app, "_tts", None),
        patch("gradio_app.browser_audio_to_wav", return_value="dummy.wav"),
        patch("os.path.exists", return_value=True),
        patch("os.remove") as mock_remove,
    ):
        steps = list(
            gradio_app.handle_audio_input(
                audio=fake_audio,
                language="en",
                chat_history=[],
                conversation_messages=[],
            )
        )
        assert len(steps) >= 3
        # Check final step
        final_step = steps[-1]
        (chatbot, chat_state, conv_msgs, transcript, audio_path, status, timing,
         asr_status, llm_status, tts_status) = final_step
        assert status.startswith("✅ Ready")
        assert transcript == "What is computer science?"
        assert conv_msgs[-1]["content"] == "the study of computation."
        assert len(chat_state) == 2  # user + assistant


def test_handle_text_input_empty():
    """Empty text query returns warning without calling LLM."""
    gradio_app._startup_errors.clear()
    steps = list(gradio_app.handle_text_input("", [], []))
    assert len(steps) == 1
    assert "Please enter a message" in steps[0][4]


def test_handle_text_input_streaming():
    """handle_text_input streams LLM response and updates conversation messages."""
    gradio_app._startup_errors.clear()
    mock_llm = MagicMock()
    mock_llm.generate_stream.return_value = iter(["Hello", " there!"])

    with (
        patch.object(gradio_app, "_llm", mock_llm),
        patch.object(gradio_app, "_tts", None),
    ):
        steps = list(
            gradio_app.handle_text_input(
                text="Hi",
                chat_history=[],
                conversation_messages=[],
            )
        )
        assert len(steps) >= 2
        final_step = steps[-1]
        (chatbot, chat_state, conv_msgs, audio_path, status, timing,
         asr_status, llm_status, tts_status) = final_step
        assert status.startswith("✅ Ready")
        assert conv_msgs[-1]["content"] == " there!"


def test_handle_clear():
    """handle_clear returns empty history and ready status."""
    cleared = gradio_app.handle_clear()
    assert cleared[:7] == ([], [], [], "", None, "✅ Ready", "")
    assert cleared[7:] == ("ASR: idle", "LLM: idle", "TTS: idle")


def test_build_ui():
    """build_ui successfully constructs Gradio Blocks application."""
    demo = gradio_app.build_ui()
    assert demo is not None
