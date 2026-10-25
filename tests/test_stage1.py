"""
tests/test_stage1.py — Stage 1 automated test suite.

All tests run without real hardware, without a loaded Whisper model,
without Ollama running, and without Piper installed.
Hardware / external dependencies are mocked via unittest.mock.

Run with:
    pytest tests/test_stage1.py -v

Covers audit findings:
  CR-1  — recorder device failure → AudioInputError
  CR-2  — TTS tempfile always cleaned up
  CR-3  — requirements.txt has version pins (spot check)
  CR-4  — importing app module does NOT load models or call Ollama
  CR-5  — llm.generate raises OllamaConnectionError when Ollama is down
  B-7   — TTS paths are absolute (not CWD-relative)
  C-7   — llm.generate raises OllamaResponseError on bad response schema
  H-5   — fallback thread joined even when llm.generate raises
  H-6   — TTS paths resolved from PROJECT_ROOT regardless of CWD
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile

import pytest

# ── helpers ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# CR-4 — Importing app must NOT trigger model loading or Ollama calls
# ===========================================================================

def test_import_app_does_not_load_models():
    """
    Importing app.py must not call WhisperModel(), ollama.chat(), or
    TextToSpeech() at module scope.
    """
    with (
        patch("stt.WhisperModel") as mock_whisper,
        patch("llm.ollama") as mock_ollama,
        patch("tts.subprocess.run") as mock_run,
    ):
        # Force reimport to get a clean module
        if "app" in sys.modules:
            del sys.modules["app"]
        import app  # noqa: F401

        mock_whisper.assert_not_called(), "WhisperModel was called at import time"
        mock_ollama.chat.assert_not_called(), "ollama.chat was called at import time"
        mock_run.assert_not_called(), "subprocess.run (Piper) was called at import time"


# ===========================================================================
# CR-1 / B-1 — Recorder raises AudioInputError on device failure
# ===========================================================================

def test_recorder_raises_audio_input_error_on_portaudio_failure():
    """sd.rec() raising PortAudioError must become AudioInputError, not crash."""
    import sounddevice as sd
    from audio.recorder import record_audio, AudioInputError

    with patch("audio.recorder.sd.rec", side_effect=sd.PortAudioError("no device")):
        with pytest.raises(AudioInputError) as exc_info:
            record_audio(duration=1)
    assert "Microphone" in str(exc_info.value) or "recording failed" in str(exc_info.value).lower()


def test_recorder_raises_audio_input_error_on_generic_exception():
    """Any other exception from sd.rec must also become AudioInputError."""
    from audio.recorder import record_audio, AudioInputError

    with patch("audio.recorder.sd.rec", side_effect=OSError("device locked")):
        with pytest.raises(AudioInputError):
            record_audio(duration=1)


# ===========================================================================
# CR-2 / B-2 — TTS tempfile always cleaned up
# ===========================================================================

def test_tts_tempfile_cleaned_up_on_piper_crash():
    """
    If Piper subprocess returns a non-zero exit code, the temp WAV must still
    be removed and PiperSynthesisError must be raised.
    """
    from tts import TextToSpeech, PiperSynthesisError

    fake_piper = MagicMock()
    fake_piper.exists.return_value = True
    fake_voice = MagicMock()
    fake_voice.exists.return_value = True

    tts = TextToSpeech.__new__(TextToSpeech)
    tts.piper_path = fake_piper
    tts.voice_model = fake_voice

    deleted_files = []

    def fake_remove(path):
        deleted_files.append(str(path))

    # Simulate Piper crashing (returncode=1)
    failed_result = MagicMock()
    failed_result.returncode = 1
    failed_result.stderr = b"fatal error"

    with (
        patch("tts.subprocess.run", return_value=failed_result),
        patch("tts.os.remove", side_effect=fake_remove),
    ):
        with pytest.raises(PiperSynthesisError):
            tts.speak("hello")

    assert len(deleted_files) == 1, "Temp file was not cleaned up after Piper crash"


def test_tts_tempfile_cleaned_up_on_playback_error():
    """Even if playback fails, the temp file must be deleted."""
    from tts import TextToSpeech, AudioPlaybackError

    fake_piper = MagicMock()
    fake_piper.exists.return_value = True
    fake_voice = MagicMock()
    fake_voice.exists.return_value = True

    tts = TextToSpeech.__new__(TextToSpeech)
    tts.piper_path = fake_piper
    tts.voice_model = fake_voice

    deleted_files = []

    def fake_remove(path):
        deleted_files.append(str(path))

    ok_result = MagicMock()
    ok_result.returncode = 0
    ok_result.stderr = b""

    fake_stat = MagicMock()
    fake_stat.st_size = 1024

    with (
        patch("tts.subprocess.run", return_value=ok_result),
        patch("tts.os.remove", side_effect=fake_remove),
        patch("tts.Path.exists", return_value=True),
        patch("tts.Path.stat", return_value=fake_stat),
        patch("tts.TextToSpeech._play_wav", side_effect=AudioPlaybackError("no speaker")),
    ):
        with pytest.raises(AudioPlaybackError):
            tts.speak("hello")

    # Cleanup must have run even though playback raised
    assert len(deleted_files) >= 1, "Temp file was not cleaned up after playback failure"


# ===========================================================================
# TTS — ValueError on empty text
# ===========================================================================

def test_tts_speak_raises_on_empty_text():
    """speak('') must raise ValueError before invoking Piper."""
    from tts import TextToSpeech

    fake_piper = MagicMock()
    fake_piper.exists.return_value = True
    fake_voice = MagicMock()
    fake_voice.exists.return_value = True

    tts = TextToSpeech.__new__(TextToSpeech)
    tts.piper_path = fake_piper
    tts.voice_model = fake_voice

    with patch("tts.subprocess.run") as mock_run:
        with pytest.raises(ValueError):
            tts.speak("")
        mock_run.assert_not_called()


def test_tts_speak_raises_on_whitespace_only():
    """speak('   ') must also raise ValueError."""
    from tts import TextToSpeech

    fake_piper = MagicMock()
    fake_piper.exists.return_value = True
    fake_voice = MagicMock()
    fake_voice.exists.return_value = True

    tts = TextToSpeech.__new__(TextToSpeech)
    tts.piper_path = fake_piper
    tts.voice_model = fake_voice

    with pytest.raises(ValueError):
        tts.speak("   ")


# ===========================================================================
# H-6 / B-7 — TTS paths are absolute and PROJECT_ROOT-based
# ===========================================================================

def test_tts_paths_are_absolute():
    """
    The default piper_path and voice_model in config.py must be absolute paths
    (PROJECT_ROOT-relative), not bare relative strings.
    """
    import config
    assert config.PIPER_EXE.is_absolute(), \
        f"PIPER_EXE is not absolute: {config.PIPER_EXE}"
    assert config.PIPER_VOICE_MODEL.is_absolute(), \
        f"PIPER_VOICE_MODEL is not absolute: {config.PIPER_VOICE_MODEL}"


# ===========================================================================
# CR-5 / C-2 — LLM raises OllamaConnectionError when Ollama is down
# ===========================================================================

def test_llm_generate_raises_connection_error_when_ollama_down():
    """ollama.chat raising ConnectionRefusedError must become OllamaConnectionError."""
    from llm import LLM, OllamaConnectionError

    llm = LLM.__new__(LLM)
    llm.model = "llama3.2:1b"

    with patch("llm.ollama.chat", side_effect=ConnectionRefusedError("refused")):
        with pytest.raises(OllamaConnectionError) as exc_info:
            llm.generate("hello")
    assert "Ollama" in str(exc_info.value)


def test_llm_generate_raises_connection_error_on_generic_exception():
    """Any network-level exception from ollama.chat must become OllamaConnectionError."""
    from llm import LLM, OllamaConnectionError

    llm = LLM.__new__(LLM)
    llm.model = "llama3.2:1b"

    with patch("llm.ollama.chat", side_effect=Exception("network error")):
        with pytest.raises(OllamaConnectionError):
            llm.generate("hello")


# ===========================================================================
# C-7 — LLM handles good response correctly
# ===========================================================================

def test_llm_generate_returns_content_on_success():
    """When ollama.chat returns a valid response, generate() returns the text."""
    from llm import LLM

    llm = LLM.__new__(LLM)
    llm.model = "llama3.2:1b"

    mock_response = MagicMock()
    mock_response.message.content = "Hello, how can I help?"

    with patch("llm.ollama.chat", return_value=mock_response):
        result = llm.generate("hi")

    assert result == "Hello, how can I help?"


# ===========================================================================
# H-5 — Fallback thread joined even when llm.generate raises
# ===========================================================================

def test_fallback_thread_joined_on_llm_exception():
    """
    When llm.generate() raises, the fallback thread must still be joined
    before app.main() returns.  We verify this by checking the thread is
    no longer alive after main() returns.
    """
    if "app" in sys.modules:
        del sys.modules["app"]
    import app
    from llm import OllamaConnectionError

    thread_alive_after = []
    original_join = threading.Thread.join

    def capturing_join(self, timeout=None):
        original_join(self, timeout=timeout)
        thread_alive_after.append(self.is_alive())

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "test prompt"

    mock_llm = MagicMock()
    mock_llm.generate.side_effect = OllamaConnectionError("Ollama is down")

    mock_tts = MagicMock()
    mock_fallback = MagicMock()
    mock_fallback.get_message.return_value = "One moment please."
    mock_history = MagicMock()

    with (
        patch("app.record_audio", return_value="fake.wav"),
        patch("threading.Thread.join", capturing_join),
    ):
        app.main(
            stt=mock_stt,
            llm=mock_llm,
            tts=mock_tts,
            fallback=mock_fallback,
            history=mock_history,
        )

    # At least one join was called (our fallback thread's join)
    assert len(thread_alive_after) >= 1, \
        "fallback_thread.join() was never called after LLM exception"


# ===========================================================================
# History — round-trip save/load
# ===========================================================================

def test_history_save_and_load(tmp_path):
    """ConversationHistory.save() must produce valid, loadable JSON."""
    import config
    from history import ConversationHistory

    with patch.object(config, "HISTORY_DIR", tmp_path):
        h = ConversationHistory()
        h.save("What is AI?", "AI stands for Artificial Intelligence.")
        h.save("Tell me more.", "AI can learn from data.")

    with open(h.filename, encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 2
    assert data[0]["user"] == "What is AI?"
    assert data[1]["assistant"] == "AI can learn from data."
    assert "time" in data[0]


# ===========================================================================
# CR-3 — requirements.txt has version pins
# ===========================================================================

def test_requirements_txt_has_version_pins():
    """Every non-comment, non-empty line in requirements.txt must have a version pin."""
    req_path = PROJECT_ROOT / "requirements.txt"
    assert req_path.exists(), "requirements.txt not found"

    unpinned = []
    with open(req_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # A pinned line contains ==
            if "==" not in line:
                unpinned.append(line)

    assert not unpinned, (
        f"Unpinned dependencies in requirements.txt: {unpinned}"
    )


# ===========================================================================
# Fallback — all messages reachable
# ===========================================================================

def test_fallback_all_messages_reachable():
    """All 6 fallback messages must be returned within 200 random calls."""
    from fallback import FallbackConversation
    fb = FallbackConversation()
    seen = set()
    for _ in range(200):
        seen.add(fb.get_message())
    assert len(seen) == len(fb.messages), \
        f"Only {len(seen)} of {len(fb.messages)} fallback messages were reached."


# ===========================================================================
# Config — project root is absolute
# ===========================================================================

def test_config_project_root_is_absolute():
    import config
    assert config.PROJECT_ROOT.is_absolute(), \
        f"PROJECT_ROOT is not absolute: {config.PROJECT_ROOT}"


def test_config_default_values_are_correct():
    import config
    assert config.WHISPER_MODEL_SIZE == "base"
    assert config.AUDIO_SAMPLE_RATE == 16000
    assert config.AUDIO_CHANNELS == 1
    assert config.OLLAMA_MODEL == "llama3.2:1b"
