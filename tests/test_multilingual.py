from unittest.mock import MagicMock, patch

import numpy as np

import config
import gradio_app
from languages import discover_piper_voices, resolve_response_language
from llm import LLM
from stt import SpeechToText, resolve_asr_language
from tts import PiperNotFoundError, TextToSpeech


def test_whisper_forwards_selected_hindi_unchanged():
    model = MagicMock()
    info = MagicMock(language="hi", language_probability=0.98)
    model.transcribe.return_value = ([MagicMock(text="नमस्ते")], info)
    with patch("stt.WhisperModel", return_value=model):
        stt = SpeechToText()
        assert stt.transcribe("hindi.wav", language="hi") == "नमस्ते"
    assert model.transcribe.call_args.kwargs["language"] == "hi"
    assert stt.last_language == "hi"


def test_whisper_auto_detection_passes_none():
    model = MagicMock()
    info = MagicMock(language="hi", language_probability=0.9)
    model.transcribe.return_value = ([MagicMock(text="नमस्ते")], info)
    with patch("stt.WhisperModel", return_value=model):
        stt = SpeechToText()
        stt.transcribe("hindi.wav", language="auto")
    assert model.transcribe.call_args.kwargs["language"] is None
    assert stt.last_language == "hi"


def test_probability_gate_does_not_reject_supported_hindi():
    model = MagicMock()
    info = MagicMock(language="hi", language_probability=0.4)
    model.transcribe.return_value = ([MagicMock(text="नमस्ते")], info)
    with patch("stt.WhisperModel", return_value=model), patch.object(config, "WHISPER_MIN_LANGUAGE_PROB", 0.7):
        stt = SpeechToText()
        assert stt.transcribe("hindi.wav", language="hi") == "नमस्ते"


def test_language_specific_assistant_instruction():
    llm = LLM()
    with patch("llm.ollama.chat", return_value=MagicMock(message=MagicMock(content="नमस्ते"))) as chat:
        llm.generate_with_history("नमस्ते", history=[{"role": "assistant", "content": "नमस्ते"}], response_language="hi")
    messages = chat.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system" and "Reply in Hindi" in messages[0]["content"]
    assert messages[-2]["content"] == "नमस्ते"


def test_whisper_model_config_is_multilingual():
    assert not str(config.WHISPER_MODEL_SIZE).endswith(".en")
    for code in ("en", "hi", "fr", "it", "es", "de", "pt"):
        assert code in config.SUPPORTED_ASR_LANGUAGES


def test_resolve_asr_language_does_not_replace_hindi_with_english():
    assert resolve_asr_language("hi") == "hi"
    assert resolve_asr_language("fr") == "fr"
    assert resolve_asr_language("en") == "en"
    assert resolve_asr_language("auto") is None


def test_resolve_response_language_does_not_default_detected_hindi_to_english():
    assert resolve_response_language("auto", "hi") == "hi"
    assert resolve_response_language("fr", "en") == "fr"
    assert resolve_response_language("auto", None) is None


def test_gradio_forwards_hindi_transcript_to_ollama():
    gradio_app._startup_errors.clear()
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "नमस्ते"
    mock_stt.last_language = "hi"
    mock_llm = MagicMock()
    mock_llm.generate_stream.return_value = iter(["नमस्ते"])
    fake_audio = (16000, np.zeros(16000, dtype=np.float32))

    with (
        patch.object(gradio_app, "_stt", mock_stt),
        patch.object(gradio_app, "_llm", mock_llm),
        patch.object(gradio_app, "_tts", None),
        patch("gradio_app.browser_audio_to_wav", return_value="dummy.wav"),
        patch("os.path.exists", return_value=True),
        patch("os.remove"),
    ):
        steps = list(
            gradio_app.handle_audio_input(
                audio=fake_audio,
                language="hi",
                chat_history=[],
                conversation_messages=[],
                response_language="hi",
            )
        )

    mock_stt.transcribe.assert_called()
    assert mock_stt.transcribe.call_args.kwargs["language"] == "hi"
    assert mock_llm.generate_stream.call_args.kwargs["response_language"] == "hi"
    assert mock_llm.generate_stream.call_args.args[0] == "नमस्ते"
    assert steps[-1][3] == "नमस्ते"


def test_voice_selection_and_missing_hindi_voice(tmp_path):
    english = tmp_path / "english.onnx"
    english.touch()
    with patch.dict(config.PIPER_VOICE_MODELS, {"en": english, "hi": tmp_path / "missing-hi.onnx"}):
        tts = object.__new__(TextToSpeech)
        assert tts.voice_for_language("en") == english
        try:
            tts.voice_for_language("hi")
        except PiperNotFoundError as exc:
            assert "Hindi" in str(exc)
            assert "English TTS is installed and will not be used as Hindi TTS" in str(exc)
        else:
            raise AssertionError("missing Hindi voice must not silently fall back to English")

    with patch.dict(config.PIPER_VOICE_MODELS, {"en": english}, clear=True):
        tts = object.__new__(TextToSpeech)
        assert tts.installed_voices() == {"en": english}
        try:
            tts.voice_for_language("ta")
        except PiperNotFoundError as exc:
            assert "ta" in str(exc)
        else:
            raise AssertionError("Tamil must not be claimed without an installed Piper voice")
        try:
            tts.voice_for_language("fr")
        except PiperNotFoundError as exc:
            assert "French" in str(exc)
        else:
            raise AssertionError("French TTS must not be claimed without an installed Piper voice")


def test_whisper_forwards_french_unchanged():
    model = MagicMock()
    info = MagicMock(language="fr", language_probability=0.95)
    model.transcribe.return_value = ([MagicMock(text="Bonjour, ça va ?")], info)
    with patch("stt.WhisperModel", return_value=model):
        stt = SpeechToText()
        assert stt.transcribe("french.wav", language="fr") == "Bonjour, ça va ?"
    assert model.transcribe.call_args.kwargs["language"] == "fr"


def test_language_specific_french_assistant_instruction():
    llm = LLM()
    with patch("llm.ollama.chat", return_value=MagicMock(message=MagicMock(content="Bonjour"))) as chat:
        llm.generate_with_history("Bonjour", history=[], response_language="fr")
    assert "Reply in French" in chat.call_args.kwargs["messages"][0]["content"]
    assert chat.call_args.kwargs["messages"][-1]["content"] == "Bonjour"


def test_piper_lang_from_filename_rejects_non_piper_names(tmp_path):
    from pathlib import Path
    from languages import piper_lang_from_filename
    assert piper_lang_from_filename(Path("fr_FR-siwis-medium.onnx")) == "fr"
    assert piper_lang_from_filename(Path("hi_IN-pratham-medium.onnx")) == "hi"
    assert piper_lang_from_filename(Path("english.onnx")) is None


def test_discover_piper_voices_from_filenames(tmp_path):
    (tmp_path / "en_US-lessac-medium.onnx").touch()
    (tmp_path / "hi_IN-pratham-medium.onnx").touch()
    found = discover_piper_voices(tmp_path)
    assert found["en"].name == "en_US-lessac-medium.onnx"
    assert found["hi"].name == "hi_IN-pratham-medium.onnx"
    assert "fr" not in found
