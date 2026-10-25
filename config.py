"""
config.py — Single source of truth for all runtime configuration.

All configurable values that were previously scattered across multiple
source files are consolidated here.  Values can be overridden via
environment variables so no source file edit is required to change a
runtime parameter.

Stage 1 fix for audit findings: D-8, M-5, A-1, H-6, B-7
"""

import os
from pathlib import Path

from languages import ASR_UI_LANGUAGES, discover_piper_voices

# ---------------------------------------------------------------------------
# Project root — resolved relative to THIS file so the app works regardless
# of which directory Python is launched from.  (Audit finding H-6 / B-7)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Whisper / ASR
# ---------------------------------------------------------------------------
# "base" is multilingual. Do not switch to "base.en" / "*.en" — those are English-only.
WHISPER_MODEL_SIZE: str = os.environ.get("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE: str = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE: str = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_BEAM_SIZE: int = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
WHISPER_LANGUAGE: str = os.environ.get("WHISPER_LANGUAGE", "en")
WHISPER_VAD_FILTER: bool = os.environ.get("WHISPER_VAD_FILTER", "true").lower() in ("true", "1", "yes")
WHISPER_MIN_LANGUAGE_PROB: float = float(os.environ.get("WHISPER_MIN_LANGUAGE_PROB", "0.0"))
# Languages the UI advertises and the probability gate will not reject.
# Whisper `base` supports these; `*.en` checkpoints do not.
SUPPORTED_ASR_LANGUAGES = ASR_UI_LANGUAGES

# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE: int = int(os.environ.get("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_CHANNELS: int = int(os.environ.get("AUDIO_CHANNELS", "1"))
AUDIO_DTYPE: str = os.environ.get("AUDIO_DTYPE", "float32")
AUDIO_RECORD_DURATION: int = int(os.environ.get("AUDIO_RECORD_DURATION", "30"))
RECORDINGS_DIR: Path = PROJECT_ROOT / os.environ.get("RECORDINGS_DIR", "recordings")
MAX_RECORDINGS_KEPT: int = int(os.environ.get("MAX_RECORDINGS_KEPT", "20"))

# ---------------------------------------------------------------------------
# TTS (Piper)
# ---------------------------------------------------------------------------
PIPER_EXE: Path = PROJECT_ROOT / os.environ.get("PIPER_EXE", "piper/piper.exe")
PIPER_VOICE_DIR: Path = PROJECT_ROOT / "Voice"
PIPER_VOICE_MODEL: Path = PROJECT_ROOT / os.environ.get(
    "PIPER_VOICE_MODEL", "Voice/en_US-lessac-medium.onnx"
)
# Known default paths (may be missing). Discovered ONNX files override these.
_PIPER_VOICE_DEFAULTS = {
    "en": PIPER_VOICE_MODEL,
    "hi": PROJECT_ROOT / os.environ.get("PIPER_HI_VOICE_MODEL", "Voice/hi_IN-pratham-medium.onnx"),
    "fr": PROJECT_ROOT / "Voice/fr_FR-siwis-medium.onnx",
    "it": PROJECT_ROOT / "Voice/it_IT-riccardo-x_low.onnx",
    "es": PROJECT_ROOT / "Voice/es_ES-davefx-medium.onnx",
    "de": PROJECT_ROOT / "Voice/de_DE-thorsten-medium.onnx",
    "pt": PROJECT_ROOT / "Voice/pt_BR-faber-medium.onnx",
}
PIPER_VOICE_MODELS = dict(_PIPER_VOICE_DEFAULTS)
PIPER_VOICE_MODELS.update(discover_piper_voices(PIPER_VOICE_DIR))
PIPER_TIMEOUT: int = int(os.environ.get("PIPER_TIMEOUT", "30"))   # seconds

# ---------------------------------------------------------------------------
# LLM (Ollama)
# ---------------------------------------------------------------------------
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT: int = int(os.environ.get("OLLAMA_TIMEOUT", "120"))  # seconds

# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------
HISTORY_DIR: Path = PROJECT_ROOT / os.environ.get("HISTORY_DIR", "history")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
