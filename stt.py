"""
stt.py — Speech-to-Text using Faster Whisper.

Stage 1 fixes applied:
  A-1  — model_size default now matches config.py (single source of truth)
  A-3  — removed the commented-out dead code block
  D-6  — replaced bare print() with logging
  M-5  — reads model config from config.py
"""

import logging
import time
from typing import Optional

from faster_whisper import WhisperModel

import config

logger = logging.getLogger(__name__)


def resolve_asr_language(language: Optional[str] = None) -> Optional[str]:
    """
    Map a UI/console language choice to the Faster Whisper `language=` argument.

    - Explicit codes such as "hi" / "en" are forwarded unchanged.
    - "auto" enables Whisper language detection (passes None).
    - language=None (console default) uses config.WHISPER_LANGUAGE, unless that
      value is also "auto".
    """
    if language == "auto":
        return None
    if language is None:
        cfg = (config.WHISPER_LANGUAGE or "").strip()
        if cfg in ("", "auto"):
            return None
        return cfg
    return language


class SpeechToText:
    """Wraps a Faster Whisper model for offline speech recognition."""

    def __init__(self, model_size: str = config.WHISPER_MODEL_SIZE) -> None:
        logger.info("Loading Whisper model '%s' (device=%s, compute=%s)…",
                    model_size, config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE)
        if str(model_size).endswith(".en"):
            logger.warning(
                "Whisper model '%s' is English-only. Hindi and other languages "
                "require a multilingual checkpoint such as 'base' (not 'base.en').",
                model_size,
            )
        self.model = WhisperModel(
            model_size,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        self.last_language: Optional[str] = None
        self.last_language_probability: Optional[float] = None
        logger.info("Whisper model ready.")

    def transcribe(self, audio_file: str, language: Optional[str] = None) -> str:
        """
        Transcribe *audio_file* and return the transcript string.

        Returns an empty string when no speech is detected.
        Never raises on an empty audio file.
        """
        start = time.time()
        selected_lang = resolve_asr_language(language)

        segments, info = self.model.transcribe(
            audio_file,
            language=selected_lang,
            beam_size=config.WHISPER_BEAM_SIZE,
            vad_filter=config.WHISPER_VAD_FILTER,
        )

        detected_language = getattr(info, "language", selected_lang)
        probability = getattr(info, "language_probability", 1.0)
        self.last_language = detected_language
        self.last_language_probability = probability

        supported = set(config.SUPPORTED_ASR_LANGUAGES)
        detected_code = (detected_language or "").split("-")[0].lower()
        if (
            config.WHISPER_MIN_LANGUAGE_PROB > 0.0
            and probability < config.WHISPER_MIN_LANGUAGE_PROB
            and detected_code not in supported
            and selected_lang is None
        ):
            logger.warning(
                "Transcript rejected: language_probability %.2f < threshold %.2f "
                "(detected=%s)",
                probability,
                config.WHISPER_MIN_LANGUAGE_PROB,
                detected_language,
            )
            return ""

        transcript = " ".join(segment.text.strip() for segment in segments)

        elapsed = time.time() - start
        logger.info(
            "Transcription complete: %.2f s | language=%s (%.2f)",
            elapsed,
            getattr(info, "language", selected_lang),
            getattr(info, "language_probability", 1.0),
        )

        return transcript
