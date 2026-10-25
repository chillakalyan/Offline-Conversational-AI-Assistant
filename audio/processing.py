"""
audio/processing.py — Utilities for handling browser-provided audio arrays.

Gradio's microphone component delivers audio as (sample_rate, numpy_array).
This module normalises, resamples, and writes that to a temporary WAV file
suitable for Faster Whisper, without requiring a sounddevice recording.

Phase C implementation — browser audio integration.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

import config

logger = logging.getLogger(__name__)


class AudioProcessingError(Exception):
    """Raised when audio cannot be preprocessed for transcription."""


def validate_audio(sample_rate: int, audio: np.ndarray) -> None:
    """
    Raise AudioProcessingError for empty, silent, or invalid audio.

    Does NOT raise for very quiet but non-zero audio — that is Whisper's job.
    """
    if audio is None or audio.size == 0:
        raise AudioProcessingError("No audio data received.")
    if sample_rate <= 0:
        raise AudioProcessingError(f"Invalid sample rate: {sample_rate}")
    duration_s = audio.shape[0] / sample_rate
    if duration_s < 0.1:
        raise AudioProcessingError(
            f"Recording too short ({duration_s:.2f} s). Please speak for at least 0.1 s."
        )


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert any channel count and dtype to mono float32 in [-1, 1]."""
    # Ensure float32
    arr = audio.astype(np.float32)
    # Normalise integer ranges
    if audio.dtype == np.int16:
        arr = arr / 32768.0
    elif audio.dtype == np.int32:
        arr = arr / 2147483648.0
    # Convert to mono
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return arr


def resample_if_needed(audio: np.ndarray, src_rate: int,
                       target_rate: int = config.AUDIO_SAMPLE_RATE) -> np.ndarray:
    """
    Resample *audio* from *src_rate* to *target_rate* using scipy.

    If rates match, returns the original array unchanged.
    """
    if src_rate == target_rate:
        return audio
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(src_rate, target_rate)
        up, down = target_rate // g, src_rate // g
        resampled = resample_poly(audio, up, down)
        logger.info("Resampled %d Hz → %d Hz (%d → %d samples)",
                    src_rate, target_rate, len(audio), len(resampled))
        return resampled.astype(np.float32)
    except Exception as exc:
        raise AudioProcessingError(f"Resampling failed: {exc}") from exc


def browser_audio_to_wav(
    gradio_audio: Optional[Tuple[int, np.ndarray]],
) -> str:
    """
    Convert a Gradio audio tuple (sample_rate, numpy_array) to a temp WAV file.

    Returns the absolute path of the written WAV file.
    The caller is responsible for deleting the file after use.

    Raises AudioProcessingError on empty, too-short, or invalid input.
    """
    if gradio_audio is None:
        raise AudioProcessingError("No audio provided. Please record or upload audio.")

    sample_rate, audio = gradio_audio
    validate_audio(sample_rate, audio)

    audio_mono = to_mono_float32(audio)
    audio_resampled = resample_if_needed(audio_mono, sample_rate,
                                         config.AUDIO_SAMPLE_RATE)

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".wav", dir=tempfile.gettempdir()
    ) as tmp:
        wav_path = tmp.name

    sf.write(wav_path, audio_resampled, config.AUDIO_SAMPLE_RATE, subtype="PCM_16")
    logger.debug("Browser audio → %s  (%.2f s @ %d Hz)",
                 wav_path, len(audio_resampled) / config.AUDIO_SAMPLE_RATE,
                 config.AUDIO_SAMPLE_RATE)
    return wav_path
