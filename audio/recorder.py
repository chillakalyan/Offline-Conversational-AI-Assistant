"""
audio/recorder.py — Microphone recording using sounddevice.

Stage 1 fixes applied:
  CR-1  — sd.rec() / sd.wait() wrapped in try/except with custom
           AudioInputError so hardware failures never crash the app (B-1)
  B-8   — explicit subtype="PCM_16" for soundfile write to avoid
           ambiguous float32 WAV format (still 16 kHz mono for Whisper)
  B-6   — recordings dir created from config; path returned for caller use
  D-6   — replaced bare print() with logging
  M-5   — sample rate, channels, duration and dir from config.py
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import sounddevice as sd
import soundfile as sf

import config

logger = logging.getLogger(__name__)


class AudioInputError(RuntimeError):
    """Raised when microphone recording fails due to a device problem."""


def prune_old_recordings(
    directory: Optional[Path] = None,
    max_keep: int = config.MAX_RECORDINGS_KEPT,
) -> int:
    """
    Remove oldest WAV files in *directory* keeping at most *max_keep* files.
    Returns the number of files deleted.
    """
    record_dir = directory or config.RECORDINGS_DIR
    if not record_dir.exists():
        return 0

    wav_files = sorted(
        record_dir.glob("*.wav"),
        key=lambda p: p.stat().st_mtime,
    )
    num_to_delete = max(0, len(wav_files) - max_keep)
    deleted = 0
    for file_path in wav_files[:num_to_delete]:
        try:
            file_path.unlink()
            deleted += 1
            logger.debug("Pruned old recording: %s", file_path.name)
        except OSError as exc:
            logger.warning("Could not delete old recording %s: %s", file_path, exc)
    return deleted


def record_audio(duration: Optional[int] = None) -> str:
    """
    Record *duration* seconds of mono audio from the default microphone.

    Returns the absolute path to the saved WAV file.

    Raises:
        AudioInputError — microphone is unavailable or recording fails.
    """
    rec_duration = duration if duration is not None else config.AUDIO_RECORD_DURATION
    recordings_dir: Path = config.RECORDINGS_DIR
    recordings_dir.mkdir(parents=True, exist_ok=True)

    filename: Path = recordings_dir / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
    )

    logger.info("🎤 Recording for %d seconds…", rec_duration)

    try:
        audio = sd.rec(
            int(rec_duration * config.AUDIO_SAMPLE_RATE),
            samplerate=config.AUDIO_SAMPLE_RATE,
            channels=config.AUDIO_CHANNELS,
            dtype=config.AUDIO_DTYPE,
        )
        sd.wait()
    except sd.PortAudioError as exc:
        raise AudioInputError(
            f"Microphone recording failed: {exc}\n"
            "Check that a microphone is connected and not in use by another app."
        ) from exc
    except Exception as exc:
        raise AudioInputError(
            f"Unexpected audio recording error: {exc}"
        ) from exc

    # Validate that we actually captured something non-trivial
    if audio is None or audio.size == 0:
        raise AudioInputError("Recorder returned empty audio data.")

    # Audio engineering (Finding A-5): Remove DC offset and guard clipping
    import numpy as np
    audio = audio - np.mean(audio, axis=0)
    audio = np.clip(audio, -1.0, 1.0)

    sf.write(
        str(filename),
        audio,
        config.AUDIO_SAMPLE_RATE,
        subtype="PCM_16",   # explicit 16-bit PCM — unambiguous for Whisper
    )

    logger.info("Recording saved → %s", filename)

    # Prune old recordings to avoid disk bloat (B-6, L-7)
    prune_old_recordings(recordings_dir)

    return str(filename)