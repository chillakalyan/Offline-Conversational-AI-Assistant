"""
tts.py — Text-to-Speech using the Piper binary.

Stage 1 fixes applied:
  CR-2  — tempfile lifecycle fixed: file is created, kept open until Piper
           finishes, then closed; finally block guarantees cleanup even on
           Piper crash (audit finding B-2)
  B-3   — subprocess errors are caught and re-raised with a clear message
  B-7   — Piper executable and voice model paths resolved via config.py
           (PROJECT_ROOT-relative) so CWD does not matter (audit H-6)
  H-3   — winsound replaced with cross-platform sounddevice playback;
           winsound kept as a fallback on Windows only if sounddevice fails
  D-6   — replaced bare print() with logging
  M-5   — reads paths and timeout from config.py
"""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import sounddevice as sd
import soundfile as sf

import config
from languages import missing_voice_message, piper_lang_from_filename

logger = logging.getLogger(__name__)


class PiperNotFoundError(RuntimeError):
    """Raised when the Piper executable or voice model cannot be found."""


class PiperSynthesisError(RuntimeError):
    """Raised when Piper fails to synthesise audio."""


class AudioPlaybackError(RuntimeError):
    """Raised when audio playback fails."""


class TextToSpeech:
    """Synthesises speech offline using the Piper TTS binary."""

    def __init__(
        self,
        piper_path: Optional[Path] = None,
        voice_model: Optional[Path] = None,
    ) -> None:
        self.piper_path: Path = Path(piper_path or config.PIPER_EXE)
        self.voice_model: Path = Path(voice_model or config.PIPER_VOICE_MODEL)
        self._validate_paths()
        logger.info(
            "TTS ready — piper=%s, voice=%s",
            self.piper_path,
            self.voice_model,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_paths(self) -> None:
        """Raise PiperNotFoundError early if required files are missing."""
        if not self.piper_path.exists():
            raise PiperNotFoundError(
                f"Piper executable not found: {self.piper_path}\n"
                "Download Piper from https://github.com/rhasspy/piper/releases\n"
                "and place piper.exe in the 'piper/' subfolder."
            )
        if not self.voice_model.exists():
            raise PiperNotFoundError(
                f"Piper voice model not found: {self.voice_model}\n"
                "Download the .onnx voice file and place it in the 'Voice/' subfolder."
            )

    def installed_voices(self) -> dict:
        """Map language code → ONNX path for voices that actually exist on disk."""
        return {
            code: Path(model)
            for code, model in config.PIPER_VOICE_MODELS.items()
            if Path(model).exists()
        }

    def voice_for_language(self, language: Optional[str]) -> Path:
        """Return an installed language-compatible voice; never fake another language."""
        code = (language or "en").split("-")[0].lower()
        if code in ("", "auto"):
            raise PiperNotFoundError(
                "Select a concrete response language before synthesizing speech. "
                "Automatic mode cannot guess a Piper voice."
            )
        installed = self.installed_voices()
        model = installed.get(code) or config.PIPER_VOICE_MODELS.get(code)
        if model is None:
            raise PiperNotFoundError(missing_voice_message(code))
        model = Path(model)
        if not model.exists():
            raise PiperNotFoundError(missing_voice_message(code, model))
        actual_lang = piper_lang_from_filename(model)
        if actual_lang and actual_lang != code:
            raise PiperNotFoundError(
                f"Voice {model.name} is a '{actual_lang}' model and will not be used for '{code}'."
            )
        return model

    def _play_wav(self, wav_path: Path) -> None:
        """Play a WAV file via sounddevice (cross-platform)."""
        try:
            data, sample_rate = sf.read(str(wav_path))
            sd.play(data, sample_rate)
            sd.wait()
        except Exception as exc:
            # Last-resort Windows fallback so audio still plays on Windows
            if sys.platform == "win32":
                logger.warning(
                    "sounddevice playback failed (%s); falling back to winsound.", exc
                )
                try:
                    import winsound  # noqa: PLC0415
                    winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
                    return
                except Exception as win_exc:
                    raise AudioPlaybackError(
                        f"Both sounddevice and winsound failed.\n"
                        f"sounddevice: {exc}\n"
                        f"winsound: {win_exc}"
                    ) from win_exc
            raise AudioPlaybackError(
                f"Audio playback failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str, language: str = "en") -> None:
        """
        Synthesise *text* with Piper and play it through the default speaker.

        Raises:
            ValueError           — *text* is empty.
            PiperSynthesisError  — Piper process failed.
            AudioPlaybackError   — Playback device error.
        """
        if not text or not text.strip():
            raise ValueError("speak() called with empty text.")

        output_file: Optional[Path] = None
        try:
            # Create a named temp file; keep it open so Piper can write to it.
            # delete=False so we control deletion in the finally block.
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            ) as tmp:
                output_file = Path(tmp.name)
            # File is now closed so Piper can open it on Windows.

            logger.debug("Synthesising %d chars with Piper → %s", len(text), output_file)

            result = subprocess.run(
                [
                    str(self.piper_path),
                    "-m", str(self.voice_for_language(language)),
                    "-f", str(output_file),
                ],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=config.PIPER_TIMEOUT,
            )

            if result.returncode != 0:
                stderr_msg = result.stderr.decode("utf-8", errors="replace").strip()
                raise PiperSynthesisError(
                    f"Piper exited with code {result.returncode}.\n"
                    f"stderr: {stderr_msg or '(empty)'}"
                )

            if not output_file.exists() or output_file.stat().st_size == 0:
                raise PiperSynthesisError(
                    "Piper completed but produced no audio output."
                )

            logger.debug("Piper synthesis OK, playing audio…")
            self._play_wav(output_file)

        except subprocess.TimeoutExpired as exc:
            raise PiperSynthesisError(
                f"Piper did not finish within {config.PIPER_TIMEOUT} seconds."
            ) from exc

        finally:
            # Always clean up the temp file — even when exceptions occur (CR-2)
            if output_file is not None and output_file.exists():
                try:
                    os.remove(output_file)
                except OSError as exc:
                    logger.warning("Could not remove temp WAV %s: %s", output_file, exc)
