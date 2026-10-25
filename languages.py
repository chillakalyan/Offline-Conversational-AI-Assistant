"""
Language catalogues for ASR, LLM replies, and installed Piper voices.

Whisper multilingual checkpoints (e.g. `base`, not `base.en`) can transcribe
many languages. Piper can only speak languages for which an ONNX voice file
is present under Voice/. This module never treats a missing voice as installed.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "fr": "French",
    "it": "Italian",
    "es": "Spanish",
    "de": "German",
    "pt": "Portuguese",
}

# Requested UI languages. Whisper `base` supports these (and many more).
ASR_UI_LANGUAGES: Tuple[str, ...] = ("en", "hi", "fr", "it", "es", "de", "pt")

# Example Piper filenames from the public rhasspy/piper voice list.
# These are hints for the user — they are NOT treated as installed.
PIPER_VOICE_HINTS: Dict[str, str] = {
    "en": "en_US-lessac-medium.onnx",
    "hi": "hi_IN-pratham-medium.onnx",
    "fr": "fr_FR-siwis-medium.onnx",
    "it": "it_IT-riccardo-x_low.onnx",
    "es": "es_ES-davefx-medium.onnx",
    "de": "de_DE-thorsten-medium.onnx",
    "pt": "pt_BR-faber-medium.onnx",
}

PIPER_VOICES_DOCS = "https://github.com/rhasspy/piper/blob/master/VOICES.md"


def resolve_response_language(
    response_language: Optional[str],
    detected: Optional[str] = None,
) -> Optional[str]:
    """Pick a concrete reply language. Never substitutes English for another code."""
    if response_language and response_language not in ("auto", ""):
        return response_language.split("-")[0].lower()
    if detected and detected not in ("auto", ""):
        return detected.split("-")[0].lower()
    return None


def language_name(code: Optional[str]) -> str:
    if not code or code == "auto":
        return "automatic detection"
    return LANGUAGE_NAMES.get(code.split("-")[0].lower(), code)


def asr_dropdown_choices() -> List[Tuple[str, str]]:
    choices: List[Tuple[str, str]] = [
        ("Automatic detection (Whisper)", "auto"),
    ]
    for code in ASR_UI_LANGUAGES:
        choices.append((LANGUAGE_NAMES[code], code))
    return choices


def response_dropdown_choices(installed_tts: Sequence[str]) -> List[Tuple[str, str]]:
    installed = {c.split("-")[0].lower() for c in installed_tts}
    choices: List[Tuple[str, str]] = [
        ("Same as recognized speech", "auto"),
    ]
    for code in ASR_UI_LANGUAGES:
        name = LANGUAGE_NAMES[code]
        if code in installed:
            choices.append((f"{name} (text + installed TTS voice)", code))
        else:
            choices.append((f"{name} (text; TTS voice not installed)", code))
    return choices


def piper_lang_from_filename(path: Path) -> Optional[str]:
    """Parse `hi_IN-pratham-medium.onnx` → `hi`."""
    match = re.match(r"^([a-z]{2,3})_[A-Za-z]{2,4}-", path.stem)
    return match.group(1) if match else None


def discover_piper_voices(voice_dir: Path) -> Dict[str, Path]:
    """Return language-code → first matching .onnx in voice_dir."""
    found: Dict[str, Path] = {}
    if not voice_dir.is_dir():
        return found
    for onnx in sorted(voice_dir.glob("*.onnx")):
        code = piper_lang_from_filename(onnx)
        if code and code not in found:
            found[code] = onnx
    return found


def missing_voice_message(code: str, expected_path: Optional[Path] = None) -> str:
    name = language_name(code)
    hint = PIPER_VOICE_HINTS.get(code, f"{code}_XX-<voice>-medium.onnx")
    path_bit = f" Expected path: {expected_path}." if expected_path else ""
    return (
        f"{name} speech output needs a local Piper voice. English TTS is installed "
        f"and will not be used as {name} TTS.{path_bit} "
        f"Download `{hint}` and its matching `.onnx.json` into the Voice/ folder. "
        f"Voice list: {PIPER_VOICES_DOCS}"
    )
