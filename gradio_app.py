"""
gradio_app.py — LockIN Voice AI Assistant — Gradio browser interface.

Separate entry point from app.py (the console interface).
Reuses all existing backend modules: stt, llm, tts, history, config.

Run with:
    python gradio_app.py

Then open http://127.0.0.1:7860 in your browser.

Design:
- Browser microphone OR file upload → Whisper ASR → Ollama LLM → Piper TTS
- Text-input bypass for testing LLM without audio
- Full conversation history maintained in Gradio State
- Streamed LLM responses displayed word-by-word
- TTS output playback via Gradio Audio component
- All processing on localhost — no cloud, no public sharing

Phase B/C/D implementation.
"""

import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np

# ---------------------------------------------------------------------------
# Logging — must be configured before importing backend modules
# ---------------------------------------------------------------------------
import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend imports — importing these modules does NOT load models
# ---------------------------------------------------------------------------
from audio.processing import browser_audio_to_wav, AudioProcessingError
from languages import (
    asr_dropdown_choices,
    language_name,
    missing_voice_message,
    response_dropdown_choices,
    resolve_response_language,
)
from llm import LLM, OllamaConnectionError, OllamaModelError, OllamaResponseError, SYSTEM_PROMPT
from stt import SpeechToText
from tts import TextToSpeech, PiperNotFoundError, PiperSynthesisError, AudioPlaybackError
from history import ConversationHistory

# ---------------------------------------------------------------------------
# Global component singletons — loaded ONCE when the server starts,
# not at module import time.
# ---------------------------------------------------------------------------
_stt: Optional[SpeechToText] = None
_llm: Optional[LLM] = None
_tts: Optional[TextToSpeech] = None
_history: Optional[ConversationHistory] = None
_startup_errors: dict = {}   # component → error message string


def _load_components() -> None:
    """
    Load all pipeline components.  Called once at server start.
    Failures are recorded but do not crash the UI — each handler
    checks _startup_errors before proceeding.
    """
    global _stt, _llm, _tts, _history

    logger.info("Loading Faster Whisper (%s)…", config.WHISPER_MODEL_SIZE)
    try:
        _stt = SpeechToText()
    except Exception as exc:
        _startup_errors["stt"] = str(exc)
        logger.error("Whisper load failed: %s", exc)

    logger.info("Connecting to Ollama (%s)…", config.OLLAMA_MODEL)
    try:
        _llm = LLM()
        _llm.check_connection()
    except (OllamaConnectionError, OllamaModelError) as exc:
        _startup_errors["llm"] = str(exc)
        logger.error("Ollama connection failed: %s", exc)
    except Exception as exc:
        _startup_errors["llm"] = str(exc)
        logger.error("LLM init failed: %s", exc)

    logger.info("Initialising Piper TTS…")
    try:
        _tts = TextToSpeech()
    except PiperNotFoundError as exc:
        _startup_errors["tts"] = str(exc)
        logger.warning("Piper not found — TTS will be unavailable: %s", exc)
    except Exception as exc:
        _startup_errors["tts"] = str(exc)
        logger.error("TTS init failed: %s", exc)

    try:
        _history = ConversationHistory()
    except Exception as exc:
        logger.warning("History init failed (non-fatal): %s", exc)

    logger.info("Component loading complete. Errors: %s",
                list(_startup_errors.keys()) or "none")


def _status_banner() -> str:
    """Build a startup status string for the UI."""
    lines = []
    checks = {
        "stt": ("🎙️ Whisper ASR", "stt"),
        "llm": (f"🧠 Ollama ({config.OLLAMA_MODEL})", "llm"),
        "tts": ("🔊 Piper TTS", "tts"),
    }
    for key, (label, err_key) in checks.items():
        if err_key in _startup_errors:
            lines.append(f"❌ {label}: {_startup_errors[err_key][:80]}…")
        else:
            lines.append(f"✅ {label}: Ready")

    lines.append("")
    if _tts is not None:
        installed = ", ".join(
            f"{language_name(code)} ({path.name})"
            for code, path in sorted(_tts.installed_voices().items())
        ) or "none"
        lines.append(f"🔊 Installed Piper voices: {installed}")
    lines.append("🖥️  Running locally at http://127.0.0.1:7860")
    lines.append("🔒 No cloud services — fully offline after model download")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: synthesise TTS and return path to WAV
# ---------------------------------------------------------------------------

_last_tts_wav: Optional[str] = None


def _audio_update(
    chat_display,
    chat_history,
    conversation_messages,
    transcript="",
    audio=None,
    status="",
    timing="",
    asr_status="—",
    llm_status="—",
    tts_status="—",
):
    return (
        chat_display,
        chat_history,
        conversation_messages,
        transcript,
        audio,
        status,
        timing,
        asr_status,
        llm_status,
        tts_status,
    )


def _text_update(
    chat_display,
    chat_history,
    conversation_messages,
    audio=None,
    status="",
    timing="",
    asr_status="ASR skipped (text)",
    llm_status="—",
    tts_status="—",
):
    return (
        chat_display,
        chat_history,
        conversation_messages,
        audio,
        status,
        timing,
        asr_status,
        llm_status,
        tts_status,
    )


def _installed_tts_codes() -> list:
    if _tts is None:
        return []
    return list(_tts.installed_voices().keys())


def _synthesise_tts(text: str, language: str = "en") -> Optional[str]:
    """
    Run Piper synthesis and return the path to the output WAV.

    The returned path is valid for playback in Gradio.
    Cleans up previous synthesized file if present to prevent disk accumulation.
    Returns None if TTS is unavailable or synthesis fails.
    """
    global _last_tts_wav
    if _tts is None:
        return None

    # Clean up previous turn's temp WAV if it exists
    if _last_tts_wav and os.path.exists(_last_tts_wav):
        try:
            os.remove(_last_tts_wav)
        except OSError:
            pass
        _last_tts_wav = None

    voice_model = _tts.voice_for_language(language)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        out_path = tmp.name
    import subprocess
    result = subprocess.run(
        [str(_tts.piper_path), "-m", str(voice_model), "-f", out_path],
        input=text.encode("utf-8"),
        capture_output=True,
        timeout=config.PIPER_TIMEOUT,
    )
    if result.returncode != 0 or not Path(out_path).exists() or Path(out_path).stat().st_size == 0:
        try:
            os.remove(out_path)
        except OSError:
            pass
        stderr_msg = result.stderr.decode("utf-8", errors="replace").strip()
        raise PiperSynthesisError(
            f"Piper failed for language '{language}' using voice {voice_model.name}. "
            f"{stderr_msg or 'No audio was produced.'}"
        )
    _last_tts_wav = out_path
    return out_path


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------

def handle_audio_input(
    audio: Optional[Tuple[int, np.ndarray]],
    language: str,
    chat_history: List[dict],
    conversation_messages: List[dict],
    response_language: str = "auto",
):
    """
    Full pipeline: browser audio → ASR → LLM (streamed) → TTS.

    Yields progressive updates so Gradio can show streaming text.
    Yields 7 elements corresponding to:
    (chatbot, chat_history_state, conversation_messages_state, transcript_box,
     audio_output, pipeline_status, timing_display)
    """
    # --- Guard: components ready? ---
    errors = []
    if "stt" in _startup_errors:
        errors.append(f"ASR unavailable: {_startup_errors['stt']}")
    if "llm" in _startup_errors:
        errors.append(f"LLM unavailable: {_startup_errors['llm']}")
    if errors:
        err_msg = "\n".join(errors)
        yield _audio_update(
            chat_history, chat_history, conversation_messages,
            status=f"❌ {err_msg}",
            asr_status="unavailable" if "stt" in _startup_errors else "idle",
            llm_status="unavailable" if "llm" in _startup_errors else "idle",
            tts_status="idle",
        )
        return

    # --- Guard: audio provided? ---
    if audio is None:
        yield _audio_update(
            chat_history, chat_history, conversation_messages,
            status="⚠️ No audio detected. Please record or upload audio first.",
            asr_status="no audio",
        )
        return

    # ------------------------------------------------------------------
    # 1. Convert browser audio → WAV
    # ------------------------------------------------------------------
    wav_path = None
    try:
        yield _audio_update(
            chat_history, chat_history, conversation_messages,
            status="🎙️ Processing audio…",
            asr_status="converting",
        )
        wav_path = browser_audio_to_wav(audio)
    except AudioProcessingError as exc:
        yield _audio_update(
            chat_history, chat_history, conversation_messages,
            status=f"❌ Audio error: {exc}",
            asr_status="failed",
        )
        return

    # ------------------------------------------------------------------
    # 2. ASR
    # ------------------------------------------------------------------
    stt_start = time.time()
    try:
        yield _audio_update(
            chat_history, chat_history, conversation_messages,
            status="📝 Transcribing…",
            asr_status=f"running ({language})",
        )
        user_text = _stt.transcribe(wav_path, language=language)
    except Exception as exc:
        logger.error("STT error: %s", exc)
        yield _audio_update(
            chat_history, chat_history, conversation_messages,
            status=f"❌ Transcription failed: {exc}",
            asr_status="failed",
        )
        return
    finally:
        if wav_path and Path(wav_path).exists():
            try:
                os.remove(wav_path)
            except OSError:
                pass

    stt_time = time.time() - stt_start

    if not user_text.strip():
        yield _audio_update(
            chat_history, chat_history, conversation_messages,
            status="⚠️ Could not detect speech. Please speak more clearly or check your microphone.",
            timing=f"ASR: {stt_time:.2f}s",
            asr_status="no speech",
        )
        return

    detected_language = getattr(_stt, "last_language", None)
    if not isinstance(detected_language, str) or not detected_language:
        detected_language = language if language != "auto" else None
    asr_label = detected_language or language
    yield _audio_update(
        chat_history, chat_history, conversation_messages,
        transcript=user_text,
        status="🧠 Generating response…",
        timing=f"ASR: {stt_time:.2f}s",
        asr_status=f"done ({asr_label}, {stt_time:.2f}s)",
        llm_status="running",
    )

    # ------------------------------------------------------------------
    # 3. LLM (streamed)
    # ------------------------------------------------------------------
    llm_start = time.time()
    accumulated_response = ""

    # Add user message to display history
    new_chat_history = chat_history + [{"role": "user", "content": user_text}]

    try:
        resolved_response_language = resolve_response_language(
            response_language, detected_language
        )
        for partial in _llm.generate_stream(
            user_text, history=conversation_messages,
            response_language=resolved_response_language or "auto",
        ):
            accumulated_response = partial
            streaming_history = new_chat_history + [
                {"role": "assistant", "content": accumulated_response + " ▌"}
            ]
            yield _audio_update(
                streaming_history, new_chat_history, conversation_messages,
                transcript=user_text,
                status="🧠 Streaming response…",
                timing=f"ASR: {stt_time:.2f}s",
                asr_status=f"done ({asr_label}, {stt_time:.2f}s)",
                llm_status="streaming",
            )
    except (OllamaConnectionError, OllamaModelError, OllamaResponseError) as exc:
        logger.error("LLM error: %s", exc)
        yield _audio_update(
            new_chat_history, new_chat_history, conversation_messages,
            transcript=user_text,
            status=f"❌ LLM error: {exc}",
            timing=f"ASR: {stt_time:.2f}s",
            asr_status=f"done ({asr_label}, {stt_time:.2f}s)",
            llm_status="failed",
        )
        return

    llm_time = time.time() - llm_start

    final_chat_history = new_chat_history + [
        {"role": "assistant", "content": accumulated_response}
    ]
    new_conv_messages = conversation_messages + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": accumulated_response},
    ]

    # Save to history log
    if _history:
        try:
            _history.save(user_text, accumulated_response)
        except Exception as exc:
            logger.warning("History save failed: %s", exc)

    # ------------------------------------------------------------------
    # 4. TTS synthesis
    # ------------------------------------------------------------------
    tts_start = time.time()
    tts_output_path = None

    tts_error = ""
    if _tts is None:
        tts_error = "Piper TTS is unavailable. Showing text-only output."
    elif not resolved_response_language:
        tts_error = (
            "No concrete response language was detected, so speech was skipped. "
            "Installed voices: " + ", ".join(_installed_tts_codes() or ["none"])
        )
    else:
        try:
            yield _audio_update(
                final_chat_history, final_chat_history, new_conv_messages,
                transcript=user_text,
                status="🔊 Synthesising speech…",
                timing=f"ASR: {stt_time:.2f}s | LLM: {llm_time:.2f}s",
                asr_status=f"done ({asr_label}, {stt_time:.2f}s)",
                llm_status=f"done ({llm_time:.2f}s)",
                tts_status=f"running ({resolved_response_language})",
            )
            tts_output_path = _synthesise_tts(accumulated_response, resolved_response_language)
        except Exception as exc:
            logger.warning("TTS synthesis failed: %s", exc)
            tts_error = str(exc)

    tts_time = time.time() - tts_start
    total_time = stt_time + llm_time + tts_time
    timing_str = (
        f"ASR: {stt_time:.2f}s | LLM: {llm_time:.2f}s | "
        f"TTS: {tts_time:.2f}s | Total: {total_time:.2f}s"
    )

    if tts_output_path and _tts is not None and resolved_response_language:
        voice_display = Path(_tts.voice_for_language(resolved_response_language)).name
        tts_status = f"done ({resolved_response_language}: {voice_display}, {tts_time:.2f}s)"
    else:
        voice_display = tts_error or "text-only"
        tts_status = f"text-only: {voice_display}"
    yield _audio_update(
        final_chat_history, final_chat_history, new_conv_messages,
        transcript=user_text,
        audio=tts_output_path,
        status=(
            f"✅ Ready | Speech: {detected_language or language} | "
            f"Response: {resolved_response_language or 'auto'} | Voice: {voice_display}"
        ),
        timing=timing_str,
        asr_status=f"done ({asr_label}, {stt_time:.2f}s)",
        llm_status=f"done ({llm_time:.2f}s)",
        tts_status=tts_status,
    )


def handle_text_input(
    text: str,
    chat_history: List[dict],
    conversation_messages: List[dict],
    response_language: str = "auto",
):
    """
    Text-only pipeline: typed input → LLM (streamed) → TTS.
    Bypasses ASR entirely — useful for LLM testing.

    Yields 6 elements corresponding to:
    (chatbot, chat_history_state, conversation_messages_state,
     audio_output, pipeline_status, timing_display)
    """
    if not text or not text.strip():
        yield _text_update(
            chat_history, chat_history, conversation_messages,
            status="⚠️ Please enter a message.",
            llm_status="idle",
        )
        return

    if "llm" in _startup_errors:
        yield _text_update(
            chat_history, chat_history, conversation_messages,
            status=f"❌ LLM unavailable: {_startup_errors['llm']}",
            llm_status="unavailable",
        )
        return

    new_chat_history = chat_history + [{"role": "user", "content": text}]
    accumulated_response = ""
    llm_start = time.time()

    try:
        resolved_response_language = resolve_response_language(response_language, None)
        for partial in _llm.generate_stream(
            text,
            history=conversation_messages,
            response_language=resolved_response_language or "auto",
        ):
            accumulated_response = partial
            streaming_history = new_chat_history + [
                {"role": "assistant", "content": accumulated_response + " ▌"}
            ]
            yield _text_update(
                streaming_history, new_chat_history, conversation_messages,
                status="🧠 Streaming…",
                llm_status="streaming",
            )
    except (OllamaConnectionError, OllamaModelError, OllamaResponseError) as exc:
        logger.error("LLM error: %s", exc)
        yield _text_update(
            new_chat_history, new_chat_history, conversation_messages,
            status=f"❌ LLM error: {exc}",
            llm_status="failed",
        )
        return

    llm_time = time.time() - llm_start

    final_chat_history = new_chat_history + [
        {"role": "assistant", "content": accumulated_response}
    ]
    new_conv_messages = conversation_messages + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": accumulated_response},
    ]

    if _history:
        try:
            _history.save(f"[TEXT] {text}", accumulated_response)
        except Exception:
            pass

    # TTS
    tts_start = time.time()
    tts_output = None
    tts_error = ""
    if not _tts:
        tts_error = "Piper TTS is unavailable. Showing text-only output."
    elif not resolved_response_language:
        tts_error = (
            "Text mode with automatic language has no matching Piper voice until you "
            "pick a response language. Installed voices: "
            + ", ".join(_installed_tts_codes() or ["none"])
        )
    else:
        try:
            tts_output = _synthesise_tts(accumulated_response, resolved_response_language)
        except Exception as exc:
            logger.warning("TTS error: %s", exc)
            tts_error = str(exc)
    tts_time = time.time() - tts_start

    timing = (f"LLM: {llm_time:.2f}s"
              + (f" | TTS: {tts_time:.2f}s" if tts_output else " | TTS: text-only"))
    if tts_output and _tts and resolved_response_language:
        voice_display = Path(_tts.voice_for_language(resolved_response_language)).name
        tts_status = f"done ({resolved_response_language}: {voice_display})"
    else:
        voice_display = tts_error or "text-only"
        tts_status = f"text-only: {voice_display}"
    status = (
        f"✅ Ready | Response: {resolved_response_language or response_language} | "
        f"Voice: {voice_display}"
    )

    yield _text_update(
        final_chat_history, final_chat_history, new_conv_messages,
        audio=tts_output,
        status=status,
        timing=timing,
        llm_status=f"done ({llm_time:.2f}s)",
        tts_status=tts_status,
    )


def handle_clear() -> Tuple:
    """Reset all UI state and conversation memory."""
    logger.info("Conversation cleared by user.")
    return [], [], [], "", None, "✅ Ready", "", "ASR: idle", "LLM: idle", "TTS: idle"


# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------

CUSTOM_THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="indigo",
    neutral_hue="slate",
)

CUSTOM_CSS = """
.status-box { font-family: monospace; font-size: 13px; }
.timing-box { font-family: monospace; font-size: 12px; color: #888; }
.transcript-box { font-style: italic; }
footer { display: none !important; }
"""


def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface."""

    with gr.Blocks(title="LockIN Voice Assistant") as demo:

        # ── Title ──────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 🎓 LockIN Voice Assistant
            **Local · Offline · Private** — Powered by Faster Whisper + Ollama + Piper TTS
            """,
        )

        # ── Startup status ─────────────────────────────────────────────
        with gr.Accordion("🔧 System Status", open=True):
            status_display = gr.Textbox(
                value=_status_banner,
                label="Component Status",
                interactive=False,
                lines=6,
                elem_classes=["status-box"],
            )
            gr.Button("🔄 Refresh Status", size="sm").click(
                fn=_status_banner, outputs=status_display
            )

        # ── Shared state ───────────────────────────────────────────────
        # chat_history: Gradio chatbot messages (display only)
        # conversation_messages: raw {"role","content"} list fed to LLM
        chat_history_state = gr.State([])
        conversation_messages_state = gr.State([])

        # ── Main layout ────────────────────────────────────────────────
        with gr.Row():

            # Left column — input
            with gr.Column(scale=1):
                gr.Markdown("### 🎙️ Voice Input")

                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Record or Upload Audio",
                )

                language_select = gr.Dropdown(
                    choices=asr_dropdown_choices(),
                    value="auto",
                    label="Speech recognition language",
                    info=(
                        "Whisper `base` is multilingual. Select Hindi (or another listed language) "
                        "to force that language; Automatic detection does not rewrite speech as English."
                    ),
                )
                response_language_select = gr.Dropdown(
                    choices=response_dropdown_choices(_installed_tts_codes()),
                    value="auto",
                    label="Assistant response language",
                    info=(
                        "Controls the LLM reply language. Speech playback is only attempted when a "
                        "matching Piper ONNX voice is installed in Voice/."
                    ),
                )

                submit_audio_btn = gr.Button(
                    "▶️ Process Audio",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown("---")
                gr.Markdown("### ⌨️ Text Input (LLM test mode)")

                text_input = gr.Textbox(
                    placeholder="Type a message to test the LLM directly…",
                    label="Text Query",
                    lines=2,
                )
                submit_text_btn = gr.Button("▶️ Send Text", variant="secondary")

                gr.Markdown("---")
                clear_btn = gr.Button("🗑️ Clear Conversation", variant="stop", size="sm")

            # Right column — output
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Conversation")

                chatbot = gr.Chatbot(
                    label="LockIN Assistant",
                    height=400,
                    avatar_images=(None, "🤖"),
                    show_label=True,
                    layout="bubble",
                )

                gr.Markdown("### 📝 Last Transcript")
                transcript_box = gr.Textbox(
                    label="Recognized Speech",
                    interactive=False,
                    lines=2,
                    elem_classes=["transcript-box"],
                    placeholder="Your speech will appear here after processing…",
                )

                gr.Markdown("### 🔊 Assistant Voice Response")
                audio_output = gr.Audio(
                    label="Synthesised Speech",
                    type="filepath",
                    interactive=False,
                    autoplay=True,
                )

        # ── Status bar ─────────────────────────────────────────────────
        with gr.Row():
            pipeline_status = gr.Textbox(
                value="✅ Ready",
                label="Pipeline Status",
                interactive=False,
                lines=2,
            )
            timing_display = gr.Textbox(
                value="",
                label="⏱ Timing",
                interactive=False,
                lines=2,
                elem_classes=["timing-box"],
            )
        with gr.Row():
            asr_status = gr.Textbox(value="ASR: idle", label="ASR", interactive=False)
            llm_status = gr.Textbox(value="LLM: idle", label="LLM", interactive=False)
            tts_status = gr.Textbox(value="TTS: idle", label="TTS", interactive=False)

        # ── Examples ───────────────────────────────────────────────────
        gr.Markdown("### 💡 Example Text Queries")
        gr.Examples(
            examples=[
                ["What courses should I focus on this semester?"],
                ["Explain the difference between supervised and unsupervised learning."],
                ["Help me understand what a Mel spectrogram is."],
                ["What is voice activity detection and why is it important?"],
                ["Give me a study tip for memorising formulas."],
            ],
            inputs=text_input,
        )

        # ── Research info accordion ─────────────────────────────────────
        with gr.Accordion("📚 About This System", open=False):
            gr.Markdown(f"""
            **ASR Model:** Faster Whisper `{config.WHISPER_MODEL_SIZE}` (CPU, int8)
            **LLM:** Ollama `{config.OLLAMA_MODEL}` (local, no internet required)
            **TTS:** Only voices found in `Voice/*.onnx` are used. This machine currently has English `en_US-lessac-medium`. Hindi, French, Italian, Spanish, German, and Portuguese speech require a matching Piper voice (ONNX + JSON) in `Voice/`. English playback is never treated as those languages.
            **Piper voices:** {missing_voice_message("hi")}
            **Audio:** {config.AUDIO_SAMPLE_RATE} Hz mono, 16-bit PCM
            **Server:** Bound to 127.0.0.1 — not accessible outside this machine

            All processing runs locally. No audio, transcripts, or conversations
            are sent to external servers.

            See `ASR_RESEARCH_ROADMAP.md` for future improvement plans.
            """)

        # ── Wire events ────────────────────────────────────────────────
        audio_outputs = [
            chatbot, chat_history_state, conversation_messages_state,
            transcript_box, audio_output, pipeline_status, timing_display,
            asr_status, llm_status, tts_status,
        ]

        submit_audio_btn.click(
            fn=handle_audio_input,
            inputs=[audio_input, language_select,
                    chat_history_state, conversation_messages_state, response_language_select],
            outputs=audio_outputs,
        )

        text_outputs = [
            chatbot, chat_history_state, conversation_messages_state,
            audio_output, pipeline_status, timing_display,
            asr_status, llm_status, tts_status,
        ]

        submit_text_btn.click(
            fn=handle_text_input,
            inputs=[text_input, chat_history_state, conversation_messages_state, response_language_select],
            outputs=text_outputs,
        )

        text_input.submit(
            fn=handle_text_input,
            inputs=[text_input, chat_history_state, conversation_messages_state, response_language_select],
            outputs=text_outputs,
        )

        clear_btn.click(
            fn=handle_clear,
            outputs=[
                chatbot, chat_history_state, conversation_messages_state,
                transcript_box, audio_output, pipeline_status, timing_display,
                asr_status, llm_status, tts_status,
            ],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  LockIN Voice Assistant — Gradio Interface")
    logger.info("=" * 60)

    _load_components()

    demo = build_ui()

    demo.launch(
        server_name="127.0.0.1",   # localhost only — no external exposure
        server_port=7860,
        share=False,               # never enable public share
        inbrowser=True,            # auto-open browser tab
        show_error=True,
        theme=CUSTOM_THEME,
        css=CUSTOM_CSS,
    )
