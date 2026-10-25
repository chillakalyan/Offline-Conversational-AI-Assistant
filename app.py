"""
app.py — Offline Conversational AI Assistant entry point.

Stage 1 fixes applied:
  CR-4  — model initialisation moved OUT of module scope into setup().
           Importing this module no longer loads Whisper, calls Ollama,
           or starts audio hardware. (audit D-7)
  H-5   — fallback thread always joined in a finally block so a crash in
           llm.generate() cannot leak a live thread. (audit C-5)
  CR-5  — LLM connectivity checked at startup via llm.check_connection().
  D-6   — replaced bare print() with logging; emojis kept in INFO messages
           so the console experience is unchanged for end-users.
  M-5   — duration and model names from config.py
  L-5   — fixed comment typo on line 41 of original
"""

import logging
import sys
import threading
import time

import config
from audio import record_audio
from audio.recorder import AudioInputError
from fallback import FallbackConversation
from history import ConversationHistory
from llm import LLM, OllamaConnectionError, OllamaModelError, OllamaResponseError
from stt import SpeechToText
from tts import TextToSpeech, PiperNotFoundError, PiperSynthesisError, AudioPlaybackError

# ---------------------------------------------------------------------------
# Logging setup — level controlled by LOG_LEVEL env var (default: INFO)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Setup — called once; returns all initialised components
# ---------------------------------------------------------------------------

def setup():
    """
    Initialise all pipeline components.

    Returns a dict of ready-to-use objects.
    Raises on unrecoverable startup errors (missing Piper, Ollama down, etc.)
    """
    print("=" * 60)
    print("      Offline Conversational AI Assistant")
    print("=" * 60)

    logger.info("🔄 Loading Faster Whisper (%s)…", config.WHISPER_MODEL_SIZE)
    stt = SpeechToText()

    logger.info("🔄 Connecting to Ollama (%s)…", config.OLLAMA_MODEL)
    llm = LLM()
    llm.check_connection()          # raises OllamaConnectionError/ModelError if down

    logger.info("🔄 Initialising Piper TTS…")
    tts = TextToSpeech()            # raises PiperNotFoundError if exe/voice missing

    fallback = FallbackConversation()
    history = ConversationHistory()

    print("\n✅ Assistant Ready!\n")

    return {
        "stt": stt,
        "llm": llm,
        "tts": tts,
        "fallback": fallback,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Single conversation turn
# ---------------------------------------------------------------------------

def main(stt: SpeechToText, llm: LLM, tts: TextToSpeech,
         fallback: FallbackConversation, history: ConversationHistory) -> None:
    """
    Run one full listen → think → speak cycle.

    All exceptions that are not fatal to the loop are caught and reported
    here so the while-loop in run() can always ask the user to continue.
    """

    # ------------------------------------------------------------------
    # 1. Record voice
    # ------------------------------------------------------------------
    try:
        audio_file = record_audio(duration=config.AUDIO_RECORD_DURATION)
    except AudioInputError as exc:
        logger.error("🎤 Recording failed: %s", exc)
        print(f"\n❌ Could not record audio: {exc}")
        return

    # ------------------------------------------------------------------
    # 2. Speech-to-text
    # ------------------------------------------------------------------
    stt_start = time.time()
    try:
        user_text = stt.transcribe(audio_file)
    except Exception as exc:
        logger.error("STT error: %s", exc)
        print(f"\n❌ Transcription error: {exc}")
        return
    stt_time = time.time() - stt_start

    print(f"\n🗣  You:\n{user_text}")

    # ------------------------------------------------------------------
    # 3. Guard — empty transcript
    # ------------------------------------------------------------------
    if not user_text.strip():
        message = "I couldn't hear anything. Please try again."
        print(f"\n🤖 Assistant:\n{message}")
        try:
            tts.speak(message)
        except (PiperSynthesisError, AudioPlaybackError) as exc:
            logger.warning("TTS failed for empty-input response: %s", exc)
        return

    # ------------------------------------------------------------------
    # 4. Fallback phrase — spoken in background while LLM generates
    # ------------------------------------------------------------------
    fallback_message = fallback.get_message()
    print(f"\n🤖 Thinking…\n🤖 Assistant: {fallback_message}")

    fallback_thread = threading.Thread(
        target=_safe_speak,
        args=(tts, fallback_message),
        daemon=True,
        name="fallback-speaker",
    )
    fallback_thread.start()

    # ------------------------------------------------------------------
    # 5. LLM response — always join fallback thread, even on exception
    # ------------------------------------------------------------------
    response: str = ""
    llm_time: float = 0.0

    try:
        llm_start = time.time()
        conv_history = history.get_history_for_llm()
        response = llm.generate_with_history(user_text, history=conv_history)
        llm_time = time.time() - llm_start
        history.save(user_text, response)
    except (OllamaConnectionError, OllamaModelError, OllamaResponseError) as exc:
        logger.error("LLM error: %s", exc)
        print(f"\n❌ LLM error: {exc}")
        try:
            tts.speak("I am having trouble connecting to the local AI model. Please ensure Ollama is running.")
        except Exception:
            pass
        return
    except Exception as exc:
        logger.error("Unexpected LLM error: %s", exc, exc_info=True)
        print(f"\n❌ Unexpected error: {exc}")
        return
    finally:
        # H-5 — always join so the thread is never leaked
        fallback_thread.join(timeout=30)
        if fallback_thread.is_alive():
            logger.warning("Fallback thread did not finish within 30 s.")

    # ------------------------------------------------------------------
    # 6. Speak final response
    # ------------------------------------------------------------------
    print(f"\n🤖 Assistant:\n{response}")
    print("\n🔊 Speaking…")

    playback_start = time.time()
    try:
        tts.speak(response)
    except (PiperSynthesisError, AudioPlaybackError) as exc:
        logger.error("TTS playback error: %s", exc)
        print(f"\n❌ TTS error: {exc}")
    playback_time = time.time() - playback_start

    # ------------------------------------------------------------------
    # 7. Performance report
    # ------------------------------------------------------------------
    ai_processing = stt_time + llm_time

    print("\n" + "=" * 50)
    print("📊 AI PIPELINE PERFORMANCE")
    print("=" * 50)
    print(f"📝 Speech-to-Text      : {stt_time:.2f} sec")
    print(f"🧠 LLM Response        : {llm_time:.2f} sec")
    print(f"⚡ AI Processing Total : {ai_processing:.2f} sec")
    print("-" * 50)
    print(f"🔊 Audio Playback      : {playback_time:.2f} sec")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_speak(tts: TextToSpeech, text: str) -> None:
    """Speak *text* without letting TTS exceptions escape the thread."""
    try:
        tts.speak(text)
    except Exception as exc:
        logger.warning("Fallback TTS failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def run() -> None:
    """Initialise components and enter the main conversation loop."""
    try:
        components = setup()
    except (OllamaConnectionError, OllamaModelError) as exc:
        print(f"\n❌ Startup failed — Ollama problem:\n{exc}")
        sys.exit(1)
    except PiperNotFoundError as exc:
        print(f"\n❌ Startup failed — Piper problem:\n{exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌ Startup failed: {exc}")
        logger.exception("Startup error")
        sys.exit(1)

    while True:
        try:
            main(**components)
        except KeyboardInterrupt:
            print("\n\n👋 Assistant Stopped.")
            break
        except Exception as exc:
            logger.error("Unhandled error in conversation loop: %s", exc, exc_info=True)
            print(f"\n❌ Error: {exc}")

        choice = input("\nPress Enter to continue or type 'q' to quit: ")
        if choice.strip().lower() == "q":
            print("\n👋 Goodbye!")
            break


# ---------------------------------------------------------------------------
# Entry point — CR-4: NO model loading at module import time
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run()