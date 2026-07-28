
from audio import record_audio
from stt import SpeechToText
from llm import LLM
from tts import TextToSpeech
from fallback import FallbackConversation
from history import ConversationHistory

import threading
import time


# ==========================================================
# Load Models Once
# ==========================================================

print("=" * 60)
print("      Offline Conversational AI Assistant")
print("=" * 60)

print("🔄 Loading Faster Whisper...")
stt = SpeechToText(model_size="base")

print("🔄 Loading Ollama LLM...")
llm = LLM(model="llama3.2:1b")

print("🔄 Loading Piper TTS...")
tts = TextToSpeech()

print("🔄 Loading Fallback Conversation...")
fallback = FallbackConversation()

print("🔄 Loading Conversation History...")
history = ConversationHistory()

print("\n✅ Assistant Ready!\n")


# ==========================================================
# Main Conversation Function
# ===============================s===========================

def main():


    # ---------------------------------------------
    # Record User Voice
    # ---------------------------------------------

    audio_file = record_audio(duration=30)
    # audio_file = record_audio()

    # ---------------------------------------------
    # Speech-to-Text
    # ---------------------------------------------

    stt_start = time.time()

    user_text = stt.transcribe(audio_file)

    stt_time = time.time() - stt_start

    print("\n🗣 You:")
    print(user_text)

    # ---------------------------------------------
    # Handle Empty Input
    # ---------------------------------------------

    if not user_text.strip():

        message = "I couldn't hear anything. Please try again."

        print("\n🤖 Assistant:")
        print(message)

        tts.speak(message)

        return

    # ---------------------------------------------
    # Fallback Conversation
    # ---------------------------------------------

    print("\n🤖 Thinking...")

    fallback_message = fallback.get_message()

    print("\n🤖 Assistant:")
    print(fallback_message)

    # Speak fallback while LLM is generating
    fallback_thread = threading.Thread(
        target=tts.speak,
        args=(fallback_message,)
    )

    fallback_thread.start()

    # ---------------------------------------------
    # LLM Response Generation
    # ---------------------------------------------

    llm_start = time.time()

    response = llm.generate(user_text)
    llm_time = time.time() - llm_start
    history.save(user_text, response)

   

    # Wait until fallback finishes speaking
    fallback_thread.join()

    # ---------------------------------------------
    # Final Response
    # ---------------------------------------------

    print("\n🤖 Assistant:")
    print(response)

    print("\n🔊 Speaking...")

    playback_start = time.time()

    tts.speak(response)

    playback_time = time.time() - playback_start

    # ---------------------------------------------
    # Performance Report
    # ---------------------------------------------

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


# ==========================================================
# Run Assistant
# ==========================================================

if __name__ == "__main__":

    while True:

        try:

            main()

        except KeyboardInterrupt:

            print("\n\n👋 Assistant Stopped.")
            break

        except Exception as e:

            print(f"\n❌ Error: {e}")

        choice = input("\nPress Enter to continue or type 'q' to quit: ")

        if choice.lower() == "q":

            print("\n👋 Goodbye!")

            break