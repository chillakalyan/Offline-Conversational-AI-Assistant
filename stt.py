# from faster_whisper import WhisperModel


# class SpeechToText:
#     def __init__(self, model_size="small"):
#         print("Loading Whisper model... (first run may take a while)")
#         self.model = WhisperModel(
#             model_size,
#             device="cpu",
#             compute_type="int8"
#         )

#     def transcribe(self, audio_file):
#         segments, info = self.model.transcribe(
#             audio_file,
#             language="en",
#             beam_size=5,
            
#         )

#         print(f"Language: {info.language} ({info.language_probability:.2f})")

#         transcript = " ".join(segment.text.strip() for segment in segments)

#         return transcript

import time
from faster_whisper import WhisperModel


class SpeechToText:
    def __init__(self, model_size="small"):
        print("Loading Whisper model...")
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )

    def transcribe(self, audio_file):

        start = time.time()

        segments, info = self.model.transcribe(
            audio_file,
            language="en",
            beam_size=5,
        )

        transcript = " ".join(segment.text.strip() for segment in segments)

        print(f"Language: {info.language} ({info.language_probability:.2f})")
        print(f"Transcription Time: {time.time() - start:.2f} seconds")

        return transcript