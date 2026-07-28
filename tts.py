import subprocess
import tempfile
import winsound
import os


class TextToSpeech:

    def __init__(self):
        self.piper_path = "piper/piper.exe"
        self.voice_model = "Voice/en_US-lessac-medium.onnx"   # or "voices/..."

    def speak(self, text):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio:
            output_file = audio.name

        subprocess.run(
            [
                self.piper_path,
                "-m",
                self.voice_model,
                "-f",
                output_file
            ],
            input=text.encode("utf-8"),
            check=True
        )

        winsound.PlaySound(output_file, winsound.SND_FILENAME)

        os.remove(output_file)