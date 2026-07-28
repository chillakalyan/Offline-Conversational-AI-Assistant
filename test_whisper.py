from faster_whisper import WhisperModel

print("Starting...")

model = WhisperModel("tiny", device="cpu", compute_type="int8")

print("Model loaded successfully!")