# Offline Conversational AI Assistant

An **Offline Conversational AI Assistant** built using **Faster Whisper**, **Ollama (Llama 3.2)**, and **Piper Text-to-Speech**. The assistant accepts voice input, converts it into text, generates intelligent responses using a local Large Language Model (LLM), and converts the response back into natural speech. The entire pipeline is designed to run **offline**, ensuring privacy and eliminating dependence on cloud services.

---

## Features

- 🎤 Voice Input (Speech Recording)
- 📝 Speech-to-Text using Faster Whisper
- 🧠 Intelligent Response Generation using Ollama (Llama 3.2)
- 🔊 Text-to-Speech using Piper
- 💬 Fallback Conversation while AI is processing
- 💾 Conversation History Logging
- 🎧 Timestamped Audio Recordings
- 📊 AI Pipeline Performance Monitoring
- 🔒 Fully Offline Execution

---

## Project Architecture

```
User Voice
     │
     ▼
Speech Recording
     │
     ▼
Speech-to-Text (Faster Whisper)
     │
     ▼
Large Language Model (Ollama - Llama 3.2)
     │
     ▼
Text-to-Speech (Piper)
     │
     ▼
Voice Response
```

---

## Project Structure

```
Offline-Conversational-AI-Assistant/

├── audio/
│   ├── recorder.py
│   ├── player.py
│   └── audio_manager.py
│
├── piper/
├── Voice/
├── models/
│
├── app.py
├── stt.py
├── llm.py
├── tts.py
├── fallback.py
├── history.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- Python 3.11+
- Faster Whisper
- Ollama
- Llama 3.2 (1B)
- Piper TTS
- SoundDevice
- SoundFile
- NumPy

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/offline-conversational-ai-assistant.git

cd offline-conversational-ai-assistant
```

---

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Install Ollama

Download and install Ollama from:

https://ollama.com

---

### 5. Download the Llama Model

```bash
ollama pull llama3.2:1b
```

---

### 6. Run the Assistant

```bash
python app.py
```

---

## How It Works

1. The assistant records the user's voice.
2. Faster Whisper converts speech into text.
3. Ollama generates an intelligent response.
4. A fallback message keeps the user engaged while the response is being generated.
5. Piper converts the generated response into speech.
6. The assistant plays the response aloud.
7. Conversation history and recordings are stored locally.

---

## AI Pipeline

```
Speech Input
      │
      ▼
Faster Whisper
      │
      ▼
Ollama Llama 3.2
      │
      ▼
Piper Text-to-Speech
      │
      ▼
Audio Output
```

---

## Performance Monitoring

The assistant reports:

- Speech-to-Text Processing Time
- LLM Response Time
- Total AI Processing Time
- Audio Playback Time

Example:

```
==================================================
📊 AI PIPELINE PERFORMANCE
==================================================
Speech-to-Text : 2.45 sec
LLM Response   : 8.73 sec
AI Total       : 11.18 sec
Audio Playback : 7.21 sec
==================================================
```

---

## AI Usage

This project uses the following AI technologies:

- **Faster Whisper** for Speech-to-Text
- **Ollama (Llama 3.2)** as the Large Language Model
- **Piper** for Text-to-Speech

During development, **ChatGPT** was used to assist with code guidance, debugging, architecture discussions, and documentation.

---

## Future Improvements

- Voice Activity Detection (Automatic Stop on Silence)
- Wake Word Detection
- Streaming LLM Responses
- GPU Acceleration
- Graphical User Interface (GUI)
- Long-Term Conversation Memory
- Multi-language Support

---

## Author

**Chilla Kalyan**

AI Engineering Internship Project

---

## License

This project is intended for educational and internship evaluation purposes.
