# Stage 1 Audit — Offline Conversational AI Assistant
### LockIN Voice AI Researcher Internship · Review Date: 2026-10-20

---

## 1. Confirmed Workspace Root

> [!IMPORTANT]
> The VS Code workspace URI opens the **outer** folder:
> `D:\Downloads\Offline-Conversational-AI-Assistant\`
> but all source files live one level deeper:
> `D:\Downloads\Offline-Conversational-AI-Assistant\Offline-Conversational-AI-Assistant\`

**All paths in this report are relative to the inner root above.**
The outer folder contains only this one subdirectory and no code.
When opening VS Code, you must open the inner folder — or set
`"folders"` in a `.code-workspace` file pointing to it — so that relative
imports (`from audio import record_audio`, `piper/piper.exe`,
`Voice/en_US-lessac-medium.onnx`) resolve correctly.

All required files are present:

| File | Present | Bytes |
|---|---|---|
| `app.py` | ✅ | 4 261 |
| `stt.py` | ✅ | 1 508 |
| `llm.py` | ✅ | 415 |
| `tts.py` | ✅ | 769 |
| `requirements.txt` | ✅ | 60 |
| `fallback.py` | ✅ | 598 |
| `history.py` | ✅ | 808 |
| `audio/recorder.py` | ✅ | 619 |
| `audio/player.py` | ✅ | 151 |
| `audio/audio_manager.py` | ✅ | 593 |
| `test_llm.py` | ✅ | 254 |
| `test_tts.py` | ✅ | 160 |
| `test_whisper.py` | ✅ | 169 |

---

## 2. Actual Architecture and Execution Flow

```
app.py (module-level, runs at import time)
 ├── SpeechToText("base")        ← WhisperModel loaded on CPU, int8
 ├── LLM("llama3.2:1b")         ← thin wrapper; no connection check
 ├── TextToSpeech()              ← paths hard-coded to piper/ & Voice/
 ├── FallbackConversation()      ← 6 hard-coded filler phrases, random.choice
 └── ConversationHistory()       ← creates history/ dir + timestamped JSON

main() — called in a while-True loop:
 1. record_audio(duration=30)    ← blocking: always records exactly 30 s
 2. stt.transcribe(audio_file)   ← segments generator consumed eagerly
 3. if text.strip() == "":       ← TTS "I couldn't hear anything" + return
 4. fallback.get_message()       ← random filler phrase
 5. Thread(tts.speak, fallback)  ← filler spoken in background
 6. llm.generate(user_text)      ← blocking ollama.chat (no system prompt,
    |                               no conversation history passed in)
    └── history.save(user, resp)
 7. fallback_thread.join()
 8. tts.speak(response)          ← subprocess piper → tempfile.wav →
    |                               winsound.PlaySound → os.remove
    └── performance report printed
```

**Key insight:** `audio_manager.py` and `audio/player.py` are **dead code** — they are never imported or used anywhere. The `play_audio` export in `audio/__init__.py` is also unused. The README lists a `models/` directory that does not exist.

---

## 3. Findings

### A. ASR (Speech-to-Text)

| ID | File | Lines | Finding |
|---|---|---|---|
| A-1 | `stt.py` | 34–38 | `model_size` parameter defaults to `"small"` in `__init__` but `app.py` (L22) passes `"base"`. Two different model sizes in play — no single source of truth. |
| A-2 | `stt.py` | 44–48 | `beam_size=5` is hard-coded; no `initial_prompt`, no `word_timestamps`, no `vad_filter`. VAD is entirely absent. |
| A-3 | `stt.py` | 1–25 | The entire original implementation is **commented out** and then re-implemented immediately below with nearly identical code. Dead commented code inflates the file and implies unfinished refactoring. |
| A-4 | `stt.py` | 50 | Segments generator is consumed with a list join but no guard against zero segments. An empty audio file returns `""` silently — caught only by `app.py:70`. |
| A-5 | `recorder.py` | 8, 22 | `SAMPLE_RATE = 16000` is correct for Whisper. However there is no normalisation, DC-offset removal, or clipping check before writing to disk. |
| A-6 | `recorder.py` | 11 | `duration=30` is a fixed wall-clock recording. If the user speaks for 3 s the remaining 27 s of silence is fed to Whisper, inflating inference time. |
| A-7 | (none) | — | No VAD library (e.g. `webrtcvad`, `silero-vad`) is present in `requirements.txt` or imported anywhere. |
| A-8 | (none) | — | No WER/CER evaluation harness exists. No reference transcripts, no evaluation dataset, no `jiwer` or equivalent. |
| A-9 | `stt.py` | 52 | Language probability is printed but not gated; a very low probability (hallucination) still gets passed to the LLM unchanged. |
| A-10 | (none) | — | Named entity recognition for names, dates, or numbers is completely absent. No post-processing normaliser. |

---

### B. Audio Engineering

| ID | File | Lines | Finding |
|---|---|---|---|
| B-1 | `recorder.py` | 22–29 | `sd.rec()` / `sd.wait()` has **no exception handler**. A missing microphone, a locked device, or a PortAudio error crashes the whole process with an unhandled `sounddevice.PortAudioError`. |
| B-2 | `tts.py` | 15–32 | `tempfile.NamedTemporaryFile(delete=False)` creates the file then **immediately closes** it before `subprocess.run` writes to it. `os.remove` on L32 is called unconditionally — if Piper crashes, the file may not exist and raises `FileNotFoundError`. |
| B-3 | `tts.py` | 18–28 | `subprocess.run(..., check=True)` raises `CalledProcessError` if Piper fails, but there is no try/except. A corrupt voice model or missing `piper.exe` terminates the app. |
| B-4 | `tts.py` | 3, 30 | `winsound` is **Windows-only**. The project is non-portable; no fallback for macOS/Linux. |
| B-5 | `audio_manager.py`, `player.py` | all | Both files are imported in `audio/__init__.py` (`play_audio` import) but **never called anywhere** — dead code. |
| B-6 | `recorder.py` | 13 | Recordings are saved to `recordings/` (excluded from git) but the function returns the path; the caller never deletes them. Disk can fill over long sessions. |
| B-7 | `tts.py` | 10–11 | Piper binary and voice model paths are relative strings. If the working directory is not the project root the process fails silently with a `FileNotFoundError`. |
| B-8 | `recorder.py` | 22–27 | `channels=1, dtype="float32"` is correct. However the file is written as float32 WAV; `soundfile` auto-promotes this to 32-bit float PCM which Whisper accepts, but no explicit format/subtype is specified, leaving behavior to `soundfile` defaults. |

---

### C. LLM and Response Generation

| ID | File | Lines | Finding |
|---|---|---|---|
| C-1 | `llm.py` | 9–17 | `ollama.chat()` is called with **no system prompt** and **no conversation history**. Every turn is treated as a fresh single-turn query. The assistant has no memory within a session. |
| C-2 | `llm.py` | 9 | No try/except around `ollama.chat`. If Ollama is not running, a `ConnectionRefusedError` or `httpx` exception propagates to `app.py:168` where it is caught by a bare `except Exception as e: print(f"❌ Error: {e}")` — the user sees a cryptic message and the loop continues. |
| C-3 | `llm.py` | 9 | No timeout specified. A hung Ollama process blocks `main()` indefinitely. |
| C-4 | `app.py` | 106 | `llm.generate(user_text)` passes only the current user turn — conversation history stored by `ConversationHistory` is **never fed back** into the LLM. History is write-only. |
| C-5 | `app.py` | 87–98 | The fallback thread is started **before** `llm.generate()` returns. If `llm.generate()` returns before the fallback thread finishes, `fallback_thread.join()` is called correctly; however if `llm.generate()` throws an exception the fallback thread is leaked and never joined. |
| C-6 | `app.py` | 172 | After any exception the loop prompts the user for input. If the exception occurred during Piper subprocess the tempfile may not have been deleted (see B-2). |
| C-7 | `llm.py` | 19 | `response["message"]["content"]` will raise `KeyError` if the Ollama response schema changes. No defensive access. |
| C-8 | (none) | — | No response latency budget or streaming. The LLM is fully blocking; first-token latency is not measured separately from total latency. |
| C-9 | `requirements.txt` | 6 | `ollama` Python SDK is pinned with no version. Upstream breaking changes (e.g. response key renames) will silently break C-7. |

---

### D. Engineering Quality

| ID | File | Lines | Finding |
|---|---|---|---|
| D-1 | `test_llm.py` | all | **Not a test** — it is an interactive REPL loop. No assertions, no pytest, no mock. Cannot be run in CI. |
| D-2 | `test_tts.py` | all | **Not a test** — single `tts.speak()` call with no assertion. Requires Piper binary present and a microphone-free audio device. |
| D-3 | `test_whisper.py` | all | **Not a test** — loads `"tiny"` model and prints success. No transcription, no assertion, no known-good audio fixture. |
| D-4 | `requirements.txt` | all | **No version pins** on any dependency. `numpy`, `sounddevice`, `soundfile`, `scipy`, `faster-whisper`, `ollama` are all unpinned. Reproducible installs are impossible. |
| D-5 | (none) | — | No `setup.py`, `pyproject.toml`, or `setup.cfg`. The package is not installable; CI cannot `pip install -e .`. |
| D-6 | (none) | — | No `logging` module used anywhere. All diagnostics use bare `print()`. There is no way to redirect, filter, or disable output. |
| D-7 | `app.py` | 17–34 | Models are loaded at **module level**, outside `if __name__ == "__main__"`. Importing `app` in a test or notebook triggers full model loading and Ollama connection. |
| D-8 | (none) | — | No `.env` / `config.py` / `config.yaml`. All configurable values (model size, piper path, voice path, sample rate, duration) are hard-coded across multiple files. |
| D-9 | `.gitignore` | all | `recordings/` and `history/` are excluded but `*.onnx` model weights in `Voice/` are not ignored. Large binary assets will be committed to git. |
| D-10 | `stt.py` | 1–25 | Large block of commented-out code shipped in production. |
| D-11 | `app.py` | 41 | Comment typo: `# ===============================s===========================` |
| D-12 | `audio/__init__.py` | 2 | `play_audio` is exported but never used; creates a misleading public API. |
| D-13 | (none) | — | No `mypy`, no `ruff`/`flake8`, no `pre-commit` hooks. No type annotations on any function. |

---

### E. Research Readiness

| ID | Area | Finding |
|---|---|---|
| E-1 | Evaluation datasets | None. No reference audio files, no transcripts, no test sentences covering names, dates, or numerals. |
| E-2 | Baseline measurements | README shows fabricated example numbers ("2.45 sec", "8.73 sec"). No real benchmark run is saved or reproducible. |
| E-3 | WER/CER | Not measured. `jiwer` is not in requirements. No evaluation script exists. |
| E-4 | Benchmark methodology | Not defined. No fixed test conditions (hardware spec, CPU/GPU, model size, temperature, repetitions). |
| E-5 | Reproducibility | Non-reproducible install (no pinned deps), Windows-only runtime (winsound), no seed for random fallback selection, no Ollama model version lock. |
| E-6 | Ablation / variants | No mechanism to switch models, beam sizes, or VAD thresholds without editing source code. |
| E-7 | Limitations section | README "Future Improvements" lists VAD, streaming, GPU — but frames them as future work without quantifying current limitations (e.g. WER on clean vs noisy audio, latency at different model sizes). |

---

## 4. Prioritised Issue List

### 🔴 Critical

| # | Issue | File(s) |
|---|---|---|
| CR-1 | No error handling on microphone recording — any device failure crashes the app | `recorder.py` |
| CR-2 | TTS tempfile race: file closed before Piper writes; unconditional `os.remove` | `tts.py` |
| CR-3 | No version pins — environment is not reproducible | `requirements.txt` |
| CR-4 | Model loading at module level prevents safe importing in tests/notebooks | `app.py` |
| CR-5 | Ollama called with no connection check, no timeout, no retry | `llm.py` |

### 🟠 High

| # | Issue | File(s) |
|---|---|---|
| H-1 | Fixed 30-second recording — unusable UX; no VAD | `recorder.py`, `app.py` |
| H-2 | No system prompt; no session history passed to LLM | `llm.py`, `app.py` |
| H-3 | `winsound` makes project Windows-only | `tts.py` |
| H-4 | No WER/CER evaluation harness whatsoever | (new file needed) |
| H-5 | Fallback thread not joined on LLM exception — resource leak | `app.py` |
| H-6 | `piper_path` and `voice_model` paths are relative strings — CWD-dependent | `tts.py` |

### 🟡 Medium

| # | Issue | File(s) |
|---|---|---|
| M-1 | Language probability not gated — hallucinated transcript forwarded to LLM | `stt.py` |
| M-2 | Commented-out dead code block in `stt.py` | `stt.py` |
| M-3 | `audio_manager.py` and `player.py` are dead code | `audio/` |
| M-4 | No `logging` module — no log levels, no file output | all |
| M-5 | No config file — all hyperparameters scattered across source | all |
| M-6 | History saved but never read back by LLM | `history.py`, `llm.py` |
| M-7 | No named-entity post-processing (names, dates, numbers) | `stt.py` |
| M-8 | Large ONNX model weights tracked by git | `Voice/` |

### 🔵 Low

| # | Issue | File(s) |
|---|---|---|
| L-1 | Tests are manual smoke scripts, not pytest | `test_*.py` |
| L-2 | No type annotations | all |
| L-3 | No linter / pre-commit hooks | (repo root) |
| L-4 | `models/` directory documented in README does not exist | `README.md` |
| L-5 | Comment typo in `app.py` line 41 | `app.py` |
| L-6 | No `__version__` or project metadata | (new file needed) |
| L-7 | `recordings/` never pruned — disk fills over time | `recorder.py` |

---

## 5. Staged Implementation Plan

> Awaiting your approval before any edits are made.

### Stage 1 — Stabilise Foundation (no new features)
**Goal:** make the app robust enough that it never silently crashes.

1. **CR-3** Pin all dependencies with `pip freeze` output in `requirements.txt`.
2. **CR-4** Wrap all model loading inside `if __name__ == "__main__"` and a `setup()` factory function.
3. **CR-1** Wrap `sd.rec()` / `sd.wait()` in try/except with a clean error message and optional retry.
4. **CR-2** Rewrite `tts.py` using a proper `with tempfile.NamedTemporaryFile` lifecycle; wrap subprocess in try/except/finally for cleanup.
5. **CR-5** Add `try/except` + configurable timeout to `llm.generate()`, plus an Ollama connectivity pre-check at startup.
6. **H-5** Fix fallback thread join inside finally block.
7. **M-4** Replace all `print()` calls with `logging` (configurable level via env var).
8. **M-5** Introduce `config.py` with a single `Config` dataclass (model size, piper path, sample rate, duration, Ollama model, etc.).

---

### Stage 2 — Audio Engineering Upgrade
1. **H-1 / A-7** Integrate `silero-vad` (or `webrtcvad`) for real-time silence detection. Replace fixed-duration recording with VAD-gated capture.
2. **A-5** Add pre-processing: DC offset removal, normalisation, optional noise gate.
3. **H-3** Replace `winsound` with `sounddevice`-based playback (already a dependency) so TTS works cross-platform.
4. **B-6** Add a rolling `recordings/` cleanup (keep last N files or prune by age).
5. **H-6** Use `pathlib.Path(__file__).parent` to resolve Piper and voice model paths absolutely.
6. Delete or refactor dead code: `audio_manager.py`, `player.py` (or integrate them into the main path).

---

### Stage 3 — ASR Hardening
1. **A-9 / M-1** Gate transcript on language probability threshold (e.g. `< 0.70` → ask user to repeat).
2. **A-2** Expose `vad_filter=True`, `initial_prompt`, and `word_timestamps` in `SpeechToText.transcribe()`.
3. **A-1** Remove default-size mismatch — single `MODEL_SIZE` constant in `config.py`.
4. **A-10** Add lightweight post-processing: number normalisation (`num2words` or regex), proper-noun capitalisation from a user-editable hotwords list.
5. **M-2** Delete commented-out code block in `stt.py`.

---

### Stage 4 — LLM and Conversational Quality
1. **H-2 / C-1** Add a system prompt and inject the rolling conversation history into `ollama.chat` messages list.
2. **C-8** Implement streaming response via `ollama.chat(stream=True)` — begin TTS on the first completed sentence rather than waiting for the full response.
3. **M-6** Wire `ConversationHistory` back into `LLM.generate()` for multi-turn coherence.
4. **C-7** Add defensive key access and schema validation for Ollama response.
5. Add model availability check (`ollama list`) at startup.

---

### Stage 5 — Testing and Research Infrastructure
1. **H-4 / E-1** Create `eval/` directory with:
   - `eval/sentences.txt` — 50 reference sentences (names, dates, numbers, conversational)
   - `eval/recordings/` — pre-recorded WAVs (or synthesis pipeline to generate them)
   - `eval/evaluate_asr.py` — WER/CER via `jiwer`, saves CSV report
2. **L-1** Rewrite all `test_*.py` as proper `pytest` modules with mocks (`unittest.mock.patch`) for audio hardware.
3. Add `pytest`, `jiwer`, `num2words` to `requirements.txt`.
4. **E-4** Create `benchmark.py` — fixed-condition latency benchmark: 10 runs, records mean ± std for STT, LLM, TTS, and total.
5. **E-6** Make model/config selectable via CLI args (`argparse`) or environment variables.
6. **D-9** Add `*.onnx` and `*.onnx.json` to `.gitignore`; document model download step.

---

### Stage 6 — Polish and Documentation
1. Update `README.md` with real benchmark numbers, hardware spec, and limitations.
2. Add `CONTRIBUTING.md` and `CHANGELOG.md`.
3. Add `pyproject.toml` with project metadata, dev dependencies, and optional `[tool.ruff]` config.
4. Add GitHub Actions / CI yaml (lint + pytest on push).

---

## 6. Proposed Tests and Acceptance Criteria

### Unit Tests (pytest, all mocked — no hardware required)

| Test | Module | Acceptance Criterion |
|---|---|---|
| `test_stt_empty_audio` | `stt.py` | Returns `""` for a silent WAV; does not raise. |
| `test_stt_low_probability` | `stt.py` | Returns `None` or raises `LowConfidenceError` when language probability < 0.70. |
| `test_stt_name_entity` | `stt.py` | "My name is John Smith" → transcript preserves "John Smith" (capitalised). |
| `test_stt_date_entity` | `stt.py` | "Today is October twentieth" → normalised to "Today is October 20". |
| `test_recorder_device_failure` | `recorder.py` | Raises `AudioInputError` (custom) with a user-readable message when `sd.rec` raises `PortAudioError`. |
| `test_recorder_vad_stops_early` | `recorder.py` | Recording terminates before `duration` when VAD detects silence. |
| `test_tts_piper_crash` | `tts.py` | Tempfile is deleted even when `subprocess.run` raises `CalledProcessError`. |
| `test_tts_empty_text` | `tts.py` | Empty string → raises `ValueError` before invoking Piper. |
| `test_llm_ollama_down` | `llm.py` | Raises `OllamaConnectionError` with a meaningful message within 5 seconds. |
| `test_llm_timeout` | `llm.py` | Returns fallback text or raises `OllamaTimeoutError` after configurable timeout. |
| `test_llm_history_injected` | `llm.py` | Conversation history is present in the messages list sent to `ollama.chat`. |
| `test_history_save_load` | `history.py` | Round-trips user+assistant turns through JSON correctly. |
| `test_fallback_distribution` | `fallback.py` | All 6 messages reachable across 100 calls (no dead phrases). |

### Integration Tests (require Ollama running locally)

| Test | Acceptance Criterion |
|---|---|
| `test_full_pipeline_smoke` | `app.main()` completes without exception given a pre-recorded WAV fixture. |
| `test_latency_budget` | STT ≤ 5 s, LLM ≤ 15 s, TTS ≤ 3 s on reference hardware (spec to be documented). |

### WER/CER Evaluation (eval harness)

| Metric | Target | Dataset |
|---|---|---|
| WER (clean speech) | < 10 % | 50 sentences, studio recording |
| WER (noisy +10 dB SNR) | < 20 % | Same sentences + white noise |
| CER (names, dates) | < 5 % | 10 named-entity sentences |

---

## 7. Commands to Reproduce Results

All commands run from the **inner project root**:
`D:\Downloads\Offline-Conversational-AI-Assistant\Offline-Conversational-AI-Assistant\`

```powershell
# Verify workspace root
Get-ChildItem -Name | Sort-Object

# Confirm Python version
python --version

# Reproduce current (unpinned) install
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Inspect installed versions (run AFTER install to create a lock)
pip freeze

# Verify Ollama model is available
ollama list

# Smoke-test Whisper loading (test_whisper.py loads "tiny", app.py uses "base")
python test_whisper.py

# Smoke-test TTS (requires piper.exe + Voice/en_US-lessac-medium.onnx)
python test_tts.py

# Smoke-test LLM (requires Ollama running with llama3.2:1b pulled)
python test_llm.py

# Run the assistant
python app.py

# Future: run pytest (after Stage 5)
# pytest tests/ -v --tb=short

# Future: run WER evaluation (after Stage 5)
# python eval/evaluate_asr.py --dataset eval/sentences.txt --output eval/results.csv
```

---

## Summary Table

| Area | Current State | Readiness for Internship |
|---|---|---|
| ASR (Faster Whisper) | Basic working integration | ⚠️ Prototype |
| VAD | Absent | ❌ Missing |
| WER/CER Evaluation | Absent | ❌ Missing |
| Audio Engineering | Functional, fragile | ⚠️ Fragile |
| LLM Integration | Working, no memory | ⚠️ Prototype |
| Error Handling | Minimal | ❌ Insufficient |
| Test Coverage | 0 real tests | ❌ None |
| Dependency Management | Unpinned | ❌ Non-reproducible |
| Research Infrastructure | Absent | ❌ Missing |
| Documentation | Basic README | ⚠️ Incomplete |

**Overall: functional proof-of-concept, not research-ready. Stages 1–3 are the minimum bar before submission.**

---
*Audit performed by static code inspection only. No packages were installed, no models were downloaded, no files were modified, and no network requests were made.*
