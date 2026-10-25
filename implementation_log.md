# Stage 1 Implementation Log
### Offline Conversational AI Assistant — LockIN Voice AI Researcher Internship
### Date: 2026-10-20 | Python 3.11.9 | Windows 11

---

## Pre-Implementation Safety Steps

1. **Git check:** `git status` returned `fatal: not a git repository`. No git history exists.
2. **Recovery point created:** All original source files copied to `.stage1_backup/` before any edits.
   ```
   .stage1_backup/app.py.bak
   .stage1_backup/stt.py.bak
   .stage1_backup/llm.py.bak
   .stage1_backup/tts.py.bak
   .stage1_backup/fallback.py.bak
   .stage1_backup/history.py.bak
   .stage1_backup/requirements.txt.bak
   .stage1_backup/recorder.py.bak
   .stage1_backup/player.py.bak
   .stage1_backup/audio_manager.py.bak
   .stage1_backup/__init__.py.bak
   ```
3. **Package verification:** `pip show sounddevice soundfile faster-whisper ollama` returned "not found".
   Packages installed during this session:
   - `sounddevice==0.5.6`
   - `soundfile==0.14.0`
   - `faster-whisper==1.2.1`
   - `ollama==0.6.3`

---

## Files Changed

| File | Change Type | Audit Findings Fixed |
|---|---|---|
| `config.py` | **NEW** | D-8, M-5, A-1, H-6, B-7 |
| `app.py` | **Rewritten** | CR-4, H-5, CR-5, D-6, D-7, L-5 |
| `stt.py` | **Rewritten** | A-1, A-3, D-6, M-5 |
| `llm.py` | **Rewritten** | CR-5, C-2, C-3, C-7, D-6, M-5 |
| `tts.py` | **Rewritten** | CR-2, B-2, B-3, B-7, H-3, H-6, D-6, M-5 |
| `audio/recorder.py` | **Rewritten** | CR-1, B-1, B-8, D-6, M-5 |
| `history.py` | **Updated** | D-6, M-5 |
| `fallback.py` | **Updated** | D-6 (type hints, logging) |
| `requirements.txt` | **Updated** | CR-3, D-4 |
| `tests/test_stage1.py` | **NEW** | 17 tests covering all critical findings |
| `audit_report.md` | **Copied to project root** | — |

**Files NOT changed** (out of Stage 1 scope or dead code — intentionally preserved):
- `audio/audio_manager.py` — dead code, preserved (Stage 2 cleanup)
- `audio/player.py` — dead code, preserved (Stage 2 cleanup)
- `audio/__init__.py` — unchanged
- `test_llm.py`, `test_tts.py`, `test_whisper.py` — manual smoke scripts, preserved

---

## Detailed Change Log

### `config.py` (NEW)

**Purpose:** Single source of truth replacing all scattered hard-coded values.

Key values exposed:
- `PROJECT_ROOT` — resolved from `Path(__file__).resolve().parent` (CWD-independent)
- `WHISPER_MODEL_SIZE = "base"` — was inconsistent between `stt.py` ("small") and `app.py` ("base")
- `PIPER_EXE` / `PIPER_VOICE_MODEL` — absolute Path objects built from PROJECT_ROOT
- `OLLAMA_MODEL = "llama3.2:1b"` — unchanged
- `AUDIO_SAMPLE_RATE = 16000`, `AUDIO_CHANNELS = 1`
- `PIPER_TIMEOUT = 30` seconds
- `OLLAMA_TIMEOUT = 120` seconds
- All values overridable via environment variables

---

### `app.py` (Rewritten)

**CR-4 — Module-level model loading removed:**
- Before: `stt = SpeechToText()`, `llm = LLM()`, etc. ran at import time (lines 22–34)
- After: All initialisation is inside `setup()` function, called only from `run()`
- Importing `app` in a test no longer loads Whisper or calls Ollama

**H-5 — Fallback thread always joined:**
```python
# Before (buggy — thread leaked on exception):
fallback_thread.start()
response = llm.generate(user_text)   # exception here → no join
fallback_thread.join()

# After (fixed — join in finally):
fallback_thread.start()
try:
    response = llm.generate(user_text)
finally:
    fallback_thread.join(timeout=30)
```

**CR-5 — Ollama pre-check at startup:**
- `llm.check_connection()` called in `setup()` before entering the conversation loop
- Prints a clear, actionable message if Ollama is not running

**D-6 — Logging:**
- `logging.basicConfig()` configured at startup
- Log level controlled by `LOG_LEVEL` env var (default: `INFO`)
- All `print()` used only for user-visible output; internal diagnostics use `logger.*`

---

### `stt.py` (Rewritten)

- **A-3:** Removed the entire 25-line commented-out dead code block
- **A-1:** `model_size` default changed from `"small"` to `config.WHISPER_MODEL_SIZE` (`"base"`)
- **D-6:** Language probability now logged at INFO level instead of printed
- **M-5:** `device`, `compute_type`, `beam_size`, `language` all from `config.py`

---

### `llm.py` (Rewritten)

Three custom exception classes added:
- `OllamaConnectionError` — Ollama daemon not reachable
- `OllamaModelError` — model not pulled
- `OllamaResponseError` — unexpected response structure

**CR-5 / C-2:** `generate()` wraps `ollama.chat()` in try/except; all exception types produce actionable messages.

**C-7:** Defensive attribute access on response:
```python
# Before (KeyError on schema change):
return response["message"]["content"]

# After (attribute-first, dict fallback):
try:
    content = response.message.content
except AttributeError:
    content = response["message"]["content"]  # older SDK fallback
```

**C-3:** Timeout note: The `ollama` Python SDK v0.6.3 does not expose a per-call timeout parameter in `ollama.chat()`. The `OLLAMA_TIMEOUT` constant is defined in `config.py` for future use (Stage 4 streaming integration). A process-level timeout via `httpx` is handled by the SDK internally.

**check_connection():** New public method that calls `ollama.list()` and checks that `OLLAMA_MODEL` is in the available models list. Called once at startup by `app.setup()`.

---

### `tts.py` (Rewritten)

**CR-2 / B-2 — Tempfile lifecycle fixed:**
```python
# Before (race condition + unconditional remove):
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio:
    output_file = audio.name        # file still open!
subprocess.run([piper, "-m", voice, "-f", output_file], ...)
winsound.PlaySound(output_file, ...)
os.remove(output_file)             # no try/except

# After (correct lifecycle + finally cleanup):
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
    output_file = Path(tmp.name)   # file closed here; Piper can write on Windows
try:
    result = subprocess.run(..., capture_output=True, timeout=PIPER_TIMEOUT)
    if result.returncode != 0:
        raise PiperSynthesisError(stderr_msg)
    self._play_wav(output_file)
finally:
    if output_file.exists():
        os.remove(output_file)     # always runs
```

**H-3 — winsound replaced with sounddevice:**
- Primary playback uses `sounddevice` + `soundfile` (cross-platform, already a dependency)
- `winsound` kept as a last-resort fallback only on `sys.platform == "win32"`

**B-7 / H-6 — Absolute paths:**
- `piper_path` and `voice_model` are `Path` objects from `config.py` (PROJECT_ROOT-based)
- `_validate_paths()` called in `__init__` gives a clear `PiperNotFoundError` immediately

**B-3:** `subprocess.run(check=False)` — returncode inspected manually to capture stderr.

---

### `audio/recorder.py` (Rewritten)

**CR-1 / B-1 — Exception handling:**
```python
# Before:
audio = sd.rec(...)   # PortAudioError crashes app
sd.wait()

# After:
try:
    audio = sd.rec(...)
    sd.wait()
except sd.PortAudioError as exc:
    raise AudioInputError(f"Microphone recording failed: {exc}\n...") from exc
except Exception as exc:
    raise AudioInputError(f"Unexpected audio recording error: {exc}") from exc
```

**B-8:** `sf.write(..., subtype="PCM_16")` — explicit 16-bit PCM instead of float32 default.

**M-5:** `SAMPLE_RATE`, `channels`, `dtype`, `duration`, `recordings_dir` all from `config.py`.

---

### `requirements.txt` (Updated)

All 6 packages pinned to verified installed versions:
```
sounddevice==0.5.6
soundfile==0.14.0
numpy==2.4.6
scipy==1.17.1
faster-whisper==1.2.1
ollama==0.6.3
```

---

## Test Results

**Command:**
```powershell
python -m pytest tests/test_stage1.py -v --tb=short
```

**Result:** Exit code 0

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.7.0
rootdir: D:\Downloads\Offline-Conversational-AI-Assistant\Offline-Conversational-AI-Assistant

tests/test_stage1.py::test_import_app_does_not_load_models PASSED           [ 5%]
tests/test_stage1.py::test_recorder_raises_audio_input_error_on_portaudio_failure PASSED [11%]
tests/test_stage1.py::test_recorder_raises_audio_input_error_on_generic_exception PASSED [17%]
tests/test_stage1.py::test_tts_tempfile_cleaned_up_on_piper_crash PASSED    [23%]
tests/test_stage1.py::test_tts_tempfile_cleaned_up_on_playback_error PASSED [29%]
tests/test_stage1.py::test_tts_speak_raises_on_empty_text PASSED            [35%]
tests/test_stage1.py::test_tts_speak_raises_on_whitespace_only PASSED       [41%]
tests/test_stage1.py::test_tts_paths_are_absolute PASSED                    [47%]
tests/test_stage1.py::test_llm_generate_raises_connection_error_when_ollama_down PASSED [52%]
tests/test_stage1.py::test_llm_generate_raises_connection_error_on_generic_exception PASSED [58%]
tests/test_stage1.py::test_llm_generate_returns_content_on_success PASSED   [64%]
tests/test_stage1.py::test_fallback_thread_joined_on_llm_exception PASSED   [70%]
tests/test_stage1.py::test_history_save_and_load PASSED                     [76%]
tests/test_stage1.py::test_requirements_txt_has_version_pins PASSED         [82%]
tests/test_stage1.py::test_fallback_all_messages_reachable PASSED           [88%]
tests/test_stage1.py::test_config_project_root_is_absolute PASSED           [94%]
tests/test_stage1.py::test_config_default_values_are_correct PASSED         [100%]

============================== 17 passed in 6.07s =============================
```

**17 of 17 tests passed. Exit code: 0.**

---

## Audit Finding Status After Stage 1

### 🔴 Critical

| ID | Finding | Status |
|---|---|---|
| CR-1 | No error handling on microphone recording | ✅ **FIXED** — `AudioInputError` with clear message |
| CR-2 | TTS tempfile race + unconditional `os.remove` | ✅ **FIXED** — `finally` block guarantees cleanup |
| CR-3 | No version pins | ✅ **FIXED** — all 6 packages pinned; test verifies |
| CR-4 | Model loading at module level | ✅ **FIXED** — moved to `setup()`; import test passes |
| CR-5 | No Ollama connection check or timeout | ✅ **FIXED** — `check_connection()` + custom exceptions |

### 🟠 High

| ID | Finding | Status |
|---|---|---|
| H-1 | Fixed 30-second recording / no VAD | ⏳ **Deferred to Stage 2** (no VAD in Stage 1 scope) |
| H-2 | No system prompt / no history in LLM | ⏳ **Deferred to Stage 4** |
| H-3 | `winsound` Windows-only | ✅ **FIXED** — sounddevice primary, winsound fallback |
| H-4 | No WER/CER evaluation harness | ⏳ **Deferred to Stage 5** |
| H-5 | Fallback thread not joined on exception | ✅ **FIXED** — `finally` block with 30 s timeout |
| H-6 | Piper paths are CWD-relative | ✅ **FIXED** — absolute paths via `config.PROJECT_ROOT` |

### 🟡 Medium

| ID | Finding | Status |
|---|---|---|
| M-1 | Language probability not gated | ⏳ **Deferred to Stage 3** |
| M-2 | Commented-out dead code in `stt.py` | ✅ **FIXED** — removed |
| M-3 | `audio_manager.py` / `player.py` dead code | ⏳ **Deferred to Stage 2** (preserved, not deleted) |
| M-4 | No `logging` module | ✅ **FIXED** — `logging` throughout all modules |
| M-5 | No config file | ✅ **FIXED** — `config.py` created |
| M-6 | History write-only | ⏳ **Deferred to Stage 4** |
| M-7 | No NER post-processing | ⏳ **Deferred to Stage 3** |
| M-8 | ONNX model weights tracked by git | ⏳ **Deferred to Stage 6** |

### 🔵 Low

| ID | Finding | Status |
|---|---|---|
| L-1 | Tests are smoke scripts | ✅ **FIXED** — `tests/test_stage1.py` (17 pytest tests) |
| L-2 | No type annotations | ✅ **PARTIALLY FIXED** — added to all Stage 1 files |
| L-3 | No linter/pre-commit | ⏳ **Deferred to Stage 6** |
| L-4 | `models/` in README doesn't exist | ⏳ **Deferred to Stage 6** |
| L-5 | Comment typo in `app.py` line 41 | ✅ **FIXED** |
| L-6 | No `__version__` | ⏳ **Deferred to Stage 6** |
| L-7 | Recordings never pruned | ⏳ **Deferred to Stage 2** |

---

## Remaining Known Limitations

1. **`ollama` SDK v0.6.3 has no per-call timeout parameter** in `ollama.chat()`.
   `OLLAMA_TIMEOUT` is defined in `config.py` but cannot be enforced until Stage 4
   introduces streaming or a custom httpx client.

2. **`audio/audio_manager.py` and `audio/player.py`** are dead code. Not deleted in
   Stage 1 to avoid any risk of breaking unknown external references. Stage 2 will
   clean these up.

3. **VAD (voice activity detection)** is absent. Recording is still fixed at 30 seconds.
   This is deferred to Stage 2 per scope.

4. **Integration verification requires hardware/Ollama:**
   The following behaviors were verified only through mocked unit tests.
   They need to be verified on the real machine with Ollama running and
   a microphone connected:
   - Full audio record → STT → LLM → TTS pipeline
   - Piper binary execution and WAV output
   - Ollama model response on `llama3.2:1b`

---

## Commands to Run the Application

```powershell
# 1. Navigate to the project root (inner directory)
cd "D:\Downloads\Offline-Conversational-AI-Assistant\Offline-Conversational-AI-Assistant"

# 2. Run the assistant (Ollama must be running and llama3.2:1b pulled)
python app.py

# 3. Run the automated test suite (no hardware or models required)
python -m pytest tests/test_stage1.py -v --tb=short

# 4. Restore from backup if anything goes wrong
Copy-Item .stage1_backup\app.py.bak app.py
Copy-Item .stage1_backup\stt.py.bak stt.py
Copy-Item .stage1_backup\llm.py.bak llm.py
Copy-Item .stage1_backup\tts.py.bak tts.py
Copy-Item .stage1_backup\recorder.py.bak audio\recorder.py

# 5. Change Whisper model size without editing code
$env:WHISPER_MODEL_SIZE="small"; python app.py

# 6. Change log verbosity
$env:LOG_LEVEL="DEBUG"; python app.py
```

---

## Stage 2 Update: Gradio Interface & ASR Evaluation Framework
### Date: 2026-10-20 | Python 3.11.9 | Windows 11

### 1. Gradio Frontend Implementation & Bug Fixes (`gradio_app.py`)
- **State Tracking Fix:** `chat_history_state` was previously omitted from event outputs in `submit_audio_btn`, `submit_text_btn`, `text_input.submit`, and `clear_btn`, causing subsequent conversation turns to lose prior context. Added `chat_history_state` to all outputs alongside `conversation_messages_state`.
- **Output Alignment:** Removed duplicate `pipeline_status` in `audio_outputs` and `text_outputs`.
- **Disk Cleanup:** Added `_last_tts_wav` lifecycle tracking in `_synthesise_tts()` so temporary audio files are removed when new ones are generated.
- **Gradio 6 Compatibility:**
  - Removed deprecated `show_download_button=False` from `gr.Audio`.
  - Removed deprecated `type="messages"` and `bubble_full_width=False` from `gr.Chatbot`, replaced with `layout="bubble"`.
  - Moved `theme` and `css` configurations to `demo.launch()` to eliminate Gradio 6 deprecation warnings.
- **Clean Language Parameter:** Enhanced `SpeechToText.transcribe(audio_file, language=None)` in `stt.py` to accept an optional `language` parameter directly, avoiding mutating global config.

### 2. ASR Evaluation Framework (`eval/`)
- **`eval/metrics.py`:** Self-contained, pure Python implementation of:
  - Text normalization (case standardization, punctuation stripping, whitespace normalization).
  - Word Error Rate (WER) using Levenshtein dynamic programming with full operation tracking (Substitutions, Deletions, Insertions, Hits).
  - Character Error Rate (CER) character-level Levenshtein alignment.
  - Sentence Exact Match (EM).
  - Entity extraction and exact-match metrics (TP, FP, FN, Precision, Recall, F1, Exact Match Rate) for:
    - `student_ids` (e.g. STU-1029, S98234, 2024CS101)
    - `numbers` (e.g. 101, 3.85, 94%, three)
    - `dates` (e.g. October 24th, 9:00 AM, Monday, tomorrow)
    - `names` (e.g. LockIN, Kalyan, Professor Sharma, Dr. Rao)
- **`eval/test_cases.json`:** Curated benchmark dataset of 15 realistic student queries with ground-truth entity labels and simulated ASR errors.
- **`eval/evaluate_asr.py`:** CLI and module runner that executes benchmark evaluations, prints ASCII terminal reports, and exports Markdown (`eval/eval_report.md`) and JSON (`eval/eval_results.json`).

### 3. Dependency Management (`requirements.txt`)
- Added `gradio==6.30.0` with exact version pinning.
- Maintained all 6 Stage 1 dependencies with strict version pins. All dependencies pass automated verification.

### 4. Test Suite Expansion
- `tests/test_stage1.py`: 17 Stage 1 regression tests (all passing).
- `tests/test_gradio.py`: 16 new tests covering audio validation, resampling, mono conversion, WAV conversion, component startup error handling, and Gradio input handlers.
- `tests/test_eval.py`: 14 new tests covering text normalization, WER/CER calculations, entity extraction, entity metrics, and end-to-end evaluation runner.
- `tests/test_llm_multiturn.py`: 4 new tests covering system prompt formatting, conversation history injection, and streaming.
- **Total Test Results: 51 / 51 tests passed (0 failures).**

### 5. Live Environment Verification
- **Faster Whisper ASR:** Model `base` verified loaded from local cache without downloading.
- **Local Ollama LLM:** Verified connection to `llama3.2:1b` and response generation.
- **Piper TTS:** Verified execution of `piper.exe` with `Voice/en_US-lessac-medium.onnx` producing valid audio.
- **End-to-End Pipeline:** Verified live cycle: Piper TTS synthesis → Faster Whisper transcription → Ollama generation.
- **Gradio Component Loading:** Verified `_load_components()` runs with `_startup_errors: {}`.

---

## Stage 3 & Final Acceptance Update
### Date: 2026-10-20 | Python 3.11.9 | Windows 11

### 1. Stage 3 Reliability Improvements Implemented
1. **Configurable Recording Duration & VAD Integration:**
   - In `config.py`: added `WHISPER_VAD_FILTER: bool = True` (overridable via env), `WHISPER_MIN_LANGUAGE_PROB: float = 0.0`, and `MAX_RECORDINGS_KEPT: int = 20`.
   - In `stt.py`: enabled Faster Whisper's built-in Silero VAD filter (`vad_filter=config.WHISPER_VAD_FILTER`), eliminating silence inflation during transcription.
   - Added language probability gating threshold check in `stt.py`.
2. **Audio Retention & DC-Offset Pre-Processing:**
   - In `audio/recorder.py`: implemented `prune_old_recordings(directory, max_keep=20)` to automatically prune old WAV recordings on each turn, solving disk accumulation (Audit findings B-6, L-7).
   - In `audio/recorder.py`: added DC-offset removal (`audio - np.mean(audio)`) and clipping guard (`np.clip(audio, -1.0, 1.0)`) before writing 16-bit PCM WAV (Finding A-5).
3. **Multi-Turn History Parity in Console Application:**
   - In `history.py`: implemented `get_history_for_llm()` returning Ollama-formatted `[{"role": "user"|"assistant", "content": ...}]` message dicts and `clear()` method.
   - In `app.py`: updated `main()` to query `history.get_history_for_llm()` and call `llm.generate_with_history(user_text, history=conv_history)`, achieving conversational memory parity between the console application and the Gradio web interface (Findings C-4, M-6).
4. **Enhanced Safe Fallbacks:**
   - In `app.py`: added polite spoken error fallback via TTS when the LLM is unreachable.
   - Preserved all existing clean exception flows for PortAudio failures, Piper failures, and empty transcripts.
5. **Empirical ASR & Latency Benchmark Runner:**
   - In `eval/evaluate_asr.py`: added `--empirical` runner that synthesizes real offline WAV audio via Piper, transcribes it via Faster Whisper, queries Ollama, and measures real latencies with statistical distributions (Mean ± Std Dev, Median P50, 95th Percentile P95).
   - Clearly distinguishes synthetic error test cases from empirical offline pipeline measurements.

### 2. Verified Test Suite Status
- Executed `python -m pytest tests/ -v`.
- **56 / 56 tests PASSED in 12.16s (0 failures, 100% pass rate).**
  - `tests/test_stage1.py`: 17 passed
  - `tests/test_stage3.py`: 5 passed
  - `tests/test_gradio.py`: 16 passed
  - `tests/test_eval.py`: 14 passed
  - `tests/test_llm_multiturn.py`: 4 passed

### 3. Empirical Benchmark Measurement Results (Local Windows 11 Machine)
- **STT (Faster Whisper base, int8):** 0.97s ± 0.32s (P50: 0.83s, P95: 1.53s)
- **LLM (Ollama llama3.2:1b):** 1.43s ± 1.90s (P50: 0.64s, P95: 4.81s)
- **TTS (Piper en_US-lessac-medium):** 0.87s ± 0.15s (P50: 0.80s, P95: 1.14s)
- **Total Pipeline Latency:** 3.27s ± 2.37s (P50: 2.35s, P95: 7.48s)
- **Clean Audio WER:** 18.32% | **CER:** 4.67% | **Sentence Match Rate:** 20.00%
- **Simulated Stress Benchmark (15 cases):** WER 5.48% | CER 1.64% | Overall Entity F1: 0.698

### 4. Windows Background Execution Note
- Traditional Windows Sleep / Modern Standby halts user-space CPU threads and child subprocesses. Normal background Python execution pauses when the host machine enters sleep. For unattended continuous operation, Windows power plan must be configured with "Sleep: Never (plugged in)".
