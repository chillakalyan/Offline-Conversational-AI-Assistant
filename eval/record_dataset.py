"""Record and validate human-speech samples for the ASR evaluation dataset."""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_RATE = 16000
CHANNELS = 1
CATEGORY_NAMES = {
    "normal_speech",
    "numbers",
    "names",
    "numbers_and_names",
    "technical_terms",
    "short_commands",
}
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "dataset"
logger = logging.getLogger(__name__)


class DatasetValidationError(ValueError):
    """Raised when dataset references, metadata, or audio are invalid."""


def _load_json(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Could not read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetValidationError(f"Expected a JSON object in {path}.")
    return value


def load_references(path: Path) -> Dict[str, Dict]:
    """Load and validate the static reference inventory, keyed by sample ID."""
    document = _load_json(path)
    samples = document.get("samples")
    if not isinstance(samples, list) or not samples:
        raise DatasetValidationError("references.json must contain a non-empty 'samples' list.")

    references: Dict[str, Dict] = {}
    required = {
        "id", "reference", "language", "category",
        "contains_numbers", "contains_names_entities",
    }
    for index, sample in enumerate(samples):
        prefix = f"references.json samples[{index}]"
        if not isinstance(sample, dict):
            raise DatasetValidationError(f"{prefix} must be an object.")
        missing = required - sample.keys()
        if missing:
            raise DatasetValidationError(f"{prefix} is missing fields: {sorted(missing)}.")
        sample_id = sample["id"]
        if not isinstance(sample_id, str) or not re.fullmatch(r"[A-Z0-9_-]+", sample_id):
            raise DatasetValidationError(f"{prefix} has an invalid sample ID.")
        if sample_id in references:
            raise DatasetValidationError(f"Duplicate reference ID: {sample_id}.")
        if not isinstance(sample["reference"], str) or not sample["reference"].strip():
            raise DatasetValidationError(f"{prefix} must have a non-empty reference transcript.")
        if not isinstance(sample["language"], str) or not sample["language"].strip():
            raise DatasetValidationError(f"{prefix} must have a language code.")
        if not isinstance(sample["category"], str) or sample["category"] not in CATEGORY_NAMES:
            raise DatasetValidationError(
                f"{prefix} has unsupported category {sample['category']!r}."
            )
        if type(sample["contains_numbers"]) is not bool:
            raise DatasetValidationError(f"{prefix}.contains_numbers must be a boolean.")
        if type(sample["contains_names_entities"]) is not bool:
            raise DatasetValidationError(
                f"{prefix}.contains_names_entities must be a boolean."
            )
        references[sample_id] = sample
    return references


def validate_audio_file(audio_path: Path) -> Dict[str, int]:
    """Verify a sample is readable mono 16 kHz PCM WAV and return its properties."""
    if not audio_path.is_file():
        raise DatasetValidationError(f"Audio file does not exist: {audio_path}.")
    try:
        info = sf.info(str(audio_path))
        with sf.SoundFile(str(audio_path)) as audio_file:
            if audio_file.frames <= 0:
                raise DatasetValidationError(f"Audio file is empty: {audio_path}.")
            audio_file.seek(0)
            first_frame = audio_file.read(frames=1, dtype="float32", always_2d=True)
            if first_frame.size == 0:
                raise DatasetValidationError(f"Audio file has no readable samples: {audio_path}.")
    except DatasetValidationError:
        raise
    except (OSError, RuntimeError, sf.LibsndfileError) as exc:
        raise DatasetValidationError(f"Audio file is unreadable: {audio_path}: {exc}") from exc

    if info.format != "WAV":
        raise DatasetValidationError(f"Audio must be WAV: {audio_path}.")
    if info.subtype != "PCM_16":
        raise DatasetValidationError(f"Audio must be 16-bit PCM: {audio_path}.")
    if info.samplerate != SAMPLE_RATE:
        raise DatasetValidationError(
            f"Audio must be {SAMPLE_RATE} Hz, got {info.samplerate}: {audio_path}."
        )
    if info.channels != CHANNELS:
        raise DatasetValidationError(
            f"Audio must be mono, got {info.channels} channels: {audio_path}."
        )
    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
    }


def validate_dataset(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> Tuple[Dict[str, Dict], List[Dict]]:
    """Validate the reference inventory and every recorded metadata/audio entry."""
    references = load_references(dataset_dir / "references.json")
    metadata_document = _load_json(dataset_dir / "metadata.json")
    samples = metadata_document.get("samples")
    if not isinstance(samples, list):
        raise DatasetValidationError("metadata.json must contain a 'samples' list.")

    recorded: List[Dict] = []
    seen_ids = set()
    seen_audio = set()
    required = {
        "id", "audio_filename", "speaker_id", "recording_condition",
        "sample_rate", "channels", "format", "subtype", "recorded_at",
    }
    for index, sample in enumerate(samples):
        prefix = f"metadata.json samples[{index}]"
        if not isinstance(sample, dict):
            raise DatasetValidationError(f"{prefix} must be an object.")
        missing = required - sample.keys()
        if missing:
            raise DatasetValidationError(f"{prefix} is missing fields: {sorted(missing)}.")
        sample_id = sample["id"]
        if not isinstance(sample_id, str):
            raise DatasetValidationError(f"{prefix}.id must be a string.")
        if sample_id not in references:
            raise DatasetValidationError(f"{prefix} references unknown ID {sample_id!r}.")
        if sample_id in seen_ids:
            raise DatasetValidationError(f"Duplicate metadata ID: {sample_id}.")
        seen_ids.add(sample_id)

        filename = sample["audio_filename"]
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or Path(filename).suffix.lower() != ".wav"
        ):
            raise DatasetValidationError(f"{prefix} has an invalid audio_filename.")
        if filename in seen_audio:
            raise DatasetValidationError(f"Duplicate audio filename: {filename}.")
        seen_audio.add(filename)

        for field in ("speaker_id", "recording_condition", "recorded_at"):
            if not isinstance(sample[field], str) or not sample[field].strip():
                raise DatasetValidationError(f"{prefix}.{field} must be a non-empty string.")
        if type(sample["sample_rate"]) is not int or sample["sample_rate"] != SAMPLE_RATE:
            raise DatasetValidationError(f"{prefix}.sample_rate must be {SAMPLE_RATE}.")
        if type(sample["channels"]) is not int or sample["channels"] != CHANNELS:
            raise DatasetValidationError(f"{prefix}.channels must be 1.")
        if sample["format"] != "WAV" or sample["subtype"] != "PCM_16":
            raise DatasetValidationError(f"{prefix} must declare WAV / PCM_16.")

        audio_path = dataset_dir / "audio" / filename
        properties = validate_audio_file(audio_path)
        if properties["sample_rate"] != sample["sample_rate"]:
            raise DatasetValidationError(f"{prefix} sample rate does not match its audio.")
        if properties["channels"] != sample["channels"]:
            raise DatasetValidationError(f"{prefix} channel count does not match its audio.")
        recorded.append({**references[sample_id], **sample, "audio_path": audio_path})
    return references, recorded


def _write_json(path: Path, document: Dict) -> None:
    """Write JSON through a sibling temporary file to avoid partial metadata."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def record_sample(
    sample_id: str,
    speaker_id: str,
    recording_condition: str,
    duration: float,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> Path:
    """Record one reference sentence and add its validated audio metadata."""
    references, recorded = validate_dataset(dataset_dir)
    if sample_id not in references:
        raise DatasetValidationError(f"Unknown sample ID: {sample_id}.")
    if any(sample["id"] == sample_id for sample in recorded):
        raise DatasetValidationError(
            f"{sample_id} already has a recording; remove its metadata/audio to re-record."
        )
    if not isinstance(speaker_id, str) or not speaker_id.strip():
        raise DatasetValidationError("speaker_id must not be empty.")
    if not isinstance(recording_condition, str) or not recording_condition.strip():
        raise DatasetValidationError("recording_condition must not be empty.")
    if not isinstance(duration, (int, float)) or duration <= 0:
        raise DatasetValidationError("duration must be greater than zero.")

    print(f"\nSpeak this sentence:\n  {references[sample_id]['reference']}")
    input(f"Press Enter to record {duration:g} seconds...")
    try:
        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
        )
        sd.wait()
    except sd.PortAudioError as exc:
        raise RuntimeError(f"Microphone recording failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Unexpected microphone recording error: {exc}") from exc
    if audio is None or audio.size == 0:
        raise RuntimeError("Microphone returned empty audio.")

    audio_dir = dataset_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sample_id}.wav"
    audio_path = audio_dir / filename
    sf.write(str(audio_path), np.asarray(audio, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")

    metadata_entry = {
        "id": sample_id,
        "audio_filename": filename,
        "speaker_id": speaker_id.strip(),
        "recording_condition": recording_condition.strip(),
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "format": "WAV",
        "subtype": "PCM_16",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_document = _load_json(dataset_dir / "metadata.json")
    original_metadata = json.dumps(metadata_document, ensure_ascii=False, indent=2) + "\n"
    try:
        validate_audio_file(audio_path)
        metadata_document["samples"].append(metadata_entry)
        _write_json(dataset_dir / "metadata.json", metadata_document)
        validate_dataset(dataset_dir)
    except Exception:
        (dataset_dir / "metadata.json").write_text(original_metadata, encoding="utf-8")
        audio_path.unlink(missing_ok=True)
        raise
    logger.info("Recorded and validated %s", audio_path)
    return audio_path


def _select_sample(references: Dict[str, Dict], requested_id: Optional[str]) -> str:
    if requested_id:
        if requested_id not in references:
            raise DatasetValidationError(f"Unknown sample ID: {requested_id}.")
        return requested_id
    print("Available prompts:")
    for number, (sample_id, sample) in enumerate(references.items(), start=1):
        print(f"{number:2}. {sample_id:12} [{sample['category']}] {sample['reference']}")
    while True:
        selection = input("Select a prompt by number or ID: ").strip()
        if selection in references:
            return selection
        if selection.isdigit() and 1 <= int(selection) <= len(references):
            return list(references)[int(selection) - 1]
        print("Invalid selection; enter a listed number or sample ID.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--sample", help="Sample ID; omit to select from the prompt list.")
    parser.add_argument("--speaker-id", help="Speaker identifier (for example speaker_01).")
    parser.add_argument("--condition", help="Recording condition (for example quiet_room).")
    parser.add_argument("--duration", type=float, default=6.0, help="Recording length in seconds.")
    parser.add_argument("--list", action="store_true", help="List prompts without recording.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        references = load_references(args.dataset_dir / "references.json")
        if args.list:
            for sample_id, sample in references.items():
                print(f"{sample_id:12} [{sample['category']}] {sample['reference']}")
            return
        sample_id = _select_sample(references, args.sample)
        speaker_id = args.speaker_id or input("Speaker ID (for example speaker_01): ").strip()
        condition = args.condition or input(
            "Recording condition (for example quiet_room): "
        ).strip()
        record_sample(
            sample_id,
            speaker_id,
            condition,
            args.duration,
            args.dataset_dir,
        )
        print(f"Saved {sample_id}.wav in {args.dataset_dir / 'audio'}.")
    except (DatasetValidationError, RuntimeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
