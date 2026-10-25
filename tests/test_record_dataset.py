import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from eval.evaluate_asr import run_evaluation, run_recorded_audio_evaluation
from eval.record_dataset import (
    DatasetValidationError,
    load_references,
    record_sample,
    validate_audio_file,
    validate_dataset,
)


def make_dataset(directory: Path, reference="Please open the study guide.") -> Path:
    (directory / "audio").mkdir(parents=True)
    (directory / "references.json").write_text(
        json.dumps({
            "samples": [{
                "id": "NORMAL-001",
                "reference": reference,
                "language": "en",
                "category": "normal_speech",
                "contains_numbers": False,
                "contains_names_entities": False,
            }]
        }),
        encoding="utf-8",
    )
    (directory / "metadata.json").write_text('{"samples": []}', encoding="utf-8")
    return directory


def write_metadata(directory: Path, filename="NORMAL-001.wav", **overrides) -> None:
    entry = {
        "id": "NORMAL-001",
        "audio_filename": filename,
        "speaker_id": "speaker_01",
        "recording_condition": "quiet_room",
        "sample_rate": 16000,
        "channels": 1,
        "format": "WAV",
        "subtype": "PCM_16",
        "recorded_at": "2026-10-22T12:00:00+00:00",
    }
    entry.update(overrides)
    (directory / "metadata.json").write_text(
        json.dumps({"samples": [entry]}),
        encoding="utf-8",
    )


def make_grouped_dataset(directory: Path) -> Path:
    (directory / "audio").mkdir(parents=True)
    samples = [
        {
            "id": "NORMAL-001",
            "reference": "I can study today.",
            "language": "en",
            "category": "normal_speech",
            "contains_numbers": False,
            "contains_names_entities": False,
        },
        {
            "id": "NUMBERS-001",
            "reference": "The accuracy is 97.06 percent.",
            "language": "en",
            "category": "numbers",
            "contains_numbers": True,
            "contains_names_entities": False,
        },
        {
            "id": "NAMES-001",
            "reference": "My name is Chilla Kalyan.",
            "language": "en",
            "category": "names",
            "contains_numbers": False,
            "contains_names_entities": True,
            "expected_entities": ["Chilla Kalyan"],
        },
    ]
    (directory / "references.json").write_text(
        json.dumps({"samples": samples}),
        encoding="utf-8",
    )
    metadata = []
    for index, sample in enumerate(samples):
        filename = f"{sample['id']}.wav"
        sf.write(
            directory / "audio" / filename,
            np.zeros(16000, dtype=np.float32),
            16000,
            subtype="PCM_16",
        )
        metadata.append({
            "id": sample["id"],
            "audio_filename": filename,
            "speaker_id": "speaker_01" if index < 2 else "speaker_02",
            "recording_condition": "quiet_room" if index == 0 else "background_noise",
            "sample_rate": 16000,
            "channels": 1,
            "format": "WAV",
            "subtype": "PCM_16",
            "recorded_at": "2026-10-22T12:00:00+00:00",
        })
    (directory / "metadata.json").write_text(
        json.dumps({"samples": metadata}),
        encoding="utf-8",
    )
    return directory


def test_reference_inventory_has_required_categories_and_thirty_samples():
    references = load_references(Path(__file__).resolve().parents[1] / "eval" / "dataset" / "references.json")

    assert len(references) >= 30
    categories = {sample["category"] for sample in references.values()}
    assert categories == {
        "normal_speech",
        "numbers",
        "names",
        "numbers_and_names",
        "technical_terms",
        "short_commands",
    }
    assert all(sample["reference"].strip() for sample in references.values())


def test_empty_metadata_is_valid_and_does_not_require_audio(tmp_path):
    dataset = make_dataset(tmp_path)

    references, recordings = validate_dataset(dataset)

    assert len(references) == 1
    assert recordings == []


def test_reference_transcript_must_exist(tmp_path):
    dataset = make_dataset(tmp_path, reference="  ")

    with pytest.raises(DatasetValidationError, match="non-empty reference"):
        validate_dataset(dataset)


def test_recording_metadata_must_have_readable_audio(tmp_path):
    dataset = make_dataset(tmp_path)
    write_metadata(dataset)

    with pytest.raises(DatasetValidationError, match="does not exist"):
        validate_dataset(dataset)


@pytest.mark.parametrize(
    ("sample_rate", "channels", "message"),
    [
        (8000, 1, "must be 16000 Hz"),
        (16000, 2, "must be mono"),
    ],
)
def test_audio_validator_rejects_wrong_rate_or_channel_count(
    tmp_path, sample_rate, channels, message
):
    wav_path = tmp_path / "wrong.wav"
    sf.write(
        wav_path,
        np.zeros((1600, channels), dtype=np.float32),
        sample_rate,
        subtype="PCM_16",
    )

    with pytest.raises(DatasetValidationError, match=message):
        validate_audio_file(wav_path)


def test_audio_validator_rejects_non_pcm16_wav(tmp_path):
    wav_path = tmp_path / "float.wav"
    sf.write(wav_path, np.zeros(1600, dtype=np.float32), 16000, subtype="FLOAT")

    with pytest.raises(DatasetValidationError, match="16-bit PCM"):
        validate_audio_file(wav_path)


def test_metadata_rejects_unsafe_audio_filename(tmp_path):
    dataset = make_dataset(tmp_path)
    write_metadata(dataset, filename="..\\outside.wav")

    with pytest.raises(DatasetValidationError, match="invalid audio_filename"):
        validate_dataset(dataset)


def test_record_sample_saves_valid_wav_and_associated_metadata(tmp_path):
    dataset = make_dataset(tmp_path)
    captured = np.full((16000 * 2, 1), 0.1, dtype=np.float32)

    with (
        patch("builtins.input", return_value=""),
        patch("eval.record_dataset.sd.rec", return_value=captured),
        patch("eval.record_dataset.sd.wait"),
    ):
        path = record_sample(
            "NORMAL-001",
            speaker_id="speaker_01",
            recording_condition="quiet_room",
            duration=2,
            dataset_dir=dataset,
        )

    assert path.exists()
    properties = validate_audio_file(path)
    assert properties["sample_rate"] == 16000
    assert properties["channels"] == 1
    assert json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))["samples"][0][
        "speaker_id"
    ] == "speaker_01"
    _, samples = validate_dataset(dataset)
    assert samples[0]["id"] == "NORMAL-001"


def test_recorded_evaluation_does_not_report_metrics_without_audio(tmp_path):
    dataset = make_dataset(tmp_path)
    stt = MagicMock()

    result = run_recorded_audio_evaluation(dataset, stt=stt)

    assert result["status"] == "not_started"
    assert "average_wer" not in result
    assert "average_cer" not in result
    assert result["evaluation_type"] == "empirical_recorded_evaluation"
    assert result["total_samples"] == 0
    assert result["valid_samples"] == 0
    assert result["corpus_wer"] is None
    assert result["corpus_cer"] is None
    assert result["category_results"] == {}
    assert result["condition_results"] == {}
    assert result["speaker_results"] == {}
    assert result["number_results"] == {}
    assert result["entity_results"] == {}
    assert result["utterances"] == []
    stt.transcribe.assert_not_called()


def test_recorded_only_runner_separates_fixture_and_real_results(tmp_path):
    output_json = tmp_path / "recorded-results.json"
    output_md = tmp_path / "recorded-report.md"
    dataset = make_dataset(tmp_path / "empty_dataset")

    with patch("eval.record_dataset.DEFAULT_DATASET_DIR", dataset):
        result = run_evaluation(
            output_json=str(output_json),
            output_md=str(output_md),
            recorded_audio=True,
        )
    saved = json.loads(output_json.read_text(encoding="utf-8"))

    assert result["fixture_evaluation"] is None
    assert result["empirical_recorded_evaluation"]["status"] == "not_started"
    assert "average_wer" not in result["empirical_recorded_evaluation"]
    assert result["empirical_recorded_evaluation"]["corpus_wer"] is None
    assert result["empirical_recorded_evaluation"]["corpus_cer"] is None
    assert saved["fixture_evaluation"] is None
    assert saved["empirical_recorded_evaluation"]["status"] == "not_started"
    assert (
        "Empirical ASR results are unavailable because no human-recorded "
        "evaluation audio has been provided."
    ) in output_md.read_text(encoding="utf-8")


def test_recorded_results_aggregate_by_category_condition_and_speaker(tmp_path):
    dataset = make_grouped_dataset(tmp_path)
    hypotheses = {
        "NORMAL-001.wav": "I can study today.",
        "NUMBERS-001.wav": "The accuracy is 97.6 percent.",
        "NAMES-001.wav": "My name is Chilla Kalyan.",
    }
    stt = MagicMock()
    stt.transcribe.side_effect = lambda path, language: hypotheses[Path(path).name]

    result = run_recorded_audio_evaluation(dataset, stt=stt)

    assert result["valid_samples"] == 3
    assert set(result["category_results"]) == {"normal_speech", "numbers", "names"}
    assert result["category_results"]["numbers"]["samples"] == 1
    assert result["category_results"]["names"]["exact_match"] == 1.0
    assert set(result["condition_results"]) == {"quiet_room", "background_noise"}
    assert result["condition_results"]["background_noise"]["samples"] == 2
    assert set(result["speaker_results"]) == {"speaker_01", "speaker_02"}
    assert result["speaker_results"]["speaker_01"]["samples"] == 2
    assert result["number_results"]["samples"] == 1
    assert result["number_results"]["exact_match"] == 0.0
    assert result["entity_results"]["samples"] == 1
    assert result["entity_results"]["precision"] == 1.0
    assert result["entity_results"]["recall"] == 1.0
    assert "number_error" in result["utterances"][1]["error_categories"]
    assert "substitution" in result["utterances"][1]["error_categories"]
    assert result["utterances"][0]["error_categories"] == ["no_error"]


def test_recorded_evaluation_transcribes_recording_before_scoring(tmp_path):
    dataset = make_dataset(tmp_path)
    audio_path = dataset / "audio" / "NORMAL-001.wav"
    sf.write(audio_path, np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
    write_metadata(dataset)
    stt = MagicMock()
    stt.transcribe.return_value = "Please open the study guide."

    result = run_recorded_audio_evaluation(dataset, stt=stt)

    stt.transcribe.assert_called_once_with(str(audio_path), language="en")
    assert result["status"] == "completed"
    assert result["total_samples"] == 1
    assert result["valid_samples"] == 1
    assert result["per_utterance_average_wer"] == 0.0
    assert result["per_utterance_average_cer"] == 0.0
    assert result["corpus_wer"] == 0.0
    assert result["corpus_cer"] == 0.0
    assert result["utterances"][0]["hypothesis"] == "Please open the study guide."
