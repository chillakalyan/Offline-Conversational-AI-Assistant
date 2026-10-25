"""
tests/test_eval.py — Automated tests for the ASR evaluation framework.

Tests:
  - Text normalization
  - WER (Word Error Rate) with substitutions, insertions, deletions
  - CER (Character Error Rate)
  - Sentence Exact Match
  - Entity extraction (numbers, names, dates, student IDs)
  - Entity evaluation metrics (Precision, Recall, F1, Exact Match Rate)
  - ASREvaluator runner end-to-end
"""

import json
from pathlib import Path
import pytest

from eval.metrics import (
    aggregate_asr_results,
    compute_cer,
    compute_sentence_em,
    compute_wer,
    evaluate_entities,
    evaluate_expected_entities,
    evaluate_numeric_content,
    extract_numeric_content,
    extract_entities,
    normalize_text,
)
from eval.evaluate_asr import ASREvaluator


# ===========================================================================
# Metric Calculations Tests
# ===========================================================================

def test_normalize_text():
    """Text normalization lowercases, strips punctuation, and normalizes space."""
    raw = "  Hello, World! This is STU-1029—test.  "
    norm = normalize_text(raw)
    assert norm == "hello world this is stu-1029 test"


def test_compute_wer_identical():
    """Identical reference and hypothesis yield 0.0 WER."""
    res = compute_wer("the quick brown fox", "the quick brown fox")
    assert res["wer"] == 0.0
    assert res["substitutions"] == 0
    assert res["deletions"] == 0
    assert res["insertions"] == 0
    assert res["hits"] == 4


def test_compute_wer_all_deleted():
    """Empty hypothesis against 4-word reference yields 1.0 WER."""
    res = compute_wer("one two three four", "")
    assert res["wer"] == 1.0
    assert res["deletions"] == 4


def test_compute_wer_all_inserted():
    """Hypothesis against empty reference yields 1.0 WER."""
    res = compute_wer("", "one two three")
    assert res["wer"] == 1.0
    assert res["insertions"] == 3


def test_compute_wer_substitutions_and_insertions():
    """Calculates mixed edit operations correctly."""
    # Ref: "speech recognition is hard" (4 words)
    # Hyp: "speech cognition was very hard" (5 words)
    # recognition -> cognition (sub)
    # is -> was (sub)
    # very (ins)
    res = compute_wer("speech recognition is hard", "speech cognition was very hard")
    assert res["ref_words"] == 4
    assert res["substitutions"] == 2
    assert res["insertions"] == 1
    assert res["wer"] == 0.75  # (2 + 1) / 4


def test_compute_cer_identical():
    """Identical strings yield 0.0 CER."""
    res = compute_cer("lockin voice", "lockin voice")
    assert res["cer"] == 0.0
    assert res["hits"] == len("lockin voice")


def test_compute_cer_edits():
    """Character errors are computed correctly."""
    res = compute_cer("cat", "cut")
    assert res["cer"] == round(1 / 3, 4)
    assert res["substitutions"] == 1


def test_compute_sentence_em():
    """Exact match returns 1.0 on matching normalized text and 0.0 otherwise."""
    assert compute_sentence_em("Hello World!", "hello world") == 1.0
    assert compute_sentence_em("Hello World", "Hello There") == 0.0


def test_corpus_wer_uses_total_errors_and_reference_words():
    utterances = [
        ASREvaluator().evaluate_item(
            {"reference": "x", "hypothesis": "x"}
        ),
        ASREvaluator().evaluate_item(
            {"reference": "a b c d", "hypothesis": "x b c d"}
        ),
    ]

    summary = aggregate_asr_results(utterances)

    assert summary["per_utterance_average_wer"] == 0.125
    assert summary["corpus_wer"] == 0.2


def test_corpus_cer_uses_total_errors_and_reference_characters():
    utterances = [
        ASREvaluator().evaluate_item(
            {"reference": "x", "hypothesis": "x"}
        ),
        ASREvaluator().evaluate_item(
            {"reference": "abcd", "hypothesis": "xbcd"}
        ),
    ]

    summary = aggregate_asr_results(utterances)

    assert summary["per_utterance_average_cer"] == 0.125
    assert summary["corpus_cer"] == 0.2


def test_empty_corpus_metrics_are_null():
    summary = aggregate_asr_results([])

    assert summary["samples"] == 0
    assert summary["corpus_wer"] is None
    assert summary["corpus_cer"] is None
    assert summary["exact_match"] is None


@pytest.mark.parametrize(
    ("reference", "hypothesis", "matches"),
    [
        ("The accuracy is 97.06 percent.", "The accuracy is 97.06%.", True),
        ("The accuracy is 97.06 percent.", "The accuracy is 97.6 percent.", False),
        ("My ID is 22CS108001.", "My ID is 22CS108001.", True),
        ("Meet on November 18th, 2026 at 10:30 AM.", "Meet on November 18, 2026 at 10:30am.", True),
        ("The result is 7.", "The result is 8.", False),
    ],
)
def test_numeric_content_matching(reference, hypothesis, matches):
    result = evaluate_numeric_content(reference, hypothesis)

    assert result["exact_match"] is matches


def test_extract_numeric_content_recognizes_numeric_types():
    values = extract_numeric_content(
        "ID 22CS108001, date November 18th, 2026, at 10:30 AM: 97.06 percent."
    )

    assert {value["type"] for value in values} == {"id", "date", "time", "percent"}
    assert ("id", "22CS108001") in {
        (value["type"], value["value"]) for value in values
    }


def test_expected_entity_matching_and_entity_precision_recall_f1():
    perfect = evaluate_expected_entities(
        ["Chilla Kalyan", "Professor Meera Rao"],
        "Chilla Kalyan met Professor Meera Rao.",
    )
    partial = evaluate_expected_entities(
        ["Chilla Kalyan", "Professor Meera Rao"],
        "Chilla Kalyan met Professor Rao.",
    )

    assert perfect["exact_match"] is True
    assert (perfect["precision"], perfect["recall"], perfect["f1"]) == (1.0, 1.0, 1.0)
    assert partial["exact_match"] is False
    assert partial["tp"] == 1
    assert partial["fn"] == 1
    assert partial["fp"] == 1
    assert (partial["precision"], partial["recall"], partial["f1"]) == (0.5, 0.5, 0.5)


# ===========================================================================
# Entity Extraction & Evaluation Tests
# ===========================================================================

def test_extract_entities_student_ids():
    """Extracts valid student IDs."""
    text = "Student STU-1029 and S98234 registered for course 2024CS101."
    entities = extract_entities(text)
    assert "STU-1029" in entities["student_ids"]
    assert "S98234" in entities["student_ids"]
    assert "2024CS101" in entities["student_ids"]


def test_extract_entities_numbers():
    """Extracts digits, decimals, and spoken number words."""
    text = "I completed 48 credits with a 3.85 GPA and three electives."
    entities = extract_entities(text)
    assert "48" in entities["numbers"]
    assert "3.85" in entities["numbers"]
    assert "three" in entities["numbers"]


def test_extract_entities_dates():
    """Extracts dates, days, and times."""
    text = "Exam is on October 24th, 2026 at 10:30 AM next Monday."
    entities = extract_entities(text)
    assert any("october 24th" in d.lower() for d in entities["dates"])
    assert any("10:30 am" in d.lower() for d in entities["dates"])
    assert "monday" in entities["dates"]


def test_extract_entities_names():
    """Extracts person names and assistant persona."""
    text = "LockIN asked Professor Sharma to meet Kalyan."
    entities = extract_entities(text)
    assert any("Professor Sharma" in n for n in entities["names"])
    assert any("LockIN" in n or "lockin" in n.lower() for n in entities["names"])
    assert any("Kalyan" in n for n in entities["names"])


def test_evaluate_entities_metrics():
    """Computes precision, recall, and distinguishes entity recall from exact match."""
    ref_entities = {
        "student_ids": ["STU-1029", "S12345"],
        "numbers": ["101"],
        "dates": [],
        "names": ["LockIN"],
    }
    hyp_entities = {
        "student_ids": ["STU-1029"],  # missed S12345 (FN=1) -> Recall=0.5, EM=0.0
        "numbers": ["101", "202"],   # extra 202 (FP=1) -> Precision=0.5, Recall=1.0, EM=0.0
        "dates": [],                  # both empty -> EM=1.0, Prec=1.0, Rec=1.0
        "names": ["LockIN"],          # identical -> EM=1.0, Prec=1.0, Rec=1.0
    }

    metrics = evaluate_entities(ref_entities, hyp_entities)

    # Student IDs: 1 TP, 0 FP, 1 FN -> Rec = 0.5, Prec = 1.0, EM = 0.0 (incomplete set)
    assert metrics["student_ids"]["tp"] == 1
    assert metrics["student_ids"]["precision"] == 1.0
    assert metrics["student_ids"]["recall"] == 0.5
    assert metrics["student_ids"]["exact_match"] == 0.0

    # Numbers: 1 TP, 1 FP, 0 FN -> Rec = 1.0, Prec = 0.5, EM = 0.0 (extra entity)
    assert metrics["numbers"]["tp"] == 1
    assert metrics["numbers"]["fp"] == 1
    assert metrics["numbers"]["precision"] == 0.5
    assert metrics["numbers"]["recall"] == 1.0
    assert metrics["numbers"]["exact_match"] == 0.0

    # Dates: both empty -> perfect match of empty set
    assert metrics["dates"]["exact_match"] == 1.0
    assert metrics["dates"]["precision"] == 1.0
    assert metrics["dates"]["recall"] == 1.0

    # Names: exact match of single item
    assert metrics["names"]["tp"] == 1
    assert metrics["names"]["exact_match"] == 1.0
    assert metrics["names"]["precision"] == 1.0
    assert metrics["names"]["recall"] == 1.0

    # Overall: 3 TP out of 4 references -> Recall = 0.75, not an overall EM
    assert metrics["overall"]["tp"] == 3
    assert metrics["overall"]["recall"] == 0.75
    assert metrics["overall"]["exact_match"] == 0.0


def test_evaluate_entities_false_positives():
    """False positives: reference is empty, hypothesis contains hallucinated entities."""
    ref_entities = {"student_ids": [], "numbers": [], "dates": [], "names": []}
    hyp_entities = {
        "student_ids": ["STU-9999"],
        "numbers": ["42"],
        "dates": [],
        "names": [],
    }

    metrics = evaluate_entities(ref_entities, hyp_entities)

    # student_ids: TP=0, FP=1, FN=0 -> Precision=0.0, Recall=0.0, EM=0.0
    assert metrics["student_ids"]["tp"] == 0
    assert metrics["student_ids"]["fp"] == 1
    assert metrics["student_ids"]["precision"] == 0.0
    assert metrics["student_ids"]["recall"] == 0.0
    assert metrics["student_ids"]["exact_match"] == 0.0

    # dates: both empty -> Precision=1.0, Recall=1.0, EM=1.0
    assert metrics["dates"]["exact_match"] == 1.0
    assert metrics["dates"]["fp"] == 0

    # overall has FPs -> EM must be 0.0
    assert metrics["overall"]["exact_match"] == 0.0
    assert metrics["overall"]["fp"] == 2


def test_evaluate_entities_false_negatives():
    """False negatives: reference contains entities, hypothesis is completely empty."""
    ref_entities = {
        "student_ids": ["STU-1029"],
        "numbers": ["100"],
        "dates": ["Monday"],
        "names": ["Alice"],
    }
    hyp_entities = {"student_ids": [], "numbers": [], "dates": [], "names": []}

    metrics = evaluate_entities(ref_entities, hyp_entities)

    for cat in ["student_ids", "numbers", "dates", "names"]:
        assert metrics[cat]["tp"] == 0
        assert metrics[cat]["fn"] == 1
        assert metrics[cat]["precision"] == 0.0
        assert metrics[cat]["recall"] == 0.0
        assert metrics[cat]["f1"] == 0.0
        assert metrics[cat]["exact_match"] == 0.0

    assert metrics["overall"]["tp"] == 0
    assert metrics["overall"]["fn"] == 4
    assert metrics["overall"]["exact_match"] == 0.0


def test_evaluate_entities_empty_sets():
    """Empty sets: when both reference and hypothesis are empty, all metrics are 1.0."""
    ref_entities = {"student_ids": [], "numbers": [], "dates": [], "names": []}
    hyp_entities = {"student_ids": [], "numbers": [], "dates": [], "names": []}

    metrics = evaluate_entities(ref_entities, hyp_entities)

    for cat in ["student_ids", "numbers", "dates", "names", "overall"]:
        assert metrics[cat]["tp"] == 0
        assert metrics[cat]["fp"] == 0
        assert metrics[cat]["fn"] == 0
        assert metrics[cat]["precision"] == 1.0
        assert metrics[cat]["recall"] == 1.0
        assert metrics[cat]["f1"] == 1.0
        assert metrics[cat]["exact_match"] == 1.0


def test_evaluate_entities_exact_vs_partial_matches():
    """Distinguishes exact match from partial token/substring match."""
    # Ref has full student ID "STU-1029", hyp only extracted number "1029"
    ref_entities = {
        "student_ids": ["STU-1029"],
        "numbers": [],
        "dates": ["October 24th, 2026"],
        "names": ["Professor Kalyan Sharma"],
    }
    hyp_entities = {
        "student_ids": ["1029"],             # partial substring match
        "numbers": [],
        "dates": ["October 24th"],           # partial date match
        "names": ["Kalyan Sharma"],          # partial name match
    }

    metrics = evaluate_entities(ref_entities, hyp_entities)

    # Partial matches should NOT count as exact true positives (exact_tp == 0)
    assert metrics["student_ids"]["tp"] == 0
    assert metrics["student_ids"]["partial_tp"] == 1
    assert metrics["student_ids"]["exact_match"] == 0.0
    assert metrics["student_ids"]["recall"] == 0.0  # strict exact recall

    assert metrics["dates"]["tp"] == 0
    assert metrics["dates"]["partial_tp"] == 1
    assert metrics["dates"]["exact_match"] == 0.0

    assert metrics["names"]["tp"] == 0
    assert metrics["names"]["partial_tp"] == 1
    assert metrics["names"]["exact_match"] == 0.0

    # Overall exact match is 0.0 due to partial matches
    assert metrics["overall"]["exact_match"] == 0.0
    assert metrics["overall"]["partial_tp"] == 3


# ===========================================================================
# End-to-End Evaluator Runner Test
# ===========================================================================

def test_asr_evaluator_sentence_em_is_not_entity_em():
    """Dataset sentence exact-match must not be overwritten by entity-set exact match."""
    evaluator = ASREvaluator()
    summary = evaluator.run(
        [
            {
                "id": "A",
                "domain": "grading_and_academics",
                "reference": "I scored 94 with Professor Sharma.",
                "hypothesis": "I scored 94 with Professor Sharma.",
                "entities": {
                    "student_ids": [],
                    "numbers": ["94"],
                    "dates": [],
                    "names": [],
                },
            },
            {
                "id": "B",
                "domain": "simulated_asr_error_spelling",
                "reference": "LockIN can you email Professor Miller by Tuesday?",
                "hypothesis": "Lock in can you email professor miller by Tuesday?",
                "entities": {
                    "student_ids": [],
                    "numbers": [],
                    "dates": ["Tuesday"],
                    "names": ["LockIN", "Professor Miller"],
                },
            },
        ]
    )
    assert summary["sentence_exact_match_rate"] == 0.5
    assert summary["entity_metrics"]["overall"]["exact_match_rate"] == 0.0


def test_asr_evaluator_run_end_to_end(tmp_path):
    """ASREvaluator runs on benchmark dataset and exports results."""
    evaluator = ASREvaluator()
    summary = evaluator.run()

    assert summary["total_cases"] >= 10
    assert 0.0 <= summary["average_wer"] <= 1.0
    assert 0.0 <= summary["average_cer"] <= 1.0
    assert "overall" in summary["entity_metrics"]
    expected_sentence_em = sum(r["sentence_em"] for r in evaluator.results) / len(evaluator.results)
    assert summary["sentence_exact_match_rate"] == round(expected_sentence_em, 4)

    # Test export
    json_path = tmp_path / "test_eval.json"
    md_path = tmp_path / "test_eval.md"

    evaluator.export_json_results(json_path)
    evaluator.export_markdown_report(md_path)

    assert json_path.exists()
    assert md_path.exists()

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "summary" in data
    assert "cases" in data


def test_markdown_report_counts_partial_matches_in_ground_truth(tmp_path):
    """Ground-truth totals include entities that only matched partially."""
    evaluator = ASREvaluator()
    evaluator.run()
    report_path = tmp_path / "eval_report.md"

    evaluator.export_markdown_report(report_path)

    report = report_path.read_text(encoding="utf-8")
    assert "| Names | 6 | 5 |" in report
    assert "| **Overall** | 33 | 30 |" in report
