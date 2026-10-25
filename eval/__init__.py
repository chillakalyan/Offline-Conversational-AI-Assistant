"""
eval — ASR evaluation framework for LockIN Voice AI Assistant.

Provides Word Error Rate (WER), Character Error Rate (CER),
Sentence Exact Match (EM), and Entity Exact Match & F1 metrics
for numbers, names, dates, and student IDs.
"""

from eval.metrics import (
    compute_wer,
    compute_cer,
    compute_sentence_em,
    extract_entities,
    evaluate_entities,
    normalize_text,
    evaluate_asr_pair,
)

__all__ = [
    "compute_wer",
    "compute_cer",
    "compute_sentence_em",
    "extract_entities",
    "evaluate_entities",
    "normalize_text",
    "evaluate_asr_pair",
]
