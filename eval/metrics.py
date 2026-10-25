"""
eval/metrics.py — Core metrics for ASR evaluation.

Implements:
  1. Text Normalization (standardized punctuation and case handling)
  2. Word Error Rate (WER) via dynamic programming Levenshtein alignment
  3. Character Error Rate (CER) via character-level Levenshtein alignment
  4. Sentence Exact Match (EM) binary metric
  5. Entity Extraction & Exact-Match Evaluation for:
     - numbers (digits, percentages, spoken number words)
     - names (person names, titles, persona names)
     - dates (calendar dates, days of week, relative dates, times)
     - student IDs (alphanumeric student/course IDs)
"""

import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Stopwords and Spoken Number Vocabulary
# ---------------------------------------------------------------------------

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
}

MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

DAYS_OF_WEEK = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}

COMMON_ENGLISH_WORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "to", "in", "for", "on",
    "with", "at", "by", "from", "up", "about", "into", "over", "after",
    "can", "could", "will", "would", "shall", "should", "may", "might",
    "what", "when", "where", "who", "whom", "which", "why", "how",
    "my", "your", "his", "her", "its", "our", "their", "please", "help",
    "student", "course", "assignment", "deadline", "grade", "exam",
}


# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str, remove_punctuation: bool = True) -> str:
    """
    Standardize text for ASR comparison:
      - Lowercase
      - Replace em-dashes and underscores with spaces
      - Remove or normalize punctuation (preserving hyphens within words if specified)
      - Collapse multiple whitespace to single space
      - Strip leading and trailing whitespace
    """
    if not text:
        return ""

    text = text.lower().strip()
    text = re.sub(r"[—–_]", " ", text)

    if remove_punctuation:
        # Keep alphanumeric characters and whitespace
        # Preserve hyphens inside alphanumeric tokens like STU-1029
        # Replace other punctuation with space
        cleaned_chars = []
        for i, char in enumerate(text):
            if char.isalnum() or char.isspace():
                cleaned_chars.append(char)
            elif char == "-" and 0 < i < len(text) - 1 and text[i - 1].isalnum() and text[i + 1].isalnum():
                cleaned_chars.append(char)
            else:
                cleaned_chars.append(" ")
        text = "".join(cleaned_chars)

    # Collapse repeated whitespace
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Levenshtein Dynamic Programming on Tokens
# ---------------------------------------------------------------------------

def _levenshtein_dp(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """
    Compute substitutions (S), deletions (D), insertions (I), hits (H)
    between ref and hyp tokens using standard Levenshtein dynamic programming.
    """
    m, n = len(ref), len(hyp)
    # d[i][j] stores the minimum edit distance between ref[:i] and hyp[:j]
    d = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(
                    d[i - 1][j],      # deletion from ref
                    d[i][j - 1],      # insertion into hyp
                    d[i - 1][j - 1],  # substitution
                )

    # Backtracking to extract individual operation counts
    i, j = m, n
    subs = 0
    dels = 0
    ins = 0
    hits = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            hits += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dels += 1
            i -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            # Fallback path if tie-breaking leads here
            if i > 0 and j > 0:
                subs += 1
                i -= 1
                j -= 1
            elif i > 0:
                dels += 1
                i -= 1
            else:
                ins += 1
                j -= 1

    return subs, dels, ins, hits


# ---------------------------------------------------------------------------
# Word Error Rate (WER)
# ---------------------------------------------------------------------------

def compute_wer(reference: str, hypothesis: str, normalize: bool = True) -> Dict[str, float]:
    """
    Calculate Word Error Rate (WER):
      WER = (Substitutions + Deletions + Insertions) / Reference_Word_Count

    Returns a dict with:
      - wer: float (0.0 to 1.0+, 0.0 being perfect match)
      - substitutions: int
      - deletions: int
      - insertions: int
      - hits: int
      - ref_words: int
      - hyp_words: int
    """
    ref_norm = normalize_text(reference) if normalize else reference.strip()
    hyp_norm = normalize_text(hypothesis) if normalize else hypothesis.strip()

    ref_tokens = ref_norm.split() if ref_norm else []
    hyp_tokens = hyp_norm.split() if hyp_norm else []

    if not ref_tokens:
        if not hyp_tokens:
            return {
                "wer": 0.0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "hits": 0,
                "ref_words": 0,
                "hyp_words": 0,
            }
        return {
            "wer": 1.0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(hyp_tokens),
            "hits": 0,
            "ref_words": 0,
            "hyp_words": len(hyp_tokens),
        }

    subs, dels, ins, hits = _levenshtein_dp(ref_tokens, hyp_tokens)
    wer = (subs + dels + ins) / len(ref_tokens)

    return {
        "wer": round(wer, 4),
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "hits": hits,
        "ref_words": len(ref_tokens),
        "hyp_words": len(hyp_tokens),
    }


# ---------------------------------------------------------------------------
# Character Error Rate (CER)
# ---------------------------------------------------------------------------

def compute_cer(reference: str, hypothesis: str, normalize: bool = True) -> Dict[str, float]:
    """
    Calculate Character Error Rate (CER):
      CER = (Substitutions + Deletions + Insertions) / Reference_Character_Count

    Returns a dict with:
      - cer: float
      - substitutions: int
      - deletions: int
      - insertions: int
      - hits: int
      - ref_chars: int
      - hyp_chars: int
    """
    ref_norm = normalize_text(reference) if normalize else reference.strip()
    hyp_norm = normalize_text(hypothesis) if normalize else hypothesis.strip()

    ref_chars = list(ref_norm)
    hyp_chars = list(hyp_norm)

    if not ref_chars:
        if not hyp_chars:
            return {
                "cer": 0.0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "hits": 0,
                "ref_chars": 0,
                "hyp_chars": 0,
            }
        return {
            "cer": 1.0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": len(hyp_chars),
            "hits": 0,
            "ref_chars": 0,
            "hyp_chars": len(hyp_chars),
        }

    subs, dels, ins, hits = _levenshtein_dp(ref_chars, hyp_chars)
    cer = (subs + dels + ins) / len(ref_chars)

    return {
        "cer": round(cer, 4),
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "hits": hits,
        "ref_chars": len(ref_chars),
        "hyp_chars": len(hyp_chars),
    }


# ---------------------------------------------------------------------------
# Sentence Exact Match (EM)
# ---------------------------------------------------------------------------

def compute_sentence_em(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """Return 1.0 if normalized reference equals normalized hypothesis, else 0.0."""
    ref_norm = normalize_text(reference) if normalize else reference.strip()
    hyp_norm = normalize_text(hypothesis) if normalize else hypothesis.strip()
    return 1.0 if ref_norm == hyp_norm else 0.0


def aggregate_asr_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-utterance metric details into corpus rates and mean rates."""
    if not results:
        return {
            "samples": 0,
            "corpus_wer": None,
            "corpus_cer": None,
            "per_utterance_average_wer": None,
            "per_utterance_average_cer": None,
            "exact_match": None,
            "reference_words": 0,
            "reference_characters": 0,
            "word_errors": 0,
            "character_errors": 0,
        }

    word_errors = sum(
        result["wer_details"]["substitutions"]
        + result["wer_details"]["deletions"]
        + result["wer_details"]["insertions"]
        for result in results
    )
    character_errors = sum(
        result["cer_details"]["substitutions"]
        + result["cer_details"]["deletions"]
        + result["cer_details"]["insertions"]
        for result in results
    )
    reference_words = sum(result["wer_details"]["ref_words"] for result in results)
    reference_characters = sum(result["cer_details"]["ref_chars"] for result in results)
    return {
        "samples": len(results),
        "corpus_wer": round(word_errors / reference_words, 4) if reference_words else None,
        "corpus_cer": (
            round(character_errors / reference_characters, 4)
            if reference_characters else None
        ),
        "per_utterance_average_wer": round(
            sum(result["wer"] for result in results) / len(results), 4
        ),
        "per_utterance_average_cer": round(
            sum(result["cer"] for result in results) / len(results), 4
        ),
        "exact_match": round(
            sum(result["sentence_em"] for result in results) / len(results), 4
        ),
        "reference_words": reference_words,
        "reference_characters": reference_characters,
        "word_errors": word_errors,
        "character_errors": character_errors,
    }


_MONTH_PATTERN = (
    r"(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|"
    r"oct|nov|dec)"
)
_NUMERIC_CONTENT_PATTERN = re.compile(
    rf"(?P<id>(?<![A-Za-z0-9])(?=[A-Za-z0-9_-]*[A-Za-z])"
    rf"(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*(?![A-Za-z0-9]))"
    rf"|(?P<date>(?<!\w)(?:\d{{1,4}}[-/]\d{{1,2}}[-/]\d{{2,4}}|"
    rf"{_MONTH_PATTERN}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?(?:\s+of)?\s+{_MONTH_PATTERN}"
    rf"(?:,?\s+\d{{4}})?)(?!\w))"
    r"|(?P<time>(?<!\d)\d{1,2}:\d{2}(?:\s*(?:a\.?m\.?|p\.?m\.?))?(?!\d))"
    r"|(?P<percent>(?<![\w.])\d+(?:[.,]\d+)?\s*(?:%|percent(?:age)?\b))"
    r"|(?P<number>(?<![\w.])\d+(?:[.,]\d+)?(?![\w]))",
    re.IGNORECASE,
)


def _canonical_numeric_value(kind: str, value: str) -> str:
    """Normalize surface formatting while preserving numeric distinctions."""
    normalized = value.casefold().strip()
    if kind == "id":
        return re.sub(r"[-_\s]", "", normalized).upper()
    if kind == "percent":
        amount = re.sub(r"\s*(?:%|percent(?:age)?)$", "", normalized).strip()
        return f"{amount.replace(',', '.') }%"
    if kind == "time":
        normalized = re.sub(r"\s+", "", normalized)
        normalized = normalized.replace(".", "")
        return normalized.upper()
    if kind == "date":
        normalized = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", normalized)
        normalized = re.sub(r"[,]+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()
    return normalized.replace(",", ".")


def extract_numeric_content(text: str) -> List[Dict[str, str]]:
    """Extract ordered numeric, date, time, percentage, and identifier spans."""
    content = []
    for match in _NUMERIC_CONTENT_PATTERN.finditer(text or ""):
        kind = match.lastgroup
        if kind is None:
            continue
        value = match.group()
        content.append({
            "type": kind,
            "value": _canonical_numeric_value(kind, value),
            "surface": value,
        })
    return content


def evaluate_numeric_content(reference: str, hypothesis: str) -> Dict[str, Any]:
    """Compare number-like spans exactly as a multiset; no transcript correction."""
    expected = extract_numeric_content(reference)
    predicted = extract_numeric_content(hypothesis)
    expected_values = Counter((item["type"], item["value"]) for item in expected)
    predicted_values = Counter((item["type"], item["value"]) for item in predicted)
    true_positive = sum((expected_values & predicted_values).values())
    false_positive = sum((predicted_values - expected_values).values())
    false_negative = sum((expected_values - predicted_values).values())
    exact_match = expected_values == predicted_values
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else (1.0 if not expected_values else 0.0)
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else (1.0 if not predicted_values else 0.0)
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "expected": expected,
        "predicted": predicted,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": exact_match,
    }


def _canonical_entity(value: str) -> str:
    return normalize_text(value).replace("-", " ")


def evaluate_expected_entities(
    expected_entities: List[str],
    hypothesis: str,
) -> Dict[str, Any]:
    """Match annotated entities against raw hypothesis text deterministically."""
    normalized_hypothesis = _canonical_entity(hypothesis)
    expected = list(dict.fromkeys(
        _canonical_entity(entity) for entity in expected_entities if entity.strip()
    ))
    found = [
        entity
        for entity in expected
        if re.search(
            rf"(?<!\w){re.escape(entity)}(?!\w)",
            normalized_hypothesis,
        )
    ]
    missing = [entity for entity in expected if entity not in found]

    extracted = extract_entities(hypothesis)
    predicted_candidates = list(dict.fromkeys(
        [_canonical_entity(entity) for entity in extracted["names"]]
        + [_canonical_entity(entity) for entity in extracted["student_ids"]]
    ))
    unexpected = [
        candidate for candidate in predicted_candidates
        if not any(
            candidate == entity
            or (
                candidate
                and re.search(
                    rf"(?<!\w){re.escape(candidate)}(?!\w)",
                    entity,
                )
            )
            for entity in expected
        )
    ]

    true_positive = len(found)
    false_positive = len(unexpected)
    false_negative = len(missing)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else (1.0 if not expected else 0.0)
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else (1.0 if not predicted_candidates else 0.0)
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "expected": expected,
        "found": found,
        "unexpected": unexpected,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": not missing and not unexpected,
    }


# ---------------------------------------------------------------------------
# Entity Extraction
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract key domain entities from text:
      - student_ids: alphanumeric ID formats (e.g. STU-1029, S12345, 2024CS101)
      - numbers: integer / decimal digits and spoken numbers
      - dates: calendar dates, days of week, relative dates, times
      - names: proper names, personas, academic titles
    """
    if not text:
        return {"student_ids": [], "numbers": [], "dates": [], "names": []}

    student_ids: List[str] = []
    numbers: List[str] = []
    dates: List[str] = []
    names: List[str] = []

    # 1. Student IDs (e.g., STU-1029, S98234, 2024CS101, ID-9082)
    id_pattern = re.compile(
        r"\b(?:[A-Za-z]{1,4}[-_]?\d{3,8}|\d{4}[A-Za-z]{2}\d{2,4})\b",
        re.IGNORECASE,
    )
    for match in id_pattern.finditer(text):
        val = match.group().strip()
        # Ensure it has both letters and numbers
        has_letters = any(c.isalpha() for c in val)
        has_digits = any(c.isdigit() for c in val)
        if has_letters and has_digits:
            student_ids.append(val.upper())

    # 2. Dates & Times
    # ISO / formatted dates: 2026-10-20, 20/10/2026, 10/20/26
    date_formatted = re.findall(r"\b\d{1,4}[-/]\d{1,2}[-/]\d{2,4}\b", text)
    dates.extend(date_formatted)

    # Word-based dates: October 20th 2026, 24th of October, Nov 5
    month_regex = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
    text_date_pattern = re.compile(
        rf"\b(?:{month_regex}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*,?\s*\d{{4}})?|\d{{1,2}}(?:st|nd|rd|th)?(?:\s+of)?\s+{month_regex}(?:\s*,?\s*\d{{4}})?)\b",
        re.IGNORECASE,
    )
    for m in text_date_pattern.finditer(text):
        dates.append(m.group().strip())

    # Days of week & relative dates
    dow_pattern = re.compile(
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|yesterday)\b",
        re.IGNORECASE,
    )
    for m in dow_pattern.finditer(text):
        dates.append(m.group().strip().lower())

    # Times: 9:00 AM, 14:30, 5:30pm
    time_pattern = re.compile(
        r"\b\d{1,2}:\d{2}(?:\s*(?:am|pm|a\.m\.|p\.m\.))?\b",
        re.IGNORECASE,
    )
    for m in time_pattern.finditer(text):
        dates.append(m.group().strip().lower())

    # 3. Numbers (digits, percentages, floats, spoken words)
    # Exclude parts of student IDs already captured
    clean_for_nums = text
    for sid in student_ids:
        clean_for_nums = re.sub(re.escape(sid), " ", clean_for_nums, flags=re.IGNORECASE)

    # Digit numbers: 100, 3.85, 94%, $50
    digit_nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", clean_for_nums)
    numbers.extend(digit_nums)

    # Word numbers: one, forty, hundred
    words = re.findall(r"\b[A-Za-z]+\b", clean_for_nums.lower())
    for w in words:
        if w in NUMBER_WORDS:
            numbers.append(w)

    # 4. Names
    # Person titles: Professor Sharma, Dr. Rao, Mr. Smith
    title_pattern = re.compile(
        r"\b(?:dr\.|prof\.|professor|mr\.|ms\.|mrs\.)\s+[A-Z][a-z]+\b",
        re.IGNORECASE,
    )
    for m in title_pattern.finditer(text):
        names.append(m.group().strip())

    # Domain specific persona names
    persona_pattern = re.compile(r"\b(?:lockin|kalyan)\b", re.IGNORECASE)
    for m in persona_pattern.finditer(text):
        names.append(m.group().strip())

    # Capitalized non-stopword tokens (if original text has capitalization)
    tokens = text.split()
    for i, tok in enumerate(tokens):
        # strip punctuation
        clean_tok = tok.strip(string.punctuation)
        if clean_tok and clean_tok[0].isupper() and len(clean_tok) > 1:
            lower = clean_tok.lower()
            if (
                lower not in COMMON_ENGLISH_WORDS
                and lower not in NUMBER_WORDS
                and lower not in MONTHS
                and lower not in DAYS_OF_WEEK
                and not any(c.isdigit() for c in clean_tok)
            ):
                # Don't duplicate if already in title matches
                if not any(clean_tok.lower() in n.lower() for n in names):
                    names.append(clean_tok)

    # Deduplicate and normalize entity lists
    return {
        "student_ids": list(dict.fromkeys(student_ids)),
        "numbers": list(dict.fromkeys(numbers)),
        "dates": list(dict.fromkeys(dates)),
        "names": list(dict.fromkeys(names)),
    }


# ---------------------------------------------------------------------------
# Entity Comparison & Metrics
# ---------------------------------------------------------------------------

def _match_entity_lists(ref_list: List[str], hyp_list: List[str]) -> Tuple[int, int, int, int]:
    """
    Match two lists of entities using case-insensitive normalization.
    Distinguishes exact matches from partial (substring/token-overlap) matches.

    Returns:
      (exact_tp, partial_tp, fp, fn)
        - exact_tp: Reference entities that matched a hypothesis entity exactly.
        - partial_tp: Remaining reference entities that partially matched a hypothesis entity.
        - fp: Hypothesis entities that matched neither exactly nor partially.
        - fn: Reference entities that remained completely unmatched.
    """
    norm_refs = [normalize_text(r) for r in ref_list if r]
    norm_hyps = [normalize_text(h) for h in hyp_list if h]

    exact_tp = 0
    hyp_pool = list(norm_hyps)
    unmatched_refs = []

    # 1. First pass: exact matches
    for r in norm_refs:
        if r in hyp_pool:
            exact_tp += 1
            hyp_pool.remove(r)
        else:
            unmatched_refs.append(r)

    # 2. Second pass: partial matches (substring or token overlap)
    partial_tp = 0
    remaining_refs = []
    for r in unmatched_refs:
        matched = False
        r_tokens = set(r.split())
        for h in list(hyp_pool):
            h_tokens = set(h.split())
            if (r in h) or (h in r) or (bool(r_tokens & h_tokens)):
                partial_tp += 1
                hyp_pool.remove(h)
                matched = True
                break
        if not matched:
            remaining_refs.append(r)

    fp = len(hyp_pool)
    fn = len(remaining_refs)
    return exact_tp, partial_tp, fp, fn


def evaluate_entities(
    ref_entities: Dict[str, List[str]],
    hyp_entities: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute Precision, Recall, F1, and Exact Match Rate for each category:
      - numbers
      - names
      - dates
      - student_ids
    Also computes an 'overall' aggregated score.

    Mathematical definitions:
      - Precision: exact_tp / (exact_tp + fp)
      - Recall: exact_tp / len(refs)  [where len(refs) = exact_tp + partial_tp + fn]
      - F1: 2 * Prec * Rec / (Prec + Rec)
      - Exact Match (sample level): 1.0 iff predicted entity set matches reference set
        exactly (exact_tp == len(refs) and fp == 0, or both sets empty); 0.0 otherwise.
      - Partial Matches: count of entities that matched partially but not exactly.
    """
    categories = ["student_ids", "numbers", "dates", "names"]
    results: Dict[str, Dict[str, Any]] = {}

    total_exact_tp = 0
    total_partial_tp = 0
    total_fp = 0
    total_fn = 0
    total_refs = 0
    total_hyps = 0

    all_categories_exact_match = True

    for cat in categories:
        refs = ref_entities.get(cat, [])
        hyps = hyp_entities.get(cat, [])
        exact_tp, partial_tp, fp, fn = _match_entity_lists(refs, hyps)

        total_exact_tp += exact_tp
        total_partial_tp += partial_tp
        total_fp += fp
        total_fn += fn
        total_refs += len(refs)
        total_hyps += len(hyps)

        # Precision: tp / (tp + fp)
        if (exact_tp + fp) > 0:
            prec = exact_tp / (exact_tp + fp)
        else:
            prec = 1.0 if (not refs and not hyps) else 0.0

        # Recall: tp / len(refs)
        if len(refs) > 0:
            rec = exact_tp / len(refs)
        else:
            rec = 1.0 if not hyps else 0.0

        # F1 Score
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        # Exact Match for this category:
        # 1.0 if reference and hypothesis sets match identically (no FP, no FN, no partials)
        # 1.0 if both are empty; 0.0 if any discrepancies exist.
        if not refs and not hyps:
            em = 1.0
        elif len(refs) == exact_tp and fp == 0 and partial_tp == 0 and fn == 0:
            em = 1.0
        else:
            em = 0.0

        if em == 0.0:
            all_categories_exact_match = False

        results[cat] = {
            "tp": exact_tp,
            "exact_tp": exact_tp,
            "partial_tp": partial_tp,
            "fp": fp,
            "fn": fn,
            "ref_count": len(refs),
            "hyp_count": len(hyps),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "exact_match": em,
            "exact_match_rate": em,  # alias for backwards compatibility
        }

    # Overall metrics across all categories
    if (total_exact_tp + total_fp) > 0:
        overall_prec = total_exact_tp / (total_exact_tp + total_fp)
    else:
        overall_prec = 1.0 if (total_refs == 0 and total_hyps == 0) else 0.0

    if total_refs > 0:
        overall_rec = total_exact_tp / total_refs
    else:
        overall_rec = 1.0 if total_hyps == 0 else 0.0

    overall_f1 = (
        (2 * overall_prec * overall_rec) / (overall_prec + overall_rec)
        if (overall_prec + overall_rec) > 0
        else 0.0
    )

    overall_em = 1.0 if all_categories_exact_match else 0.0

    results["overall"] = {
        "tp": total_exact_tp,
        "exact_tp": total_exact_tp,
        "partial_tp": total_partial_tp,
        "fp": total_fp,
        "fn": total_fn,
        "ref_count": total_refs,
        "hyp_count": total_hyps,
        "precision": round(overall_prec, 4),
        "recall": round(overall_rec, 4),
        "f1": round(overall_f1, 4),
        "exact_match": overall_em,
        "exact_match_rate": overall_em,
    }

    return results


# ---------------------------------------------------------------------------
# End-to-End Pair Evaluator
# ---------------------------------------------------------------------------

def evaluate_asr_pair(
    reference: str,
    hypothesis: str,
    expected_entities: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """
    Evaluate a single (reference, hypothesis) pair on all metrics:
      - WER
      - CER
      - Sentence Exact Match
      - Entity metrics for numbers, names, dates, student IDs
    """
    wer_res = compute_wer(reference, hypothesis)
    cer_res = compute_cer(reference, hypothesis)
    em = compute_sentence_em(reference, hypothesis)

    ref_ents = expected_entities or extract_entities(reference)
    hyp_ents = extract_entities(hypothesis)

    ent_res = evaluate_entities(ref_ents, hyp_ents)

    return {
        "reference": reference,
        "hypothesis": hypothesis,
        "wer": wer_res["wer"],
        "cer": cer_res["cer"],
        "sentence_exact_match": em,
        "wer_details": wer_res,
        "cer_details": cer_res,
        "ref_entities": ref_ents,
        "hyp_entities": hyp_ents,
        "entity_metrics": ent_res,
    }
