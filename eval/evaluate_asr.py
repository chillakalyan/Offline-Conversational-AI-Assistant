"""
eval/evaluate_asr.py — ASR evaluation harness for LockIN Voice AI Assistant.

Evaluates ASR outputs against reference ground-truth transcripts, measuring:
  - Word Error Rate (WER)
  - Character Error Rate (CER)
  - Sentence Exact Match (EM)
  - Entity-level Exact Match, Precision, Recall, and F1 for:
      * Numbers
      * Names
      * Dates / Times
      * Student IDs

Usage:
  # Run the fixture-only evaluation (not a real ASR benchmark):
  python eval/evaluate_asr.py

  # Evaluate human-recorded samples after adding them with record_dataset.py:
  python eval/evaluate_asr.py --recorded-audio

  # Custom fixture dataset and output report:
  python eval/evaluate_asr.py --dataset eval/test_cases.json --output-md eval/eval_report.md
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to sys.path so config and eval can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import (
    compute_cer,
    compute_sentence_em,
    compute_wer,
    aggregate_asr_results,
    evaluate_expected_entities,
    evaluate_numeric_content,
    evaluate_entities,
    extract_entities,
    normalize_text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ASREvaluator:
    """Evaluation runner for ASR benchmarks and transcripts."""

    def __init__(self, dataset_path: Optional[Path] = None) -> None:
        self.dataset_path = dataset_path or (PROJECT_ROOT / "eval" / "test_cases.json")
        self.results: List[Dict] = []
        self.summary: Dict = {}

    def load_dataset(self) -> List[Dict]:
        """Load benchmark cases from JSON."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded %d test cases from %s", len(data), self.dataset_path)
        return data

    def evaluate_item(self, item: Dict, hypothesis_text: Optional[str] = None) -> Dict:
        """Evaluate a single test case."""
        case_id = item.get("id", "UNKNOWN")
        domain = item.get("domain", "general")
        ref_text = item.get("reference", "")
        hyp_text = hypothesis_text if hypothesis_text is not None else item.get("hypothesis", "")

        wer_info = compute_wer(ref_text, hyp_text)
        cer_info = compute_cer(ref_text, hyp_text)
        em = compute_sentence_em(ref_text, hyp_text)

        # Use pre-annotated entities if provided, otherwise extract from reference
        ref_entities = item.get("entities") or extract_entities(ref_text)
        hyp_entities = extract_entities(hyp_text)

        entity_metrics = evaluate_entities(ref_entities, hyp_entities)

        return {
            "id": case_id,
            "domain": domain,
            "reference": ref_text,
            "hypothesis": hyp_text,
            "wer": wer_info["wer"],
            "cer": cer_info["cer"],
            "sentence_em": em,
            "wer_details": wer_info,
            "cer_details": cer_info,
            "ref_entities": ref_entities,
            "hyp_entities": hyp_entities,
            "entity_metrics": entity_metrics,
        }

    def run(self, data: Optional[List[Dict]] = None) -> Dict:
        """Run full evaluation on loaded dataset."""
        dataset = data or self.load_dataset()
        self.results = [self.evaluate_item(item) for item in dataset]

        n = len(self.results)
        if n == 0:
            self.summary = {
                "total_cases": 0,
                "average_wer": None,
                "average_cer": None,
                "corpus_wer": None,
                "corpus_cer": None,
                "sentence_exact_match_rate": None,
                "entity_metrics": {},
            }
            return self.summary

        aggregate = aggregate_asr_results(self.results)

        # Aggregate entity statistics across all cases
        categories = ["student_ids", "numbers", "dates", "names", "overall"]
        entity_summary = {}

        for cat in categories:
            total_tp = sum(r["entity_metrics"][cat]["tp"] for r in self.results)
            total_partial = sum(r["entity_metrics"][cat].get("partial_tp", 0) for r in self.results)
            total_fp = sum(r["entity_metrics"][cat]["fp"] for r in self.results)
            total_fn = sum(r["entity_metrics"][cat]["fn"] for r in self.results)
            total_refs = sum(
                r["entity_metrics"][cat].get("ref_count", r["entity_metrics"][cat]["tp"] + r["entity_metrics"][cat]["fn"])
                for r in self.results
            )

            if (total_tp + total_fp) > 0:
                prec = total_tp / (total_tp + total_fp)
            else:
                prec = 1.0 if total_refs == 0 else 0.0

            if total_refs > 0:
                rec = total_tp / total_refs
            else:
                rec = 1.0 if total_fp == 0 else 0.0

            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            # Dataset-level Exact Match Rate: fraction of cases where entity set matched exactly
            cat_em_rate = sum(r["entity_metrics"][cat]["exact_match"] for r in self.results) / n

            entity_summary[cat] = {
                "tp": total_tp,
                "partial_tp": total_partial,
                "fp": total_fp,
                "fn": total_fn,
                "ref_count": total_refs,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "exact_match_rate": round(cat_em_rate, 4),
            }

        self.summary = {
            "total_cases": n,
            "average_wer": aggregate["per_utterance_average_wer"],
            "average_cer": aggregate["per_utterance_average_cer"],
            "corpus_wer": aggregate["corpus_wer"],
            "corpus_cer": aggregate["corpus_cer"],
            "sentence_exact_match_rate": aggregate["exact_match"],
            "entity_metrics": entity_summary,
        }
        return self.summary

    def format_console_report(self) -> str:
        """Format an ASCII report for terminal output."""
        if not self.summary:
            return "No evaluation summary available. Call run() first."

        s = self.summary
        ent = s["entity_metrics"]

        lines = [
            "=" * 70,
            "                ASR FIXTURE EVALUATION REPORT",
            "=" * 70,
            f"Test Cases Evaluated  : {s['total_cases']}",
            f"Per-utterance average WER      : {s['average_wer'] * 100:.2f} %",
            f"Per-utterance average CER      : {s['average_cer'] * 100:.2f} %",
            f"Corpus WER                     : {s['corpus_wer'] * 100:.2f} %",
            f"Corpus CER                     : {s['corpus_cer'] * 100:.2f} %",
            f"Sentence Exact Match (EM)     : {s['sentence_exact_match_rate'] * 100:.2f} %",
            "-" * 70,
            "ENTITY-LEVEL EXACT MATCH & EXTRACTION METRICS",
            "-" * 70,
            f"{'Entity Category':<15} | {'Exact Match':<11} | {'Precision':<9} | {'Recall':<9} | {'F1':<6}",
            "-" * 70,
        ]

        display_cats = [
            ("Student IDs", "student_ids"),
            ("Numbers", "numbers"),
            ("Dates", "dates"),
            ("Names", "names"),
            ("Overall Entities", "overall"),
        ]

        for label, cat in display_cats:
            m = ent[cat]
            lines.append(
                f"{label:<15} | {m['exact_match_rate'] * 100:6.1f} %    | {m['precision'] * 100:6.1f} %  | {m['recall'] * 100:6.1f} %  | {m['f1']:.3f}"
            )

        lines.append("=" * 70)
        return "\n".join(lines)

    def export_markdown_report(self, output_path: Path) -> None:
        """Export the fixture evaluation report to markdown."""
        s = self.summary
        ent = s["entity_metrics"]

        md_lines = [
            "# ASR Fixture Evaluation Report",
            "### LockIN Voice AI Assistant — Offline Conversational AI Assistant",
            "",
            "> Fixture-only results: hypotheses are prewritten test data, not Faster-Whisper "
            "transcriptions of human-recorded speech. Do not cite these scores as model "
            "accuracy. Use `--recorded-audio` for the real recorded-speech evaluation.",
            "",
            "## Evaluation Overview",
            "",
            "This report evaluates prewritten fixture hypotheses only. These "
            "results are **not real ASR model performance**.",
            "",
            "## Dataset",
            "",
            f"- Fixture test cases: {s['total_cases']}",
            "",
            "## Overall WER/CER",
            "",
            "| Metric | Score | Target |",
            "|---|---|---|",
            f"| **Per-utterance average WER** | **{s['average_wer'] * 100:.2f} %** | — |",
            f"| **Per-utterance average CER** | **{s['average_cer'] * 100:.2f} %** | — |",
            f"| **Corpus WER** | **{s['corpus_wer'] * 100:.2f} %** | — |",
            f"| **Corpus CER** | **{s['corpus_cer'] * 100:.2f} %** | — |",
            f"| **Sentence Exact Match** | **{s['sentence_exact_match_rate'] * 100:.2f} %** | — |",
            f"| **Entity Overall F1** | **{ent['overall']['f1']:.3f}** | > 0.90 |",
            f"| **Entity Exact Match Rate** | **{ent['overall']['exact_match_rate'] * 100:.2f} %** | > 90.0 % |",
            "",
            "## Entity Recognition Breakdown",
            "",
            "| Category | Ground Truth Entities | Exact Matches (TP) | Precision | Recall | F1 Score | Exact Match Rate |",
            "|---|---|---|---|---|---|---|",
        ]

        cats = [
            ("Student IDs", "student_ids"),
            ("Numbers", "numbers"),
            ("Dates / Times", "dates"),
            ("Names", "names"),
            ("**Overall**", "overall"),
        ]

        for label, cat in cats:
            m = ent[cat]
            # Include partial matches: they still correspond to ground-truth
            # references even though they are not counted as exact true positives.
            total_ref = m["ref_count"]
            md_lines.append(
                f"| {label} | {total_ref} | {m['tp']} | {m['precision'] * 100:.1f} % | {m['recall'] * 100:.1f} % | {m['f1']:.3f} | {m['exact_match_rate'] * 100:.1f} % |"
            )

        md_lines.extend([
            "",
            "## Detailed Per-Case Evaluation",
            "",
            "| ID | Domain | WER | CER | Sentence Match | Key Entities |",
            "|---|---|---|---|---|---|",
        ])

        for r in self.results:
            match_icon = "✅" if r["sentence_em"] == 1.0 else "❌"
            ents = []
            for c in ["student_ids", "numbers", "dates", "names"]:
                if r["ref_entities"].get(c):
                    ents.append(f"{c}: {', '.join(r['ref_entities'][c])}")
            ent_str = "; ".join(ents) if ents else "none"
            md_lines.append(
                f"| `{r['id']}` | {r['domain']} | {r['wer'] * 100:.1f} % | {r['cer'] * 100:.1f} % | {match_icon} | {ent_str} |"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        logger.info("Exported markdown evaluation report to %s", output_path)

    def export_json_results(self, output_path: Path) -> None:
        """Export raw evaluation results to JSON."""
        data = {
            "summary": self.summary,
            "cases": self.results,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Exported JSON evaluation results to %s", output_path)


    def run_empirical_audio_benchmark(self, num_samples: int = 5) -> Dict:
        """
        Run a synthetic Piper-to-Whisper smoke benchmark:
          1. Synthesizes WAV audio from reference sentence via Piper TTS.
          2. Transcribes WAV audio via Faster Whisper ASR.
          3. Evaluates synthetic-speech WER, CER, and entity metrics.
          4. Measures STT, LLM, TTS, and total response latencies.

        This is not human-recorded evaluation and must not be presented as a
        representative ASR accuracy benchmark.
        Cleans up all temporary audio files.
        """
        import os
        import subprocess
        import tempfile
        import time
        from stt import SpeechToText
        from llm import LLM
        import config

        data = self.load_dataset()
        # Select clean benchmark samples (exclude simulated error cases)
        clean_cases = [c for c in data if not c.get("domain", "").startswith("simulated_asr_error")][:num_samples]

        logger.info("Starting synthetic Piper audio smoke test on %d cases...", len(clean_cases))
        stt = SpeechToText()
        llm = LLM()

        empirical_results = []
        latencies = {
            "tts": [],
            "stt": [],
            "llm": [],
            "total": [],
        }

        for case in clean_cases:
            ref = case["reference"]
            cid = case["id"]
            logger.info("Evaluating [%s]: %s", cid, ref[:40])

            # 1. Synthesize audio via Piper
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                wav_path = tmp.name

            try:
                t0 = time.perf_counter()
                res = subprocess.run(
                    [str(config.PIPER_EXE), "-m", str(config.PIPER_VOICE_MODEL), "-f", wav_path],
                    input=ref.encode("utf-8"),
                    capture_output=True,
                    timeout=config.PIPER_TIMEOUT,
                )
                tts_lat = time.perf_counter() - t0
                wav_ok = Path(wav_path).exists() and Path(wav_path).stat().st_size > 44
                if res.returncode != 0 or not wav_ok:
                    logger.warning("Piper failed on %s: %s", cid, res.stderr.decode("utf-8", "ignore"))
                    continue
                try:
                    import soundfile as sf
                    with sf.SoundFile(wav_path) as check:
                        if check.frames <= 0:
                            raise ValueError("empty wav")
                except Exception as exc:
                    logger.warning("Skipping %s: Piper wrote an unreadable WAV (%s)", cid, exc)
                    continue

                # 2. Transcribe audio via Faster Whisper
                t1 = time.perf_counter()
                hyp_text = stt.transcribe(wav_path)
                stt_lat = time.perf_counter() - t1

                # 3. Query LLM
                t2 = time.perf_counter()
                reply = llm.generate(hyp_text)
                llm_lat = time.perf_counter() - t2

                tot_lat = stt_lat + llm_lat + tts_lat

                latencies["tts"].append(tts_lat)
                latencies["stt"].append(stt_lat)
                latencies["llm"].append(llm_lat)
                latencies["total"].append(tot_lat)

                # 4. Evaluate pair accuracy
                evaluated = self.evaluate_item(case, hypothesis_text=hyp_text)
                evaluated["latencies"] = {
                    "tts_sec": round(tts_lat, 3),
                    "stt_sec": round(stt_lat, 3),
                    "llm_sec": round(llm_lat, 3),
                    "total_sec": round(tot_lat, 3),
                }
                empirical_results.append(evaluated)

            finally:
                if os.path.exists(wav_path):
                    try:
                        os.remove(wav_path)
                    except OSError:
                        pass

        def _stats(arr: List[float]) -> Dict[str, float]:
            if not arr:
                return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0}
            import statistics
            arr_sorted = sorted(arr)
            mean_val = statistics.mean(arr)
            std_val = statistics.stdev(arr) if len(arr) > 1 else 0.0
            p50 = arr_sorted[len(arr_sorted) // 2]
            p95 = arr_sorted[min(len(arr_sorted) - 1, int(len(arr_sorted) * 0.95))]
            return {
                "mean": round(mean_val, 3),
                "std": round(std_val, 3),
                "p50": round(p50, 3),
                "p95": round(p95, 3),
            }

        n = len(empirical_results)
        avg_wer = sum(r["wer"] for r in empirical_results) / n if n else 0.0
        avg_cer = sum(r["cer"] for r in empirical_results) / n if n else 0.0
        em_rate = sum(r["sentence_em"] for r in empirical_results) / n if n else 0.0

        latency_summary = {
            "stt_sec": _stats(latencies["stt"]),
            "llm_sec": _stats(latencies["llm"]),
            "tts_sec": _stats(latencies["tts"]),
            "total_sec": _stats(latencies["total"]),
        }

        empirical_summary = {
            "sample_count": n,
            "average_wer": round(avg_wer, 4),
            "average_cer": round(avg_cer, 4),
            "sentence_exact_match_rate": round(em_rate, 4),
            "latency": latency_summary,
            "cases": empirical_results,
        }
        return empirical_summary


def format_recorded_markdown_report(summary: Dict) -> str:
    """Render the recorded-audio results and explicit unavailability state."""
    lines = [
        "# Empirical Recorded-Speech ASR Evaluation",
        "",
        "## 1. Evaluation Overview",
        "",
        "This section reports raw Faster-Whisper transcriptions of validated "
        "human-recorded audio. No transcript correction or LLM judgment is used.",
        "",
        "## 2. Dataset",
        "",
        f"- Evaluation type: `{summary['evaluation_type']}`",
        f"- Total metadata samples: {summary['total_samples']}",
        f"- Valid transcribed samples: {summary['valid_samples']}",
        "",
        "## 3. Overall WER/CER",
        "",
    ]
    if summary["valid_samples"] == 0:
        lines.extend([
            "Empirical ASR results are unavailable because no human-recorded "
            "evaluation audio has been provided.",
            "",
            "- Corpus WER: unavailable",
            "- Corpus CER: unavailable",
            "- Per-utterance average WER/CER: unavailable",
            "",
            "## 4. Category Results",
            "",
            "No categories have valid recorded audio.",
            "",
            "## 5. Number Accuracy",
            "",
            "Unavailable; no recorded utterances were evaluated.",
            "",
            "## 6. Name/Entity Accuracy",
            "",
            "Unavailable; no recorded utterances were evaluated.",
            "",
            "## 7. Recording Condition Results",
            "",
            "No recording conditions have valid audio.",
            "",
            "## 8. Speaker Results",
            "",
            "No speakers have valid audio.",
            "",
            "## 9. Per-Utterance Results",
            "",
            "No utterances were evaluated.",
            "",
            "## 10. Limitations",
            "",
            "Fixture hypotheses are not included in this report. Add genuine "
            "microphone recordings and metadata before interpreting ASR performance.",
        ])
        return "\n".join(lines) + "\n"

    lines.extend([
        f"- Valid recordings: {summary['valid_samples']}",
        f"- Per-utterance average WER: "
        f"{summary['per_utterance_average_wer'] * 100:.2f}%",
        f"- Per-utterance average CER: "
        f"{summary['per_utterance_average_cer'] * 100:.2f}%",
        f"- Corpus WER: {summary['corpus_wer'] * 100:.2f}%",
        f"- Corpus CER: {summary['corpus_cer'] * 100:.2f}%",
        f"- Exact sentence match: {summary['exact_sentence_match'] * 100:.2f}%",
        "",
        "## 4. Category Results",
        "",
        "| Category | Samples | Corpus WER | Corpus CER | Exact sentence match |",
        "|---|---:|---:|---:|---:|",
    ])
    for category, metrics in summary["category_results"].items():
        lines.append(
            f"| {category} | {metrics['samples']} | {metrics['wer'] * 100:.2f}% | "
            f"{metrics['cer'] * 100:.2f}% | {metrics['exact_match'] * 100:.2f}% |"
        )

    def append_special_metrics(title: str, metrics: Dict) -> None:
        lines.extend(["", title, ""])
        if not metrics:
            lines.append("No samples in this dataset are annotated for this metric.")
            return
        lines.extend([
            "| Samples | Precision | Recall | F1 | Exact match |",
            "|---:|---:|---:|---:|---:|",
            f"| {metrics['samples']} | {metrics['precision']:.3f} | "
            f"{metrics['recall']:.3f} | {metrics['f1']:.3f} | "
            f"{metrics['exact_match'] * 100:.2f}% |",
        ])

    append_special_metrics("## 5. Number Accuracy", summary["number_results"])
    append_special_metrics("## 6. Name/Entity Accuracy", summary["entity_results"])

    lines.extend([
        "",
        "## 7. Recording Condition Results",
        "",
        "| Condition | Samples | Corpus WER | Corpus CER | Exact sentence match |",
        "|---|---:|---:|---:|---:|",
    ])
    for condition, metrics in summary["condition_results"].items():
        lines.append(
            f"| {condition} | {metrics['samples']} | {metrics['wer'] * 100:.2f}% | "
            f"{metrics['cer'] * 100:.2f}% | {metrics['exact_match'] * 100:.2f}% |"
        )

    lines.extend([
        "",
        "## 8. Speaker Results",
        "",
        "| Speaker ID | Samples | Corpus WER | Corpus CER | Exact sentence match |",
        "|---|---:|---:|---:|---:|",
    ])
    for speaker, metrics in summary["speaker_results"].items():
        lines.append(
            f"| {speaker} | {metrics['samples']} | {metrics['wer'] * 100:.2f}% | "
            f"{metrics['cer'] * 100:.2f}% | {metrics['exact_match'] * 100:.2f}% |"
        )

    lines.extend([
        "",
        "## 9. Per-Utterance Results",
        "",
        "| ID | Speaker | Condition | Category | WER | CER | Error categories | Reference | Raw hypothesis |",
        "|---|---|---|---|---:|---:|---|---|---|",
    ])
    for utterance in summary["utterances"]:
        lines.append(
            f"| {utterance['id']} | {utterance['speaker_id']} | "
            f"{utterance['recording_condition']} | {utterance['category']} | "
            f"{utterance['wer'] * 100:.2f}% | {utterance['cer'] * 100:.2f}% | "
            f"{', '.join(utterance['error_categories'])} | "
            f"{utterance['reference']} | {utterance['hypothesis']} |"
        )
    lines.extend([
        "",
        "## 10. Limitations",
        "",
        "Number content is compared using deterministic surface-form parsing; "
        "name/entity precision includes candidates from the existing heuristic "
        "entity extractor. Corpus results describe this recorded sample only and "
        "may not generalize across speakers, devices, or acoustic conditions. "
        "Whisper output is scored as returned. No correction is applied; WER/CER "
        "use the metric module's documented case and punctuation normalization.",
    ])
    return "\n".join(lines) + "\n"


def run_evaluation(
    dataset_path: Optional[str] = None,
    output_json: Optional[str] = None,
    output_md: Optional[str] = None,
    empirical: bool = False,
    num_samples: int = 5,
    recorded_audio: bool = False,
    include_fixtures: Optional[bool] = None,
) -> Dict:
    """Convenience runner function for programmatic and CLI use."""
    dataset = Path(dataset_path) if dataset_path else None
    evaluator = ASREvaluator(dataset_path=dataset)
    if include_fixtures is None:
        include_fixtures = not recorded_audio
    fixture_summary = evaluator.run() if include_fixtures else None

    if fixture_summary is not None:
        print(evaluator.format_console_report())

    recorded_summary = None
    if recorded_audio:
        print("\n" + "=" * 70)
        print("       EVALUATING HUMAN-RECORDED DATASET AUDIO")
        print("=" * 70)
        recorded_summary = run_recorded_audio_evaluation()
        if recorded_summary["status"] == "not_started":
            print(recorded_summary["message"])
        else:
            print(f"Recorded samples transcribed : {recorded_summary['valid_samples']}")
            print(
                "Per-utterance average WER    : "
                f"{recorded_summary['per_utterance_average_wer'] * 100:.2f} %"
            )
            print(
                "Per-utterance average CER    : "
                f"{recorded_summary['per_utterance_average_cer'] * 100:.2f} %"
            )
            print(f"Corpus WER                   : {recorded_summary['corpus_wer'] * 100:.2f} %")
            print(f"Corpus CER                   : {recorded_summary['corpus_cer'] * 100:.2f} %")
            print("Category results:")
            for category, metrics in recorded_summary["category_results"].items():
                print(
                    f"  {category}: n={metrics['samples']}, "
                    f"WER={metrics['wer'] * 100:.2f}%, "
                    f"CER={metrics['cer'] * 100:.2f}%, "
                    f"exact={metrics['exact_match'] * 100:.2f}%"
                )
            print("=" * 70)

    synthetic_summary = None
    if empirical:
        print("\n" + "=" * 70)
        print("          RUNNING SYNTHETIC PIPER AUDIO SMOKE TEST")
        print("=" * 70)
        synthetic_summary = evaluator.run_empirical_audio_benchmark(num_samples=num_samples)
        lat = synthetic_summary["latency"]
        print(f"Synthetic samples tested     : {synthetic_summary['sample_count']}")
        print(f"Synthetic speech WER         : {synthetic_summary['average_wer'] * 100:.2f} %")
        print(f"Synthetic speech CER         : {synthetic_summary['average_cer'] * 100:.2f} %")
        print(f"Synthetic sentence match     : {synthetic_summary['sentence_exact_match_rate'] * 100:.2f} %")
        print("-" * 70)
        print(f"STT Latency (Whisper)         : {lat['stt_sec']['mean']:.2f}s ± {lat['stt_sec']['std']:.2f}s  (P50: {lat['stt_sec']['p50']:.2f}s, P95: {lat['stt_sec']['p95']:.2f}s)")
        print(f"LLM Latency (Ollama)          : {lat['llm_sec']['mean']:.2f}s ± {lat['llm_sec']['std']:.2f}s  (P50: {lat['llm_sec']['p50']:.2f}s, P95: {lat['llm_sec']['p95']:.2f}s)")
        print(f"TTS Latency (Piper)           : {lat['tts_sec']['mean']:.2f}s ± {lat['tts_sec']['std']:.2f}s  (P50: {lat['tts_sec']['p50']:.2f}s, P95: {lat['tts_sec']['p95']:.2f}s)")
        print(f"Total Pipeline Latency        : {lat['total_sec']['mean']:.2f}s ± {lat['total_sec']['std']:.2f}s  (P50: {lat['total_sec']['p50']:.2f}s, P95: {lat['total_sec']['p95']:.2f}s)")
        print("=" * 70)

    if output_json:
        export_data = {
            "fixture_evaluation": fixture_summary,
            "empirical_recorded_evaluation": recorded_summary,
            "synthetic_audio_smoke_evaluation": synthetic_summary,
        }
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        logger.info("Saved combined evaluation JSON to %s", out_p)

    if output_md:
        out_p = Path(output_md)
        if fixture_summary is not None:
            evaluator.export_markdown_report(out_p)
        else:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text("", encoding="utf-8")
        if recorded_summary:
            with open(out_p, "a", encoding="utf-8") as f:
                if fixture_summary is not None:
                    f.write("\n\n---\n\n")
                f.write(format_recorded_markdown_report(recorded_summary))
        if synthetic_summary:
            # Append an explicitly non-human-recorded Piper smoke-test section.
            lat = synthetic_summary["latency"]
            emp_lines = [
                "",
                "## Synthetic Piper Audio Smoke Test",
                "*(Piper-generated speech, not human-recorded ASR benchmark data)*",
                "",
                f"- **Samples Evaluated:** {synthetic_summary['sample_count']}",
                f"- **Synthetic Speech WER:** **{synthetic_summary['average_wer'] * 100:.2f} %**",
                f"- **Synthetic Speech CER:** **{synthetic_summary['average_cer'] * 100:.2f} %**",
                f"- **Sentence Match Rate:** **{synthetic_summary['sentence_exact_match_rate'] * 100:.2f} %**",
                "",
                "### Empirical Latency Breakdown",
                "",
                "| Pipeline Stage | Mean Latency | Std Dev | P50 (Median) | P95 |",
                "|---|---|---|---|---|",
                f"| **Speech-to-Text (Whisper base)** | {lat['stt_sec']['mean']:.2f} s | {lat['stt_sec']['std']:.2f} s | {lat['stt_sec']['p50']:.2f} s | {lat['stt_sec']['p95']:.2f} s |",
                f"| **LLM Generation (Ollama llama3.2:1b)** | {lat['llm_sec']['mean']:.2f} s | {lat['llm_sec']['std']:.2f} s | {lat['llm_sec']['p50']:.2f} s | {lat['llm_sec']['p95']:.2f} s |",
                f"| **TTS Synthesis (Piper)** | {lat['tts_sec']['mean']:.2f} s | {lat['tts_sec']['std']:.2f} s | {lat['tts_sec']['p50']:.2f} s | {lat['tts_sec']['p95']:.2f} s |",
                f"| **Total Pipeline Latency** | **{lat['total_sec']['mean']:.2f} s** | **{lat['total_sec']['std']:.2f} s** | **{lat['total_sec']['p50']:.2f} s** | **{lat['total_sec']['p95']:.2f} s** |",
                "",
            ]
            with open(out_p, "a", encoding="utf-8") as f:
                f.write("\n".join(emp_lines))
            logger.info("Appended empirical benchmark report to %s", out_p)

    return {
        "fixture_evaluation": fixture_summary,
        "empirical_recorded_evaluation": recorded_summary,
        "synthetic_audio_smoke_evaluation": synthetic_summary,
    }


def run_recorded_audio_evaluation(
    dataset_dir: Optional[Path] = None,
    stt=None,
) -> Dict:
    """Transcribe only validated human-recorded dataset audio, then score it."""
    from eval.record_dataset import DEFAULT_DATASET_DIR, validate_dataset

    root = dataset_dir or DEFAULT_DATASET_DIR
    references, samples = validate_dataset(root)
    for sample_id, reference in references.items():
        if reference["contains_names_entities"]:
            entities = reference.get("expected_entities")
            if (
                not isinstance(entities, list)
                or not entities
                or any(not isinstance(entity, str) or not entity.strip() for entity in entities)
            ):
                raise ValueError(
                    f"Reference {sample_id} must define non-empty expected_entities."
                )
    if not samples:
        return {
            "evaluation_type": "empirical_recorded_evaluation",
            "status": "not_started",
            "total_samples": 0,
            "valid_samples": 0,
            "corpus_wer": None,
            "corpus_cer": None,
            "per_utterance_average_wer": None,
            "per_utterance_average_cer": None,
            "exact_sentence_match": None,
            "category_results": {},
            "condition_results": {},
            "speaker_results": {},
            "number_results": {},
            "entity_results": {},
            "utterances": [],
            "message": (
                "Empirical ASR results are unavailable because no human-recorded "
                "evaluation audio has been provided."
            ),
        }

    if stt is None:
        from stt import SpeechToText
        stt = SpeechToText()

    evaluator = ASREvaluator()
    utterances = []
    for sample in samples:
        reference = references[sample["id"]]["reference"]
        hypothesis = stt.transcribe(
            str(sample["audio_path"]),
            language=sample["language"],
        )
        item = {
            "id": sample["id"],
            "domain": sample["category"],
            "reference": reference,
        }
        result = evaluator.evaluate_item(item, hypothesis_text=hypothesis)
        number_metrics = (
            evaluate_numeric_content(reference, hypothesis)
            if sample["contains_numbers"] else None
        )
        expected_entities = sample.get("expected_entities", [])
        entity_metrics = (
            evaluate_expected_entities(expected_entities, hypothesis)
            if sample["contains_names_entities"] else None
        )
        operations = {
            operation: result["wer_details"][operation]
            for operation in ("substitutions", "deletions", "insertions")
        }
        error_categories = []
        if number_metrics is not None and not number_metrics["exact_match"]:
            error_categories.append("number_error")
        if entity_metrics is not None and not entity_metrics["exact_match"]:
            error_categories.append("name_entity_error")
        if operations["insertions"]:
            error_categories.append("insertion")
        if operations["deletions"]:
            error_categories.append("deletion")
        if operations["substitutions"]:
            error_categories.append("substitution")
        if not error_categories:
            error_categories.append("no_error")

        result.update({
            "audio_filename": sample["audio_filename"],
            "speaker_id": sample["speaker_id"],
            "recording_condition": sample["recording_condition"],
            "language": sample["language"],
            "category": sample["category"],
            "contains_numbers": sample["contains_numbers"],
            "contains_names_entities": sample["contains_names_entities"],
            "expected_entities": expected_entities,
            "number_metrics": number_metrics,
            "entity_metrics": entity_metrics,
            "edit_operations": operations,
            "error_categories": error_categories,
        })
        utterances.append(result)

    def grouped_text_metrics(group: List[Dict]) -> Dict:
        aggregate = aggregate_asr_results(group)
        return {
            "samples": aggregate["samples"],
            "wer": aggregate["corpus_wer"],
            "cer": aggregate["corpus_cer"],
            "exact_match": aggregate["exact_match"],
        }

    def grouped_results(field: str) -> Dict[str, Dict]:
        groups: Dict[str, List[Dict]] = {}
        for utterance in utterances:
            groups.setdefault(utterance[field], []).append(utterance)
        return {
            name: grouped_text_metrics(group)
            for name, group in sorted(groups.items())
        }

    numeric_results = [
        utterance["number_metrics"]
        for utterance in utterances
        if utterance["number_metrics"] is not None
    ]
    entity_results = [
        utterance["entity_metrics"]
        for utterance in utterances
        if utterance["entity_metrics"] is not None
    ]

    def summarize_special_metrics(metric_results: List[Dict]) -> Dict:
        if not metric_results:
            return {}
        tp = sum(metric["tp"] for metric in metric_results)
        fp = sum(metric["fp"] for metric in metric_results)
        fn = sum(metric["fn"] for metric in metric_results)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "samples": len(metric_results),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "exact_match": round(
                sum(metric["exact_match"] for metric in metric_results)
                / len(metric_results),
                4,
            ),
        }

    aggregate = aggregate_asr_results(utterances)
    category_results = grouped_results("category")
    condition_results = grouped_results("recording_condition")
    speaker_results = grouped_results("speaker_id")
    return {
        "evaluation_type": "empirical_recorded_evaluation",
        "status": "completed",
        "source": "human_recorded_audio_transcribed_by_faster_whisper",
        "total_samples": len(samples),
        "valid_samples": len(utterances),
        "per_utterance_average_wer": aggregate["per_utterance_average_wer"],
        "per_utterance_average_cer": aggregate["per_utterance_average_cer"],
        "corpus_wer": aggregate["corpus_wer"],
        "corpus_cer": aggregate["corpus_cer"],
        "exact_sentence_match": aggregate["exact_match"],
        "category_results": category_results,
        "condition_results": condition_results,
        "speaker_results": speaker_results,
        "number_results": summarize_special_metrics(numeric_results),
        "entity_results": summarize_special_metrics(entity_results),
        "utterances": utterances,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASR transcripts and benchmarks.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to evaluation dataset JSON.")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save JSON evaluation results.")
    parser.add_argument("--output-md", type=str, default=None, help="Path to save Markdown evaluation report.")
    parser.add_argument(
        "--recorded-audio",
        action="store_true",
        help="Transcribe and score human-recorded samples in eval/dataset/ (fixture scores are skipped).",
    )
    parser.add_argument(
        "--include-fixtures",
        action="store_true",
        help="Also run the separate fixture evaluation when --recorded-audio is set.",
    )
    parser.add_argument(
        "--empirical",
        action="store_true",
        help="Run legacy synthetic Piper-to-Whisper smoke test (not human-recorded evaluation).",
    )
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples for empirical benchmark.")
    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        output_json=args.output_json,
        output_md=args.output_md,
        empirical=args.empirical,
        num_samples=args.num_samples,
        recorded_audio=args.recorded_audio,
        include_fixtures=(True if args.include_fixtures else None),
    )


if __name__ == "__main__":
    main()
