# ASR Fixture Evaluation Report
### LockIN Voice AI Assistant — Offline Conversational AI Assistant

> Fixture-only results: hypotheses are prewritten test data, not Faster-Whisper
> transcriptions of human-recorded speech. Do not cite these scores as model
> accuracy. Use `python eval/evaluate_asr.py --recorded-audio` to evaluate
> validated human-recorded dataset samples.

## Summary Metrics

| Metric | Score | Target |
|---|---|---|
| **Average WER** | **5.48 %** | < 10.0 % (clean) |
| **Average CER** | **1.64 %** | < 5.0 % |
| **Sentence Exact Match** | **73.33 %** | — |
| **Entity Overall F1** | **0.706** | > 0.90 |
| **Entity Exact Match Rate** | **26.67 %** | > 90.0 % |

## Entity Recognition Breakdown

| Category | Ground Truth Entities | Exact Matches (TP) | Precision | Recall | F1 Score | Exact Match Rate |
|---|---|---|---|---|---|---|
| Student IDs | 5 | 4 | 100.0 % | 80.0 % | 0.889 | 93.3 % |
| Numbers | 10 | 9 | 50.0 % | 90.0 % | 0.643 | 60.0 % |
| Dates / Times | 12 | 12 | 100.0 % | 100.0 % | 1.000 | 100.0 % |
| Names | 6 | 5 | 27.8 % | 83.3 % | 0.417 | 33.3 % |
| **Overall** | 33 | 30 | 57.7 % | 90.9 % | 0.706 | 26.7 % |

## Detailed Per-Case Evaluation

| ID | Domain | WER | CER | Sentence Match | Key Entities |
|---|---|---|---|---|---|
| `CASE-01` | student_identification | 0.0 % | 0.0 % | ✅ | student_ids: STU-1029 |
| `CASE-02` | grading_and_academics | 0.0 % | 0.0 % | ✅ | numbers: 94, 100, 101; names: Professor Sharma |
| `CASE-03` | deadlines_and_scheduling | 0.0 % | 0.0 % | ✅ | dates: October 24th, 5:00 PM |
| `CASE-04` | student_identification | 0.0 % | 0.0 % | ✅ | student_ids: S98234; names: Kalyan |
| `CASE-05` | grading_and_academics | 0.0 % | 0.0 % | ✅ | numbers: 3.85, 48 |
| `CASE-06` | assistant_persona | 0.0 % | 0.0 % | ✅ | dates: Monday; names: LockIN |
| `CASE-07` | faculty_and_advising | 0.0 % | 0.0 % | ✅ | numbers: 402; dates: tomorrow, 10:30 AM; names: Dr. Rao |
| `CASE-08` | course_enrollment | 0.0 % | 0.0 % | ✅ | student_ids: 2024CS101; numbers: three |
| `CASE-09` | complex_query | 0.0 % | 0.0 % | ✅ | student_ids: ID-9082; dates: November 15th, 2026 |
| `CASE-10` | general_student_support | 0.0 % | 0.0 % | ✅ | dates: 9:00 PM, Friday, 8:00 AM, Saturday |
| `CASE-11` | simulated_asr_error_homophone | 25.0 % | 6.8 % | ❌ | numbers: two, 204 |
| `CASE-12` | simulated_asr_error_spelling | 25.0 % | 2.1 % | ❌ | dates: Tuesday; names: LockIN, Professor Miller |
| `CASE-13` | simulated_asr_error_id_hyphen | 22.2 % | 2.1 % | ❌ | student_ids: STU-5542 |
| `CASE-14` | simulated_asr_error_deletion | 10.0 % | 13.6 % | ❌ | dates: December 12th |
| `CASE-15` | numerical_data | 0.0 % | 0.0 % | ✅ | numbers: 75% |