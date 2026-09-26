# Amazon Business Entity Resolution — Kaggle Migration State

**Date:** 2026-09-26  
**Status:** In Progress (Migration Phase)  
**Host:** Windows (Local Control / Dev) -> Kaggle (Remote GPU/CPU Computation)

---

## CURRENT STAGE
**Transition from Local Prototyping to Remote Cloud Execution.**  
Local pilot experimentation (data auditing, normalization unit tests, 50k pilot blocking recall, and pilot LightGBM training) has successfully demonstrated pipeline viability. Computation on the full ~24M total records cannot run on local Windows hardware and is being migrated entirely to Kaggle.

---

## COMPLETED
1. **Stage 0: Data Audit & Schema Verification (`reports/data_audit.md`)**
   - Full referential integrity checked across 7 TSV files.
   - Discovered that **0.000%** of matches cross country boundaries in the ground truth (strictly intra-country matching).
2. **Stage 1: Multi-Country Normalization (`src/normalization.py`)**
   - Implemented legal suffix stripping across US, India, and France (`LLC`, `PVT LTD`, `SARL`, `SAS`, `SA`, `EURL`, etc.).
   - Implemented address normalization, token extraction, house number, and postal code regexes.
   - Comprehensive unit tests passed (`src/test_normalization.py`).
3. **Stage 2: Multi-Strategy Blocking Evaluation (`reports/blocking_evaluation.md`)**
   - Evaluated 15 independent blocking strategies on 50k S1 pilot entities.
   - Combined union recall reached 86.69% with ~1,050 average candidates per S1.
4. **Stage 4: Pairwise Feature Engineering (`src/features.py`)**
   - Name similarities (exact, no-suffix, token Jaccard, char 3-gram/4-gram Jaccard, Levenshtein, length ratios).
   - Address similarities (token overlap, char n-grams, postal match, house number match).
   - Meta-features (blocker score, candidate rank, country match).
5. **Stage 5: Competition Metric Definition (`src/metrics.py`)**
   - Exact implementation of macro-averaged $F_{0.5}$ at $S_1$ entity level (Precision weighted 2x Recall).
   - Correct singleton edge-case handling ($1.0$ for true empty, $0.0$ for false match).
6. **Kaggle CLI & Dataset Discovery**
   - Installed Kaggle CLI in project virtual environment.
   - Configured authentication via Kaggle API access token (`KGAT_...`).
   - Located existing public Kaggle dataset containing the complete challenge data: `satwiksps/amazon-ml-challenge-2026`.

---

## INCOMPLETE
1. Full-scale candidate blocking on the entire training set (2,206,821 S1 vs 5,034,616 S2 and 5,285,603 S3).
2. Large-scale LightGBM training with hard negatives from blocking candidate pool.
3. Fine threshold optimization and singleton protection tuning for Macro $F_{0.5}$.
4. Full test set inference across all 1,732,544 test $S_1$ records (including unseen country France) against 4,887,273 S2 and 5,082,316 S3 records.
5. Generation of final `matching_results.tsv` and `candidate_pairs.tsv`.
6. Official validation via `student_resource/utils/validate_submission.py`.

---

## PARTIAL ARTIFACTS
- `reports/data_audit.md`: Schema, row count, null check, and country partition findings.
- `reports/dataset_profile.md` & `reports/dataset_profile.json`: Statistical distribution profiles.
- `reports/blocking_evaluation.md`: Blocker recall curves on 50k pilot.
- `kaggle/kernel-metadata.json`: Initial Kaggle kernel configuration.

---

## REUSABLE CODE
- `src/normalization.py`: Name & address normalization, legal suffixes, phonetic rules.
- `src/blocking.py`: Multi-strategy candidate generation and inverted indexes.
- `src/features.py`: Numerical feature calculations for candidate pairs.
- `src/metrics.py`: Macro $F_{0.5}$ evaluation function.
- `src/train.py`: Polars Rust vectorization engine, index builder, threshold optimizer.
- `student_resource/utils/validate_submission.py`: Competition formatting validator.

---

## KNOWN PROBLEMS & DESIGN CONSTRAINTS
1. **Memory Ceiling:** Even with 30 GB RAM on Kaggle, dense pairwise matrices ($S_1 \times S_2$ or $S_1 \times S_3$) would require petabytes and crash immediately. Everything must be processed via country-partitioned inverted indexes and chunked batch streaming.
2. **Open-Set Country (`France`):** Test set includes France which does not appear in training data. Country filtering must remain dynamic (never hardcode `US`/`India`).
3. **High Precision Asymmetry ($F_{0.5}$):** Precision counts twice as much as recall. Low thresholds that pick up marginal matches will severely degrade the score. Optimal threshold is typically high ($T \approx 0.75 - 0.85$).
4. **Singletons:** 5.58% of S1 entities have zero matches. False positives on singletons receive a score of 0.0. Strong singleton protection logic is mandatory.

---

## NEXT STAGE
1. Verify Kaggle CLI installation (`kaggle --version`) and authentication (`kaggle kernels list --mine`).
2. Construct a lightweight **smoke-test notebook** in `kaggle/` to verify Kaggle compute environment (CPU, RAM, GPU, dataset path discovery) without running heavy computation.
3. Push smoke-test kernel to Kaggle and monitor execution until completion.
4. Report smoke test findings and proceed with full modular pipeline migration.
