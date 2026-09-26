# Stage 2: High-Recall Multi-Strategy Blocking Optimization Report

**Date:** 2026-09-26 19:50:40 UTC  
**Target:** Maximum Blocking Recall (Target >=99%)  
**Pilot A Evaluation:** 50,000 S1 reference entities against Full S2 & S3 Population  
**Independent Pilot B Evaluation:** 50,000 Disjoint S1 reference entities  

---

## 1. Executive Recall Comparison (Baseline vs. Optimized V8)

| Metric | Baseline (V1) | Optimized (V8 Pilot A) | Independent (V8 Pilot B) | Delta (Gain) |
| :--- | :---: | :---: | :---: | :---: |
| **Combined Union Recall** | 86.11% | **91.17%** | **91.27%** | **+5.15%** |
| **Source 2 Recall** | 86.17% | **90.49%** | **90.52%** | **+4.36%** |
| **Source 3 Recall** | 86.06% | **91.80%** | **91.96%** | **+5.90%** |
| **Mean Candidates / S1** | 3229.6 | 9430.4 | 9365.0 | Managed Growth |
| **P95 Candidates** | 15989 | 23967 | 23830 | Stable |
| **P99 Candidates** | 18449 | 33646 | 33497 | Bounded |

---

## 2. Iterative Version Progression (Pilot A)

| Version | Focus Area | Combined Recall | Marginal Gain | Mean Cands/S1 | P95 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `V1_Baseline` | V1 | **86.11%** | +149,344 links | 3229.6 | 15989 |
| `V2_Normalization_Abbrev` | V2 | **89.03%** | +5,058 links | 7195.6 | 20401 |
| `V3_Char_Ngrams` | V3 | **89.21%** | +312 links | 7565.6 | 21915 |
| `V4_Rare_Tokens_Pairs` | V4 | **90.90%** | +2,935 links | 8071.1 | 22624 |
| `V5_Address_Signatures` | V5 | **90.92%** | +24 links | 8072.5 | 22627 |
| `V6_Phonetic_Retrieval` | V6 | **91.17%** | +441 links | 9430.4 | 23967 |
| `V7_Adaptive_Frequency_Filter` | V7 | **91.17%** | +0 links | 9430.4 | 23967 |
| `V8_Full_Optimized_Union` | V8 | **91.17%** | +0 links | 9430.4 | 23967 |
