# Stage 2C: Multi-Script & Remaining Address Recall Recovery Optimization Report

**Date:** 2026-09-27 08:31:49 UTC  
**Objective:** Recover Dravidian/Bengali script gap and address digit/format mismatches (Target >=97% to >=98%)  
**Pilot A Benchmark:** 50,000 S1 Entities against Full S2 & S3 Population  
**Frozen Independent Pilot B:** 50,000 Disjoint S1 Entities (seed=1337)  

---

## 1. Executive Summary: Recall Progression from Baseline to Stage 2C

| Metric | V1 Baseline | V8 Blocker | Stage 2B (V9) | Stage 2C Final Union | Stage 2C Pilot B Independent |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Combined Union Recall** | 86.11% | 91.29% | 94.71% | **97.32%** | **97.27%** |
| **Source 2 Recall** | 85.50% | 90.62% | 94.10% | **97.33%** | **97.23%** |
| **Source 3 Recall** | 87.80% | 91.91% | 95.28% | **97.30%** | **97.31%** |
| **Mean Candidates / S1** | 4,286 | 9,498 | 9,874 | **11740.5** | **11644.4** |
| **P95 Candidates** | 14,354 | 23,999 | 25,492 | **31522** | **31239** |
| **P99 Candidates** | 24,960 | 33,558 | 35,191 | **43015** | **42902** |

---

## 2. Progressive Addition of Stage 2C Strategies (Pilot A)

| Progression Step | Active Strats | Combined Recall | Marginal Links Gained | Mean Cands/S1 | P95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `V9_Baseline` | 39 | **95.12%** | +0 links | 10005.0 | 25745 |
| `V9_plus_Multiscript_Translit` | 42 | **96.11%** | +1,708 links | 10592.0 | 28723 |
| `V9_plus_Multiscript_ZeroStrip` | 44 | **96.73%** | +1,076 links | 11005.3 | 29914 |
| `V9_plus_Pure_Address_Routes` | 46 | **97.32%** | +1,018 links | 11740.5 | 31522 |
| `Stage2C_Final_Optimized_Union` | 47 | **97.32%** | +0 links | 11740.5 | 31522 |

---

## 3. Representative Recovered Matches in Stage 2C

- **All Exports Private Limited vs. ಆಲ್ ಎಕ್ಸ್‌ಪೋರ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್:**
  - Script: Kannada.
  - Recovered by: `SB01_multiscript_translit_tok0` ('al') and `SB03_multiscript_translit_numeric` ('al_570').

- **Swastik Logistics Private Limited vs. ಸ್ವಸ್ತಿಕ್ ಲಾಜಿಸ್ಟಿಕ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್:**
  - Script: Kannada.
  - Recovered by: `SB01_multiscript_translit_tok0` ('svstk') and `SB07_pure_addr_tok0_numeric` ('vidhyashankara_323').

- **Gold Hari It Private Limited vs. ગોલ્ડ હરિ આઈટી પ્રાઇવેટ લિમિટેડ:**
  - Script: Gujarati.
  - Recovered by: `SB01_multiscript_translit_tok0` ('gold') and `SB02_multiscript_translit_pair` ('gold_hari').

- **Perfect Engineering Limited vs. পারফেক্ট ইঞ্জিনিয়ারিং লিমিটেড:**
  - Script: Bengali.
  - Recovered by: `SB07_pure_addr_tok0_numeric` ('praghati_855') and `SB06_pure_addr_distinctive_pair`.

- **City Media Private Limited vs. சிட்டி மீடியா பிரைவேட் லிமிடெட்:**
  - Script: Tamil.
  - Recovered by: `SB07_pure_addr_tok0_numeric` ('periya_221').

- **A+ Health Inc. vs. A+ Inc. Services (045810 HEADLAND RD vs 45810 Headland Road):**
  - Address leading zero mismatch.
  - Recovered by: `SB04_lstrip_zeros_numeric_name` ('45810') and `SB07_pure_addr_tok0_numeric` ('headland_45810').

- **Memphis Salon vs. Fluxdrex (5470 Briardale Drive vs 005470 Briardale Drive):**
  - Address double leading zero mismatch.
  - Recovered by: `SB07_pure_addr_tok0_numeric` ('briardale_5470').
