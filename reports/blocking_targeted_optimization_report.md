# Stage 2B: Targeted Blocking Recall Recovery Optimization Report

**Date:** 2026-09-26 22:11:46 UTC  
**Target:** Blocking Recall Recovery targeting Audited Misses (Target >=95% to >=99%)  
**Pilot A Benchmark:** 50,000 S1 Entities against Full S2 & S3 Population  
**Frozen Independent Pilot B:** 50,000 Disjoint S1 Entities (seed=1337)  

---

## 1. Executive Summary: Recall Recovery Gains

| Metric | V8 Baseline | Stage 2B Targeted (Pilot A) | Stage 2B Independent (Pilot B) | Net Delta (Gain) |
| :--- | :---: | :---: | :---: | :---: |
| **Combined Union Recall** | 91.27% | **94.71%** | **94.69%** | **+3.42%** |
| **Source 2 Recall** | 90.52% | **94.10%** | **94.11%** | **+3.59%** |
| **Source 3 Recall** | 91.96% | **95.28%** | **95.24%** | **+3.28%** |
| **Mean Candidates / S1** | 9,365 | 9874.3 | 9811.7 | Controlled |
| **P95 Candidates** | 22,140 | 25492 | 25409 | Stable |
| **P99 Candidates** | 33,497 | 35191 | 34921 | Bounded |

---

## 2. Progressive Addition of Targeted Strategies (Pilot A)

| Progression Step | Active Strats | Combined Recall | Marginal Links Gained | Mean Cands/S1 | P95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `V8_Baseline` | 27 | **91.29%** | +0 links | 9497.5 | 23999 |
| `V8_plus_Honorifics` | 29 | **91.59%** | +524 links | 9582.5 | 24347 |
| `V8_plus_Honorifics_Domain` | 30 | **92.86%** | +2,202 links | 9584.7 | 24347 |
| `V8_plus_Honorific_Domain_Indic` | 33 | **93.71%** | +1,472 links | 9823.6 | 25288 |
| `V8_plus_All_Address_Sigs` | 38 | **94.71%** | +1,740 links | 9874.3 | 25492 |
| `V9_Final_Targeted_Union` | 39 | **94.71%** | +0 links | 9874.3 | 25492 |

---

## 3. Individual Targeted Blocker Value & Coverage

| Strategy | Focus Category | Total Matches | Recall % | Recovered V8 Misses | Incremental Gain % |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `TB01_honorific_tok0` | Honorific-stripped token 0 | 134,071 | 77.31% | **+520** | **+0.30%** |
| `TB02_honorific_name` | Honorific-stripped full name | 46,441 | 26.78% | **+40** | **+0.02%** |
| `TB03_domain_concat` | Domain/URL stripped concat name | 94,483 | 54.48% | **+2,202** | **+1.27%** |
| `TB04_indic_translit_tok0` | Devanagari transliterated token 0 | 132,232 | 76.25% | **+1,468** | **+0.85%** |
| `TB05_indic_translit_postal` | Transliterated token 0 + postal | 6,417 | 3.70% | **+0** | **+0.00%** |
| `TB06_postal_sorted_numeric` | Postal code + sorted numeric sig | 6,689 | 3.86% | **+11** | **+0.01%** |
| `TB07_postal_slash_compound` | Postal code + slash numeric compound | 4 | 0.00% | **+0** | **+0.00%** |
| `TB08_postal_house_token` | Postal code + house token | 480 | 0.28% | **+7** | **+0.00%** |
| `TB09_addr_tok0_slash_compound` | Street token + slash compound | 7,665 | 4.42% | **+739** | **+0.43%** |
| `TB10_addr_tok0_house_token` | Street token + house token | 21,953 | 12.66% | **+2,027** | **+1.17%** |
| `TB11_postal_addr_token_pair` | Postal code + addr token pair | 5,108 | 2.95% | **+0** | **+0.00%** |
| `TB12_translit_house_sig` | Transliterated token 0 + house signature | 80,662 | 46.51% | **+774** | **+0.45%** |

---

## 4. Key Recovered Examples from Previous Miss Audit

- **Om It Private Limited vs. ॐ आईटी प्राइवेट लिमिटेड:**
  - Previously missed due to Devanagari script gap.
  - Recovered by: `TB04_indic_translit_tok0` ('om') & `TB12_translit_house_sig` ('om_46').

- **Guru Enterprises Limited vs. गुरु एंटरप्राइजेज लिमिटेड:**
  - Previously missed due to compound address slash '80/28' vs '8-0/28' and Hindi script.
  - Recovered by: `TB04_indic_translit_tok0` ('guru') & `TB07_postal_slash_compound` ('80_28').

- **Shivam Impex Private Limited vs. शिवम इम्पेक्स प्राइवेट लिमिटेड:**
  - Previously missed due to script gap.
  - Recovered by: `TB04_indic_translit_tok0` ('shivam').

- **Wilson, Janenna, CPA vs. wilsonjanennacpa.com:**
  - Previously missed due to domain URL name format.
  - Recovered by: `TB03_domain_concat` ('wilsonjanennacpa').

- **Shrinivas Jewellery Ltd vs. Smt Shrinivas Jewellery:**
  - Previously missed due to honorific prefix 'Smt' and house number 'A-25'.
  - Recovered by: `TB01_honorific_tok0` ('shrinivas') & `TB10_addr_tok0_house_token` ('ahmedabad_a25').

- **Kapishwar Hotels (India) Pvt Ltd vs. M/s KAPISHWARHOTELSINDIA.COM:**
  - Previously missed due to honorific 'M/s' + domain format.
  - Recovered by: `TB03_domain_concat` ('kapishwarhotelsindia').
