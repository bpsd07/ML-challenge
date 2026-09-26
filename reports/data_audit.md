# Stage 0: Data Audit & Referential Integrity Report
**Status:** COMPLETE  
**Audit Execution Time:** 60.23s  
---

## 1. File Structural & Schema Audit
| File | Rows | Cols | Unique IDs | Duplicate Rows | Null Names | Null Addrs | Null Country |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train_source1.tsv` | 2,206,821 | 4 | 2,206,821 | 0 | 0 | 0 | 0 |
| `train_source2.tsv` | 5,034,616 | 4 | 5,034,616 | 0 | 0 | 168,967 | 0 |
| `train_source3.tsv` | 5,285,603 | 4 | 5,285,603 | 0 | 0 | 175,916 | 0 |
| `train_ground_truth.tsv` | 2,206,821 | 2 | 2,206,821 | 0 | - | - | - |
| `test_source1.tsv` | 1,732,544 | 4 | 1,732,544 | 0 | 0 | 0 | 0 |
| `test_source2.tsv` | 4,887,273 | 4 | 4,887,273 | 0 | 0 | 129,408 | 0 |
| `test_source3.tsv` | 5,082,316 | 4 | 5,082,316 | 0 | 0 | 136,098 | 0 |

## 2. Ground Truth Integrity & Referential Validity
- **Total S1 Reference Entities in Ground Truth:** 2,206,821
- **Missing S1 from Ground Truth:** 0 (Every train S1 entity has exactly one row)
- **Duplicate S1 Rows in Ground Truth:** 0
- **Intra-list Duplicate IDs:** 0
- **Invalid S2 IDs referenced:** 0
- **Invalid S3 IDs referenced:** 0
- **Unknown / Malformed IDs referenced:** 0

## 3. Ground Truth Cross-Country Matching Analysis
> [!IMPORTANT]
> **Country Partitioning Decision Rule:**
> - Total S1-S2 True Pairs: 3,693,619 | Same Country: 3,693,619 | Cross-Country: **0 (0.0%)**
> - Total S1-S3 True Pairs: 3,944,746 | Same Country: 3,944,746 | Cross-Country: **0 (0.0%)**

**FINDING:** In the official ground truth, **EXACTLY 0.000%** of matches cross country boundaries! Business entity resolution is strictly intra-country. Partitioning candidate generation by country (`country == country`) is mathematically lossless on the ground truth while immediately pruning >50% of the candidate space. However, candidate generation logic must treat `country` as open-set so France test records are partitioned within France.

## 4. Legal Suffix & Numeric Token Distributions
### Train Source 1 Top Legal Suffixes
- `LIMITED`: 23,526 occurrences in 100k sample
- `PRIVATE`: 19,545 occurrences in 100k sample
- `LLC`: 16,142 occurrences in 100k sample
- `INC`: 10,937 occurrences in 100k sample
- `LTD`: 6,716 occurrences in 100k sample
- `PVT`: 5,474 occurrences in 100k sample
- `LLP`: 1,871 occurrences in 100k sample
- `CORP`: 1,567 occurrences in 100k sample
- `CO`: 1,233 occurrences in 100k sample
- `CORPORATION`: 677 occurrences in 100k sample

### Address Numeric Token Penetration
- S1 Addresses containing numeric digits (house numbers, postal codes): **96.54%**
- Average numeric tokens per address when present: **1.61**
