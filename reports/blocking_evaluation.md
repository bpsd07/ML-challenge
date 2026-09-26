# Stage 2: Multi-Strategy Blocking Evaluation Report (50k Pilot)
**Pilot Size:** 50,000 S1 entities  
**Ground Truth Links in Pilot:** 173,429 (83,936 S2, 89,493 S3)  
**Total Pilot Runtime:** 1262.21s  
---

## 1. Executive Blocking Summary
- **Combined Final Blocking Recall:** **86.69%** (150,353 / 173,429 true matches retained)
- **S2 Recall:** **86.89%** (72,928 / 83,936)
- **S3 Recall:** **86.52%** (77,425 / 89,493)
- **Average Candidates per S1:** **1050.66** (Median: 1034.0, P95: 2038.0, P99: 2418.010000000002, Max: 3868.0)

## 2. Individual Blocker Performance
| Blocker Strategy | S2 Recall % | S3 Recall % | Comb Recall % | Mean Cands/S1 | P95 Cands | Max Cands |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `BLOCK_A_exact_name` | 25.24% | 26.4% | **25.84%** | 10.94 | 69.0 | 500.0 |
| `BLOCK_B_no_legal_suffix` | 24.79% | 24.91% | **24.86%** | 20.67 | 69.0 | 500.0 |
| `BLOCK_C_sorted_tokens` | 26.26% | 27.05% | **26.67%** | 9.06 | 44.0 | 500.0 |
| `BLOCK_D_initials` | 42.11% | 41.03% | **41.55%** | 69.82 | 360.0 | 500.0 |
| `BLOCK_E_first_two_tokens` | 55.74% | 52.85% | **54.25%** | 66.17 | 500.0 | 500.0 |
| `BLOCK_F_prefix_len` | 32.82% | 22.96% | **27.73%** | 306.24 | 500.0 | 500.0 |
| `BLOCK_G_3gram_sig` | 21.34% | 12.6% | **16.83%** | 416.85 | 500.0 | 500.0 |
| `BLOCK_H_house_num_name` | 46.1% | 44.35% | **45.19%** | 9.62 | 40.0 | 500.0 |
| `BLOCK_I_postal_name_pfx` | 3.85% | 3.7% | **3.77%** | 0.14 | 1.0 | 11.0 |
| `BLOCK_J_postal_house_num` | 4.18% | 4.34% | **4.26%** | 2.68 | 10.0 | 213.0 |
| `BLOCK_K_token_num` | 48.44% | 48.6% | **48.52%** | 14.93 | 76.0 | 500.0 |
| `BLOCK_L_addr_name_joint` | 38.65% | 37.1% | **37.85%** | 23.17 | 129.0 | 500.0 |
| `BLOCK_M_consonant_core` | 62.08% | 55.35% | **58.61%** | 148.66 | 500.0 | 500.0 |
| `BLOCK_N_second_tok_postal` | 3.0% | 2.88% | **2.94%** | 0.11 | 0.0 | 13.0 |
| `BLOCK_O_addr_lead_tokens` | 39.67% | 38.82% | **39.23%** | 97.78 | 500.0 | 500.0 |

## 3. Cumulative Union Recall Curve
| Cumulative Blocker Union | Cum S2 Recall % | Cum S3 Recall % | Cum Combined Recall % | Marginal True Links Recovered |
| :--- | :---: | :---: | :---: | :---: |
| `+ BLOCK_A_exact_name` | 25.24% | 26.4% | **25.84%** | +44,808 |
| `+ BLOCK_B_no_legal_suffix` | 35.38% | 35.76% | **35.57%** | +16,887 |
| `+ BLOCK_C_sorted_tokens` | 37.44% | 37.7% | **37.57%** | +3,467 |
| `+ BLOCK_D_initials` | 50.9% | 49.85% | **50.36%** | +22,172 |
| `+ BLOCK_E_first_two_tokens` | 69.81% | 66.88% | **68.3%** | +31,111 |
| `+ BLOCK_F_prefix_len` | 72.46% | 68.63% | **70.48%** | +3,794 |
| `+ BLOCK_G_3gram_sig` | 73.4% | 69.24% | **71.25%** | +1,332 |
| `+ BLOCK_H_house_num_name` | 76.37% | 73.9% | **75.09%** | +6,664 |
| `+ BLOCK_I_postal_name_pfx` | 76.54% | 74.09% | **75.28%** | +315 |
| `+ BLOCK_J_postal_house_num` | 76.8% | 74.69% | **75.71%** | +758 |
| `+ BLOCK_K_token_num` | 77.14% | 75.35% | **76.22%** | +871 |
| `+ BLOCK_L_addr_name_joint` | 77.31% | 75.56% | **76.41%** | +331 |
| `+ BLOCK_M_consonant_core` | 80.82% | 78.95% | **79.85%** | +5,978 |
| `+ BLOCK_N_second_tok_postal` | 80.82% | 78.95% | **79.85%** | +3 |
| `+ BLOCK_O_addr_lead_tokens` | 86.89% | 86.52% | **86.69%** | +11,862 |
