# Stage 2C: Multi-Script & Address Failure Analysis

**Evaluated Pilot:** Pilot A (50,000 S1 Entities against Full S2/S3 Population)  
**Baseline V9 Matches Recovered:** 164,971 (95.12%)  
**Stage 2C Matches Recovered:** **168,773** (**97.32%**)  
**Net Misses Recovered in Stage 2C:** **+3,802**  
**Remaining Missed Matches:** **4,656** (2.68%)  

---

## 1. Actual Script Distribution in Missed Ground Truth Pairs

| Script | Missed GT Pairs | % of Misses | Representative Example |
| :--- | :---: | :---: | :--- |
| `Latin` | **4,055** | 47.94% | S1: "Bay League Corp" <-> Cand: "Haloectoecto One aka Bay League Corp" |
| `Devanagari` | **2,367** | 27.99% | S1: "High Enterprises Private Limited" <-> Cand: "हाई एंटरप्राइजेज प्राइवेट लिमिटेड" |
| `Tamil` | **359** | 4.24% | S1: "City Media Private Limited" <-> Cand: "சிட்டி மீடியா பிரைவேட் லிமிடெட்" |
| `Kannada` | **352** | 4.16% | S1: "All Exports Private Limited" <-> Cand: "ಆಲ್ ಎಕ್ಸ್‌ಪೋರ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್" |
| `Telugu` | **329** | 3.89% | S1: "Silver Producer Limited" <-> Cand: "సిల్వర్ ప్రొడ్యూసర్ లిమిటెడ్" |
| `Bengali` | **324** | 3.83% | S1: "Alpha Ventures Private Limited" <-> Cand: "আলফা ভেঞ্চারস প্রাইভেট লিমিটেড" |
| `Gujarati` | **270** | 3.19% | S1: "Innovative Royal Infotech Private Limited" <-> Cand: "ઇનોવેટિવ રોયલ ઇન્ફોટેક પ્રાઇવેટ લિમિટેડ" |
| `Malayalam` | **222** | 2.62% | S1: "Sree Services" <-> Cand: "ശ്രീ സർവീസസ്" |
| `Odia` | **91** | 1.08% | S1: "Vision Constructions Private Limited" <-> Cand: "ଭିଜନ୍ କନଷ୍ଟ୍ରକସନ୍ସ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍" |
| `Gurmukhi` | **89** | 1.05% | S1: "Dynamic Services LLP" <-> Cand: "ਡਾਇਨਾਮਿਕ ਸਰਵਿਸਿਜ਼ ਐਲਐਲਪੀ" |

---

## 2. Individual Stage 2C Blocker Coverage Matrix

| Strategy | Focus Category | Total Matches | Recall % | Recovered V9 Misses | Incremental Gain % | Mean Cands/S1 | Gain/1k Cands |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `SB01_multiscript_translit_tok0` | Multi-script transliterated distinctive token 0 | 138,523 | 79.87% | **+1,708** | **+0.98%** | 7291.7 | 0.00 |
| `SB02_multiscript_translit_pair` | Multi-script transliterated token pair | 99,176 | 57.19% | **+0** | **+0.00%** | 494.3 | 0.00 |
| `SB03_multiscript_translit_numeric` | Multi-script translit token + house/numeric | 85,075 | 49.05% | **+0** | **+0.00%** | 7.2 | 0.00 |
| `SB04_lstrip_zeros_numeric_name` | Zero-stripped numeric atom + name token | 107,153 | 61.78% | **+13** | **+0.01%** | 60.9 | 0.00 |
| `SB05_pure_addr_postal_distinctive` | Country + Postal + distinctive street token | 5,601 | 3.23% | **+0** | **+0.00%** | 0.2 | 0.00 |
| `SB06_pure_addr_distinctive_pair` | Country + two distinctive address tokens | 102,587 | 59.15% | **+2,260** | **+1.30%** | 838.2 | 0.05 |
| `SB07_pure_addr_tok0_numeric` | Street token + numeric atom (no postal) | 92,866 | 53.55% | **+1,267** | **+0.73%** | 478.2 | 0.05 |
| `SB08_name_tok0_city_locality` | Name token 0 + street/locality token | 88,817 | 51.21% | **+8** | **+0.00%** | 26.7 | 0.01 |
