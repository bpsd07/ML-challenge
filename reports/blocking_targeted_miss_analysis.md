# Stage 2B: Targeted Miss Analysis & Remaining Failure Audit

**Evaluated Pilot:** Pilot A (50,000 S1 Entities against Full S2/S3 Population)  
**Total True Ground Truth Pairs:** 173,429  
**Baseline V8 Matches Recovered:** 158,321 (91.29%)  
**Baseline V8 Missed Matches:** 15,108 (8.71%)  
**Final Targeted Union Matches Recovered:** **164,259** (**94.71%**)  
**Net Missed Links Recovered by Stage 2B:** **+5,938** (39.30% reduction in misses)  
**Remaining Missed Matches:** **9,170** (5.29%)  

---

## 1. Coverage Matrix of Targeted Strategies

| Strategy | Focus Failure Mode | Total Matches | Recall % | Misses Recovered | Incremental Recall % | Mean Cands/S1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `TB01_honorific_tok0` | Honorific-stripped token 0 | 134,071 | 77.31% | **+520** | **+0.30%** | 4994.1 |
| `TB02_honorific_name` | Honorific-stripped full name | 46,441 | 26.78% | **+40** | **+0.02%** | 11.3 |
| `TB03_domain_concat` | Domain/URL stripped concat name | 94,483 | 54.48% | **+2,202** | **+1.27%** | 58.7 |
| `TB04_indic_translit_tok0` | Devanagari transliterated token 0 | 132,232 | 76.25% | **+1,468** | **+0.85%** | 5097.4 |
| `TB05_indic_translit_postal` | Transliterated token 0 + postal | 6,417 | 3.70% | **+0** | **+0.00%** | 0.1 |
| `TB06_postal_sorted_numeric` | Postal code + sorted numeric sig | 6,689 | 3.86% | **+11** | **+0.01%** | 1.9 |
| `TB07_postal_slash_compound` | Postal code + slash numeric compound | 4 | 0.00% | **+0** | **+0.00%** | 0.0 |
| `TB08_postal_house_token` | Postal code + house token | 480 | 0.28% | **+7** | **+0.00%** | 0.0 |
| `TB09_addr_tok0_slash_compound` | Street token + slash compound | 7,665 | 4.42% | **+739** | **+0.43%** | 1.0 |
| `TB10_addr_tok0_house_token` | Street token + house token | 21,953 | 12.66% | **+2,027** | **+1.17%** | 57.6 |
| `TB11_postal_addr_token_pair` | Postal code + addr token pair | 5,108 | 2.95% | **+0** | **+0.00%** | 0.1 |
| `TB12_translit_house_sig` | Transliterated token 0 + house signature | 80,662 | 46.51% | **+774** | **+0.45%** | 5.9 |

---

## 2. Remaining Failure Mode Distribution Table

| Rank | Failure Mode Category | Count | % of Remaining Misses | % of Ground Truth | Nature of Miss |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `7. address typo / digit-format` | **4,089** | 44.59% | 2.36% | Irreducible / Truncated Data |
| 2 | `6. transliteration / script gap` | **3,205** | 34.95% | 1.85% | Irreducible / Truncated Data |
| 3 | `11. house-number issue` | **1,418** | 15.46% | 0.82% | Irreducible / Truncated Data |
| 4 | `3. name abbreviation` | **154** | 1.68% | 0.09% | Irreducible / Truncated Data |
| 5 | `8. address truncation` | **144** | 1.57% | 0.08% | Irreducible / Truncated Data |
| 6 | `2. name typo` | **112** | 1.22% | 0.06% | Irreducible / Truncated Data |
| 7 | `9. missing address` | **27** | 0.29% | 0.02% | Irreducible / Truncated Data |
| 8 | `10. postal mismatch` | **21** | 0.23% | 0.01% | Irreducible / Truncated Data |

---

## 3. Representative Remaining Miss Examples by Category

### Category: `7. address typo / digit-format`
- **Example 1:**
  - **S1 (S1-316488472):** "All Exports Private Limited" | Address: "No 570, J P Nagar, 9Th Phase, Bangalore, Karnataka" (India)
  - **Candidate (S2-347291141):** "ಆಲ್ ಎಕ್ಸ್‌ಪೋರ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್" | Address: "#570, J P NAGAR, 9TH PHASE, BANGALORE, Karnataka"
- **Example 2:**
  - **S1 (S1-575396121):** "Swastik Logistics Private Limited" | Address: "323, Vidhyashankara Layout, Mysore, Karnataka" (India)
  - **Candidate (S2-490390745):** "ಸ್ವಸ್ತಿಕ್ ಲಾಜಿಸ್ಟಿಕ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್" | Address: "MYSORE, C-323, Karnataka"
- **Example 3:**
  - **S1 (S1-575396121):** "Swastik Logistics Private Limited" | Address: "323, Vidhyashankara Layout, Mysore, Karnataka" (India)
  - **Candidate (S3-7557659):** "ಸ್ವಸ್ತಿಕ್ ಲಾಜಿಸ್ಟಿಕ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್" | Address: "Mysore, Vidhyashankara Layout, KA, 323"

### Category: `6. transliteration / script gap`
- **Example 1:**
  - **S1 (S1-762056751):** "High Enterprises Private Limited" | Address: "Flat 903, Bhoomi Heights, Vazira Naka, Lt Road, Borivali (W), Opp Ganesh Temple, Mumbai, Mumbai City, Maharashtra" (India)
  - **Candidate (S3-228146696):** "हाई एंटरप्राइजेज प्राइवेट लिमिटेड" | Address: "Flat D/903, Bhoomi Heights, Vazira Naka, Lt Road, Borivali (W), Opp Ganesh Temple, Bombay, Mumbai City, MH"
- **Example 2:**
  - **S1 (S1-543644400):** "Aditya Finance Private Limited" | Address: "Jalna, C/O.H.No 95 At.Po.Dhoksal, Jalna, Maharashtra" (India)
  - **Candidate (S2-1158470):** "आदित्य Finance प्राइवेट लिमिटेड" | Address: "C/O.H.NO A-95 AT.PO.DHOKSAL, JALNA, Maharashtra"
- **Example 3:**
  - **S1 (S1-543644400):** "Aditya Finance Private Limited" | Address: "Jalna, C/O.H.No 95 At.Po.Dhoksal, Jalna, Maharashtra" (India)
  - **Candidate (S2-321709045):** "आदित्य फाइनेंस प्राइवेट लिमिटेड" | Address: "C/O.H.NO A-95 AT.PO.DHOKSAL, JALNA, Maharashtra"

### Category: `2. name typo`
- **Example 1:**
  - **S1 (S1-878607831):** "Consultants International Private Limited" | Address: "A/P - Indapur Nagar, Parishad Hall, Hall Number 11 Al -Indapur, Pune, Maharashtra" (India)
  - **Candidate (S3-306389241):** "Consultants Private Limited Partners" | Address: "Pune, A/p - Indapur Nagar, Pune, MH"
- **Example 2:**
  - **S1 (S1-332964443):** "Sree Consultants" | Address: "17/5, Purameri Post, Vadakara Grameen Bank, Kozhikode, Kerala" (India)
  - **Candidate (S2-116748146):** "ശ്രീ Consultants" | Address: "H.NO 301 17/5, KOZHIKODE, Kerala"
- **Example 3:**
  - **S1 (S1-33408672):** "A+ Health Inc." | Address: "45810 Headland Road, Woodland, WA" (US)
  - **Candidate (S2-780481760):** "A+ Inc. Services" | Address: "045810 HEADLAND RD, WOODLAND, WA"

### Category: `11. house-number issue`
- **Example 1:**
  - **S1 (S1-968901184):** "Gold Hari It Private Limited" | Address: "B/12, Ashray Bunglows, Behind Yash Complex, Geri Compound Road, Gotri, Vadodara, Gujarat" (India)
  - **Candidate (S2-447950788):** "ગોલ્ડ હરિ આઈટી પ્રાઇવેટ લિમિટેડ" | Address: "Gujarat, VADODARA, #142  B/12, VADODARA"
- **Example 2:**
  - **S1 (S1-302167224):** "Crystal Rithm Inc." | Address: "2365 300th Street, Lehigh, IA" (US)
  - **Candidate (S3-852690422):** "Crysta1 Inc. Partners" | Address: "2365b 300th St, Lehigh, Iowa"
- **Example 3:**
  - **S1 (S1-366973583):** "Memphis Salon" | Address: "5470 Briardale Drive, Memphis, TN" (US)
  - **Candidate (S3-729987446):** "Fluxdrex" | Address: "005470 Briardale Drive, Memphis, Tennessee"

### Category: `8. address truncation`
- **Example 1:**
  - **S1 (S1-468967762):** "City Media Private Limited" | Address: "221 Kms Garden Periya Puthur, Salem, Tamil Nadu" (India)
  - **Candidate (S2-570814193):** "சிட்டி மீடியா பிரைவேட் லிமிடெட்" | Address: "DOOR NO 221 KMS GARDEN PERIYA PUTHUR, SALEM, Tamil Nadu"
- **Example 2:**
  - **S1 (S1-805655437):** "Perfect Engineering Limited" | Address: "Praghati Nurshing Home, Iskon Mandir Road Nr Siliguri Club, Baneswar More, Ektiasal, Sevoke Rd, Siliguri, Darjeeling, West Bengal" (India)
  - **Candidate (S2-568924647):** "পারফেক্ট ইঞ্জিনিয়ারিং লিমিটেড" | Address: "PLOT 855 PRAGHATI NURSHING HOME, ISKON MANDIR ROAD NR SILIGURI CLUB, BANESWAR MORE, EKTIASAL, SEVOKE RD, SILIGURI, DARJEELING, West Bengal"
- **Example 3:**
  - **S1 (S1-551458522):** "Future Vision Consultants Private Limited" | Address: "No. 18, Old No. 109, Hari Brindhavan 8Th 'B' Main, 30Th Cross, 4Th Block, Jay, Bangalore, Karnataka" (India)
  - **Candidate (S2-459510378):** "ಫ್ಯೂಚರ್ ವಿಷನ್ ಕನ್ಸಲ್ಟೆಂಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್" | Address: "18, OLD NO. 109, HARI BRINDHAVAN 8TH 'B' MAIN, 30TH CROSS, 4TH BLOCK, JAY, BANGALORE, Karnataka"

### Category: `3. name abbreviation`
- **Example 1:**
  - **S1 (S1-823555031):** "Electric LLC" | Address: "2016 Turkey Creek Road, Hurricane, WV" (US)
  - **Candidate (S2-271263893):** "ECECCRRIC LLC" | Address: "TURKEY CREEK ROAD, HURRICANE, WV"
- **Example 2:**
  - **S1 (S1-854593023):** "Vision Clinic" | Address: "TX, Rosenberg, 1620 Sunbury Drive" (US)
  - **Candidate (S2-128335916):** "vclinic.com" | Address: "1620 SUNBURY DRIVE, ROSENBERG, TX"
- **Example 3:**
  - **S1 (S1-547962563):** "Baon Drugs" | Address: "Norbourne Estates, 419 Breckenridge Ln, KY" (US)
  - **Candidate (S3-540735288):** "bdrúgs.com" | Address: "NULL, Kentucky, Breckenridge Ln, Norbourne Estates"

### Category: `9. missing address`
- **Example 1:**
  - **S1 (S1-770880268):** "Lyrial Armada Inc" | Address: "121 Flegal Lane, Frederick County, VA" (US)
  - **Candidate (S2-602381171):** "Lyria1 Armada" | Address: ""
- **Example 2:**
  - **S1 (S1-452811033):** "Aadharshila Services Private Limited" | Address: "381/G Mukkanniyil Arcade, Ayavana Panchayath, Muvattupuzha, Ernakulam, Kerala" (India)
  - **Candidate (S2-319979325):** "Aadharshi1a Services Private" | Address: ""
- **Example 3:**
  - **S1 (S1-802790446):** "International Consultants Private Limited" | Address: "Flat No 153, Mohan Nagar 4, Datta Madir Dahanukarwadi Road, Kandivali West, Mumbai, Mumbai City, Maharashtra" (India)
  - **Candidate (S2-443611004):** "International Private Limited Center" | Address: ""

### Category: `10. postal mismatch`
- **Example 1:**
  - **S1 (S1-511610035):** "Choice Dynamic Precious Associates" | Address: "Prince William County, VA, 15190 Wentwood Lane" (US)
  - **Candidate (S3-710004770):** "dynamicprecious.com" | Address: "15404 Wentwood Lane, Prince William County, Virginia"
- **Example 2:**
  - **S1 (S1-205109999):** "Satedyne International LLC" | Address: "Houston, TX, 11411 Glenwolde Drive" (US)
  - **Candidate (S2-549635761):** "sinternational.com" | Address: "11411-11413 GLENWOLDE DR, HOUSTON, TX"
- **Example 3:**
  - **S1 (S1-427038655):** "Constantia Lovett, DO, M.D., P.C." | Address: "13874 Elkhorn Road, Garfield, AR" (US)
  - **Candidate (S3-644819733):** "Jaxyuma" | Address: "13876 Elkhorn Road, Garfield, Arkansas"

