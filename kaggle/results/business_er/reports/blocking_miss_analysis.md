# Step 1: Detailed Audit of Baseline Missed Matches (13.31% Failure Analysis)
**Evaluated Pilot:** Pilot A (50,000 S1 Entities against Full S2/S3 Population)  
**Total True Ground Truth Pairs:** 173,429  
**Baseline True Matches Recovered:** 149,344 (86.11%)  
**Baseline Missed Matches:** **24,085** (13.89%)  

---

## 1. Failure Mode Distribution Table

| Rank | Failure Mode Category | Count | % of All Misses | % of Ground Truth | Primary Remediation Strategy |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `7. address typo` | **9,514** | 39.50% | 5.49% | Address Signature & Numeric Overlap Blocks |
| 2 | `6. transliteration` | **8,732** | 36.25% | 5.03% | Phonetic Consonant Skeleton & Soundex |
| 3 | `11. house-number issue` | **3,423** | 14.21% | 1.97% | Address Signature & Numeric Overlap Blocks |
| 4 | `9. missing address` | **802** | 3.33% | 0.46% | Address Signature & Numeric Overlap Blocks |
| 5 | `8. address truncation` | **531** | 2.20% | 0.31% | Address Signature & Numeric Overlap Blocks |
| 6 | `3. name abbreviation` | **396** | 1.64% | 0.23% | Abbreviation Expansion Dictionary |
| 7 | `2. name typo` | **301** | 1.25% | 0.17% | Character N-Grams & Rare Tokens |
| 8 | `5. word reorder` | **228** | 0.95% | 0.13% | Distinctive Token Pairs (Unordered) |
| 9 | `4. legal suffix difference` | **124** | 0.51% | 0.07% | Extended Legal Suffix Normalization |
| 10 | `10. postal mismatch` | **34** | 0.14% | 0.02% | Address Signature & Numeric Overlap Blocks |

---

## 2. Representative Miss Examples by Category

### Category: `7. address typo`
- **Example 1:**
  - **S1 (S1-2349009):** "Om It Private Limited" | Address: "S/O V K Gupta Pvt Shop No-46, Pno-S-513 T/F Vikash Marg, School Blk Shakarpur, Delhi, East Delhi, Delhi" (India)
  - **Candidate (S3-32352083):** "ॐ आईटी प्राइवेट लिमिटेड" | Address: "DL, Delhi, Door No 825 S/o V K Gupta Pvt Shop No-46, Pno-s-513 T/f Vikash Marg, School Blk Shakarpur, East Delhi"
- **Example 2:**
  - **S1 (S1-724160390):** "Shivam Impex Private Limited" | Address: "Second Floor, S-2, Plot No. 284, Radhika Heights, Mp Nagar, Huzur, Bhopal, Madhya Pradesh" (India)
  - **Candidate (S2-917012722):** "शिवम इम्पेक्स प्राइवेट लिमिटेड" | Address: "SECOND FLOR, HUZUR, Madhya Pradesh"
- **Example 3:**
  - **S1 (S1-131565832):** "Wilson, Janenna, CPA" | Address: "300 6th Street, City Of Racine, WI" (US)
  - **Candidate (S3-725293703):** "wilsonjanennacpa.com - 7713691326" | Address: "300 Sixth Street, Racine, Wisconsin"

### Category: `8. address truncation`
- **Example 1:**
  - **S1 (S1-2349009):** "Om It Private Limited" | Address: "S/O V K Gupta Pvt Shop No-46, Pno-S-513 T/F Vikash Marg, School Blk Shakarpur, Delhi, East Delhi, Delhi" (India)
  - **Candidate (S2-176865260):** "ओम आईटी प्राइवेट लिमिटेड" | Address: "21 S/O V K GUPTA PVT SHOP NO-46, PNO-S-513 T/F VIKASH MARG, SCHOOL BLK SHAKARPUR, DELHI, EAST DELHI, Delhi"
- **Example 2:**
  - **S1 (S1-878607831):** "Consultants International Private Limited" | Address: "A/P - Indapur Nagar, Parishad Hall, Hall Number 11 Al -Indapur, Pune, Maharashtra" (India)
  - **Candidate (S2-387073947):** "consultantsinternational.com" | Address: "PLOT 763 A/P - INDAPUR NAGAR, PARISHAD HALL, HALL NUMBER 11 AL -INDAPUR, PUNE, Maharashtra"
- **Example 3:**
  - **S1 (S1-574395205):** "Kapishwar Hotels (India) Pvt Ltd" | Address: "C/O Buddha Singh, Devipura-I, Bulandshahar, Bulandshahr, Uttar Pradesh" (India)
  - **Candidate (S2-292515055):** "M/s KAPISHWARHOTELSINDIA.COM" | Address: "C/O BUDDHA SINGH, DEVIPURA-I, BULANDSHAHAR, BULANDSHAHR, Uttar Pradesh"

### Category: `6. transliteration`
- **Example 1:**
  - **S1 (S1-425276878):** "Shrinivas Jewellery Ltd" | Address: "Ahmedabad, A-25, Jay Ambe Nagar Soc., Opp. Love Kush Tower, Nr. Udgam School, Thaltej, Gujarat, Ahmedabad" (India)
  - **Candidate (S3-386367044):** "Smt Shrinivas Jewellery" | Address: "A-25, Ahmedabad, Thaltaj, GJ"
- **Example 2:**
  - **S1 (S1-194756870):** "Bismillah College" | Address: "4/106, Vikalp Khand, Gomti Nagar, Lucknow, Uttar Pradesh" (India)
  - **Candidate (S3-315452609):** "M/s Bismillah College Co" | Address: "UP, Gomti Nagar, 4/106, Lucknow"
- **Example 3:**
  - **S1 (S1-586455267):** "Narasaraopet Patil Pvt Ltd" | Address: "Flat No 303, 3Rd Floor, Shanmukha Residency, Behind Sunshine School, Narasaraopet, Narasaraopet, Guntur, Andhra Pradesh" (India)
  - **Candidate (S3-893690436):** "The Narasaraopet Patil Pvt (Ltd)" | Address: "#A-303, 3Rd Floor, Shanmukha Residency, Behind Sunshine School, Narasaraopet, Narasaraopet, Guntur, ఆంధ్రప్రదేశ్"

### Category: `11. house-number issue`
- **Example 1:**
  - **S1 (S1-998962540):** "Guru Enterprises Limited" | Address: "80/28, Malviya Nagar, New Delhi, Delhi" (India)
  - **Candidate (S3-446264434):** "गुरु एंटरप्राइजेज लिमिटेड" | Address: "8-0/28, DL, Malviya Nagar, New Delhi"
- **Example 2:**
  - **S1 (S1-998962540):** "Guru Enterprises Limited" | Address: "80/28, Malviya Nagar, New Delhi, Delhi" (India)
  - **Candidate (S2-492402151):** "गुरु एंटरप्राइजेज लिमिटेड" | Address: "H.NO G-755 80/28, NEW DELHI, MALVIYA NAGAR, Delhi"
- **Example 3:**
  - **S1 (S1-998962540):** "Guru Enterprises Limited" | Address: "80/28, Malviya Nagar, New Delhi, Delhi" (India)
  - **Candidate (S3-485363505):** "गुरु एंटरप्राइजेज लिमिटेड" | Address: "8-0/28, Malviya Nagar, New Delhi, DL"

### Category: `9. missing address`
- **Example 1:**
  - **S1 (S1-496230732):** "Shubham Energy Corporation" | Address: "House No. 97 Floor 1St Block E Mansarowar Garden, New Delhi, Delhi" (India)
  - **Candidate (S3-981578172):** "Sri Shubham Energy Corporation" | Address: ""
- **Example 2:**
  - **S1 (S1-171507333):** "Veva LLC" | Address: "1131 Carlos Avenue, Fl 0, Wichita, KS" (US)
  - **Candidate (S2-596021262):** "VEVA" | Address: ""
- **Example 3:**
  - **S1 (S1-125929882):** "Alpha Estate LLP" | Address: "Plot No: 216/594, Orissa, Bhubaneswar, Paikanagar, Unit-8, Khordha" (India)
  - **Candidate (S3-234520187):** "Mr Alpha Estate L.L.P." | Address: ""

### Category: `5. word reorder`
- **Example 1:**
  - **S1 (S1-955011909):** "Glance Medical Centre" | Address: "Nr Venugopala Swamy Gudi, Andhra Pradesh, Guntur, Phanidhram, Sattenapalli Mandal, Guntur" (India)
  - **Candidate (S3-942446961):** "Centre Centre Glance-Medical" | Address: "Block D-65 Nr Venugopala Swamy Gudi, Phanidhram, Sattenapalli Mandal, Guntur, AP"
- **Example 2:**
  - **S1 (S1-329273968):** "India Ayansh Mining Ltd" | Address: "Noa4/524/2 1Floor 5 Cross, Mahalakshmi Layout, Bangalore North, Bangalore, Karnataka" (India)
  - **Candidate (S3-919962364):** "Ayansh India Ltd Mining" | Address: "H.no 11 Noa4/524/2 1Floor 5 Cross, Mahalakshmi Layout, Bangalore, Bengaluru, KA"
- **Example 3:**
  - **S1 (S1-623863439):** "Ridge Initiative Care" | Address: "11840 Glenwood Drive, Locust, NC" (US)
  - **Candidate (S3-826086779):** "Ridge Ridge Initiative Care" | Address: "1184 Glenwood Dr, Locust, North Carolina"

### Category: `2. name typo`
- **Example 1:**
  - **S1 (S1-878607831):** "Consultants International Private Limited" | Address: "A/P - Indapur Nagar, Parishad Hall, Hall Number 11 Al -Indapur, Pune, Maharashtra" (India)
  - **Candidate (S3-306389241):** "Consultants Private Limited Partners" | Address: "Pune, A/p - Indapur Nagar, Pune, MH"
- **Example 2:**
  - **S1 (S1-132298909):** "At Holdings" | Address: "Flat No. 201, Hyderabad, Uma Residential Enclave Road No. 9, Banjara Hills, Telangana, Hyderabad" (India)
  - **Candidate (S3-731144365):** "Sri At Holdings" | Address: "Flat No. 201, Uma Residential Enclave Road No. 9, Banjara Hills, Hyderabad, TG"
- **Example 3:**
  - **S1 (S1-132298909):** "At Holdings" | Address: "Flat No. 201, Hyderabad, Uma Residential Enclave Road No. 9, Banjara Hills, Telangana, Hyderabad" (India)
  - **Candidate (S2-207593729):** "The At Holdings" | Address: "H.NO 201 , UMA RESIDENTIAL ENCLAVE ROAD NO. 9, BANJARA HILLS, HYDERABAD, Telangana"

### Category: `3. name abbreviation`
- **Example 1:**
  - **S1 (S1-247383698):** "Smith Trust" | Address: "1123 Inn Keepers Way, Cornelius, NC" (US)
  - **Candidate (S2-808445838):** "smithtrust.com" | Address: "001123 INN KEEPERS WAY, CORNELIUS, NC"
- **Example 2:**
  - **S1 (S1-823555031):** "Electric LLC" | Address: "2016 Turkey Creek Road, Hurricane, WV" (US)
  - **Candidate (S2-271263893):** "ECECCRRIC LLC" | Address: "TURKEY CREEK ROAD, HURRICANE, WV"
- **Example 3:**
  - **S1 (S1-854593023):** "Vision Clinic" | Address: "TX, Rosenberg, 1620 Sunbury Drive" (US)
  - **Candidate (S2-128335916):** "vclinic.com" | Address: "1620 SUNBURY DRIVE, ROSENBERG, TX"

### Category: `4. legal suffix difference`
- **Example 1:**
  - **S1 (S1-785333318):** "Palinex" | Address: "32033 Sunripe Circle, Accomack County, VA" (US)
  - **Candidate (S3-490746292):** "Palinex Co" | Address: "032033 Sunripe Circle, PO Box 936, Accomack County, Virginia"
- **Example 2:**
  - **S1 (S1-800177993):** "Sirius LLC" | Address: "300 Camp Street, Iuka, IL" (US)
  - **Candidate (S2-975847561):** "Sirius" | Address: "CAMP STREET, IUKA, IL"
- **Example 3:**
  - **S1 (S1-463711638):** "Ab & Co" | Address: "C/O Laxman Ram Kachhawaha Chopta Bazar, Nayamandir K Niche, Pipar, City, Jodhpur, Rajasthan" (India)
  - **Candidate (S3-782229549):** "Ab &" | Address: "C/o Laxman Ram Kachhawaha Chopta Bazar, Jaipur, City, RJ"

### Category: `10. postal mismatch`
- **Example 1:**
  - **S1 (S1-517584595):** "American Network Inc" | Address: "10507 Chatham Ridge Way, Spotsylvania County, VA" (US)
  - **Candidate (S2-493009960):** "AMERICANNETWORK.COM" | Address: "10507-10509 Chatham Ridge Way, SPOTSYLVANIA, VA"
- **Example 2:**
  - **S1 (S1-511610035):** "Choice Dynamic Precious Associates" | Address: "Prince William County, VA, 15190 Wentwood Lane" (US)
  - **Candidate (S3-710004770):** "dynamicprecious.com" | Address: "15404 Wentwood Lane, Prince William County, Virginia"
- **Example 3:**
  - **S1 (S1-205109999):** "Satedyne International LLC" | Address: "Houston, TX, 11411 Glenwolde Drive" (US)
  - **Candidate (S2-549635761):** "sinternational.com" | Address: "11411-11413 GLENWOLDE DR, HOUSTON, TX"

