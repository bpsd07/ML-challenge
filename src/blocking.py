from __future__ import annotations
"""
Stage 2: High-Recall Multi-Strategy Blocking Engine (V2 - Stage 2B Targeted Blocking)
Implements 39 specialized high-recall, precision-conscious blocking strategies:
- Baseline V8 strategies (B01 to B27): Exact, suffixes, n-grams, rare token pairs, phonetic skeletons, address signatures.
- Targeted Stage 2B strategies (TB01 to TB12):
  * Devanagari / Indic transliterated token & postal/numeric keys
  * Domain / URL stripping & concatenation keys
  * Honorific stripping keys
  * Slash-aware numeric compound keys (80/28, 38/2, etc.)
  * Alphanumeric house token keys (Pno-S-513, A-25, H.No G-755, Door No 825, #A-303, etc.)
  * Pure address composite keys for script gap recovery
Eliminates arbitrary candidate truncations while maintaining strict frequency caps to prevent candidate explosions.
"""

import sys
import time
import re
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
try:
    import polars as pl
except ImportError:
    pl = None
try:
    import numpy as np
except Exception:
    np = None

from normalization import (
    normalize_business_name,
    normalize_business_address,
    extract_postal_code,
    extract_house_number,
    extract_address_signatures,
    strip_domain_tld,
    strip_honorifics,
    transliterate_devanagari,
    get_consonant_skeleton,
    soundex,
    GENERIC_STOPWORDS
)

# Key frequency threshold: keys appearing more than this across the source are skipped
# as uninformative stopwords to avoid memory explosion, while informative keys retain 100% of records.
MAX_KEY_FREQUENCY = 15000


class HighRecallBlocker:
    """
    High-Recall Partitioned Inverted Index Blocker.
    Supports individual blocker ablation, cumulative union, and unique contribution auditing.
    """

    def __init__(self, max_key_frequency: int = MAX_KEY_FREQUENCY):
        self.max_key_frequency = max_key_frequency
        # index: strategy -> country -> key -> list of candidate_ids
        self.indices = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.key_counts = defaultdict(lambda: defaultdict(Counter))
        self.indexed_candidate_count = 0

    def generate_keys_for_record(
        self,
        name_norm: dict,
        addr_norm: dict,
        country: str,
        active_versions: Optional[Set[int]] = None
    ) -> Dict[str, List[str]]:
        """
        Generate blocking keys for all implemented strategies.
        Optionally filter by version level (1 to 8) or targeted strategies.
        """
        keys = defaultdict(list)
        cntry = country or "UNKNOWN"

        name_alnum = name_norm["name_alnum"]
        no_suf = name_norm["name_no_legal_suffix"]
        tokens = name_norm["name_tokens"]
        sorted_tokens = name_norm["name_tokens_sorted"]
        initials = name_norm["name_initials"]
        c3_grams = name_norm["name_char_3gram"]
        c4_grams = name_norm["name_char_4gram"]
        distinctive_toks = name_norm["distinctive_tokens"]
        phonetic_toks = name_norm["phonetic_tokens"]
        soundex_toks = name_norm["soundex_tokens"]
        name_abbrev = name_norm["name_abbrev_expanded"]

        domain_clean = name_norm.get("domain_clean")
        no_honorific_distinctive = name_norm.get("no_honorific_distinctive", [])
        name_no_honorific = name_norm.get("name_no_honorific")
        translit_distinctive = name_norm.get("transliterated_distinctive", [])
        translit_phonetic = name_norm.get("transliterated_phonetic", [])
        concat_distinctive = name_norm.get("name_concat_distinctive", "")

        addr_alnum = addr_norm["address_alnum"]
        addr_tokens = addr_norm["address_tokens"]
        postal = addr_norm["postal_code"]
        house_num = addr_norm["house_number"]
        numeric_tokens = list(addr_norm.get("numeric_tokens", []))
        addr_distinctive = addr_norm.get("address_distinctive_tokens", [])

        slash_compounds = list(addr_norm.get("slash_compounds", []))
        house_tokens = list(addr_norm.get("house_tokens", []))
        sorted_num_sig = addr_norm.get("sorted_numeric_sig")
        house_sig = addr_norm.get("house_signature")

        # -------------------------------------------------------------
        # VERSION 1: BASELINE STRATEGIES
        # -------------------------------------------------------------
        if active_versions is None or 1 in active_versions:
            # 1. Exact normalized name
            if name_alnum:
                keys["B01_exact_name"].append(name_alnum)

            # 2. Name without legal suffix
            if no_suf and no_suf != name_alnum:
                keys["B02_no_legal_suffix"].append(no_suf)

            # 3. Sorted name tokens
            if sorted_tokens and sorted_tokens != name_alnum:
                keys["B03_sorted_tokens"].append(sorted_tokens)

            # 4. Initials + first token prefix
            if len(tokens) >= 2 and len(initials) >= 2:
                keys["B04_initials_pfx"].append(f"{initials[:4]}_{tokens[0][:4]}")

            # 5. First two tokens
            if len(tokens) >= 2:
                keys["B05_first_two_tokens"].append(f"{tokens[0]}_{tokens[1]}")

            # 6. First 4 chars + length bucket
            if len(name_alnum) >= 4:
                keys["B06_prefix4_len"].append(f"{name_alnum[:4]}_{len(name_alnum)//4}")

            # 7. House number + first name token
            if house_num and tokens:
                keys["B07_house_tok0"].append(f"{house_num}_{tokens[0]}")

            # 8. Postal code + 3-char name prefix
            if postal and len(name_alnum) >= 3:
                keys["B08_postal_name_pfx"].append(f"{postal}_{name_alnum[:3]}")

            # 9. Postal code + House number
            if postal and house_num:
                keys["B09_postal_house"].append(f"{postal}_{house_num}")

            # 10. First name token + numeric token
            if tokens and numeric_tokens:
                keys["B10_token_numeric"].append(f"{tokens[0]}_{numeric_tokens[0]}")

            # 11. Address leading tokens
            if len(addr_tokens) >= 2:
                keys["B11_addr_lead_tokens"].append(f"{addr_tokens[0]}_{addr_tokens[1]}")

        # -------------------------------------------------------------
        # VERSION 2: FIX NORMALIZATION & COMPOUND WORDS & ABBREVIATIONS
        # -------------------------------------------------------------
        if active_versions is None or 2 in active_versions:
            # 12. Abbreviation-expanded name
            if name_abbrev and name_abbrev != name_alnum:
                keys["B12_name_abbrev_expanded"].append(name_abbrev)

            # 13. Compound / concatenated name without spaces (e.g., "wal mart" vs "walmart")
            name_no_space = "".join(tokens)
            if len(name_no_space) >= 5 and name_no_space != name_alnum:
                keys["B13_name_no_space"].append(name_no_space)

            # 14. First distinctive non-stopword token (handles "The Home Depot" -> "home")
            if distinctive_toks:
                keys["B14_first_distinctive_token"].append(distinctive_toks[0])

        # -------------------------------------------------------------
        # VERSION 3: CHARACTER N-GRAM SHINGLES (RECOVERS TYPOS & EDITS)
        # -------------------------------------------------------------
        if active_versions is None or 3 in active_versions:
            # 15. Prefix 4-gram + Suffix 4-gram
            if len(name_alnum) >= 8:
                keys["B15_prefix_suffix_4gram"].append(f"{name_alnum[:4]}_{name_alnum[-4:]}")

            # 16. Skip-gram: first 2 chars + middle 2 chars
            if len(name_alnum) >= 6:
                mid = len(name_alnum) // 2
                keys["B16_skip_gram"].append(f"{name_alnum[:2]}_{name_alnum[mid:mid+2]}_{name_alnum[-2:]}")

            # 17. Two rarest 3-grams from distinctive tokens
            if len(distinctive_toks) >= 1:
                t0 = distinctive_toks[0]
                if len(t0) >= 5:
                    keys["B17_distinctive_tok_ngrams"].append(f"{t0[:3]}_{t0[-3:]}")

        # -------------------------------------------------------------
        # VERSION 4: RARE-TOKEN BLOCKS & TOKEN PAIRS (RECOVERS WORD REORDERS)
        # -------------------------------------------------------------
        if active_versions is None or 4 in active_versions:
            # 18. Any pair of distinctive tokens
            if len(distinctive_toks) >= 2:
                for i in range(min(3, len(distinctive_toks))):
                    for j in range(i + 1, min(4, len(distinctive_toks))):
                        pair = sorted([distinctive_toks[i], distinctive_toks[j]])
                        keys["B18_distinctive_token_pair"].append(f"{pair[0]}_{pair[1]}")

            # 19. First + Last distinctive token
            if len(distinctive_toks) >= 2:
                keys["B19_first_last_distinctive"].append(f"{distinctive_toks[0]}_{distinctive_toks[-1]}")

            # 20. Second distinctive token + house number
            if len(distinctive_toks) >= 2 and house_num:
                keys["B20_tok1_house"].append(f"{distinctive_toks[1]}_{house_num}")

        # -------------------------------------------------------------
        # VERSION 5: ADDRESS SIGNATURES & LOCALITY
        # -------------------------------------------------------------
        if active_versions is None or 5 in active_versions:
            # 21. Postal code + first distinctive name token
            if postal and distinctive_toks:
                keys["B21_postal_distinctive_tok"].append(f"{postal}_{distinctive_toks[0]}")

            # 22. Street distinctive token + first distinctive name token
            if addr_distinctive and distinctive_toks:
                keys["B22_street_name_joint"].append(f"{addr_distinctive[0]}_{distinctive_toks[0]}")

            # 23. Numeric token signature + first distinctive token
            if len(numeric_tokens) >= 2 and distinctive_toks:
                sorted_nums = "_".join(sorted(numeric_tokens)[:2])
                keys["B23_numeric_sig_name"].append(f"{sorted_nums}_{distinctive_toks[0]}")

        # -------------------------------------------------------------
        # VERSION 6: PHONETIC RETRIEVAL (INDIAN TRANSLITERATIONS & PHONETICS)
        # -------------------------------------------------------------
        if active_versions is None or 6 in active_versions:
            # 24. Consonant skeleton of first distinctive token
            if phonetic_toks:
                keys["B24_phonetic_tok0"].append(phonetic_toks[0][:6])

            # 25. Phonetic pair (first 2 distinctive tokens)
            if len(phonetic_toks) >= 2:
                p_pair = sorted([phonetic_toks[0][:5], phonetic_toks[1][:5]])
                keys["B25_phonetic_pair"].append(f"{p_pair[0]}_{p_pair[1]}")

            # 26. Soundex token 0 + postal code
            if soundex_toks and postal:
                keys["B26_soundex_postal"].append(f"{soundex_toks[0]}_{postal}")

            # 27. Consonant skeleton + house number
            if phonetic_toks and house_num:
                keys["B27_phonetic_house"].append(f"{phonetic_toks[0][:5]}_{house_num}")

        # -------------------------------------------------------------
        # STAGE 2B: TARGETED RECALL RECOVERY BLOCKERS (TB01 to TB12)
        # -------------------------------------------------------------
        # TB01: Honorific-stripped first distinctive token
        # Handles "Smt Shrinivas Jewellery", "M/s Bismillah College", "Mr Alpha Estate", "Sri At Holdings"
        if no_honorific_distinctive:
            keys["TB01_honorific_tok0"].append(no_honorific_distinctive[0])
        elif distinctive_toks:
            keys["TB01_honorific_tok0"].append(distinctive_toks[0])

        # TB02: Honorific-stripped full normalized name
        if name_no_honorific:
            h_alnum = re.sub(r'[^a-z0-9\s]', ' ', name_no_honorific.lower()).strip()
            if h_alnum:
                keys["TB02_honorific_name"].append(h_alnum)
        elif name_alnum:
            keys["TB02_honorific_name"].append(name_alnum)

        # TB03: Domain / URL normalized name matching concatenated tokens
        # Handles "wilsonjanennacpa.com", "consultantsinternational.com", "kapishwarhotelsindia.com"
        if domain_clean:
            keys["TB03_domain_concat"].append(domain_clean)
        no_suf_concat = no_suf.replace(" ", "")
        if len(no_suf_concat) >= 5:
            keys["TB03_domain_concat"].append(no_suf_concat)
        if concat_distinctive and concat_distinctive != no_suf_concat:
            keys["TB03_domain_concat"].append(concat_distinctive)

        # TB04: Indic / Devanagari transliterated distinctive token 0
        # Handles "ॐ आईटी प्राइवेट लिमिटेड" -> "om", "शिवम इम्पेक्स" -> "shivam", "गुरु एंटरप्राइजेज" -> "guru"
        if translit_distinctive:
            keys["TB04_indic_translit_tok0"].append(translit_distinctive[0])
        elif distinctive_toks:
            keys["TB04_indic_translit_tok0"].append(distinctive_toks[0])

        # TB05: Indic transliterated token 0 + postal code
        # Couples transliterated token with postal for high precision
        if translit_distinctive and postal:
            keys["TB05_indic_translit_postal"].append(f"{translit_distinctive[0]}_{postal}")
        elif distinctive_toks and postal:
            keys["TB05_indic_translit_postal"].append(f"{distinctive_toks[0]}_{postal}")

        # TB06: Postal code + sorted numeric signature
        # Handles complex compound addresses under the same postal PIN code
        if postal and sorted_num_sig:
            keys["TB06_postal_sorted_numeric"].append(f"{postal}_{sorted_num_sig}")

        # TB07: Postal code + slash-aware numeric compound (e.g. 80/28, 38/2, 1098/100, 4/524/2)
        if postal and slash_compounds:
            for sc in slash_compounds:
                keys["TB07_postal_slash_compound"].append(f"{postal}_{sc}")

        # TB08: Postal code + alphanumeric house token (e.g. Pno-S-513, A-25, H.No G-755, #A-303, Door No 825)
        if postal and house_tokens:
            for ht in house_tokens:
                keys["TB08_postal_house_token"].append(f"{postal}_{ht}")

        # TB09: Street distinctive token + slash-aware numeric compound
        # Handles cases where postal code is missing or slightly discordant
        if addr_distinctive and slash_compounds:
            for sc in slash_compounds:
                keys["TB09_addr_tok0_slash_compound"].append(f"{addr_distinctive[0]}_{sc}")

        # TB10: Street distinctive token + alphanumeric house token
        if addr_distinctive and house_tokens:
            for ht in house_tokens:
                keys["TB10_addr_tok0_house_token"].append(f"{addr_distinctive[0]}_{ht}")

        # TB11: Postal code + distinctive address token pair (pure address blocking for script gap)
        if postal and len(addr_distinctive) >= 2:
            pair = sorted([addr_distinctive[0], addr_distinctive[1]])
            keys["TB11_postal_addr_token_pair"].append(f"{postal}_{pair[0]}_{pair[1]}")

        # TB12: Indic transliterated token 0 + house signature / numeric atom
        if translit_distinctive and house_sig:
            keys["TB12_translit_house_sig"].append(f"{translit_distinctive[0]}_{house_sig}")
        elif distinctive_toks and house_sig:
            keys["TB12_translit_house_sig"].append(f"{distinctive_toks[0]}_{house_sig}")

        # -------------------------------------------------------------
        # STAGE 2C: MULTI-SCRIPT & REMAINING ADDRESS STRATEGIES (SB01 to SB08)
        # -------------------------------------------------------------
        # SB01: Multi-script transliterated distinctive token 0 + phonetic skeleton
        # Covers Kannada, Telugu, Tamil, Bengali, Gujarati, Malayalam, Odia, Gurmukhi
        if translit_distinctive:
            keys["SB01_multiscript_translit_tok0"].append(translit_distinctive[0])
            if translit_phonetic:
                keys["SB01_multiscript_translit_tok0"].append(translit_phonetic[0])
        elif distinctive_toks:
            keys["SB01_multiscript_translit_tok0"].append(distinctive_toks[0])
            if phonetic_toks:
                keys["SB01_multiscript_translit_tok0"].append(phonetic_toks[0])

        # SB02: Multi-script transliterated token pair
        if len(translit_distinctive) >= 2:
            pair = sorted([translit_distinctive[0], translit_distinctive[1]])
            keys["SB02_multiscript_translit_pair"].append(f"{pair[0]}_{pair[1]}")
        elif len(distinctive_toks) >= 2:
            pair = sorted([distinctive_toks[0], distinctive_toks[1]])
            keys["SB02_multiscript_translit_pair"].append(f"{pair[0]}_{pair[1]}")

        # SB03: Multi-script transliterated token 0 + house signature / numeric atom
        tok0 = translit_distinctive[0] if translit_distinctive else (distinctive_toks[0] if distinctive_toks else "")
        if tok0:
            if house_sig:
                keys["SB03_multiscript_translit_numeric"].append(f"{tok0}_{house_sig}")
            elif numeric_tokens:
                keys["SB03_multiscript_translit_numeric"].append(f"{tok0}_{numeric_tokens[0]}")

        # SB04: Zero-stripped numeric atom + name token 0 (recovers 045810 -> 45810, 005470 -> 5470)
        if tok0 and numeric_tokens:
            for num in numeric_tokens[:2]:
                norm_num = num.lstrip('0')
                if norm_num:
                    keys["SB04_lstrip_zeros_numeric_name"].append(f"{tok0}_{norm_num}")

        # SB05: Country + Postal + first distinctive address token (Pure address route)
        if postal and addr_distinctive:
            keys["SB05_pure_addr_postal_distinctive"].append(f"{cntry}_{postal}_{addr_distinctive[0]}")

        # SB06: Country + two distinctive address tokens (Pure address route)
        if len(addr_distinctive) >= 2:
            a_pair = sorted([addr_distinctive[0], addr_distinctive[1]])
            keys["SB06_pure_addr_distinctive_pair"].append(f"{cntry}_{a_pair[0]}_{a_pair[1]}")

        # SB07: Street distinctive token + numeric atom (without postal)
        # Recovers cases where postal is missing or discordant
        if addr_distinctive and numeric_tokens:
            for num in numeric_tokens[:2]:
                norm_num = num.lstrip('0')
                if norm_num:
                    keys["SB07_pure_addr_tok0_numeric"].append(f"{addr_distinctive[0]}_{norm_num}")

        # SB08: Name distinctive token 0 + street / locality token
        if tok0 and addr_distinctive:
            keys["SB08_name_tok0_city_locality"].append(f"{tok0}_{addr_distinctive[0]}")

        return keys

    def index_candidate_source(self, df_candidates: pl.DataFrame, source_name: str):
        """
        Index full candidate source into memory-efficient country-partitioned inverted indexes.
        """
        print(f"Indexing {len(df_candidates):,} records from {source_name}...")
        t0 = time.time()
        count = 0

        for row in df_candidates.select(["entity_id", "business_name", "business_address", "country"]).iter_rows():
            ent_id = row[0]
            raw_name = row[1]
            raw_addr = row[2]
            cntry = str(row[3] or "UNKNOWN")

            name_norm = normalize_business_name(raw_name, country=cntry)
            addr_norm = normalize_business_address(raw_addr, country=cntry)

            keys_dict = self.generate_keys_for_record(name_norm, addr_norm, cntry)

            for strat, key_list in keys_dict.items():
                strat_index = self.indices[strat][cntry]
                for k in key_list:
                    plist = strat_index[k]
                    # Never drop records arbitrarily; limit only huge generic uninformative keys
                    if len(plist) < self.max_key_frequency:
                        plist.append(ent_id)

            count += 1
            if count % 1000000 == 0:
                print(f"  Indexed {count:,} records in {time.time()-t0:.1f}s...")

        self.indexed_candidate_count += count
        print(f"Finished indexing {source_name} ({count:,} records) in {time.time() - t0:.2f}s.")

    def query_s1_entity(
        self,
        name_norm: dict,
        addr_norm: dict,
        country: str,
        enabled_strategies: Optional[Set[str]] = None
    ) -> Tuple[Set[str], Dict[str, Set[str]]]:
        """
        Retrieve candidates for a single S1 entity.
        Returns:
            union_candidates: set of candidate IDs across all enabled strategies
            strat_candidates: dict mapping strategy -> candidate IDs
        """
        cntry = country or "UNKNOWN"
        keys_dict = self.generate_keys_for_record(name_norm, addr_norm, cntry)

        union_candidates = set()
        strat_candidates = {}

        for strat, key_list in keys_dict.items():
            if enabled_strategies is not None and strat not in enabled_strategies:
                continue

            strat_index = self.indices[strat].get(cntry, {})
            cands_for_strat = set()
            for k in key_list:
                posting = strat_index.get(k, [])
                if posting:
                    cands_for_strat.update(posting)

            if cands_for_strat:
                strat_candidates[strat] = cands_for_strat
                union_candidates.update(cands_for_strat)

        return union_candidates, strat_candidates
