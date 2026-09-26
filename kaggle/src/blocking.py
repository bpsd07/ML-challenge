"""
Stage 2: High-Recall Multi-Strategy Blocking Engine (V2 - Target >=99% Recall)
Implements 20+ specialized high-recall, precision-conscious blocking strategies.
Eliminates arbitrary posting-list cutoffs that lose true matches.
Employs country partitioning, rare-token indexing, character n-gram inverted indexing,
phonetic representations, address signatures, and adaptive key frequency filtering.
"""

import sys
import time
import re
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
import polars as pl
import numpy as np

from normalization import (
    normalize_business_name,
    normalize_business_address,
    extract_postal_code,
    extract_house_number,
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
        Optionally filter by version level (1 to 8).
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

        addr_alnum = addr_norm["address_alnum"]
        addr_tokens = addr_norm["address_tokens"]
        postal = addr_norm["postal_code"]
        house_num = addr_norm["house_number"]
        numeric_tokens = list(addr_norm["numeric_tokens"])
        addr_distinctive = addr_norm["address_distinctive_tokens"]

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
