"""
Stage 3: Advanced Pairwise Feature Engineering Engine
Generates rich, robust, non-leaking pairwise similarity features between S1 and S2/S3 candidates.
Covers name similarity, address similarity, structured components, interaction terms,
and blocker-agreement meta-features.
"""

import re
import difflib
from typing import Dict, List, Set, Optional, Tuple
import numpy as np


def fast_jaccard(set_a: Set, set_b: Set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection_len = len(set_a.intersection(set_b))
    if intersection_len == 0:
        return 0.0
    union_len = len(set_a.union(set_b))
    return intersection_len / union_len if union_len > 0 else 0.0


def fast_cosine_set(set_a: Set, set_b: Set) -> float:
    """Compute cosine similarity between two binary n-gram sets."""
    if not set_a or not set_b:
        return 0.0
    intersection_len = len(set_a.intersection(set_b))
    if intersection_len == 0:
        return 0.0
    denom = (len(set_a) * len(set_b)) ** 0.5
    return intersection_len / denom if denom > 0 else 0.0


def fast_containment(set_a: Set, set_b: Set) -> float:
    """Compute containment (overlap divided by size of smaller set)."""
    if not set_a or not set_b:
        return 0.0
    min_size = min(len(set_a), len(set_b))
    if min_size == 0:
        return 0.0
    return len(set_a.intersection(set_b)) / min_size


def fast_edit_similarity(str_a: str, str_b: str) -> float:
    """Compute fast normalized edit similarity using difflib SequenceMatcher."""
    if str_a == str_b:
        return 1.0
    if not str_a or not str_b:
        return 0.0
    return difflib.SequenceMatcher(None, str_a, str_b).ratio()


# Blocker family definitions for provenance categorization
FAMILY_EXACT = {"B01_exact_name", "B02_no_legal_suffix", "B03_sorted_tokens", "B13_name_no_space"}
FAMILY_TOKEN = {"B05_first_two_tokens", "B14_first_distinctive_token", "B18_distinctive_token_pair", "B19_first_last_distinctive", "TB01_honorific_tok0"}
FAMILY_POSTAL = {"B08_postal_name_pfx", "B09_postal_house", "B21_postal_distinctive_tok", "B26_soundex_postal", "TB05_indic_translit_postal", "TB06_postal_sorted_numeric", "TB07_postal_slash_compound", "TB08_postal_house_token", "TB11_postal_addr_token_pair"}
FAMILY_NUMERIC = {"B07_house_tok0", "B10_token_numeric", "B20_tok1_house", "B23_numeric_sig_name", "B27_phonetic_house", "TB12_translit_house_sig", "SB03_multiscript_translit_numeric", "SB04_lstrip_zeros_numeric_name", "SB07_pure_addr_tok0_numeric"}
FAMILY_PHONETIC = {"B24_phonetic_tok0", "B25_phonetic_pair", "B26_soundex_postal", "B27_phonetic_house"}
FAMILY_INDIC = {"TB04_indic_translit_tok0", "TB05_indic_translit_postal", "TB12_translit_house_sig", "SB01_multiscript_translit_tok0", "SB02_multiscript_translit_pair", "SB03_multiscript_translit_numeric"}
FAMILY_PURE_ADDR = {"SB05_pure_addr_postal_distinctive", "SB06_pure_addr_distinctive_pair", "SB07_pure_addr_tok0_numeric"}


def compute_pairwise_features(
    s1_name_norm: dict,
    s1_addr_norm: dict,
    cand_name_norm: dict,
    cand_addr_norm: dict,
    cand_id: str,
    country: str,
    blockers_fired: int = 1,
    cand_rank: int = 1,
    strategies_fired: Optional[Set[str]] = None
) -> Dict[str, float]:
    """
    Compute dense numerical feature dictionary for an (S1, Candidate) pair.
    Returns 57 numerical features.
    """
    if strategies_fired is None:
        strategies_fired = set()

    # 1. NAME FEATURES
    s1_n_raw = s1_name_norm.get("name_raw", "")
    cand_n_raw = cand_name_norm.get("name_raw", "")

    s1_n_alnum = s1_name_norm.get("name_alnum", "")
    cand_n_alnum = cand_name_norm.get("name_alnum", "")

    s1_no_suf = s1_name_norm.get("name_no_legal_suffix", "")
    cand_no_suf = cand_name_norm.get("name_no_legal_suffix", "")

    s1_tokens = s1_name_norm.get("name_token_set", set())
    cand_tokens = cand_name_norm.get("name_token_set", set())

    name_exact = 1.0 if s1_n_alnum and s1_n_alnum == cand_n_alnum else 0.0
    name_no_suffix_exact = 1.0 if s1_no_suf and s1_no_suf == cand_no_suf else 0.0
    name_sorted_exact = 1.0 if s1_name_norm.get("name_tokens_sorted") == cand_name_norm.get("name_tokens_sorted") else 0.0

    name_edit_sim = fast_edit_similarity(s1_n_alnum, cand_n_alnum)
    name_tok_jaccard = fast_jaccard(s1_tokens, cand_tokens)
    name_tok_containment = fast_containment(s1_tokens, cand_tokens)
    name_tok_overlap = float(len(s1_tokens.intersection(cand_tokens)))

    s1_c2 = set([s1_n_alnum[i:i+2] for i in range(len(s1_n_alnum)-1)]) if len(s1_n_alnum) >= 2 else set()
    cand_c2 = set([cand_n_alnum[i:i+2] for i in range(len(cand_n_alnum)-1)]) if len(cand_n_alnum) >= 2 else set()
    name_c2_jaccard = fast_jaccard(s1_c2, cand_c2)

    s1_c3 = s1_name_norm.get("name_char_3gram", set())
    cand_c3 = cand_name_norm.get("name_char_3gram", set())
    name_c3_jaccard = fast_jaccard(s1_c3, cand_c3)
    name_c3_cosine = fast_cosine_set(s1_c3, cand_c3)

    s1_c4 = s1_name_norm.get("name_char_4gram", set())
    cand_c4 = cand_name_norm.get("name_char_4gram", set())
    name_c4_jaccard = fast_jaccard(s1_c4, cand_c4)
    name_c4_cosine = fast_cosine_set(s1_c4, cand_c4)

    len_s1_n = max(1, len(s1_n_alnum))
    len_cand_n = max(1, len(cand_n_alnum))
    name_len_diff = float(abs(len(s1_n_alnum) - len(cand_n_alnum)))
    name_len_ratio = min(len_s1_n, len_cand_n) / max(len_s1_n, len_cand_n)
    name_tok_diff = float(abs(len(s1_tokens) - len(cand_tokens)))

    # First token similarity
    s1_tok0 = s1_name_norm.get("name_tokens", [""])[0] if s1_name_norm.get("name_tokens") else ""
    cand_tok0 = cand_name_norm.get("name_tokens", [""])[0] if cand_name_norm.get("name_tokens") else ""
    name_first_tok_sim = fast_edit_similarity(s1_tok0, cand_tok0)

    # Distinctive token overlap
    s1_dist = set(s1_name_norm.get("distinctive_tokens", []))
    cand_dist = set(cand_name_norm.get("distinctive_tokens", []))
    name_distinctive_overlap = float(len(s1_dist.intersection(cand_dist)))

    # Initials
    s1_init = s1_name_norm.get("name_initials", "")
    cand_init = cand_name_norm.get("name_initials", "")
    initials_match = 1.0 if s1_init and s1_init == cand_init else 0.0

    # Phonetics
    s1_ph = s1_name_norm.get("phonetic_tokens", [])
    cand_ph = cand_name_norm.get("phonetic_tokens", [])
    phonetic_tok0_match = 1.0 if (s1_ph and cand_ph and s1_ph[0] == cand_ph[0]) else 0.0
    phonetic_jaccard = fast_jaccard(set(s1_ph), set(cand_ph))

    # Transliteration
    s1_trans = s1_name_norm.get("name_transliterated", "")
    cand_trans = cand_name_norm.get("name_transliterated", "")
    translit_name_sim = fast_edit_similarity(s1_trans, cand_trans) if (s1_trans and cand_trans) else 0.0

    s1_script = s1_name_norm.get("detected_script", "Latin")
    cand_script = cand_name_norm.get("detected_script", "Latin")
    native_vs_translit_agree = 1.0 if (s1_script != cand_script and translit_name_sim >= 0.7) else 0.0

    # Legal suffix agreement
    s1_suf = s1_name_norm.get("legal_suffix", "")
    cand_suf = cand_name_norm.get("legal_suffix", "")
    if s1_suf and cand_suf:
        legal_suffix_agree = 1.0 if s1_suf == cand_suf else 0.0
    elif not s1_suf and not cand_suf:
        legal_suffix_agree = 0.5
    else:
        legal_suffix_agree = 0.0

    # Domain TLD containment
    domain_name_contain = 1.0 if (s1_tok0 and s1_tok0 in cand_n_alnum) else 0.0

    # 2. ADDRESS FEATURES
    s1_a_alnum = s1_addr_norm.get("address_alnum", "")
    cand_a_alnum = cand_addr_norm.get("address_alnum", "")

    s1_a_tokens = set(s1_addr_norm.get("address_tokens", []))
    cand_a_tokens = set(cand_addr_norm.get("address_tokens", []))

    addr_exact = 1.0 if s1_a_alnum and s1_a_alnum == cand_a_alnum else 0.0
    addr_tok_jaccard = fast_jaccard(s1_a_tokens, cand_a_tokens)
    addr_tok_containment = fast_containment(s1_a_tokens, cand_a_tokens)
    addr_tok_overlap = float(len(s1_a_tokens.intersection(cand_a_tokens)))

    addr_c3_jaccard = fast_jaccard(s1_addr_norm.get("address_char_3gram", set()), cand_addr_norm.get("address_char_3gram", set()))
    addr_c4_jaccard = fast_jaccard(s1_addr_norm.get("address_char_4gram", set()), cand_addr_norm.get("address_char_4gram", set()))

    len_s1_a = max(1, len(s1_a_alnum))
    len_cand_a = max(1, len(cand_a_alnum))
    addr_len_diff = float(abs(len(s1_a_alnum) - len(cand_a_alnum)))
    addr_len_ratio = min(len_s1_a, len_cand_a) / max(len_s1_a, len_cand_a)

    # Distinctive address tokens
    s1_a_dist = set(s1_addr_norm.get("addr_distinctive", []))
    cand_a_dist = set(cand_addr_norm.get("addr_distinctive", []))
    addr_distinctive_overlap = float(len(s1_a_dist.intersection(cand_a_dist)))

    # Numeric atoms and house numbers
    s1_nums = set(s1_addr_norm.get("numeric_tokens", []))
    cand_nums = set(cand_addr_norm.get("numeric_tokens", []))
    s1_nums_norm = set([n.lstrip('0') for n in s1_nums if n.lstrip('0')])
    cand_nums_norm = set([n.lstrip('0') for n in cand_nums if n.lstrip('0')])

    numeric_atom_overlap = float(len(s1_nums_norm.intersection(cand_nums_norm)))
    numeric_atom_jaccard = fast_jaccard(s1_nums_norm, cand_nums_norm)

    s1_hn = s1_addr_norm.get("house_number", "")
    cand_hn = cand_addr_norm.get("house_number", "")
    house_num_exact = 1.0 if (s1_hn and cand_hn and s1_hn == cand_hn) else 0.0
    house_num_norm_match = 1.0 if (s1_hn and cand_hn and s1_hn.lstrip('0') == cand_hn.lstrip('0')) else 0.0

    # Postal codes
    s1_post = s1_addr_norm.get("postal_code", "")
    cand_post = cand_addr_norm.get("postal_code", "")
    postal_exact = 1.0 if (s1_post and cand_post and s1_post == cand_post) else 0.0
    postal_prefix2_match = 1.0 if (s1_post and cand_post and s1_post[:2] == cand_post[:2]) else 0.0
    postal_prefix3_match = 1.0 if (s1_post and cand_post and s1_post[:3] == cand_post[:3]) else 0.0
    postal_present_both = 1.0 if (s1_post and cand_post) else 0.0
    postal_mismatch = 1.0 if (s1_post and cand_post and s1_post != cand_post) else 0.0

    # Missing candidate address
    cand_addr_missing = 1.0 if not cand_a_alnum else 0.0

    # 3. CROSS-FIELD INTERACTIONS
    name_x_addr = name_tok_jaccard * addr_tok_jaccard
    both_exact = 1.0 if (name_exact == 1.0 and addr_exact == 1.0) else 0.0
    both_strong = 1.0 if (name_tok_jaccard >= 0.6 and addr_tok_jaccard >= 0.4) else 0.0
    exact_name_partial_addr = 1.0 if (name_exact == 1.0 and addr_tok_jaccard >= 0.25) else 0.0
    partial_name_exact_addr = 1.0 if (name_tok_jaccard >= 0.5 and addr_exact == 1.0) else 0.0

    # Numeric consistency between name and address
    name_nums = set(re.findall(r'\b\d+\b', s1_n_raw))
    name_addr_numeric_consistent = 1.0 if (name_nums and name_nums.intersection(cand_nums)) else 0.0

    # 4. PROVENANCE & BLOCKER META FEATURES
    source_is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
    blocker_votes_count = float(blockers_fired)
    cand_rank_by_votes = float(cand_rank)

    # Blocker families
    f_exact = 1.0 if (strategies_fired and strategies_fired.intersection(FAMILY_EXACT)) else 0.0
    f_token = 1.0 if (strategies_fired and strategies_fired.intersection(FAMILY_TOKEN)) else 0.0
    f_postal = 1.0 if (strategies_fired and strategies_fired.intersection(FAMILY_POSTAL)) else 0.0
    f_numeric = 1.0 if (strategies_fired and strategies_fired.intersection(FAMILY_NUMERIC)) else 0.0
    f_phonetic = 1.0 if (strategies_fired and strategies_fired.intersection(FAMILY_PHONETIC)) else 0.0
    f_indic = 1.0 if (strategies_fired and strategies_fired.intersection(FAMILY_INDIC)) else 0.0
    f_pure_addr = 1.0 if (strategies_fired and strategies_fired.intersection(FAMILY_PURE_ADDR)) else 0.0
    blocker_families_count = float(f_exact + f_token + f_postal + f_numeric + f_phonetic + f_indic + f_pure_addr)

    return {
        # Name features
        "name_exact": name_exact,
        "name_no_suffix_exact": name_no_suffix_exact,
        "name_sorted_exact": name_sorted_exact,
        "name_edit_sim": round(name_edit_sim, 4),
        "name_tok_jaccard": round(name_tok_jaccard, 4),
        "name_tok_containment": round(name_tok_containment, 4),
        "name_tok_overlap": name_tok_overlap,
        "name_c2_jaccard": round(name_c2_jaccard, 4),
        "name_c3_jaccard": round(name_c3_jaccard, 4),
        "name_c3_cosine": round(name_c3_cosine, 4),
        "name_c4_jaccard": round(name_c4_jaccard, 4),
        "name_c4_cosine": round(name_c4_cosine, 4),
        "name_len_diff": name_len_diff,
        "name_len_ratio": round(name_len_ratio, 4),
        "name_tok_diff": name_tok_diff,
        "name_first_tok_sim": round(name_first_tok_sim, 4),
        "name_distinctive_overlap": name_distinctive_overlap,
        "initials_match": initials_match,
        "phonetic_tok0_match": phonetic_tok0_match,
        "phonetic_jaccard": round(phonetic_jaccard, 4),
        "translit_name_sim": round(translit_name_sim, 4),
        "native_vs_translit_agree": native_vs_translit_agree,
        "legal_suffix_agree": legal_suffix_agree,
        "domain_name_contain": domain_name_contain,

        # Address features
        "addr_exact": addr_exact,
        "addr_tok_jaccard": round(addr_tok_jaccard, 4),
        "addr_tok_containment": round(addr_tok_containment, 4),
        "addr_tok_overlap": addr_tok_overlap,
        "addr_c3_jaccard": round(addr_c3_jaccard, 4),
        "addr_c4_jaccard": round(addr_c4_jaccard, 4),
        "addr_len_diff": addr_len_diff,
        "addr_len_ratio": round(addr_len_ratio, 4),
        "addr_distinctive_overlap": addr_distinctive_overlap,
        "numeric_atom_overlap": numeric_atom_overlap,
        "numeric_atom_jaccard": round(numeric_atom_jaccard, 4),
        "house_num_exact": house_num_exact,
        "house_num_norm_match": house_num_norm_match,
        "postal_exact": postal_exact,
        "postal_prefix2_match": postal_prefix2_match,
        "postal_prefix3_match": postal_prefix3_match,
        "postal_present_both": postal_present_both,
        "postal_mismatch": postal_mismatch,
        "cand_addr_missing": cand_addr_missing,

        # Cross-field / interactions
        "name_x_addr": round(name_x_addr, 4),
        "both_exact": both_exact,
        "both_strong": both_strong,
        "exact_name_partial_addr": exact_name_partial_addr,
        "partial_name_exact_addr": partial_name_exact_addr,
        "name_addr_numeric_consistent": name_addr_numeric_consistent,

        # Provenance & meta
        "source_is_s2": source_is_s2,
        "blocker_votes_count": blocker_votes_count,
        "cand_rank_by_votes": cand_rank_by_votes,
        "blocker_families_count": blocker_families_count,
        "fired_exact_family": f_exact,
        "fired_token_family": f_token,
        "fired_postal_family": f_postal,
        "fired_numeric_family": f_numeric,
        "fired_phonetic_family": f_phonetic,
        "fired_indic_family": f_indic,
        "fired_pure_addr_family": f_pure_addr
    }


FEATURE_NAMES = list(compute_pairwise_features(
    {}, {}, {}, {}, "S2-0", "US", 1, 1, set()
).keys())
