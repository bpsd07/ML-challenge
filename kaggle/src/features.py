"""
Stage 4: Pairwise Feature Engineering Engine
Generates rich, robust, non-leaking pairwise similarity features between S1 and S2/S3 candidates.
Covers name similarity, address similarity, structured components, interaction terms,
and blocker-agreement meta-features.
"""

import re
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


def fast_containment(set_a: Set, set_b: Set) -> float:
    """Compute containment (overlap divided by size of smaller set)."""
    if not set_a or not set_b:
        return 0.0
    min_size = min(len(set_a), len(set_b))
    if min_size == 0:
        return 0.0
    return len(set_a.intersection(set_b)) / min_size


def compute_pairwise_features(
    s1_name_norm: dict,
    s1_addr_norm: dict,
    cand_name_norm: dict,
    cand_addr_norm: dict,
    cand_id: str,
    country: str,
    blockers_fired: int = 1,
    cand_rank: int = 1
) -> Dict[str, float]:
    """
    Compute dense numerical feature dictionary for an (S1, Candidate) pair.
    """
    # 1. Name Features
    s1_n_raw = s1_name_norm["name_raw"]
    cand_n_raw = cand_name_norm["name_raw"]

    s1_n_alnum = s1_name_norm["name_alnum"]
    cand_n_alnum = cand_name_norm["name_alnum"]

    s1_no_suf = s1_name_norm["name_no_legal_suffix"]
    cand_no_suf = cand_name_norm["name_no_legal_suffix"]

    s1_tokens = s1_name_norm["name_token_set"]
    cand_tokens = cand_name_norm["name_token_set"]

    name_exact = 1.0 if s1_n_alnum and s1_n_alnum == cand_n_alnum else 0.0
    name_no_suffix_exact = 1.0 if s1_no_suf and s1_no_suf == cand_no_suf else 0.0
    name_sorted_exact = 1.0 if s1_name_norm["name_tokens_sorted"] == cand_name_norm["name_tokens_sorted"] else 0.0

    name_tok_jaccard = fast_jaccard(s1_tokens, cand_tokens)
    name_tok_containment = fast_containment(s1_tokens, cand_tokens)
    name_tok_overlap = float(len(s1_tokens.intersection(cand_tokens)))

    name_c3_jaccard = fast_jaccard(s1_name_norm["name_char_3gram"], cand_name_norm["name_char_3gram"])
    name_c4_jaccard = fast_jaccard(s1_name_norm["name_char_4gram"], cand_name_norm["name_char_4gram"])

    len_s1_n = max(1, len(s1_n_alnum))
    len_cand_n = max(1, len(cand_n_alnum))
    name_len_ratio = min(len_s1_n, len_cand_n) / max(len_s1_n, len_cand_n)
    name_tok_diff = abs(len(s1_tokens) - len(cand_tokens))

    initials_match = 1.0 if s1_name_norm["name_initials"] and s1_name_norm["name_initials"] == cand_name_norm["name_initials"] else 0.0

    # 2. Address Features
    s1_a_alnum = s1_addr_norm["address_alnum"]
    cand_a_alnum = cand_addr_norm["address_alnum"]

    s1_a_tokens = set(s1_addr_norm["address_tokens"])
    cand_a_tokens = set(cand_addr_norm["address_tokens"])

    addr_exact = 1.0 if s1_a_alnum and s1_a_alnum == cand_a_alnum else 0.0
    addr_tok_jaccard = fast_jaccard(s1_a_tokens, cand_a_tokens)
    addr_tok_containment = fast_containment(s1_a_tokens, cand_a_tokens)
    addr_tok_overlap = float(len(s1_a_tokens.intersection(cand_a_tokens)))

    addr_c3_jaccard = fast_jaccard(s1_addr_norm["address_char_3gram"], cand_addr_norm["address_char_3gram"])

    len_s1_a = max(1, len(s1_a_alnum))
    len_cand_a = max(1, len(cand_a_alnum))
    addr_len_ratio = min(len_s1_a, len_cand_a) / max(len_s1_a, len_cand_a)

    # Structured address components
    s1_post = s1_addr_norm["postal_code"]
    cand_post = cand_addr_norm["postal_code"]
    postal_match = 1.0 if (s1_post and cand_post and s1_post == cand_post) else 0.0
    postal_missing = 1.0 if (not s1_post or not cand_post) else 0.0

    s1_hn = s1_addr_norm["house_number"]
    cand_hn = cand_addr_norm["house_number"]
    house_num_match = 1.0 if (s1_hn and cand_hn and s1_hn == cand_hn) else 0.0

    s1_num_digits = s1_addr_norm["numeric_tokens"]
    cand_num_digits = cand_addr_norm["numeric_tokens"]
    digit_jaccard = fast_jaccard(s1_num_digits, cand_num_digits)
    digit_overlap = float(len(s1_num_digits.intersection(cand_num_digits)))

    # Missing address flag
    cand_addr_missing = 1.0 if not cand_a_alnum else 0.0

    # 3. Interactions & Cross-Features
    name_x_addr = name_tok_jaccard * addr_tok_jaccard
    max_similarity = max(name_tok_jaccard, addr_tok_jaccard)
    both_strong = 1.0 if (name_tok_jaccard >= 0.6 and addr_tok_jaccard >= 0.4) else 0.0
    exact_name_partial_addr = 1.0 if (name_exact == 1.0 and addr_tok_jaccard >= 0.25) else 0.0
    partial_name_exact_addr = 1.0 if (name_tok_jaccard >= 0.5 and addr_exact == 1.0) else 0.0
    both_exact = 1.0 if (name_exact == 1.0 and addr_exact == 1.0) else 0.0

    # 4. Meta & Contextual Features
    source_is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
    cntry_us = 1.0 if country == "US" else 0.0
    cntry_in = 1.0 if country == "India" else 0.0
    cntry_fr = 1.0 if country == "France" else 0.0

    return {
        "name_exact": name_exact,
        "name_no_suffix_exact": name_no_suffix_exact,
        "name_sorted_exact": name_sorted_exact,
        "name_tok_jaccard": round(name_tok_jaccard, 4),
        "name_tok_containment": round(name_tok_containment, 4),
        "name_tok_overlap": name_tok_overlap,
        "name_c3_jaccard": round(name_c3_jaccard, 4),
        "name_c4_jaccard": round(name_c4_jaccard, 4),
        "name_len_ratio": round(name_len_ratio, 4),
        "name_tok_diff": float(name_tok_diff),
        "initials_match": initials_match,
        "addr_exact": addr_exact,
        "addr_tok_jaccard": round(addr_tok_jaccard, 4),
        "addr_tok_containment": round(addr_tok_containment, 4),
        "addr_tok_overlap": addr_tok_overlap,
        "addr_c3_jaccard": round(addr_c3_jaccard, 4),
        "addr_len_ratio": round(addr_len_ratio, 4),
        "postal_match": postal_match,
        "postal_missing": postal_missing,
        "house_num_match": house_num_match,
        "digit_jaccard": round(digit_jaccard, 4),
        "digit_overlap": digit_overlap,
        "cand_addr_missing": cand_addr_missing,
        "name_x_addr": round(name_x_addr, 4),
        "max_similarity": round(max_similarity, 4),
        "both_strong": both_strong,
        "exact_name_partial_addr": exact_name_partial_addr,
        "partial_name_exact_addr": partial_name_exact_addr,
        "both_exact": both_exact,
        "blockers_fired_count": float(blockers_fired),
        "cand_rank": float(cand_rank),
        "source_is_s2": source_is_s2,
        "cntry_us": cntry_us,
        "cntry_in": cntry_in,
        "cntry_fr": cntry_fr
    }
