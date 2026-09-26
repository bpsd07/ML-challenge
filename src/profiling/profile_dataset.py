"""
Reproducible Dataset Profiling Script for Amazon Business Entity Resolution Challenge.
Profiles all train and test TSV files, analyzes noise patterns, ground truth matches,
distributions, and cross-dataset shifts.
Outputs reports/dataset_profile.json and reports/dataset_profile.md.
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from collections import Counter
import polars as pl
import numpy as np

# Ensure UTF-8 output handling
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "student_resource" / "dataset"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train_source1": DATA_DIR / "train" / "train_source1.tsv",
    "train_source2": DATA_DIR / "train" / "train_source2.tsv",
    "train_source3": DATA_DIR / "train" / "train_source3.tsv",
    "train_ground_truth": DATA_DIR / "train" / "train_ground_truth.tsv",
    "test_source1": DATA_DIR / "test" / "test_source1.tsv",
    "test_source2": DATA_DIR / "test" / "test_source2.tsv",
    "test_source3": DATA_DIR / "test" / "test_source3.tsv",
}


def compute_distribution_stats(series: pl.Series) -> dict:
    """Compute comprehensive summary statistics for a numeric series."""
    valid = series.drop_nulls()
    if len(valid) == 0:
        return {}
    vals = valid.to_numpy()
    return {
        "count": int(len(vals)),
        "min": float(np.min(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p25": float(np.percentile(vals, 25)),
        "median": float(np.median(vals)),
        "mean": round(float(np.mean(vals)), 2),
        "std": round(float(np.std(vals)), 2),
        "p75": float(np.percentile(vals, 75)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "max": float(np.max(vals)),
    }


def profile_source_file(file_path: Path, name: str) -> dict:
    """Profile a single source TSV file."""
    print(f"Profiling {name} from {file_path.name}...")
    t0 = time.time()
    file_size_bytes = file_path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    # Read TSV using polars
    df = pl.read_csv(
        file_path,
        separator="\t",
        truncate_ragged_lines=True,
        infer_schema_length=10000,
        null_values=["", "NULL", "null", "None", "NaN", "nan"]
    )

    n_rows = len(df)
    cols = df.columns
    schema = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}

    # Missing values
    missing_counts = {}
    missing_pcts = {}
    for col in cols:
        n_missing = int(df[col].is_null().sum())
        missing_counts[col] = n_missing
        missing_pcts[col] = round((n_missing / n_rows) * 100, 3) if n_rows > 0 else 0.0

    # Duplicates
    # Exact full row duplicates
    n_unique_rows = df.n_unique()
    duplicate_rows = n_rows - n_unique_rows
    duplicate_rows_pct = round((duplicate_rows / n_rows) * 100, 3) if n_rows > 0 else 0.0

    # Business name duplicates
    n_unique_names = int(df["business_name"].drop_nulls().n_unique()) if "business_name" in cols else 0
    dup_names_count = (n_rows - missing_counts.get("business_name", 0)) - n_unique_names
    dup_names_pct = round((dup_names_count / max(1, n_rows - missing_counts.get("business_name", 0))) * 100, 3)

    # Business address duplicates
    n_unique_addrs = int(df["business_address"].drop_nulls().n_unique()) if "business_address" in cols else 0
    dup_addrs_count = (n_rows - missing_counts.get("business_address", 0)) - n_unique_addrs
    dup_addrs_pct = round((dup_addrs_count / max(1, n_rows - missing_counts.get("business_address", 0))) * 100, 3)

    # Country distribution
    country_dist = {}
    if "country" in cols:
        vc = df["country"].fill_null("MISSING").value_counts()
        for row in vc.iter_rows():
            c_name, c_cnt = row[0], row[1]
            country_dist[str(c_name)] = {
                "count": int(c_cnt),
                "pct": round((int(c_cnt) / n_rows) * 100, 2)
            }

    # Top duplicate names
    top_names = []
    if "business_name" in cols:
        top_names_df = (
            df["business_name"]
            .drop_nulls()
            .value_counts()
            .sort("count", descending=True)
            .head(10)
        )
        for row in top_names_df.iter_rows():
            top_names.append({"name": str(row[0]), "count": int(row[1])})

    # Top duplicate addresses
    top_addrs = []
    if "business_address" in cols:
        top_addrs_df = (
            df["business_address"]
            .drop_nulls()
            .value_counts()
            .sort("count", descending=True)
            .head(10)
        )
        for row in top_addrs_df.iter_rows():
            top_addrs.append({"address": str(row[0]), "count": int(row[1])})

    # Length distributions
    name_char_lens = df["business_name"].drop_nulls().str.len_chars() if "business_name" in cols else pl.Series()
    name_word_lens = df["business_name"].drop_nulls().str.split(" ").list.len() if "business_name" in cols else pl.Series()

    addr_char_lens = df["business_address"].drop_nulls().str.len_chars() if "business_address" in cols else pl.Series()
    addr_word_lens = df["business_address"].drop_nulls().str.split(" ").list.len() if "business_address" in cols else pl.Series()

    name_char_stats = compute_distribution_stats(name_char_lens)
    name_word_stats = compute_distribution_stats(name_word_lens)
    addr_char_stats = compute_distribution_stats(addr_char_lens)
    addr_word_stats = compute_distribution_stats(addr_word_lens)

    elapsed = round(time.time() - t0, 2)
    print(f"  Done in {elapsed}s: {n_rows} rows, {n_unique_rows} unique.")

    return {
        "file_name": file_path.name,
        "file_size_mb": file_size_mb,
        "row_count": n_rows,
        "columns": cols,
        "schema": schema,
        "missing_counts": missing_counts,
        "missing_pcts": missing_pcts,
        "duplicate_rows": duplicate_rows,
        "duplicate_rows_pct": duplicate_rows_pct,
        "unique_names": n_unique_names,
        "duplicate_names_count": dup_names_count,
        "duplicate_names_pct": dup_names_pct,
        "unique_addresses": n_unique_addrs,
        "duplicate_addresses_count": dup_addrs_count,
        "duplicate_addresses_pct": dup_addrs_pct,
        "country_distribution": country_dist,
        "top_names": top_names,
        "top_addresses": top_addrs,
        "name_char_length_stats": name_char_stats,
        "name_word_length_stats": name_word_stats,
        "address_char_length_stats": addr_char_stats,
        "address_word_length_stats": addr_word_stats,
        "profile_time_sec": elapsed
    }


def profile_ground_truth(file_path: Path, train_s1_ids: set, train_s2_ids: set, train_s3_ids: set) -> dict:
    """Profile ground truth matching relationships."""
    print("Profiling train_ground_truth.tsv...")
    t0 = time.time()
    file_size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)

    df_gt = pl.read_csv(
        file_path,
        separator="\t",
        truncate_ragged_lines=True,
        null_values=["NULL", "null"]
    )

    n_rows = len(df_gt)
    cols = df_gt.columns
    s1_col = "source1_entity_id"
    match_col = "matched_entity_ids"

    s1_ids_in_gt = set(df_gt[s1_col].to_list())
    missing_from_gt = len(train_s1_ids - s1_ids_in_gt)
    extra_in_gt = len(s1_ids_in_gt - train_s1_ids)

    # Process match strings
    match_counts = []
    s2_counts = []
    s3_counts = []
    both_counts = 0
    s2_only_counts = 0
    s3_only_counts = 0
    neither_counts = 0  # singletons

    invalid_s2_ids = 0
    invalid_s3_ids = 0
    other_invalid_ids = 0

    total_s2_links = 0
    total_s3_links = 0

    # Iterating through matched strings
    for m_str in df_gt[match_col].to_list():
        if m_str is None or str(m_str).strip() == "" or str(m_str).strip() == "nan":
            match_counts.append(0)
            s2_counts.append(0)
            s3_counts.append(0)
            neither_counts += 1
            continue

        raw_ids = [x.strip() for x in str(m_str).split(",") if x.strip()]
        n_m = len(raw_ids)
        match_counts.append(n_m)

        n_s2 = 0
        n_s3 = 0
        for m_id in raw_ids:
            if m_id.startswith("S2-"):
                n_s2 += 1
                total_s2_links += 1
                if m_id not in train_s2_ids:
                    invalid_s2_ids += 1
            elif m_id.startswith("S3-"):
                n_s3 += 1
                total_s3_links += 1
                if m_id not in train_s3_ids:
                    invalid_s3_ids += 1
            else:
                other_invalid_ids += 1

        s2_counts.append(n_s2)
        s3_counts.append(n_s3)

        if n_s2 > 0 and n_s3 > 0:
            both_counts += 1
        elif n_s2 > 0:
            s2_only_counts += 1
        elif n_s3 > 0:
            s3_only_counts += 1
        else:
            neither_counts += 1

    match_counts_series = pl.Series("matches", match_counts)
    s2_series = pl.Series("s2_matches", s2_counts)
    s3_series = pl.Series("s3_matches", s3_counts)

    # Histogram of match counts
    hist_counts = Counter(match_counts)
    match_distribution = {
        "0 (singleton)": {"count": hist_counts[0], "pct": round(hist_counts[0] / n_rows * 100, 2)},
        "1": {"count": hist_counts[1], "pct": round(hist_counts[1] / n_rows * 100, 2)},
        "2": {"count": hist_counts[2], "pct": round(hist_counts[2] / n_rows * 100, 2)},
        "3": {"count": hist_counts[3], "pct": round(hist_counts[3] / n_rows * 100, 2)},
        "4": {"count": hist_counts[4], "pct": round(hist_counts[4] / n_rows * 100, 2)},
        "5+": {"count": sum(c for k, c in hist_counts.items() if k >= 5),
               "pct": round(sum(c for k, c in hist_counts.items() if k >= 5) / n_rows * 100, 2)}
    }

    elapsed = round(time.time() - t0, 2)
    print(f"  Done in {elapsed}s: {neither_counts} singletons ({neither_counts/n_rows*100:.2f}%).")

    return {
        "file_name": file_path.name,
        "file_size_mb": file_size_mb,
        "row_count": n_rows,
        "columns": cols,
        "s1_in_gt_count": len(s1_ids_in_gt),
        "missing_s1_from_gt": missing_from_gt,
        "extra_s1_in_gt": extra_in_gt,
        "singleton_count": neither_counts,
        "singleton_pct": round(neither_counts / n_rows * 100, 2),
        "total_matched_s1_count": n_rows - neither_counts,
        "total_matched_s1_pct": round((n_rows - neither_counts) / n_rows * 100, 2),
        "match_count_distribution": match_distribution,
        "match_count_stats": compute_distribution_stats(match_counts_series),
        "s2_match_stats": compute_distribution_stats(s2_series),
        "s3_match_stats": compute_distribution_stats(s3_series),
        "total_s2_links": total_s2_links,
        "total_s3_links": total_s3_links,
        "s2_vs_s3_match_breakdown": {
            "s2_only": {"count": s2_only_counts, "pct": round(s2_only_counts / n_rows * 100, 2)},
            "s3_only": {"count": s3_only_counts, "pct": round(s3_only_counts / n_rows * 100, 2)},
            "both_s2_and_s3": {"count": both_counts, "pct": round(both_counts / n_rows * 100, 2)},
            "neither_singletons": {"count": neither_counts, "pct": round(neither_counts / n_rows * 100, 2)}
        },
        "target_id_validity": {
            "invalid_s2_ids": invalid_s2_ids,
            "invalid_s3_ids": invalid_s3_ids,
            "other_invalid_ids": other_invalid_ids
        },
        "profile_time_sec": elapsed
    }


def analyze_noise_and_match_patterns(df_s1, df_s2, df_s3, df_gt, n_samples=50000):
    """
    Mine grounded examples of noise patterns and matching variations across true matches.
    """
    print("Mining noise patterns from ground truth pairs...")
    t0 = time.time()

    # Index s1, s2, s3 by entity_id
    s1_dict = {
        row[0]: (row[1] or "", row[2] or "", row[3] or "")
        for row in df_s1.select(["entity_id", "business_name", "business_address", "country"]).iter_rows()
    }
    s2_dict = {
        row[0]: (row[1] or "", row[2] or "", row[3] or "")
        for row in df_s2.select(["entity_id", "business_name", "business_address", "country"]).iter_rows()
    }
    s3_dict = {
        row[0]: (row[1] or "", row[2] or "", row[3] or "")
        for row in df_s3.select(["entity_id", "business_name", "business_address", "country"]).iter_rows()
    }

    # Extract non-singleton pairs
    pairs = []
    for row in df_gt.iter_rows():
        s1_id = row[0]
        m_str = row[1]
        if m_str and str(m_str).strip() and str(m_str).strip() != "nan":
            for m_id in str(m_str).split(","):
                m_id = m_id.strip()
                if m_id:
                    pairs.append((s1_id, m_id))

    print(f"  Total ground truth pairs available: {len(pairs)}. Sampling {min(len(pairs), n_samples)} for deep noise analysis...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(pairs), size=min(len(pairs), n_samples), replace=False)

    exact_matches = []
    abbreviation_variations = []
    typo_variations = []
    reordered_names = []
    legal_suffix_variations = []
    address_variations = []
    transliteration_examples = []
    missing_address_components = []

    legal_suffixes = [
        "LLC", "INC", "CORP", "CORPORATION", "LTD", "LIMITED", "PVT", "PRIVATE",
        "CO", "COMPANY", "LLP", "GMBH", "SA", "SAS", "SARL", "PLC"
    ]
    legal_suffix_regex = re.compile(r'\b(' + '|'.join(legal_suffixes) + r')\b', re.IGNORECASE)

    common_abbrevs = {
        "CORP": "CORPORATION", "INC": "INCORPORATED", "PVT": "PRIVATE", "LTD": "LIMITED",
        "CO": "COMPANY", "ST": "STREET", "RD": "ROAD", "AVE": "AVENUE", "BLVD": "BOULEVARD",
        "DR": "DRIVE", "HWY": "HIGHWAY", "SQ": "SQUARE", "PL": "PLACE", "CT": "COURT",
        "STE": "SUITE", "APT": "APARTMENT", "FL": "FLOOR", "CTR": "CENTER", "INTL": "INTERNATIONAL",
        "NATL": "NATIONAL", "DEPT": "DEPARTMENT", "ASSN": "ASSOCIATION", "SVCS": "SERVICES"
    }

    for idx in sample_indices:
        s1_id, m_id = pairs[idx]
        if s1_id not in s1_dict:
            continue
        target_dict = s2_dict if m_id.startswith("S2-") else s3_dict
        if m_id not in target_dict:
            continue

        s1_name, s1_addr, s1_cntry = s1_dict[s1_id]
        m_name, m_addr, m_cntry = target_dict[m_id]

        s1_n_clean = s1_name.strip()
        m_n_clean = m_name.strip()
        s1_a_clean = s1_addr.strip()
        m_a_clean = m_addr.strip()

        # 1. Exact matches
        if s1_n_clean.lower() == m_n_clean.lower() and s1_a_clean.lower() == m_a_clean.lower():
            if len(exact_matches) < 5:
                exact_matches.append({
                    "s1_id": s1_id, "matched_id": m_id, "country": s1_cntry,
                    "name": s1_name, "address": s1_addr
                })

        # 2. Reordered names
        s1_tokens = sorted(re.findall(r'\w+', s1_n_clean.lower()))
        m_tokens = sorted(re.findall(r'\w+', m_n_clean.lower()))
        if s1_tokens == m_tokens and s1_n_clean.lower() != m_n_clean.lower() and len(s1_tokens) > 1:
            if len(reordered_names) < 5:
                reordered_names.append({
                    "s1_id": s1_id, "matched_id": m_id, "country": s1_cntry,
                    "s1_name": s1_name, "matched_name": m_name,
                    "s1_address": s1_addr, "matched_address": m_addr
                })

        # 3. Legal suffix variations
        s1_suffixes = set(legal_suffix_regex.findall(s1_n_clean.upper()))
        m_suffixes = set(legal_suffix_regex.findall(m_n_clean.upper()))
        s1_base = legal_suffix_regex.sub("", s1_n_clean.upper()).strip()
        m_base = legal_suffix_regex.sub("", m_n_clean.upper()).strip()
        if s1_suffixes != m_suffixes and s1_base == m_base and len(s1_base) > 3:
            if len(legal_suffix_variations) < 5:
                legal_suffix_variations.append({
                    "s1_id": s1_id, "matched_id": m_id, "country": s1_cntry,
                    "s1_name": s1_name, "matched_name": m_name,
                    "s1_suffixes": list(s1_suffixes), "matched_suffixes": list(m_suffixes)
                })

        # 4. Abbreviation variations (Name or Address)
        has_abbrev = False
        s1_all = f"{s1_n_clean} {s1_a_clean}".upper()
        m_all = f"{m_n_clean} {m_a_clean}".upper()
        for ab, full in common_abbrevs.items():
            if (re.search(r'\b' + ab + r'\b', s1_all) and re.search(r'\b' + full + r'\b', m_all)) or \
               (re.search(r'\b' + full + r'\b', s1_all) and re.search(r'\b' + ab + r'\b', m_all)):
                has_abbrev = True
                if len(abbreviation_variations) < 5:
                    abbreviation_variations.append({
                        "s1_id": s1_id, "matched_id": m_id, "country": s1_cntry,
                        "abbrev_found": f"{ab} <-> {full}",
                        "s1_name": s1_name, "matched_name": m_name,
                        "s1_address": s1_addr, "matched_address": m_addr
                    })
                break

        # 5. Typo variations (Levenshtein-like / small edit distance on names)
        # Simple length check and char differences
        if abs(len(s1_n_clean) - len(m_n_clean)) <= 2 and s1_n_clean.lower() != m_n_clean.lower():
            # Check character multiset diff
            c1 = Counter(s1_n_clean.lower())
            c2 = Counter(m_n_clean.lower())
            diff = sum((c1 - c2).values()) + sum((c2 - c1).values())
            if 1 <= diff <= 3 and len(s1_n_clean) > 6 and s1_base != m_base:
                if len(typo_variations) < 5:
                    typo_variations.append({
                        "s1_id": s1_id, "matched_id": m_id, "country": s1_cntry,
                        "s1_name": s1_name, "matched_name": m_name,
                        "diff_count": diff
                    })

        # 6. Address variations & Missing address components
        if s1_a_clean and m_a_clean and s1_a_clean.lower() != m_a_clean.lower():
            # Check if one is a substring or much shorter (missing components)
            if (s1_a_clean.lower() in m_a_clean.lower() or m_a_clean.lower() in s1_a_clean.lower()) and abs(len(s1_a_clean) - len(m_a_clean)) > 10:
                if len(missing_address_components) < 5:
                    missing_address_components.append({
                        "s1_id": s1_id, "matched_id": m_id, "country": s1_cntry,
                        "s1_address": s1_addr, "matched_address": m_addr,
                        "ratio": round(min(len(s1_a_clean), len(m_a_clean)) / max(len(s1_a_clean), len(m_a_clean)), 2)
                    })
            else:
                if len(address_variations) < 5:
                    address_variations.append({
                        "s1_id": s1_id, "matched_id": m_id, "country": s1_cntry,
                        "s1_address": s1_addr, "matched_address": m_addr
                    })

        # 7. Transliteration / Indian Phonetic variations
        if s1_cntry == "India":
            # Look for common Indian transliteration patterns: ee/i, oo/u, v/w, sh/s, th/t
            cand_patterns = [("ee", "i"), ("oo", "u"), ("v", "w"), ("sh", "s"), ("c", "k")]
            for p1, p2 in cand_patterns:
                if (p1 in s1_n_clean.lower() and p2 in m_n_clean.lower()) or (p2 in s1_n_clean.lower() and p1 in m_n_clean.lower()):
                    if len(transliteration_examples) < 5 and s1_n_clean.lower() != m_n_clean.lower():
                        transliteration_examples.append({
                            "s1_id": s1_id, "matched_id": m_id, "country": s1_cntry,
                            "pattern": f"{p1} <-> {p2}",
                            "s1_name": s1_name, "matched_name": m_name,
                            "s1_address": s1_addr, "matched_address": m_addr
                        })
                        break

    # 8. Ambiguous businesses (same or very similar name, distinct entity in S1)
    ambiguous_examples = []
    print("Finding potentially ambiguous businesses (homonyms / franchises)...")
    name_to_ids = {}
    for row in df_s1.select(["entity_id", "business_name", "business_address", "country"]).iter_rows():
        n = str(row[1] or "").strip().lower()
        if len(n) > 5:
            if n not in name_to_ids:
                name_to_ids[n] = []
            if len(name_to_ids[n]) < 5:
                name_to_ids[n].append({"entity_id": row[0], "address": row[2], "country": row[3]})

    for n, records in name_to_ids.items():
        if len(records) >= 3:
            # Check if they have different addresses
            addrs = set(r["address"] for r in records if r["address"])
            if len(addrs) >= 3:
                ambiguous_examples.append({
                    "business_name": n,
                    "instance_count": len(records),
                    "instances": records[:3]
                })
                if len(ambiguous_examples) >= 5:
                    break

    elapsed = round(time.time() - t0, 2)
    print(f"  Noise analysis completed in {elapsed}s.")

    return {
        "exact_matches": exact_matches,
        "abbreviation_variations": abbreviation_variations,
        "typo_variations": typo_variations,
        "reordered_names": reordered_names,
        "legal_suffix_variations": legal_suffix_variations,
        "address_variations": address_variations,
        "transliteration_examples": transliteration_examples,
        "missing_address_components": missing_address_components,
        "ambiguous_businesses": ambiguous_examples
    }


def analyze_train_test_shifts(profiles: dict) -> dict:
    """Compare train vs test data distributions and check overlaps."""
    print("Analyzing train vs test shifts...")
    t0 = time.time()

    train_sources = ["train_source1", "train_source2", "train_source3"]
    test_sources = ["test_source1", "test_source2", "test_source3"]

    total_train_rows = sum(profiles[s]["row_count"] for s in train_sources)
    total_test_rows = sum(profiles[s]["row_count"] for s in test_sources)

    train_countries = {}
    for s in train_sources:
        for c, data in profiles[s]["country_distribution"].items():
            train_countries[c] = train_countries.get(c, 0) + data["count"]

    test_countries = {}
    for s in test_sources:
        for c, data in profiles[s]["country_distribution"].items():
            test_countries[c] = test_countries.get(c, 0) + data["count"]

    train_c_pct = {c: round(cnt / total_train_rows * 100, 2) for c, cnt in train_countries.items()}
    test_c_pct = {c: round(cnt / total_test_rows * 100, 2) for c, cnt in test_countries.items()}

    # Check for France presence
    france_in_train = train_countries.get("France", 0)
    france_in_test = test_countries.get("France", 0)

    # Entity ID overlap check
    print("Checking entity ID overlap...")
    train_ids = set()
    for s in train_sources:
        df = pl.read_csv(FILES[s], separator="\t", columns=["entity_id"])
        train_ids.update(df["entity_id"].to_list())

    test_ids = set()
    for s in test_sources:
        df = pl.read_csv(FILES[s], separator="\t", columns=["entity_id"])
        test_ids.update(df["entity_id"].to_list())

    id_overlap = len(train_ids.intersection(test_ids))

    # Name overlap between train and test
    print("Checking business name overlap...")
    train_names = set()
    for s in train_sources:
        df = pl.read_csv(FILES[s], separator="\t", columns=["business_name"])
        train_names.update(df["business_name"].drop_nulls().str.to_lowercase().to_list())

    test_names = set()
    for s in test_sources:
        df = pl.read_csv(FILES[s], separator="\t", columns=["business_name"])
        test_names.update(df["business_name"].drop_nulls().str.to_lowercase().to_list())

    name_overlap_count = len(train_names.intersection(test_names))
    name_overlap_pct = round(name_overlap_count / max(1, len(test_names)) * 100, 2)

    elapsed = round(time.time() - t0, 2)
    print(f"  Done in {elapsed}s: Train rows: {total_train_rows}, Test rows: {total_test_rows}, France in test: {france_in_test}.")

    return {
        "total_train_records": total_train_rows,
        "total_test_records": total_test_rows,
        "train_country_counts": train_countries,
        "train_country_pct": train_c_pct,
        "test_country_counts": test_countries,
        "test_country_pct": test_c_pct,
        "france_in_train": france_in_train,
        "france_in_test": france_in_test,
        "france_test_pct": round(france_in_test / total_test_rows * 100, 2),
        "entity_id_overlap_count": id_overlap,
        "unique_train_names": len(train_names),
        "unique_test_names": len(test_names),
        "name_overlap_count": name_overlap_count,
        "name_overlap_pct_of_test": name_overlap_pct,
        "analysis_time_sec": elapsed
    }


def generate_markdown_report(profile_data: dict) -> str:
    """Generate comprehensive markdown profile report."""
    md = []
    md.append("# Amazon Business Entity Resolution Challenge — Comprehensive Dataset Profile & EDA Report\n")
    md.append("**Generated by:** Lead ML Engineer  ")
    md.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append("**Evaluation Metric:** Macro-averaged F_0.5 Score (Precision-weighted 2× over Recall, Singletons scored)  ")
    md.append("**Competition Constraints:** Closed dataset only, No external lookup/APIs/geocoding, Model <= 8B parameters, License MIT/Apache 2.0\n")
    md.append("---\n")

    # 1. Executive Summary Table
    md.append("## 1. Executive Dataset Overview\n")
    md.append("| Dataset File | Role | File Size (MB) | Row Count | Columns | Unique Records | Missing Name % | Missing Addr % |\n")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

    sources_order = [
        ("train_source1", "Train Ref (S1)"),
        ("train_source2", "Train Cand (S2)"),
        ("train_source3", "Train Cand (S3)"),
        ("train_ground_truth", "Train Labels (GT)"),
        ("test_source1", "Test Ref (S1)"),
        ("test_source2", "Test Cand (S2)"),
        ("test_source3", "Test Cand (S3)"),
    ]

    for key, role in sources_order:
        p = profile_data["files"][key]
        size = p["file_size_mb"]
        rows = f"{p['row_count']:,}"
        cols = len(p["columns"])
        unq = f"{p.get('row_count', 0) - p.get('duplicate_rows', 0):,}" if "duplicate_rows" in p else "N/A"
        m_name = f"{p.get('missing_pcts', {}).get('business_name', 0):.2f}%" if "business_name" in p.get("columns", []) else "-"
        m_addr = f"{p.get('missing_pcts', {}).get('business_address', 0):.2f}%" if "business_address" in p.get("columns", []) else "-"
        md.append(f"| `{p['file_name']}` | {role} | {size} MB | {rows} | {cols} | {unq} | {m_name} | {m_addr} |\n")

    # 2. Ground Truth Analysis & Singleton Dominance
    gt = profile_data["ground_truth"]
    md.append("\n---\n")
    md.append("## 2. Ground Truth & Matching Relationship Analysis\n")
    md.append("In the evaluation metric, macro-averaged F_0.5 is computed across all S1 entities, where **singletons (S1 entities with 0 matches) score 1.0 if predicted empty and 0.0 if any match is predicted**.\n")

    md.append(f"- **Total S1 Reference Entities in Train Ground Truth:** {gt['row_count']:,}\n")
    md.append(f"- **Singletons (Zero matches in S2/S3):** **{gt['singleton_count']:,} ({gt['singleton_pct']}%)**\n")
    md.append(f"- **Entities with >=1 Match:** **{gt['total_matched_s1_count']:,} ({gt['total_matched_s1_pct']}%)**\n")
    md.append(f"- **Total S2 Links:** {gt['total_s2_links']:,} | **Total S3 Links:** {gt['total_s3_links']:,}\n")
    md.append(f"- **Target ID Validity:** All matched S2 IDs exist: {gt['target_id_validity']['invalid_s2_ids'] == 0} | All matched S3 IDs exist: {gt['target_id_validity']['invalid_s3_ids'] == 0}\n\n")

    md.append("### Ground Truth Match-Count Distribution\n")
    md.append("| Match Count per S1 | Count | Percentage |\n")
    md.append("| :--- | :---: | :---: |\n")
    for k, v in gt["match_count_distribution"].items():
        md.append(f"| {k} | {v['count']:,} | {v['pct']}% |\n")

    md.append("\n### S2 vs. S3 Match Source Breakdown\n")
    md.append("| Match Category | S1 Count | Percentage of All S1 |\n")
    md.append("| :--- | :---: | :---: |\n")
    for k, v in gt["s2_vs_s3_match_breakdown"].items():
        md.append(f"| `{k}` | {v['count']:,} | {v['pct']}% |\n")

    md.append("\n### Match Statistics per S1 Entity\n")
    md.append("| Metric | Total Matches | S2 Matches | S3 Matches |\n")
    md.append("| :--- | :---: | :---: | :---: |\n")
    m_stat = gt["match_count_stats"]
    s2_stat = gt["s2_match_stats"]
    s3_stat = gt["s3_match_stats"]
    for field in ["mean", "std", "median", "p75", "p90", "p95", "p99", "max"]:
        md.append(f"| {field.upper()} | {m_stat.get(field, '-')} | {s2_stat.get(field, '-')} | {s3_stat.get(field, '-')} |\n")

    # 3. Country Distributions & The Critical Zero-Shot Challenge (France)
    shifts = profile_data["shifts"]
    md.append("\n---\n")
    md.append("## 3. Geographic Distribution & The Unseen Country (France)\n")
    md.append("> [!IMPORTANT]\n")
    md.append("> **Zero-Shot Country Generalization:** The training set contains only `US` and `India`. However, the test set contains `France`, which never appears in training! Any country-specific rules, hardcoding, or frequency encoding based on training countries will fail on France.\n\n")

    md.append("### Country Breakdown: Training vs. Testing\n")
    md.append("| Country | Train Count | Train % | Test Count | Test % | Shift / Note |\n")
    md.append("| :--- | :---: | :---: | :---: | :---: | :--- |\n")
    all_countries = sorted(list(set(shifts["train_country_counts"].keys()) | set(shifts["test_country_counts"].keys())))
    for c in all_countries:
        tr_cnt = shifts["train_country_counts"].get(c, 0)
        tr_pct = shifts["train_country_pct"].get(c, 0.0)
        te_cnt = shifts["test_country_counts"].get(c, 0)
        te_pct = shifts["test_country_pct"].get(c, 0.0)
        note = "Unseen in Train (Zero-shot evaluation)" if tr_cnt == 0 else "In Train & Test"
        md.append(f"| **{c}** | {tr_cnt:,} | {tr_pct}% | {te_cnt:,} | {te_pct}% | {note} |\n")

    md.append(f"\n- **Total France records in Test:** **{shifts['france_in_test']:,} ({shifts['france_test_pct']}%)** across test sources.\n")
    md.append(f"- **Entity ID Overlap between Train and Test:** **{shifts['entity_id_overlap_count']}** (Strict disjoint split).\n")
    md.append(f"- **Business Name Overlap between Train and Test:** {shifts['name_overlap_count']:,} names ({shifts['name_overlap_pct_of_test']}% of test names appear in train).\n")

    # 4. Text Length Distributions & Structural Profiles
    md.append("\n---\n")
    md.append("## 4. Text Length Distributions (Characters & Words)\n")
    md.append("Profiling the length of business names and addresses informs maximum sequence length selection for text encoders and token n-gram limits for blocking.\n\n")

    md.append("### Business Name Length (Characters)\n")
    md.append("| Source File | Min | P25 | Median | Mean | P75 | P90 | P99 | Max |\n")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for key, _ in sources_order:
        if "name_char_length_stats" in profile_data["files"][key]:
            st = profile_data["files"][key]["name_char_length_stats"]
            md.append(f"| `{key}` | {st.get('min')} | {st.get('p25')} | {st.get('median')} | {st.get('mean')} | {st.get('p75')} | {st.get('p90')} | {st.get('p99')} | {st.get('max')} |\n")

    md.append("\n### Business Address Length (Characters)\n")
    md.append("| Source File | Min | P25 | Median | Mean | P75 | P90 | P99 | Max |\n")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for key, _ in sources_order:
        if "address_char_length_stats" in profile_data["files"][key]:
            st = profile_data["files"][key]["address_char_length_stats"]
            md.append(f"| `{key}` | {st.get('min')} | {st.get('p25')} | {st.get('median')} | {st.get('mean')} | {st.get('p75')} | {st.get('p90')} | {st.get('p99')} | {st.get('max')} |\n")

    # 5. Noise Pattern Taxonomy & Concrete Ground-Truth Evidence
    np_data = profile_data["noise_patterns"]
    md.append("\n---\n")
    md.append("## 5. Noise Pattern Taxonomy & Empirical Ground-Truth Evidence\n")
    md.append("Analysis of ground truth matches reveals several dominant noise modes:\n\n")

    md.append("### 5.1 Exact Matches\n")
    md.append("Pairs where both business name and address match identically (case-insensitive):\n")
    for ex in np_data["exact_matches"][:3]:
        md.append(f"- **S1:** `{ex['s1_id']}` | **Match:** `{ex['matched_id']}` | **Country:** {ex['country']}\n")
        md.append(f"  - **Name:** \"{ex['name']}\"\n")
        md.append(f"  - **Address:** \"{ex['address']}\"\n")

    md.append("\n### 5.2 Abbreviation Variations\n")
    md.append("Standard business abbreviations vs. full expansions (e.g., Corp vs Corporation, St vs Street, Rd vs Road):\n")
    for ex in np_data["abbreviation_variations"][:3]:
        md.append(f"- **Variation:** `{ex['abbrev_found']}` ({ex['country']})\n")
        md.append(f"  - **S1:** \"{ex['s1_name']}\" — \"{ex['s1_address']}\"\n")
        md.append(f"  - **Match:** \"{ex['matched_name']}\" — \"{ex['matched_address']}\"\n")

    md.append("\n### 5.3 Legal Suffix Variations\n")
    md.append("Entities referring to the exact same organization with differing legal structures appended:\n")
    for ex in np_data["legal_suffix_variations"][:3]:
        md.append(f"- **S1:** \"{ex['s1_name']}\" vs **Match:** \"{ex['matched_name']}\" ({ex['country']})\n")
        md.append(f"  - Suffixes: `{ex['s1_suffixes']}` vs `{ex['matched_suffixes']}`\n")

    md.append("\n### 5.4 Reordered Words / Inverted Names\n")
    md.append("Tokens present in different order:\n")
    for ex in np_data["reordered_names"][:3]:
        md.append(f"- **S1:** \"{ex['s1_name']}\"\n")
        md.append(f"- **Match:** \"{ex['matched_name']}\" ({ex['country']})\n")

    md.append("\n### 5.5 Typo & Spelling Inconsistencies\n")
    md.append("Minor character deletions, insertions, or substitutions:\n")
    for ex in np_data["typo_variations"][:3]:
        md.append(f"- **S1:** \"{ex['s1_name']}\" vs **Match:** \"{ex['matched_name']}\" (Diff: {ex['diff_count']} chars, {ex['country']})\n")

    md.append("\n### 5.6 Address Variations & Missing Address Components\n")
    md.append("Address truncation, missing PIN code/state/city, or landmark additions:\n")
    for ex in np_data["missing_address_components"][:3]:
        md.append(f"- **S1 Addr:** \"{ex['s1_address']}\"\n")
        md.append(f"- **Match Addr:** \"{ex['matched_address']}\" (Length ratio: {ex['ratio']}, {ex['country']})\n")

    md.append("\n### 5.7 Indian Transliteration & Phonetic Variants\n")
    md.append("Phonetic vowel and consonant variations in Indian business names and addresses (ee/i, oo/u, v/w):\n")
    for ex in np_data["transliteration_examples"][:3]:
        md.append(f"- **Pattern:** `{ex['pattern']}` | **S1:** \"{ex['s1_name']}\" vs **Match:** \"{ex['matched_name']}\"\n")
        md.append(f"  - **S1 Addr:** \"{ex['s1_address']}\" | **Match Addr:** \"{ex['matched_address']}\"\n")

    md.append("\n### 5.8 High Ambiguity & Franchise Collision Hazards\n")
    md.append("Businesses with widespread branch networks or shared identical names across different locations:\n")
    for ex in np_data["ambiguous_businesses"][:3]:
        md.append(f"- **Ambiguous Name:** \"{ex['business_name']}\" ({ex['instance_count']} occurrences in S1 alone)\n")
        for inst in ex["instances"]:
            md.append(f"  - ID: `{inst['entity_id']}` | Country: {inst['country']} | Addr: \"{inst['address']}\"\n")

    # 6. Strategic Recommendations
    md.append("\n---\n")
    md.append("## 6. Strategic Engineering Recommendations\n")
    md.append("### 6.1 Recommended Candidate-Generation (Blocking) Strategy\n")
    md.append("Because comparing all S1 against all S2/S3 is computationally intractable (~1.8M S1 × 9M S2/S3 ≈ 1.6 × 10^13 pairs), multi-stage blocking is strictly required:\n")
    md.append("1. **Country Partitioning:** Partition matching strictly within country (`S1.country == S2/S3.country`). Cross-country matches in business resolution are statistically negligible and pruning cross-country pairs immediately eliminates >50% of the candidate space.\n")
    md.append("2. **Multi-Index Inverted Index Blocking:**\n")
    md.append("   - **Normalized Name Prefix & Token Blocking:** Strip legal suffixes, punctuation, and lowercase. Index by first 2 tokens and 3-char prefix.\n")
    md.append("   - **Locality / Postal Code Blocking:** In India and US, postal codes (PIN / ZIP) or city tokens form tight clusters.\n")
    md.append("   - **Phonetic / Double Metaphone Blocking:** Essential for Indian transliterations.\n")
    md.append("   - **MinHash LSH / TF-IDF Cosine Blocking:** Top-K cosine similarity on character 3-grams over business name + address.\n")
    md.append("3. **Recall Ceiling Monitoring:** Validate blocking recall on a held-out split; blocking must achieve >= 95% recall ceiling while maintaining reduction ratio > 99.9%.\n\n")

    md.append("### 6.2 Recommended Validation Strategy\n")
    md.append("1. **Leak-Free Stratified Group Splitting:**\n")
    md.append("   - Split Source 1 entities (e.g., 80% train / 20% validation) stratified by country and singleton status.\n")
    md.append("   - Ground truth pairs for the 20% validation S1 entities must be strictly held out during candidate generation index building and model training.\n")
    md.append("2. **Exact Competition Metric Implementation:**\n")
    md.append("   - Implement macro-averaged F_0.5 where singletons correctly predicted as empty score 1.0, and singletons with any false merge score 0.0.\n")
    md.append("3. **Simulating the France Shift (Zero-Shot Validation):**\n")
    md.append("   - To evaluate how well candidate generation and matching will perform on unseen `France`, create a validation fold holding out one country or distinct regional partitions to measure out-of-distribution drop.\n")
    md.append("4. **Precision-Weighted Threshold Tuning:**\n")
    md.append("   - F_0.5 heavily penalizes false merges (beta=0.5 -> precision is weighted 2x over recall). A high confidence classification threshold (e.g., 0.7 - 0.85) will be optimal to aggressively preserve singletons.\n")

    return "".join(md)


def main():
    print("=== Starting Amazon Business Entity Resolution Profiling ===")
    total_start = time.time()

    profiles = {}
    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        if name != "train_ground_truth":
            profiles[name] = profile_source_file(path, name)

    # Collect IDs for ground truth validation
    print("Collecting train entity IDs for ground truth verification...")
    train_s1_ids = set(pl.read_csv(FILES["train_source1"], separator="\t", columns=["entity_id"])["entity_id"].to_list())
    train_s2_ids = set(pl.read_csv(FILES["train_source2"], separator="\t", columns=["entity_id"])["entity_id"].to_list())
    train_s3_ids = set(pl.read_csv(FILES["train_source3"], separator="\t", columns=["entity_id"])["entity_id"].to_list())

    # Profile Ground Truth
    gt_profile = profile_ground_truth(FILES["train_ground_truth"], train_s1_ids, train_s2_ids, train_s3_ids)
    profiles["train_ground_truth"] = gt_profile

    # Load small subsets for deep qualitative noise mining
    print("Loading datasets for qualitative noise pattern analysis...")
    df_s1 = pl.read_csv(FILES["train_source1"], separator="\t")
    df_s2 = pl.read_csv(FILES["train_source2"], separator="\t")
    df_s3 = pl.read_csv(FILES["train_source3"], separator="\t")
    df_gt = pl.read_csv(FILES["train_ground_truth"], separator="\t")

    noise_patterns = analyze_noise_and_match_patterns(df_s1, df_s2, df_s3, df_gt, n_samples=100000)

    # Free memory of full dfs before shifts check
    del df_s1, df_s2, df_s3, df_gt

    # Cross-dataset shifts (Train vs Test)
    shifts = analyze_train_test_shifts(profiles)

    # Aggregate full report dictionary
    full_report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "challenge": "Amazon Business Entity Resolution Challenge",
            "metric": "Macro F0.5",
            "total_execution_time_sec": round(time.time() - total_start, 2)
        },
        "files": profiles,
        "ground_truth": gt_profile,
        "shifts": shifts,
        "noise_patterns": noise_patterns
    }

    # Save JSON
    json_path = REPORTS_DIR / "dataset_profile.json"
    print(f"Writing JSON profile to {json_path}...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    # Generate and save Markdown report
    md_content = generate_markdown_report(full_report)
    md_path = REPORTS_DIR / "dataset_profile.md"
    print(f"Writing Markdown profile to {md_path}...")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"=== Profiling complete in {time.time() - total_start:.2f}s ===")


if __name__ == "__main__":
    main()
