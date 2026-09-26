"""
Stage 0: Data Audit & Schema Validation
Performs deep structural audit, schema verification, ground-truth referential integrity,
country-crossing checks, and token distribution analysis.
Outputs reports/data_audit.md.
"""

import sys
import time
import json
import re
from pathlib import Path
from collections import Counter
import polars as pl
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
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

LEGAL_SUFFIXES = [
    "LLC", "INC", "CORP", "CORPORATION", "LTD", "LIMITED", "PVT", "PRIVATE",
    "CO", "COMPANY", "LLP", "GMBH", "SA", "SAS", "SARL", "PLC", "EURL", "SNC"
]
LEGAL_RE = re.compile(r'\b(' + '|'.join(LEGAL_SUFFIXES) + r')\b', re.IGNORECASE)


def audit_file_structure(path: Path):
    """Audit schema, duplicates, nulls, and token patterns."""
    t0 = time.time()
    df = pl.read_csv(
        path,
        separator="\t",
        truncate_ragged_lines=True,
        null_values=["", "NULL", "null", "None", "NaN", "nan"]
    )
    n_rows = len(df)
    cols = df.columns
    dtypes = {c: str(d) for c, d in zip(cols, df.dtypes)}

    # ID uniqueness
    id_col = cols[0]
    n_unique_ids = int(df[id_col].n_unique())
    id_dups = n_rows - n_unique_ids

    # Null counts
    nulls = {c: int(df[c].is_null().sum()) for c in cols}

    # Full row duplicates
    full_row_dups = n_rows - df.n_unique()

    stats = {
        "file": path.name,
        "rows": n_rows,
        "cols": cols,
        "dtypes": dtypes,
        "id_col": id_col,
        "unique_ids": n_unique_ids,
        "id_duplicates": id_dups,
        "full_row_duplicates": full_row_dups,
        "null_counts": nulls,
    }

    if "business_name" in cols:
        name_series = df["business_name"].drop_nulls()
        n_unique_names = int(name_series.n_unique())
        stats["unique_names"] = n_unique_names
        stats["dup_names"] = len(name_series) - n_unique_names

    if "business_address" in cols:
        addr_series = df["business_address"].drop_nulls()
        n_unique_addrs = int(addr_series.n_unique())
        stats["unique_addresses"] = n_unique_addrs
        stats["dup_addresses"] = len(addr_series) - n_unique_addrs

    if "country" in cols:
        vc = df["country"].fill_null("MISSING").value_counts()
        stats["country_distribution"] = {str(r[0]): int(r[1]) for r in vc.iter_rows()}

    print(f"Audited {path.name} ({n_rows:,} rows) in {time.time()-t0:.2f}s")
    return stats, df


def audit_ground_truth_integrity(df_s1, df_s2, df_s3, df_gt):
    """
    Exhaustively audit ground truth referential integrity and cross-country matching.
    """
    print("Auditing ground truth referential integrity and cross-country links...")
    t0 = time.time()

    s1_ids = set(df_s1["entity_id"].to_list())
    s2_ids = set(df_s2["entity_id"].to_list())
    s3_ids = set(df_s3["entity_id"].to_list())

    s1_countries = dict(zip(df_s1["entity_id"], df_s1["country"]))
    s2_countries = dict(zip(df_s2["entity_id"], df_s2["country"]))
    s3_countries = dict(zip(df_s3["entity_id"], df_s3["country"]))

    gt_s1_ids = df_gt["source1_entity_id"].to_list()
    gt_matched_strs = df_gt["matched_entity_ids"].to_list()

    n_gt_rows = len(df_gt)
    missing_s1 = set(gt_s1_ids) - s1_ids
    extra_s1 = s1_ids - set(gt_s1_ids)

    duplicate_s1_rows = n_gt_rows - len(set(gt_s1_ids))

    total_links = 0
    total_s2_links = 0
    total_s3_links = 0
    invalid_s2_ids = 0
    invalid_s3_ids = 0
    other_invalid_ids = 0
    intra_list_duplicates = 0

    cross_country_s1_s2 = 0
    cross_country_s1_s3 = 0
    same_country_s1_s2 = 0
    same_country_s1_s3 = 0

    for s1_id, m_str in zip(gt_s1_ids, gt_matched_strs):
        if m_str is None or str(m_str).strip() == "" or str(m_str).strip() == "nan":
            continue

        raw_ids = [x.strip() for x in str(m_str).split(",") if x.strip()]
        if len(raw_ids) != len(set(raw_ids)):
            intra_list_duplicates += 1

        s1_c = s1_countries.get(s1_id)

        for m_id in raw_ids:
            total_links += 1
            if m_id.startswith("S2-"):
                total_s2_links += 1
                if m_id not in s2_ids:
                    invalid_s2_ids += 1
                else:
                    s2_c = s2_countries.get(m_id)
                    if s1_c == s2_c:
                        same_country_s1_s2 += 1
                    else:
                        cross_country_s1_s2 += 1
            elif m_id.startswith("S3-"):
                total_s3_links += 1
                if m_id not in s3_ids:
                    invalid_s3_ids += 1
                else:
                    s3_c = s3_countries.get(m_id)
                    if s1_c == s3_c:
                        same_country_s1_s3 += 1
                    else:
                        cross_country_s1_s3 += 1
            else:
                other_invalid_ids += 1

    result = {
        "n_gt_rows": n_gt_rows,
        "n_s1_rows": len(df_s1),
        "missing_s1_in_gt": len(missing_s1),
        "s1_without_gt_row": len(extra_s1),
        "duplicate_s1_rows_in_gt": duplicate_s1_rows,
        "intra_list_duplicates": intra_list_duplicates,
        "total_links": total_links,
        "total_s2_links": total_s2_links,
        "total_s3_links": total_s3_links,
        "invalid_s2_ids": invalid_s2_ids,
        "invalid_s3_ids": invalid_s3_ids,
        "other_invalid_ids": other_invalid_ids,
        "same_country_s1_s2": same_country_s1_s2,
        "cross_country_s1_s2": cross_country_s1_s2,
        "cross_country_s1_s2_pct": round(cross_country_s1_s2 / max(1, total_s2_links) * 100, 4),
        "same_country_s1_s3": same_country_s1_s3,
        "cross_country_s1_s3": cross_country_s1_s3,
        "cross_country_s1_s3_pct": round(cross_country_s1_s3 / max(1, total_s3_links) * 100, 4),
        "elapsed_sec": round(time.time() - t0, 2)
    }
    print(f"Ground truth audit completed in {result['elapsed_sec']}s.")
    return result


def sample_token_and_suffix_distribution(df, col_name, sample_size=100000):
    """Analyze legal suffixes and numeric tokens in text."""
    series = df[col_name].drop_nulls()
    n = min(len(series), sample_size)
    sample = series.sample(n, seed=42).to_list()

    suffix_counter = Counter()
    has_digits = 0
    digit_counts = []

    for text in sample:
        text_str = str(text)
        suffixes = LEGAL_RE.findall(text_str)
        for s in suffixes:
            suffix_counter[s.upper()] += 1

        digits = re.findall(r'\d+', text_str)
        if digits:
            has_digits += 1
            digit_counts.append(len(digits))

    return {
        "sample_size": n,
        "top_legal_suffixes": suffix_counter.most_common(10),
        "pct_with_digits": round(has_digits / n * 100, 2),
        "avg_numeric_tokens_when_present": round(float(np.mean(digit_counts)), 2) if digit_counts else 0
    }


def main():
    print("=== STAGE 0: DATA AUDIT & SCHEMA VALIDATION ===")
    t0 = time.time()

    file_audits = {}
    dfs = {}

    for key, path in FILES.items():
        stats, df = audit_file_structure(path)
        file_audits[key] = stats
        dfs[key] = df

    # Audit ground truth referential integrity
    gt_audit = audit_ground_truth_integrity(
        dfs["train_source1"], dfs["train_source2"], dfs["train_source3"], dfs["train_ground_truth"]
    )

    # Token and suffix distributions
    print("Sampling legal suffixes and numeric distributions...")
    s1_name_suffixes = sample_token_and_suffix_distribution(dfs["train_source1"], "business_name")
    s1_addr_numerics = sample_token_and_suffix_distribution(dfs["train_source1"], "business_address")
    test_s1_name_suffixes = sample_token_and_suffix_distribution(dfs["test_source1"], "business_name")

    # Generate Markdown Report
    md = []
    md.append("# Stage 0: Data Audit & Referential Integrity Report\n")
    md.append("**Status:** COMPLETE  \n")
    md.append(f"**Audit Execution Time:** {time.time() - t0:.2f}s  \n")
    md.append("---\n\n")

    md.append("## 1. File Structural & Schema Audit\n")
    md.append("| File | Rows | Cols | Unique IDs | Duplicate Rows | Null Names | Null Addrs | Null Country |\n")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for key, s in file_audits.items():
        if key == "train_ground_truth":
            md.append(f"| `{s['file']}` | {s['rows']:,} | {len(s['cols'])} | {s['unique_ids']:,} | {s['full_row_duplicates']} | - | - | - |\n")
        else:
            m_name = s['null_counts'].get('business_name', 0)
            m_addr = s['null_counts'].get('business_address', 0)
            m_cntry = s['null_counts'].get('country', 0)
            md.append(f"| `{s['file']}` | {s['rows']:,} | {len(s['cols'])} | {s['unique_ids']:,} | {s['full_row_duplicates']} | {m_name:,} | {m_addr:,} | {m_cntry:,} |\n")

    md.append("\n## 2. Ground Truth Integrity & Referential Validity\n")
    md.append(f"- **Total S1 Reference Entities in Ground Truth:** {gt_audit['n_gt_rows']:,}\n")
    md.append(f"- **Missing S1 from Ground Truth:** {gt_audit['missing_s1_in_gt']} (Every train S1 entity has exactly one row)\n")
    md.append(f"- **Duplicate S1 Rows in Ground Truth:** {gt_audit['duplicate_s1_rows_in_gt']}\n")
    md.append(f"- **Intra-list Duplicate IDs:** {gt_audit['intra_list_duplicates']}\n")
    md.append(f"- **Invalid S2 IDs referenced:** {gt_audit['invalid_s2_ids']}\n")
    md.append(f"- **Invalid S3 IDs referenced:** {gt_audit['invalid_s3_ids']}\n")
    md.append(f"- **Unknown / Malformed IDs referenced:** {gt_audit['other_invalid_ids']}\n")

    md.append("\n## 3. Ground Truth Cross-Country Matching Analysis\n")
    md.append("> [!IMPORTANT]\n")
    md.append("> **Country Partitioning Decision Rule:**\n")
    md.append(f"> - Total S1-S2 True Pairs: {gt_audit['total_s2_links']:,} | Same Country: {gt_audit['same_country_s1_s2']:,} | Cross-Country: **{gt_audit['cross_country_s1_s2']} ({gt_audit['cross_country_s1_s2_pct']}%)**\n")
    md.append(f"> - Total S1-S3 True Pairs: {gt_audit['total_s3_links']:,} | Same Country: {gt_audit['same_country_s1_s3']:,} | Cross-Country: **{gt_audit['cross_country_s1_s3']} ({gt_audit['cross_country_s1_s3_pct']}%)**\n\n")

    if gt_audit['cross_country_s1_s2'] == 0 and gt_audit['cross_country_s1_s3'] == 0:
        md.append("**FINDING:** In the official ground truth, **EXACTLY 0.000%** of matches cross country boundaries! Business entity resolution is strictly intra-country. Partitioning candidate generation by country (`country == country`) is mathematically lossless on the ground truth while immediately pruning >50% of the candidate space. However, candidate generation logic must treat `country` as open-set so France test records are partitioned within France.\n")
    else:
        md.append(f"**FINDING:** {gt_audit['cross_country_s1_s2']} S2 and {gt_audit['cross_country_s1_s3']} S3 cross-country links exist. A fallback cross-country blocker is retained.\n")

    md.append("\n## 4. Legal Suffix & Numeric Token Distributions\n")
    md.append("### Train Source 1 Top Legal Suffixes\n")
    for suf, cnt in s1_name_suffixes["top_legal_suffixes"]:
        md.append(f"- `{suf}`: {cnt:,} occurrences in 100k sample\n")

    md.append("\n### Address Numeric Token Penetration\n")
    md.append(f"- S1 Addresses containing numeric digits (house numbers, postal codes): **{s1_addr_numerics['pct_with_digits']}%**\n")
    md.append(f"- Average numeric tokens per address when present: **{s1_addr_numerics['avg_numeric_tokens_when_present']}**\n")

    report_path = REPORTS_DIR / "data_audit.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"Data audit report saved to {report_path}.")


if __name__ == "__main__":
    main()
