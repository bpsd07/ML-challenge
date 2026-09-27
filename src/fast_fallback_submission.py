"""
Ultra-Fast High-Precision Fallback Submission Generator
Executes vectorized exact & clean-name entity resolution in Polars.
Generates 100% compliant matching_results.tsv and candidate_pairs.tsv in <60 seconds.
"""

import sys
import os
import time
import re
from pathlib import Path
import polars as pl

BASE_DIR = Path(__file__).resolve().parent.parent
STUDENT_RES = BASE_DIR / "student_resource"
OUTPUT_DIR = STUDENT_RES / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = STUDENT_RES / "dataset" / "test"


def clean_name_expr(col_name: str) -> pl.Expr:
    return (
        pl.col(col_name)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def clean_suffix_expr(col_name: str) -> pl.Expr:
    # Strip common legal suffixes
    return (
        pl.col(col_name)
        .str.replace_all(r"\b(pvt|ltd|limited|private|llc|inc|corp|corporation|co|company|sarl|sas|sa|eurl|llp)\b", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def main():
    print("=" * 80)
    print("RUNNING ULTRA-FAST HIGH-PRECISION FALLBACK SUBMISSION PIPELINE")
    print("=" * 80)
    t0 = time.time()

    p_s1 = DATA_DIR / "test_source1.tsv"
    p_s2 = DATA_DIR / "test_source2.tsv"
    p_s3 = DATA_DIR / "test_source3.tsv"

    print(f"Loading test source files...")
    df_s1 = pl.read_csv(p_s1, separator="\t", columns=["entity_id", "business_name", "country"])
    df_s2 = pl.read_csv(p_s2, separator="\t", columns=["entity_id", "business_name", "country"])
    df_s3 = pl.read_csv(p_s3, separator="\t", columns=["entity_id", "business_name", "country"])

    n_s1 = len(df_s1)
    print(f"Loaded: S1 ({n_s1:,}), S2 ({len(df_s2):,}), S3 ({len(df_s3):,}) in {time.time()-t0:.2f}s")

    # Add normalized name expressions
    df_s1 = df_s1.with_columns([
        clean_name_expr("business_name").alias("norm_name")
    ]).with_columns([
        clean_suffix_expr("norm_name").alias("clean_name")
    ])

    df_s2 = df_s2.with_columns([
        clean_name_expr("business_name").alias("norm_name")
    ]).with_columns([
        clean_suffix_expr("norm_name").alias("clean_name")
    ])

    df_s3 = df_s3.with_columns([
        clean_name_expr("business_name").alias("norm_name")
    ]).with_columns([
        clean_suffix_expr("norm_name").alias("clean_name")
    ])

    # Filter out empty clean names
    df_s2_clean = df_s2.filter((pl.col("clean_name").str.len_chars() >= 3) & (pl.col("clean_name") != ""))
    df_s3_clean = df_s3.filter((pl.col("clean_name").str.len_chars() >= 3) & (pl.col("clean_name") != ""))

    print("Joining S1 with S2 on (clean_name, country)...")
    join_s2 = df_s1.select(["entity_id", "clean_name", "country"]).join(
        df_s2_clean.select(["entity_id", "clean_name", "country"]).rename({"entity_id": "cand_id"}),
        on=["clean_name", "country"],
        how="inner"
    )

    print("Joining S1 with S3 on (clean_name, country)...")
    join_s3 = df_s1.select(["entity_id", "clean_name", "country"]).join(
        df_s3_clean.select(["entity_id", "clean_name", "country"]).rename({"entity_id": "cand_id"}),
        on=["clean_name", "country"],
        how="inner"
    )

    # Filter out overly generic keys (frequency cap to prevent low-precision explosions)
    s2_key_freq = join_s2.group_by("clean_name").count()
    s2_good_keys = s2_key_freq.filter(pl.col("count") <= 10).select("clean_name")
    join_s2 = join_s2.join(s2_good_keys, on="clean_name", how="inner")

    s3_key_freq = join_s3.group_by("clean_name").count()
    s3_good_keys = s3_key_freq.filter(pl.col("count") <= 10).select("clean_name")
    join_s3 = join_s3.join(s3_good_keys, on="clean_name", how="inner")

    print(f"Found {len(join_s2):,} S1-S2 pairs and {len(join_s3):,} S1-S3 pairs")

    # Combine pairs
    all_pairs = pl.concat([
        join_s2.select(["entity_id", "cand_id"]),
        join_s3.select(["entity_id", "cand_id"])
    ])

    # Group by S1 entity_id
    matches_by_s1 = (
        all_pairs
        .group_by("entity_id")
        .agg(pl.col("cand_id").unique().slice(0, 10))
    )

    # Join back with all S1 entities to guarantee EVERY S1 entity has exactly one row
    final_table = (
        df_s1.select("entity_id")
        .join(matches_by_s1, on="entity_id", how="left")
        .with_columns(
            pl.col("cand_id").fill_null([])
        )
    )

    # Convert to comma-separated strings with proper null for singletons
    print("Formatting output TSVs...")
    formatted = final_table.select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.when(pl.col("cand_id").list.len() > 0)
        .then(pl.col("cand_id").list.join(","))
        .otherwise(None)
        .alias("matched_entity_ids")
    ])

    out_matching_path = OUTPUT_DIR / "matching_results.tsv"
    out_candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"

    # Write matching_results.tsv
    formatted.write_csv(out_matching_path, separator="\t", null_value="")
    print(f"Wrote matching_results.tsv ({out_matching_path.stat().st_size / (1024*1024):.2f} MB)")

    # Candidate pairs: same candidates
    cand_formatted = formatted.rename({"matched_entity_ids": "candidate_entity_ids"})
    cand_formatted.write_csv(out_candidate_path, separator="\t", null_value="")
    print(f"Wrote candidate_pairs.tsv ({out_candidate_path.stat().st_size / (1024*1024):.2f} MB)")

    n_rows = len(formatted)
    n_singletons = len(formatted.filter(pl.col("matched_entity_ids") == ""))
    n_matched = n_rows - n_singletons

    print(f"\nSummary:")
    print(f"  Total S1 rows:      {n_rows:,} (matches required {n_s1:,}: {n_rows == n_s1})")
    print(f"  Entities with match: {n_matched:,} ({n_matched/n_rows*100:.2f}%)")
    print(f"  Singletons (empty): {n_singletons:,} ({n_singletons/n_rows*100:.2f}%)")
    print(f"Pipeline executed in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
