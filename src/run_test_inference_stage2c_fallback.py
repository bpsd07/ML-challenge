"""
Stage 2C Frozen Blocker Fallback Test Inference Engine
Produces valid, high-precision matching_results.tsv and candidate_pairs.tsv
directly from frozen 47-strategy multi-script blocker if Stage 3 model is unavailable.
"""

import sys
import os
import gc
import time
import json
from pathlib import Path
from collections import defaultdict, Counter
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalization import (
    normalize_business_name,
    normalize_business_address
)
from blocking import HighRecallBlocker, MAX_KEY_FREQUENCY

BASE_DIR = Path(__file__).resolve().parent.parent
STUDENT_RES = BASE_DIR / "student_resource"
OUTPUT_DIR = STUDENT_RES / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_SEARCH_ROOTS = [
    Path("/kaggle/input"),
    STUDENT_RES / "dataset" / "test",
    STUDENT_RES / "dataset",
    STUDENT_RES,
    BASE_DIR
]


def find_file(filename: str) -> Path:
    for root in DATA_SEARCH_ROOTS:
        if root.exists():
            matches = list(root.rglob(filename))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"Required test dataset file not found: {filename}")


def run_stage2c_fallback_inference(
    min_vote_threshold: int = 3,
    top_k_candidates: int = 40
):
    print("=" * 80)
    print("STAGE 2C FALLBACK: FULL TEST SET INFERENCE PIPELINE")
    print("=" * 80)
    t0_start = time.time()

    # Locate and load test files
    p_s1 = find_file("test_source1.tsv")
    p_s2 = find_file("test_source2.tsv")
    p_s3 = find_file("test_source3.tsv")

    print(f"\nLoading Test Source 1: {p_s1}")
    df_s1 = pl.read_csv(p_s1, separator="\t")
    print(f"Test S1 entities to resolve: {len(df_s1):,}")

    print(f"\nLoading Test Source 2: {p_s2}")
    df_s2 = pl.read_csv(p_s2, separator="\t")
    print(f"Test S2 records: {len(df_s2):,}")

    print(f"\nLoading Test Source 3: {p_s3}")
    df_s3 = pl.read_csv(p_s3, separator="\t")
    print(f"Test S3 records: {len(df_s3):,}")

    # Index Test Candidates using Stage 2C Blocker
    blocker = HighRecallBlocker(max_key_frequency=MAX_KEY_FREQUENCY)
    blocker.index_candidate_source(df_s2, "Test Source 2")
    blocker.index_candidate_source(df_s3, "Test Source 3")

    out_matching_path = OUTPUT_DIR / "matching_results.tsv"
    out_candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"

    print(f"\nWriting final output files to:")
    print(f"  Matching Results: {out_matching_path}")
    print(f"  Candidate Pairs:  {out_candidate_path}")

    f_match = open(out_matching_path, "w", encoding="utf-8")
    f_cand = open(out_candidate_path, "w", encoding="utf-8")

    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    BATCH_SIZE = 5000
    n_total_s1 = len(df_s1)
    n_batches = (n_total_s1 + BATCH_SIZE - 1) // BATCH_SIZE

    total_matches_predicted = 0
    total_singletons = 0
    t0_infer = time.time()

    for b_idx in range(n_batches):
        b_slice = df_s1.slice(b_idx * BATCH_SIZE, BATCH_SIZE)

        for row in b_slice.iter_rows():
            sid = row[0]
            raw_name = row[1] or ""
            raw_addr = row[2] or ""
            cntry = str(row[3] or "UNKNOWN")

            s1_n_norm = normalize_business_name(raw_name, country=cntry)
            s1_a_norm = normalize_business_address(raw_addr, country=cntry)

            cands, strat_map = blocker.query_s1_entity(s1_n_norm, s1_a_norm, cntry)

            cand_votes = Counter()
            for strat, clist in strat_map.items():
                cand_votes.update(clist)

            top_cands_votes = cand_votes.most_common(top_k_candidates)
            top_cands = [c for c, _ in top_cands_votes]

            # High precision filtering for matches:
            # Candidates with high consensus across multiple blocker strategies
            matched_cands = [c for c, v in top_cands_votes if v >= min_vote_threshold]
            if not matched_cands and top_cands_votes:
                top_cand, top_v = top_cands_votes[0]
                if top_v >= 2:
                    matched_cands = [top_cand]

            cand_str = ",".join(top_cands)
            match_str = ",".join(matched_cands)

            f_cand.write(f"{sid}\t{cand_str}\n")
            f_match.write(f"{sid}\t{match_str}\n")

            if matched_cands:
                total_matches_predicted += len(matched_cands)
            else:
                total_singletons += 1

        if (b_idx + 1) % 10 == 0 or (b_idx + 1) == n_batches:
            elapsed = time.time() - t0_infer
            processed_s1 = min((b_idx + 1) * BATCH_SIZE, n_total_s1)
            rate = processed_s1 / elapsed
            rem_s1 = n_total_s1 - processed_s1
            eta_sec = rem_s1 / rate if rate > 0 else 0
            print(f"  Processed {processed_s1:,}/{n_total_s1:,} entities ({processed_s1/n_total_s1*100:.1f}%) | Speed: {rate:.1f} S1/s | ETA: {eta_sec/60:.1f}m")

    f_match.close()
    f_cand.close()

    print(f"\nFallback Inference Completed in {time.time()-t0_infer:.2f}s:")
    print(f"  Total S1 Entities Processed: {n_total_s1:,}")
    print(f"  Total Singletons (0 matches): {total_singletons:,} ({total_singletons/n_total_s1*100:.2f}%)")
    print(f"  Total Matches Predicted:     {total_matches_predicted:,} (Mean: {total_matches_predicted/max(1, n_total_s1-total_singletons):.2f}/active S1)")
    print(f"Total Runtime: {time.time()-t0_start:.2f}s")


if __name__ == "__main__":
    run_stage2c_fallback_inference()
