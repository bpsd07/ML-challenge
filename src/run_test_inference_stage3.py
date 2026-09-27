"""
Full Test Inference Engine (Stage 3 LightGBM + Entity Decision Policy)
Generates:
  - output/matching_results.tsv
  - output/candidate_pairs.tsv
Complies with ML Challenge 2026 submission specifications.
Runs country-partitioned streaming to operate in <7 GB RAM.
"""

import sys
import os
import gc
import time
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import polars as pl
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalization import (
    normalize_business_name,
    normalize_business_address,
    detect_script,
    transliterate_indic
)
from blocking import HighRecallBlocker, MAX_KEY_FREQUENCY
from features import compute_pairwise_features, FEATURE_NAMES

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


def run_stage3_test_inference(
    model_path: str = "checkpoints/ml/stage3_lgb_model.txt",
    checkpoint_path: str = "checkpoints/ml/stage3_model_checkpoint.json",
    top_k_candidates: int = 40
):
    print("=" * 80)
    print("STAGE 3: FULL TEST SET INFERENCE PIPELINE (COUNTRY-PARTITIONED)")
    print("=" * 80)
    t0_start = time.time()

    # 1. Load trained model and decision policy
    p_model = Path(model_path)
    if not p_model.exists():
        p_model = BASE_DIR / model_path
    if not p_model.exists():
        raise FileNotFoundError(f"Trained LightGBM model not found at {model_path}")

    p_chk = Path(checkpoint_path)
    if not p_chk.exists():
        p_chk = BASE_DIR / checkpoint_path
    if not p_chk.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    print(f"Loading LightGBM model from: {p_model}")
    model = lgb.Booster(model_file=str(p_model))

    print(f"Loading decision policy from: {p_chk}")
    with open(p_chk, "r", encoding="utf-8") as f:
        chk_data = json.load(f)

    policy = chk_data.get("entity_decision_policy", {})
    opt_t_abs = float(policy.get("optimal_t_abs", 0.65))
    opt_t_guard = float(policy.get("optimal_t_guard", 0.65))
    opt_r_rel = float(policy.get("optimal_r_rel", 0.75))

    print(f"Decision Policy Parameters:")
    print(f"  Absolute Threshold (T_abs):        {opt_t_abs:.2f}")
    print(f"  Singleton Guard Threshold (T_guard): {opt_t_guard:.2f}")
    print(f"  Relative Drop-off Ratio (R_rel):     {opt_r_rel:.2f}")

    # 2. Locate and load test files
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

    # Output files
    out_matching_path = OUTPUT_DIR / "matching_results.tsv"
    out_candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"

    print(f"\nWriting final output files to:")
    print(f"  Matching Results: {out_matching_path}")
    print(f"  Candidate Pairs:  {out_candidate_path}")

    f_match = open(out_matching_path, "w", encoding="utf-8")
    f_cand = open(out_candidate_path, "w", encoding="utf-8")

    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    total_matches_predicted = 0
    total_singletons = 0
    total_s1_processed = 0

    # Strict 0.000% cross-country matching partition
    # Process France first (smallest), then US, then India
    countries = ["France", "US", "India"]
    
    for cntry_name in countries:
        df_s1_cntry = df_s1.filter(pl.col("country") == cntry_name)
        df_s2_cntry = df_s2.filter(pl.col("country") == cntry_name)
        df_s3_cntry = df_s3.filter(pl.col("country") == cntry_name)

        n_s1_cntry = len(df_s1_cntry)
        print(f"\n" + "=" * 60)
        print(f"PROCESSING COUNTRY: {cntry_name}")
        print(f"  S1: {n_s1_cntry:,} | S2: {len(df_s2_cntry):,} | S3: {len(df_s3_cntry):,}")
        print("=" * 60)

        # Index only this country's candidates (<7 GB RAM footprint)
        blocker = HighRecallBlocker(max_key_frequency=MAX_KEY_FREQUENCY)
        blocker.index_candidate_source(df_s2_cntry, f"Test S2 ({cntry_name})")
        blocker.index_candidate_source(df_s3_cntry, f"Test S3 ({cntry_name})")

        def fetch_cand_metadata_batch(cand_ids_needed: set):
            s2_ids = [c for c in cand_ids_needed if c.startswith("S2-")]
            s3_ids = [c for c in cand_ids_needed if c.startswith("S3-")]

            batch_lookup = {}
            if s2_ids:
                df_sub = df_s2_cntry.filter(pl.col("entity_id").is_in(s2_ids))
                for row in df_sub.select(["entity_id", "business_name", "business_address", "country"]).iter_rows():
                    cid = row[0]
                    c_cntry = str(row[3] or cntry_name)
                    n_norm = normalize_business_name(row[1] or "", country=c_cntry)
                    a_norm = normalize_business_address(row[2] or "", country=c_cntry)
                    batch_lookup[cid] = (n_norm, a_norm, c_cntry)
                del df_sub

            if s3_ids:
                df_sub = df_s3_cntry.filter(pl.col("entity_id").is_in(s3_ids))
                for row in df_sub.select(["entity_id", "business_name", "business_address", "country"]).iter_rows():
                    cid = row[0]
                    c_cntry = str(row[3] or cntry_name)
                    n_norm = normalize_business_name(row[1] or "", country=c_cntry)
                    a_norm = normalize_business_address(row[2] or "", country=c_cntry)
                    batch_lookup[cid] = (n_norm, a_norm, c_cntry)
                del df_sub

            return batch_lookup

        BATCH_SIZE = 5000
        n_batches = (n_s1_cntry + BATCH_SIZE - 1) // BATCH_SIZE
        t0_cntry = time.time()

        for b_idx in range(n_batches):
            b_slice = df_s1_cntry.slice(b_idx * BATCH_SIZE, BATCH_SIZE)

            batch_queries = []
            batch_needed_cands = set()

            for row in b_slice.iter_rows():
                sid = row[0]
                raw_name = row[1] or ""
                raw_addr = row[2] or ""
                cntry = str(row[3] or cntry_name)

                s1_n_norm = normalize_business_name(raw_name, country=cntry)
                s1_a_norm = normalize_business_address(raw_addr, country=cntry)

                cands, strat_map = blocker.query_s1_entity(s1_n_norm, s1_a_norm, cntry)

                cand_votes = Counter()
                for strat, clist in strat_map.items():
                    cand_votes.update(clist)

                top_cands_votes = cand_votes.most_common(top_k_candidates)
                top_cands = [c for c, _ in top_cands_votes]
                cand_ranks = {c: r + 1 for r, c in enumerate(top_cands)}

                cands_set = set(top_cands)
                entity_cand_strats = {c: set() for c in cands_set}
                entity_cand_votes = {c: cand_votes[c] for c in cands_set}
                entity_cand_ranks = {c: cand_ranks[c] for c in cands_set}

                for strat, clist in strat_map.items():
                    for c in clist:
                        if c in entity_cand_strats:
                            entity_cand_strats[c].add(strat)

                batch_needed_cands.update(cands_set)

                batch_queries.append({
                    "sid": sid,
                    "s1_n_norm": s1_n_norm,
                    "s1_a_norm": s1_a_norm,
                    "cntry": cntry,
                    "top_cands": top_cands,
                    "cand_votes": entity_cand_votes,
                    "cand_ranks": entity_cand_ranks,
                    "cand_strats": entity_cand_strats
                })

            batch_metadata = fetch_cand_metadata_batch(batch_needed_cands)

            for q in batch_queries:
                sid = q["sid"]
                s1_n = q["s1_n_norm"]
                s1_a = q["s1_a_norm"]
                cntry = q["cntry"]
                scored_cands = q["top_cands"]

                cand_str = ",".join(scored_cands)
                f_cand.write(f"{sid}\t{cand_str}\n")

                if not scored_cands:
                    f_match.write(f"{sid}\t\n")
                    total_singletons += 1
                    total_s1_processed += 1
                    continue

                feat_rows = []
                valid_cands = []
                for cid in scored_cands:
                    if cid in batch_metadata:
                        c_n, c_a, _ = batch_metadata[cid]
                        v_count = q["cand_votes"].get(cid, 1)
                        r_rank = q["cand_ranks"].get(cid, 1)
                        feats = compute_pairwise_features(
                            s1_n, s1_a, c_n, c_a, cid, cntry,
                            blockers_fired=v_count, cand_rank=r_rank, strategies_fired=q["cand_strats"].get(cid, set())
                        )
                        feat_rows.append([feats[f] for f in FEATURE_NAMES])
                        valid_cands.append(cid)

                if not feat_rows:
                    f_match.write(f"{sid}\t\n")
                    total_singletons += 1
                    total_s1_processed += 1
                    continue

                X_entity = np.array(feat_rows, dtype=np.float32)
                probs = model.predict(X_entity)

                cand_probs = list(zip(valid_cands, probs))
                cand_probs.sort(key=lambda x: x[1], reverse=True)
                p_max = cand_probs[0][1]

                # Policy: Singleton guard + absolute threshold + relative margin
                if p_max < opt_t_guard:
                    f_match.write(f"{sid}\t\n")
                    total_singletons += 1
                else:
                    matched_ids = []
                    for cid, p in cand_probs:
                        if p >= opt_t_abs and p >= (p_max * opt_r_rel):
                            matched_ids.append(cid)

                    match_str = ",".join(matched_ids)
                    f_match.write(f"{sid}\t{match_str}\n")
                    if matched_ids:
                        total_matches_predicted += len(matched_ids)
                    else:
                        total_singletons += 1

                total_s1_processed += 1

            del batch_metadata, batch_queries, batch_needed_cands
            gc.collect()

            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == n_batches:
                elapsed = time.time() - t0_cntry
                proc_cntry = min((b_idx + 1) * BATCH_SIZE, n_s1_cntry)
                rate = proc_cntry / elapsed if elapsed > 0 else 0
                rem_cntry = n_s1_cntry - proc_cntry
                eta_m = (rem_cntry / rate) / 60 if rate > 0 else 0
                print(f"  [{cntry_name}] {proc_cntry:,}/{n_s1_cntry:,} ({proc_cntry/n_s1_cntry*100:.1f}%) | {rate:.1f} S1/s | ETA: {eta_m:.1f}m")

        del blocker, df_s1_cntry, df_s2_cntry, df_s3_cntry
        gc.collect()
        print(f"Finished {cntry_name} in {time.time()-t0_cntry:.1f}s.")

    f_match.close()
    f_cand.close()

    print(f"\n" + "=" * 80)
    print(f"TEST INFERENCE COMPLETE in {time.time()-t0_start:.2f}s:")
    print(f"  Total S1 Entities Processed: {total_s1_processed:,}")
    print(f"  Total Singletons (0 matches): {total_singletons:,} ({total_singletons/total_s1_processed*100:.2f}%)")
    print(f"  Total Matches Predicted:     {total_matches_predicted:,} (Mean: {total_matches_predicted/max(1, total_s1_processed-total_singletons):.2f}/active S1)")
    print("=" * 80)


if __name__ == "__main__":
    run_stage3_test_inference()
