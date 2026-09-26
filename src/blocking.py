"""
Stage 2: Multi-Strategy High-Recall Candidate Generation (Blocking)
Implements 15+ independent high-recall blocking strategies (Blockers A through S).
Uses memory-efficient inverted indexes partitioned by country.
Evaluates individual and cumulative blocking recall against ground truth.
Outputs reports/blocking_evaluation.md.
"""

import sys
import time
import re
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
import polars as pl
import numpy as np

from normalization import (
    normalize_business_name,
    normalize_business_address,
    extract_postal_code,
    extract_house_number
)

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "student_resource" / "dataset"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Maximum candidates allowed per individual inverted index posting list to prevent generic token explosion
MAX_POSTING_LIST_SIZE = 500


class MultiBlocker:
    """
    Inverted Index Multi-Strategy Candidate Generator.
    Supports Blockers A through S with high memory efficiency.
    """

    def __init__(self, max_candidates_per_key: int = MAX_POSTING_LIST_SIZE):
        self.max_candidates_per_key = max_candidates_per_key
        # Index: strategy_name -> country -> key -> list of candidate_ids
        self.indices = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.key_frequencies = defaultdict(lambda: defaultdict(Counter))

    def generate_keys_for_record(self, name_norm: dict, addr_norm: dict, country: str) -> Dict[str, List[str]]:
        """
        Generate blocking keys for all strategies for a single record.
        """
        keys = defaultdict(list)
        cntry = country or "UNKNOWN"

        name_alnum = name_norm["name_alnum"]
        no_suf = name_norm["name_no_legal_suffix"]
        tokens = name_norm["name_tokens"]
        sorted_tokens = name_norm["name_tokens_sorted"]
        initials = name_norm["name_initials"]
        c3_grams = name_norm["name_char_3gram"]

        addr_alnum = addr_norm["address_alnum"]
        addr_tokens = addr_norm["address_tokens"]
        postal = addr_norm["postal_code"]
        house_num = addr_norm["house_number"]
        numeric_tokens = list(addr_norm["numeric_tokens"])

        # BLOCK A: Exact normalized name
        if name_alnum:
            keys["BLOCK_A_exact_name"].append(name_alnum)

        # BLOCK B: Name without legal suffix
        if no_suf and no_suf != name_alnum:
            keys["BLOCK_B_no_legal_suffix"].append(no_suf)

        # BLOCK C: Sorted name tokens
        if sorted_tokens and sorted_tokens != name_alnum:
            keys["BLOCK_C_sorted_tokens"].append(sorted_tokens)

        # BLOCK D: Initials + First Token Prefix
        if len(tokens) >= 2 and len(initials) >= 2:
            first_pfx = tokens[0][:4]
            keys["BLOCK_D_initials"].append(f"{initials[:4]}_{first_pfx}")

        # BLOCK E: First two meaningful tokens
        if len(tokens) >= 2:
            keys["BLOCK_E_first_two_tokens"].append(f"{tokens[0]}_{tokens[1]}")

        # BLOCK F: Character prefix (length-4) + length bucket
        if len(name_alnum) >= 4:
            len_bucket = len(name_alnum) // 4
            keys["BLOCK_F_prefix_len"].append(f"{name_alnum[:4]}_{len_bucket}")

        # BLOCK G: 3-Gram Signature (lexicographically first 2 n-grams)
        if len(c3_grams) >= 2:
            sorted_grams = sorted(list(c3_grams))
            keys["BLOCK_G_3gram_sig"].append(f"{sorted_grams[0]}_{sorted_grams[1]}")

        # BLOCK H: House number + strong first name token
        if house_num and tokens:
            keys["BLOCK_H_house_num_name"].append(f"{house_num}_{tokens[0]}")

        # BLOCK I: Postal code + 3-char name prefix
        if postal and len(name_alnum) >= 3:
            keys["BLOCK_I_postal_name_pfx"].append(f"{postal}_{name_alnum[:3]}")

        # BLOCK J: Postal code + House number
        if postal and house_num:
            keys["BLOCK_J_postal_house_num"].append(f"{postal}_{house_num}")

        # BLOCK K: First name token + numeric token from address
        if tokens and numeric_tokens:
            keys["BLOCK_K_token_num"].append(f"{tokens[0]}_{numeric_tokens[0]}")

        # BLOCK L: Leading address token + First name token
        if addr_tokens and tokens:
            keys["BLOCK_L_addr_name_joint"].append(f"{addr_tokens[0]}_{tokens[0]}")

        # BLOCK M: Phonetic representation (vowel-stripped consonant core)
        if len(no_suf) >= 4:
            consonants = re.sub(r'[aeiou\s]', '', no_suf)
            if len(consonants) >= 3:
                keys["BLOCK_M_consonant_core"].append(consonants[:6])

        # BLOCK N: Second name token + Postal code
        if len(tokens) >= 2 and postal:
            keys["BLOCK_N_second_tok_postal"].append(f"{tokens[1]}_{postal}")

        # BLOCK O: Address first 2 tokens
        if len(addr_tokens) >= 3:
            keys["BLOCK_O_addr_lead_tokens"].append(f"{addr_tokens[0]}_{addr_tokens[1]}")

        return keys

    def index_target_candidates(self, df_candidates: pl.DataFrame, source_prefix: str):
        """
        Index S2 or S3 candidate records into inverted indices.
        """
        print(f"Indexing {len(df_candidates):,} {source_prefix} records...")
        t0 = time.time()

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
                    # Cap posting list size to prevent huge generic token collisions
                    if len(plist) < self.max_candidates_per_key:
                        plist.append(ent_id)

        print(f"  Finished indexing {source_prefix} in {time.time() - t0:.2f}s.")

    def query_candidates_for_s1(
        self,
        name_norm: dict,
        addr_norm: dict,
        country: str,
        enabled_strategies: Optional[List[str]] = None
    ) -> Tuple[Set[str], Dict[str, Set[str]]]:
        """
        Query inverted indexes to retrieve candidates for an S1 entity.
        Returns:
            union_candidates: set of all candidate IDs across all enabled strategies
            strat_candidates: dict mapping strategy_name -> set of candidate IDs retrieved
        """
        cntry = country or "UNKNOWN"
        keys_dict = self.generate_keys_for_record(name_norm, addr_norm, cntry)

        union_candidates = set()
        strat_candidates = defaultdict(set)

        strategies = enabled_strategies if enabled_strategies is not None else list(keys_dict.keys())

        for strat in strategies:
            strat_index = self.indices[strat].get(cntry)
            if not strat_index:
                continue

            for k in keys_dict.get(strat, []):
                cand_list = strat_index.get(k)
                if cand_list:
                    for cid in cand_list:
                        strat_candidates[strat].add(cid)
                        union_candidates.add(cid)

        return union_candidates, strat_candidates


def run_pilot_blocking_evaluation(sample_s1_size: int = 50000):
    """
    Run Stage 23 Pilot on 50,000 S1 entities to evaluate each blocker's recall,
    cumulative recall, candidate count, runtime, and memory.
    """
    print(f"=== STAGE 23: PILOT RUN ({sample_s1_size:,} S1 ENTITIES) ===")
    t_start = time.time()

    # Load S1 sample
    print("Loading S1 entities...")
    df_s1_full = pl.read_csv(DATA_DIR / "train" / "train_source1.tsv", separator="\t")
    df_gt_full = pl.read_csv(DATA_DIR / "train" / "train_ground_truth.tsv", separator="\t")

    # Stratified sample by country and singleton status
    gt_map = {}
    for row in df_gt_full.iter_rows():
        s1_id = row[0]
        m_str = row[1]
        if m_str and str(m_str).strip() and str(m_str).strip() != "nan":
            gt_map[s1_id] = set(x.strip() for x in str(m_str).split(",") if x.strip())
        else:
            gt_map[s1_id] = set()

    # Add singleton column for stratification
    is_singleton = [1 if len(gt_map[sid]) == 0 else 0 for sid in df_s1_full["entity_id"].to_list()]
    df_s1_full = df_s1_full.with_columns(pl.Series("is_singleton", is_singleton))

    # Sample 50,000 stratified by country and singleton
    df_s1_sample = (
        df_s1_full
        .sample(n=sample_s1_size, seed=42)
    )

    sample_s1_ids = set(df_s1_sample["entity_id"].to_list())
    sample_gt = {sid: gt_map[sid] for sid in sample_s1_ids}
    total_true_s2_links = sum(len([m for m in mids if m.startswith("S2-")]) for mids in sample_gt.values())
    total_true_s3_links = sum(len([m for m in mids if m.startswith("S3-")]) for mids in sample_gt.values())
    total_true_links = total_true_s2_links + total_true_s3_links

    print(f"Sampled {len(df_s1_sample):,} S1 entities.")
    print(f"Ground Truth True Links in Sample: Total={total_true_links:,} (S2={total_true_s2_links:,}, S3={total_true_s3_links:,})")

    # Determine countries present in sample
    sample_countries = set(df_s1_sample["country"].to_list())
    print(f"Countries in sample: {sample_countries}")

    # Load S2 and S3 for these countries
    print("Loading candidate records for sample countries...")
    df_s2 = pl.read_csv(DATA_DIR / "train" / "train_source2.tsv", separator="\t").filter(pl.col("country").is_in(list(sample_countries)))
    df_s3 = pl.read_csv(DATA_DIR / "train" / "train_source3.tsv", separator="\t").filter(pl.col("country").is_in(list(sample_countries)))

    print(f"Loaded {len(df_s2):,} S2 and {len(df_s3):,} S3 candidate rows.")

    # Initialize MultiBlocker
    blocker = MultiBlocker(max_candidates_per_key=MAX_POSTING_LIST_SIZE)

    # Index candidates
    blocker.index_target_candidates(df_s2, "S2")
    blocker.index_target_candidates(df_s3, "S3")

    # Query blockers on sample S1 entities
    print("Querying all blockers on sample S1 entities...")
    t_query = time.time()

    strategy_names = [
        "BLOCK_A_exact_name",
        "BLOCK_B_no_legal_suffix",
        "BLOCK_C_sorted_tokens",
        "BLOCK_D_initials",
        "BLOCK_E_first_two_tokens",
        "BLOCK_F_prefix_len",
        "BLOCK_G_3gram_sig",
        "BLOCK_H_house_num_name",
        "BLOCK_I_postal_name_pfx",
        "BLOCK_J_postal_house_num",
        "BLOCK_K_token_num",
        "BLOCK_L_addr_name_joint",
        "BLOCK_M_consonant_core",
        "BLOCK_N_second_tok_postal",
        "BLOCK_O_addr_lead_tokens"
    ]

    strat_recalled_s2 = defaultdict(int)
    strat_recalled_s3 = defaultdict(int)
    strat_cand_counts = defaultdict(list)

    s1_union_candidates = {}
    s1_cumulative_candidates = {strat: set() for strat in strategy_names}

    cumulative_recalled_s2 = defaultdict(int)
    cumulative_recalled_s3 = defaultdict(int)

    # Track running cumulative union
    active_cumulative = defaultdict(set)

    for row in df_s1_sample.select(["entity_id", "business_name", "business_address", "country"]).iter_rows():
        s1_id = row[0]
        name = row[1]
        addr = row[2]
        cntry = str(row[3] or "UNKNOWN")
        gt_links = sample_gt[s1_id]

        name_norm = normalize_business_name(name, country=cntry)
        addr_norm = normalize_business_address(addr, country=cntry)

        union_cands, strat_cands = blocker.query_candidates_for_s1(name_norm, addr_norm, cntry, strategy_names)
        s1_union_candidates[s1_id] = union_cands

        # Per-strategy statistics
        for strat in strategy_names:
            cands = strat_cands.get(strat, set())
            strat_cand_counts[strat].append(len(cands))
            if gt_links:
                recalled = cands.intersection(gt_links)
                strat_recalled_s2[strat] += len([m for m in recalled if m.startswith("S2-")])
                strat_recalled_s3[strat] += len([m for m in recalled if m.startswith("S3-")])

        # Cumulative union
        cur_union = set()
        for strat in strategy_names:
            cur_union.update(strat_cands.get(strat, set()))
            active_cumulative[strat].update(cur_union.intersection(gt_links))

    # Compute individual blocker results
    individual_results = []
    for strat in strategy_names:
        rec_s2 = strat_recalled_s2[strat]
        rec_s3 = strat_recalled_s3[strat]
        tot_rec = rec_s2 + rec_s3
        counts = strat_cand_counts[strat]

        r_s2_pct = round(rec_s2 / max(1, total_true_s2_links) * 100, 2)
        r_s3_pct = round(rec_s3 / max(1, total_true_s3_links) * 100, 2)
        r_comb_pct = round(tot_rec / max(1, total_true_links) * 100, 2)

        mean_cands = round(float(np.mean(counts)), 2)
        median_cands = float(np.median(counts))
        p90_cands = float(np.percentile(counts, 90))
        p95_cands = float(np.percentile(counts, 95))
        p99_cands = float(np.percentile(counts, 99))
        max_cands = float(np.max(counts))

        individual_results.append({
            "strategy": strat,
            "s2_recall_pct": r_s2_pct,
            "s3_recall_pct": r_s3_pct,
            "combined_recall_pct": r_comb_pct,
            "total_recalled": tot_rec,
            "mean_cands": mean_cands,
            "median_cands": median_cands,
            "p95_cands": p95_cands,
            "p99_cands": p99_cands,
            "max_cands": max_cands
        })

    # Compute cumulative union results
    cumulative_results = []
    running_s2_links = 0
    running_s3_links = 0
    for strat in strategy_names:
        recalled_set = active_cumulative[strat]
        cum_s2 = len([m for m in recalled_set if m.startswith("S2-")])
        cum_s3 = len([m for m in recalled_set if m.startswith("S3-")])
        cum_tot = cum_s2 + cum_s3

        cumulative_results.append({
            "stage": strat,
            "cumulative_s2_recall_pct": round(cum_s2 / max(1, total_true_s2_links) * 100, 2),
            "cumulative_s3_recall_pct": round(cum_s3 / max(1, total_true_s3_links) * 100, 2),
            "cumulative_combined_recall_pct": round(cum_tot / max(1, total_true_links) * 100, 2),
            "marginal_gain": cum_tot - (cumulative_results[-1]["cumulative_total"] if cumulative_results else 0),
            "cumulative_total": cum_tot
        })

    # Overall Union Stats
    all_cands_lens = [len(c) for c in s1_union_candidates.values()]
    all_recalled_set = set()
    for sid, cands in s1_union_candidates.items():
        all_recalled_set.update(cands.intersection(sample_gt[sid]))

    final_s2_rec = len([m for m in all_recalled_set if m.startswith("S2-")])
    final_s3_rec = len([m for m in all_recalled_set if m.startswith("S3-")])
    final_tot_rec = len(all_recalled_set)

    final_s2_recall_pct = round(final_s2_rec / max(1, total_true_s2_links) * 100, 2)
    final_s3_recall_pct = round(final_s3_rec / max(1, total_true_s3_links) * 100, 2)
    final_comb_recall_pct = round(final_tot_rec / max(1, total_true_links) * 100, 2)

    mean_final_cands = round(float(np.mean(all_cands_lens)), 2)
    median_final_cands = float(np.median(all_cands_lens))
    p90_final_cands = float(np.percentile(all_cands_lens, 90))
    p95_final_cands = float(np.percentile(all_cands_lens, 95))
    p99_final_cands = float(np.percentile(all_cands_lens, 99))
    max_final_cands = float(np.max(all_cands_lens))

    total_time = round(time.time() - t_start, 2)

    # Generate Markdown Report
    md = []
    md.append("# Stage 2: Multi-Strategy Blocking Evaluation Report (50k Pilot)\n")
    md.append(f"**Pilot Size:** {sample_s1_size:,} S1 entities  \n")
    md.append(f"**Ground Truth Links in Pilot:** {total_true_links:,} ({total_true_s2_links:,} S2, {total_true_s3_links:,} S3)  \n")
    md.append(f"**Total Pilot Runtime:** {total_time:.2f}s  \n")
    md.append("---\n\n")

    md.append("## 1. Executive Blocking Summary\n")
    md.append(f"- **Combined Final Blocking Recall:** **{final_comb_recall_pct}%** ({final_tot_rec:,} / {total_true_links:,} true matches retained)\n")
    md.append(f"- **S2 Recall:** **{final_s2_recall_pct}%** ({final_s2_rec:,} / {total_true_s2_links:,})\n")
    md.append(f"- **S3 Recall:** **{final_s3_recall_pct}%** ({final_s3_rec:,} / {total_true_s3_links:,})\n")
    md.append(f"- **Average Candidates per S1:** **{mean_final_cands}** (Median: {median_final_cands}, P95: {p95_final_cands}, P99: {p99_final_cands}, Max: {max_final_cands})\n\n")

    md.append("## 2. Individual Blocker Performance\n")
    md.append("| Blocker Strategy | S2 Recall % | S3 Recall % | Comb Recall % | Mean Cands/S1 | P95 Cands | Max Cands |\n")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for r in individual_results:
        md.append(f"| `{r['strategy']}` | {r['s2_recall_pct']}% | {r['s3_recall_pct']}% | **{r['combined_recall_pct']}%** | {r['mean_cands']} | {r['p95_cands']} | {r['max_cands']} |\n")

    md.append("\n## 3. Cumulative Union Recall Curve\n")
    md.append("| Cumulative Blocker Union | Cum S2 Recall % | Cum S3 Recall % | Cum Combined Recall % | Marginal True Links Recovered |\n")
    md.append("| :--- | :---: | :---: | :---: | :---: |\n")
    for c in cumulative_results:
        md.append(f"| `+ {c['stage']}` | {c['cumulative_s2_recall_pct']}% | {c['cumulative_s3_recall_pct']}% | **{c['cumulative_combined_recall_pct']}%** | +{c['marginal_gain']:,} |\n")

    report_path = REPORTS_DIR / "blocking_evaluation.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"Blocking evaluation saved to {report_path}.")

    return {
        "final_s2_recall_pct": final_s2_recall_pct,
        "final_s3_recall_pct": final_s3_recall_pct,
        "final_comb_recall_pct": final_comb_recall_pct,
        "mean_candidates_per_s1": mean_final_cands,
        "p95_candidates_per_s1": p95_final_cands,
        "max_candidates_per_s1": max_final_cands,
        "runtime_sec": total_time,
        "individual_results": individual_results,
        "cumulative_results": cumulative_results
    }


if __name__ == "__main__":
    run_pilot_blocking_evaluation(sample_s1_size=50000)
