import json
from pathlib import Path

def create_kaggle_notebook():
    cells = []

    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Header
    add_md("""# Amazon ML Challenge 2026 - Business Entity Resolution Pipeline
### End-to-End Distributed Entity Resolution with Polars Rust Vectorization & LightGBM
This notebook runs training, macro $F_{0.5}$ threshold optimization, and full-scale test inference on Kaggle GPU/CPU.

- **Deduplicated Reference Source:** Source 1 ($S_1$)
- **Matching Sources:** Source 2 ($S_2$) and Source 3 ($S_3$)
- **Countries:** US, India, France (open set)
- **Evaluation Metric:** Macro $F_{0.5}$ across all $S_1$ entities (Precision weighted 2x Recall)
""")

    # Cell 1: Environment & Path Setup
    add_code("""import sys
import os
import gc
import time
import math
import re
import csv
import zipfile
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

print(f"Python: {sys.version}")
print(f"Polars: {pl.__version__}")
print(f"LightGBM: {lgb.__version__}")

# Locate dataset directory
DATA_DIRS = [
    Path("/kaggle/input/amazon-ml-challenge-2026/dataset"),
    Path("/kaggle/input/amazon-ml-challenge-2026"),
    Path("./student_resource/dataset"),
    Path("../student_resource/dataset")
]

DATA_DIR = None
for p in DATA_DIRS:
    if (p / "train" / "train_source1.tsv").exists() or (p / "train_source1.tsv").exists():
        DATA_DIR = p
        break

if DATA_DIR is None:
    # Check recursive
    for root in [Path("/kaggle/input"), Path(".")]:
        matches = list(root.glob("**/train_source1.tsv"))
        if matches:
            DATA_DIR = matches[0].parent.parent
            break

print(f"Dataset root identified at: {DATA_DIR}")
TRAIN_DIR = DATA_DIR / "train" if (DATA_DIR / "train").exists() else DATA_DIR
TEST_DIR = DATA_DIR / "test" if (DATA_DIR / "test").exists() else DATA_DIR

OUTPUT_DIR = Path("/kaggle/working/output") if Path("/kaggle/working").exists() else Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Outputs will be written to: {OUTPUT_DIR}")
""")

    # Cell 2: Vectorized Preprocessing & Normalization
    add_md("""## 1. High-Performance Polars Rust Preprocessing
Vectorizes text normalization, suffix removal (US, India, France corporate designations), and structural extraction (postal codes, house numbers).
""")

    add_code("""LEGAL_SUFFIXES_REGEX = (
    r'\\b(llc|inc|corp|corporation|ltd|limited|pvt|private|co|company|llp|plc|'
    r'sarl|sas|sa|eurl|snc|sci|gie|gmbh|ag|spa|srl|sl)\\b'
)

def vectorize_source(df: pl.DataFrame) -> pl.DataFrame:
    \"\"\"Vectorized preprocessing using Polars native multi-threaded engine.\"\"\"
    df_clean = df.with_columns([
        pl.col("business_name").fill_null("").str.to_lowercase()
          .str.replace_all(r'[^a-z0-9\\s]', ' ').str.replace_all(r'\\s+', ' ')
          .str.strip_chars().alias("name_clean"),
        pl.col("business_address").fill_null("").str.to_lowercase()
          .str.replace_all(r'[^a-z0-9\\s]', ' ').str.replace_all(r'\\s+', ' ')
          .str.strip_chars().alias("addr_clean"),
        pl.col("country").fill_null("UNKNOWN").alias("country_norm")
    ])

    df_feats = df_clean.with_columns([
        pl.col("name_clean")
          .str.replace_all(LEGAL_SUFFIXES_REGEX, ' ')
          .str.replace_all(r'\\s+', ' ').str.strip_chars().alias("name_no_suffix"),
        pl.col("name_clean").str.slice(0, 4).alias("pfx4"),
        pl.col("business_address").str.extract(r'\\b([0-9]{5,6})\\b', 1).alias("postal"),
        pl.col("addr_clean").str.extract(r'^(\\d+)', 1).alias("house_num"),
    ])

    tokens_expr = pl.col("name_clean").str.split(" ")
    df_final = df_feats.with_columns([
        tokens_expr.list.get(0, null_on_oob=True).alias("tok0"),
        tokens_expr.list.get(1, null_on_oob=True).alias("tok1"),
        tokens_expr.list.len().alias("n_toks"),
    ])
    return df_final
""")

    # Cell 3: Inverted Index & Fast Candidate Retrieval
    add_md("""## 2. Partitioned Inverted Indexes (Blocking)
Generates high-recall candidate sets while strictly partitioning by country to bound comparison complexity.
""")

    add_code("""def build_fast_inverted_indexes(df_cand_vec: pl.DataFrame, max_posting_size: int = 400):
    t0 = time.time()
    idx_name = defaultdict(list)
    idx_nosuf = defaultdict(list)
    idx_tok0 = defaultdict(list)
    idx_postal = defaultdict(list)
    idx_pfx4 = defaultdict(list)

    for row in df_cand_vec.select([
        "entity_id", "country_norm", "name_clean", "name_no_suffix", "tok0", "postal", "pfx4"
    ]).iter_rows():
        cid, cntry, name_c, nosuf, t0_val, post, p4 = row
        c_pfx = str(cntry)

        if name_c:
            k = f"{c_pfx}|{name_c}"
            if len(idx_name[k]) < max_posting_size:
                idx_name[k].append(cid)

        if nosuf and nosuf != name_c:
            k = f"{c_pfx}|{nosuf}"
            if len(idx_nosuf[k]) < max_posting_size:
                idx_nosuf[k].append(cid)

        if t0_val and len(t0_val) >= 4:
            k = f"{c_pfx}|{t0_val}"
            if len(idx_tok0[k]) < max_posting_size:
                idx_tok0[k].append(cid)

        if post and p4:
            k = f"{c_pfx}|{post}_{p4[:3]}"
            if len(idx_postal[k]) < max_posting_size:
                idx_postal[k].append(cid)

        if p4 and len(name_c) >= 5:
            k = f"{c_pfx}|{p4}_{len(name_c)//4}"
            if len(idx_pfx4[k]) < max_posting_size:
                idx_pfx4[k].append(cid)

    print(f"Inverted indexes built in {time.time()-t0:.2f}s across {len(df_cand_vec):,} candidate entities.")
    return {
        "name": idx_name,
        "nosuf": idx_nosuf,
        "tok0": idx_tok0,
        "postal": idx_postal,
        "pfx4": idx_pfx4
    }


def query_candidates_fast(s1_row, indexes, max_return: int = 50):
    cid, cntry, name_c, nosuf, t0_val, post, p4 = s1_row
    c_pfx = str(cntry)
    scores = defaultdict(int)

    if name_c:
        for c in indexes["name"].get(f"{c_pfx}|{name_c}", []):
            scores[c] += 5
    if nosuf:
        for c in indexes["nosuf"].get(f"{c_pfx}|{nosuf}", []):
            scores[c] += 4
    if post and p4:
        for c in indexes["postal"].get(f"{c_pfx}|{post}_{p4[:3]}", []):
            scores[c] += 3
    if t0_val and len(t0_val) >= 4:
        for c in indexes["tok0"].get(f"{c_pfx}|{t0_val}", []):
            scores[c] += 2
    if p4 and len(name_c) >= 5:
        for c in indexes["pfx4"].get(f"{c_pfx}|{p4}_{len(name_c)//4}", []):
            scores[c] += 1

    if not scores:
        return []

    # Sort candidates by match frequency score
    sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_return]
    return sorted_cands
""")

    # Cell 4: Feature Extraction Function
    add_md("""## 3. High-Speed Pairwise Feature Extraction
Calculates token Jaccard, character n-gram overlaps, Levenshtein similarities, postal code & house number matches.
""")

    add_code("""def fast_levenshtein_ratio(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    l1, l2 = len(s1), len(s2)
    # Quick prefix/suffix check
    if s1 in s2 or s2 in s1:
        return min(l1, l2) / max(l1, l2)
    # Approximate ratio for fast inference
    p1 = {s1[i:i+2] for i in range(len(s1)-1)}
    p2 = {s2[i:i+2] for i in range(len(s2)-1)}
    if not p1 or not p2:
        return 0.0
    return 2.0 * len(p1 & p2) / (len(p1) + len(p2))


def compute_pair_features(s1_meta, cand_meta, blocker_score: int):
    # Name similarities
    s1_name, c_name = s1_meta["name_alnum"], cand_meta["name_alnum"]
    s1_toks, c_toks = s1_meta["name_token_set"], cand_meta["name_token_set"]

    exact_name = 1.0 if s1_name == c_name and s1_name else 0.0
    name_nosuf_match = 1.0 if s1_meta["name_no_legal_suffix"] == cand_meta["name_no_legal_suffix"] and s1_meta["name_no_legal_suffix"] else 0.0

    # Token Jaccard
    u_toks = s1_toks | c_toks
    tok_jaccard = (len(s1_toks & c_toks) / len(u_toks)) if u_toks else 0.0

    # Char n-grams
    s1_3g, c_3g = s1_meta["name_char_3gram"], cand_meta["name_char_3gram"]
    u_3g = s1_3g | c_3g
    char3_jaccard = (len(s1_3g & c_3g) / len(u_3g)) if u_3g else 0.0

    s1_4g, c_4g = s1_meta["name_char_4gram"], cand_meta["name_char_4gram"]
    u_4g = s1_4g | c_4g
    char4_jaccard = (len(s1_4g & c_4g) / len(u_4g)) if u_4g else 0.0

    lev_ratio = fast_levenshtein_ratio(s1_name, c_name)
    len_diff = abs(len(s1_name) - len(c_name)) / max(len(s1_name), len(c_name), 1)

    # Address features
    s1_addr, c_addr = s1_meta["address_alnum"], cand_meta["address_alnum"]
    s1_a_toks, c_a_toks = set(s1_meta["address_tokens"]), set(cand_meta["address_tokens"])
    u_a_toks = s1_a_toks | c_a_toks
    addr_tok_jaccard = (len(s1_a_toks & c_a_toks) / len(u_a_toks)) if u_a_toks else 0.0

    s1_a_3g, c_a_3g = s1_meta["address_char_3gram"], cand_meta["address_char_3gram"]
    u_a_3g = s1_a_3g | c_a_3g
    addr_char3_jaccard = (len(s1_a_3g & c_a_3g) / len(u_a_3g)) if u_a_3g else 0.0

    # Structural matches
    postal_match = 1.0 if s1_meta["postal_code"] and s1_meta["postal_code"] == cand_meta["postal_code"] else 0.0
    house_match = 1.0 if s1_meta["house_number"] and s1_meta["house_number"] == cand_meta["house_number"] else 0.0
    country_match = 1.0 if s1_meta["country"] == cand_meta["country"] else 0.0

    return [
        exact_name,
        name_nosuf_match,
        tok_jaccard,
        char3_jaccard,
        char4_jaccard,
        lev_ratio,
        len_diff,
        addr_tok_jaccard,
        addr_char3_jaccard,
        postal_match,
        house_match,
        country_match,
        float(blocker_score)
    ]

FEATURE_NAMES = [
    "exact_name", "name_nosuf_match", "tok_jaccard", "char3_jaccard", "char4_jaccard",
    "lev_ratio", "len_diff", "addr_tok_jaccard", "addr_char3_jaccard",
    "postal_match", "house_match", "country_match", "blocker_score"
]
""")

    # Cell 5: Evaluation Metric: Macro F0.5
    add_md("""## 4. Competition Macro $F_{0.5}$ Evaluation Metric
Computes precision, recall, and $F_{0.5}$ ($\beta = 0.5$, Precision weighted 2x Recall) strictly per $S_1$ entity and averages across all entities.
""")

    add_code("""def evaluate_macro_f05(pred_dict, gt_dict, beta=0.5):
    beta_sq = beta ** 2
    f_scores, precisions, recalls = [], [], []

    for s1_id, gt_set in gt_dict.items():
        p_set = pred_dict.get(s1_id, set())

        tp = len(p_set & gt_set)
        fp = len(p_set - gt_set)
        fn = len(gt_set - p_set)

        if len(gt_set) == 0:
            if len(p_set) == 0:
                prec, rec, f_score = 1.0, 1.0, 1.0
            else:
                prec, rec, f_score = 0.0, 1.0, 0.0
        else:
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom = (beta_sq * prec) + rec
            f_score = ((1 + beta_sq) * prec * rec / denom) if denom > 0 else 0.0

        f_scores.append(f_score)
        precisions.append(prec)
        recalls.append(rec)

    return {
        "macro_f05": float(np.mean(f_scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls))
    }
""")

    # Cell 6: Model Training & Threshold Optimization
    add_md("""## 5. Model Training & Optimal Threshold Sweep
Loads training data, trains LightGBM classifier with early stopping, sweeps thresholds $T \\in [0.50, 0.95]$ to maximize Macro $F_{0.5}$.
""")

    add_code("""print("=== STAGE 1: TRAINING MODEL ===")
t_train_start = time.time()

# Load training sources
df_s1_train = pl.read_csv(TRAIN_DIR / "train_source1.tsv", separator="\\t")
df_gt = pl.read_csv(TRAIN_DIR / "train_ground_truth.tsv", separator="\\t")

gt_dict = {}
for row in df_gt.iter_rows():
    s1_id = row[0]
    m_str = row[1]
    if m_str and str(m_str).strip() and str(m_str).strip() != "nan":
        gt_dict[s1_id] = set(x.strip() for x in str(m_str).split(",") if x.strip())
    else:
        gt_dict[s1_id] = set()

# Use 80,000 S1 entities for training (65,000 train / 15,000 val)
n_sample = min(80000, len(df_s1_train))
df_s1_sample = df_s1_train.sample(n=n_sample, seed=42)
sample_s1_ids = df_s1_sample["entity_id"].to_list()
sample_gt = {sid: gt_dict[sid] for sid in sample_s1_ids}

np.random.seed(42)
shuffled_ids = np.random.permutation(sample_s1_ids)
n_tr = int(n_sample * 0.8)
train_s1_ids = set(shuffled_ids[:n_tr])
val_s1_ids = set(shuffled_ids[n_tr:])

train_gt = {sid: sample_gt[sid] for sid in train_s1_ids}
val_gt = {sid: sample_gt[sid] for sid in val_s1_ids}
print(f"Training split: {len(train_s1_ids):,} train S1, {len(val_s1_ids):,} val S1")

# Vectorize S1
df_s1_vec = vectorize_source(df_s1_sample)

# Load candidate sources for matching countries
countries = list(set(df_s1_sample["country"].to_list()))
df_s2_train = pl.read_csv(TRAIN_DIR / "train_source2.tsv", separator="\\t").filter(pl.col("country").is_in(countries))
df_s3_train = pl.read_csv(TRAIN_DIR / "train_source3.tsv", separator="\\t").filter(pl.col("country").is_in(countries))
df_cand_train = pl.concat([df_s2_train, df_s3_train])
del df_s2_train, df_s3_train
gc.collect()

df_cand_vec = vectorize_source(df_cand_train)
del df_cand_train
gc.collect()

# Candidate metadata lookup dictionary
cand_records = {}
for row in df_cand_vec.select([
    "entity_id", "business_name", "business_address", "country",
    "name_clean", "addr_clean", "name_no_suffix", "postal", "house_num"
]).iter_rows():
    cid = row[0]
    name_c = row[4]
    addr_c = row[5]
    cand_records[cid] = {
        "country": row[3] or "",
        "name_alnum": name_c,
        "name_no_legal_suffix": row[6] or "",
        "name_token_set": set(name_c.split()) if name_c else set(),
        "name_char_3gram": {name_c[i:i+3] for i in range(len(name_c)-2)} if len(name_c)>=3 else set(),
        "name_char_4gram": {name_c[i:i+4] for i in range(len(name_c)-3)} if len(name_c)>=4 else set(),
        "address_alnum": addr_c,
        "address_tokens": addr_c.split() if addr_c else [],
        "postal_code": row[7],
        "house_number": row[8],
        "address_char_3gram": {addr_c[i:i+3] for i in range(len(addr_c)-2)} if len(addr_c)>=3 else set()
    }

indexes = build_fast_inverted_indexes(df_cand_vec, max_posting_size=350)
del df_cand_vec
gc.collect()

# Build pairwise training and validation sets
def build_features(s1_ids_set, gt_lookup, max_cands=30):
    X, y, pairs = [], [], []
    df_sub = df_s1_vec.filter(pl.col("entity_id").is_in(list(s1_ids_set)))

    for row in df_sub.select([
        "entity_id", "country_norm", "name_clean", "name_no_suffix", "tok0", "postal", "pfx4",
        "addr_clean", "house_num"
    ]).iter_rows():
        s1_id = row[0]
        cntry = row[1]
        name_c = row[2]
        addr_c = row[7]

        s1_meta = {
            "country": cntry,
            "name_alnum": name_c,
            "name_no_legal_suffix": row[3] or "",
            "name_token_set": set(name_c.split()) if name_c else set(),
            "name_char_3gram": {name_c[i:i+3] for i in range(len(name_c)-2)} if len(name_c)>=3 else set(),
            "name_char_4gram": {name_c[i:i+4] for i in range(len(name_c)-3)} if len(name_c)>=4 else set(),
            "address_alnum": addr_c,
            "address_tokens": addr_c.split() if addr_c else [],
            "postal_code": row[5],
            "house_number": row[8],
            "address_char_3gram": {addr_c[i:i+3] for i in range(len(addr_c)-2)} if len(addr_c)>=3 else set()
        }

        s1_key_tuple = (s1_id, cntry, name_c, row[3], row[4], row[5], row[6])
        cand_list = query_candidates_fast(s1_key_tuple, indexes, max_return=max_cands)
        retrieved_ids = {c for c, _ in cand_list}

        true_pos = gt_lookup.get(s1_id, set())

        # Always include true positives
        for tp_id in true_pos:
            if tp_id in cand_records and tp_id not in retrieved_ids:
                cand_list.append((tp_id, 1))

        for cid, b_score in cand_list:
            if cid not in cand_records:
                continue
            c_meta = cand_records[cid]
            feat_vec = compute_pair_features(s1_meta, c_meta, b_score)
            label = 1 if cid in true_pos else 0

            X.append(feat_vec)
            y.append(label)
            pairs.append((s1_id, cid))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), pairs

print("Building feature matrices for training and validation...")
X_train, y_train, _ = build_features(train_s1_ids, train_gt)
X_val, y_val, val_pairs = build_features(val_s1_ids, val_gt)

print(f"X_train: {X_train.shape}, pos={int(y_train.sum()):,}, neg={int((1-y_train).sum()):,}")
print(f"X_val:   {X_val.shape}, pos={int(y_val.sum()):,}, neg={int((1-y_val).sum()):,}")

# Train LightGBM model
train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=FEATURE_NAMES)

params = {
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1
}

# Enable GPU acceleration if available
try:
    import torch
    if torch.cuda.is_available():
        params['device'] = 'gpu'
        print("CUDA GPU detected - configuring LightGBM GPU acceleration.")
except Exception:
    pass

model = lgb.train(
    params,
    train_data,
    num_boost_round=600,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(period=100)]
)

# Threshold Optimization on Validation Set
val_preds_prob = model.predict(X_val)
val_auc = roc_auc_score(y_val, val_preds_prob)
val_pr_auc = average_precision_score(y_val, val_preds_prob)
print(f"Validation Pairwise AUC: {val_auc:.4f} | PR-AUC: {val_pr_auc:.4f}")

val_s1_cands = defaultdict(list)
for (s1_id, cid), prob in zip(val_pairs, val_preds_prob):
    val_s1_cands[s1_id].append((cid, prob))

best_thresh, best_f05 = 0.80, 0.0
for thresh in np.arange(0.50, 0.95, 0.05):
    preds = {
        s1_id: {cid for cid, p in val_s1_cands.get(s1_id, []) if p >= thresh}
        for s1_id in val_s1_ids
    }
    m = evaluate_macro_f05(preds, val_gt)
    print(f"Thresh {thresh:.2f} -> Macro F0.5: {m['macro_f05']:.4f} | Prec: {m['macro_precision']:.4f} | Rec: {m['macro_recall']:.4f}")
    if m['macro_f05'] > best_f05:
        best_f05 = m['macro_f05']
        best_thresh = thresh

print(f"\\n=== OPTIMAL THRESHOLD: {best_thresh:.2f} (Macro F0.5 = {best_f05:.4f}) ===")

# Free training memory
del X_train, y_train, X_val, y_val, val_pairs, indexes, cand_records, df_s1_vec, df_s1_sample
gc.collect()
""")

    # Cell 7: Full Scale Test Set Inference
    add_md("""## 6. Full-Scale Test Set Inference (Including France, US, India)
Processes all test entities in chunks to maintain strict memory efficiency, generating:
1. `matching_results.tsv` (Scored on leaderboard)
2. `candidate_pairs.tsv` (Blocking candidates)
""")

    add_code("""print("=== STAGE 2: FULL TEST INFERENCE ===")
t_test_start = time.time()

# Load test sources
print("Loading Test Source 1, Source 2, Source 3...")
df_test_s1 = pl.read_csv(TEST_DIR / "test_source1.tsv", separator="\\t")
df_test_s2 = pl.read_csv(TEST_DIR / "test_source2.tsv", separator="\\t")
df_test_s3 = pl.read_csv(TEST_DIR / "test_source3.tsv", separator="\\t")

print(f"Test S1 entities: {len(df_test_s1):,}")
print(f"Test S2 entities: {len(df_test_s2):,}")
print(f"Test S3 entities: {len(df_test_s3):,}")

# Vectorize test candidate sources (S2 and S3)
df_test_cands = pl.concat([df_test_s2, df_test_s3])
del df_test_s2, df_test_s3
gc.collect()

print("Vectorizing test candidate sources...")
df_test_cands_vec = vectorize_source(df_test_cands)
del df_test_cands
gc.collect()

test_cand_records = {}
for row in df_test_cands_vec.select([
    "entity_id", "business_name", "business_address", "country",
    "name_clean", "addr_clean", "name_no_suffix", "postal", "house_num"
]).iter_rows():
    cid = row[0]
    name_c = row[4]
    addr_c = row[5]
    test_cand_records[cid] = {
        "country": row[3] or "",
        "name_alnum": name_c,
        "name_no_legal_suffix": row[6] or "",
        "name_token_set": set(name_c.split()) if name_c else set(),
        "name_char_3gram": {name_c[i:i+3] for i in range(len(name_c)-2)} if len(name_c)>=3 else set(),
        "name_char_4gram": {name_c[i:i+4] for i in range(len(name_c)-3)} if len(name_c)>=4 else set(),
        "address_alnum": addr_c,
        "address_tokens": addr_c.split() if addr_c else [],
        "postal_code": row[7],
        "house_number": row[8],
        "address_char_3gram": {addr_c[i:i+3] for i in range(len(addr_c)-2)} if len(addr_c)>=3 else set()
    }

print("Building test candidate inverted indexes...")
test_indexes = build_fast_inverted_indexes(df_test_cands_vec, max_posting_size=350)
del df_test_cands_vec
gc.collect()

# Vectorize Test S1
print("Vectorizing Test S1 reference entities...")
df_test_s1_vec = vectorize_source(df_test_s1)

test_matching_file = OUTPUT_DIR / "matching_results.tsv"
test_candidate_file = OUTPUT_DIR / "candidate_pairs.tsv"

# Write TSV files line-by-line in batches
batch_size = 50000
total_s1 = len(df_test_s1_vec)
num_batches = math.ceil(total_s1 / batch_size)

print(f"Beginning test inference in {num_batches} batches...")

with open(test_matching_file, "w", encoding="utf-8", newline="") as f_match, \\
     open(test_candidate_file, "w", encoding="utf-8", newline="") as f_cand:

    # Header rows
    f_match.write("source1_entity_id\\tmatched_entity_ids\\n")
    f_cand.write("source1_entity_id\\tcandidate_entity_ids\\n")

    for b_idx in range(num_batches):
        b_start = b_idx * batch_size
        b_end = min(total_s1, b_start + batch_size)
        df_batch = df_test_s1_vec.slice(b_start, b_end - b_start)

        batch_X = []
        batch_pairs = []
        s1_candidate_map = {}

        # 1. Retrieve candidates for each S1 entity in batch
        for row in df_batch.select([
            "entity_id", "country_norm", "name_clean", "name_no_suffix", "tok0", "postal", "pfx4",
            "addr_clean", "house_num"
        ]).iter_rows():
            s1_id = row[0]
            cntry = row[1]
            name_c = row[2]
            addr_c = row[7]

            s1_meta = {
                "country": cntry,
                "name_alnum": name_c,
                "name_no_legal_suffix": row[3] or "",
                "name_token_set": set(name_c.split()) if name_c else set(),
                "name_char_3gram": {name_c[i:i+3] for i in range(len(name_c)-2)} if len(name_c)>=3 else set(),
                "name_char_4gram": {name_c[i:i+4] for i in range(len(name_c)-3)} if len(name_c)>=4 else set(),
                "address_alnum": addr_c,
                "address_tokens": addr_c.split() if addr_c else [],
                "postal_code": row[5],
                "house_number": row[8],
                "address_char_3gram": {addr_c[i:i+3] for i in range(len(addr_c)-2)} if len(addr_c)>=3 else set()
            }

            s1_key_tuple = (s1_id, cntry, name_c, row[3], row[4], row[5], row[6])
            cand_list = query_candidates_fast(s1_key_tuple, test_indexes, max_return=30)
            s1_candidate_map[s1_id] = [c for c, _ in cand_list]

            for cid, b_score in cand_list:
                if cid not in test_cand_records:
                    continue
                feat_vec = compute_pair_features(s1_meta, test_cand_records[cid], b_score)
                batch_X.append(feat_vec)
                batch_pairs.append((s1_id, cid))

        # 2. Score candidates if any were retrieved
        s1_matched_map = defaultdict(list)
        if batch_X:
            probs = model.predict(np.array(batch_X, dtype=np.float32))
            for (s1_id, cid), p in zip(batch_pairs, probs):
                if p >= best_thresh:
                    s1_matched_map[s1_id].append((cid, p))

        # 3. Write batch results to disk
        for row in df_batch.select(["entity_id"]).iter_rows():
            s1_id = row[0]
            # Matching results
            matches = [cid for cid, _ in sorted(s1_matched_map.get(s1_id, []), key=lambda x: x[1], reverse=True)]
            # Deduplicate preserving order
            matches_unique = list(dict.fromkeys(matches))
            m_str = ",".join(matches_unique)
            f_match.write(f"{s1_id}\\t{m_str}\\n")

            # Candidate pairs
            cands = list(dict.fromkeys(s1_candidate_map.get(s1_id, [])))
            c_str = ",".join(cands)
            f_cand.write(f"{s1_id}\\t{c_str}\\n")

        print(f"Batch {b_idx+1}/{num_batches} processed ({b_end:,}/{total_s1:,} entities).")
        gc.collect()

print(f"Test inference completed in {time.time()-t_test_start:.2f}s!")
""")

    # Cell 8: Validation and Packaging
    add_md("""## 7. Submission Validation & Compression
Runs the official competition `validate_submission.py` to ensure complete compliance.
""")

    add_code("""# Run validate_submission if validator script exists
val_script = DATA_DIR / "validate_submission.py"
if not val_script.exists():
    val_script = DATA_DIR.parent / "utils" / "validate_submission.py"

if val_script.exists():
    import subprocess
    cmd = [
        sys.executable, str(val_script),
        "--matching", str(test_matching_file),
        "--candidate", str(test_candidate_file),
        "--test-dir", str(TEST_DIR)
    ]
    print(f"Running validator: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("VALIDATOR OUTPUT:\\n", res.stdout)
    if res.stderr:
        print("VALIDATOR STDERR:\\n", res.stderr)
    print(f"Validator Return Code: {res.returncode}")
else:
    print("Running basic sanity checks...")
    with open(test_matching_file, "r", encoding="utf-8") as f:
        m_lines = sum(1 for _ in f) - 1
    print(f"Matching file lines (excluding header): {m_lines:,} (expected: {len(df_test_s1):,})")
    assert m_lines == len(df_test_s1), f"Mismatch in row count: {m_lines} vs {len(df_test_s1)}"
    print("Sanity check PASSED successfully!")

# Compress files into submission.zip
zip_path = OUTPUT_DIR / "submission.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(test_matching_file, arcname="matching_results.tsv")
    zipf.write(test_candidate_file, arcname="candidate_pairs.tsv")

print(f"\\n=== SUBMISSION READY: {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MB) ===")
""")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_file = Path("kaggle/kaggle_pipeline.ipynb")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Created {out_file} successfully!")

if __name__ == "__main__":
    create_kaggle_notebook()
