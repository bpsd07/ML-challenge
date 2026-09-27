"""
Blocking Recall Efficiency Benchmark
=====================================
Target: >95% blocking recall (fraction of true matches recovered by blocker).

Usage:
  python benchmark_recall.py --data_dir <path_to_dataset>

If dataset not found, runs a synthetic smoke test to validate logic.
Reports:
  - Blocking recall % per strategy
  - Cumulative union recall curve
  - Mean/P95/P99 candidates per S1
  - Whether 95% target is met
  - Which strategies contribute most
"""

import sys
import os
import re
import time
import math
import random
import unicodedata
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import argparse
import numpy as np

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=None,
                    help='Path to dataset root (contains train/ subdir with *.tsv files)')
parser.add_argument('--sample', type=int, default=5000,
                    help='Number of S1 entities to sample for recall measurement')
parser.add_argument('--synthetic', action='store_true',
                    help='Force synthetic data (for CI/smoke testing)')
args, _ = parser.parse_known_args()

# ── Normalisation (self-contained copy) ──────────────────────────────────────
_SUFFIXES = [
    "PRIVATE LIMITED","PVT LTD","PVT LIMITED",
    "LLC","INC","INCORPORATED","CORP","CORPORATION","LTD","LIMITED",
    "CO","COMPANY","LLP","PLC","HOLDINGS","GROUP","ENTERPRISES","INDUSTRIES",
    "PVT","PRIVATE","SARL","SAS","SA","EURL","SNC","SCI","SCOP","GIE",
    "GMBH","AG","SPA","SRL","SL",
]
_SUFFIX_RE = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in sorted(_SUFFIXES, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)
_PHONETIC = [('ee','i'),('oo','u'),('sh','s'),('ch','c'),('dh','d'),
             ('th','t'),('bh','b'),('ph','f'),('kh','k'),('gh','g'),('w','v')]
GENERIC_STOPS = {
    'the','and','of','in','for','to','at','by','on','a','an','co','company',
    'services','service','enterprises','enterprise','group','holdings',
    'solutions','systems','associates','consultants','consulting','management',
    'international','national','india','us','usa','france','corporation',
    'limited','private','llc','inc','pvt','ltd','sa','sas','sarl',
}

def _strip_accents(t):
    return ''.join(c for c in unicodedata.normalize('NFKD', t) if not unicodedata.combining(c))

def _consonant_skeleton(w):
    w = w.lower()
    for pat, rep in _PHONETIC:
        w = w.replace(pat, rep)
    first = w[0] if w else ''
    rest = re.sub(r'[aeiouy\s]', '', w[1:])
    return re.sub(r'(.)\1+', r'\1', first + rest)

def _char_ngrams(text, n):
    t = re.sub(r'\s+', ' ', text).strip()
    if len(t) < n:
        return frozenset([t]) if t else frozenset()
    return frozenset(t[i:i+n] for i in range(len(t)-n+1))

def normalize_name(raw):
    raw = raw or ''
    asc = _strip_accents(raw.lower().strip())
    alnum = re.sub(r'[^a-z0-9\s]', ' ', asc)
    alnum = re.sub(r'\s+', ' ', alnum).strip()
    nosuf = _SUFFIX_RE.sub(' ', alnum)
    nosuf = re.sub(r'\s+', ' ', nosuf).strip()
    toks = [t for t in alnum.split() if t]
    toks_nosuf = [t for t in nosuf.split() if t]
    distinct = [t for t in toks_nosuf if t not in GENERIC_STOPS and len(t) >= 4]
    cskel = [_consonant_skeleton(t) for t in toks_nosuf if len(t) >= 3]
    initials = ''.join(t[0] for t in toks if t)
    c3 = _char_ngrams(nosuf, 3)
    c4 = _char_ngrams(nosuf, 4)
    suffix_tokens = set(alnum.split()) - set(nosuf.split())
    return {
        'alnum': alnum, 'no_suffix': nosuf, 'tokens': toks,
        'tokens_nosuf': toks_nosuf, 'sorted_tokens': ' '.join(sorted(toks)),
        'sorted_nosuf': ' '.join(sorted(toks_nosuf)),
        'distinctive': distinct, 'consonant_skeletons': cskel,
        'initials': initials, 'c3grams': c3, 'c4grams': c4,
        'pfx4': alnum[:4] if len(alnum) >= 4 else alnum,
        'pfx6': alnum[:6] if len(alnum) >= 6 else alnum,
        'n_toks': len(toks), 'has_suffix': alnum != nosuf,
        'suffix_str': ' '.join(sorted(suffix_tokens)),
    }

def normalize_address(raw):
    raw = raw or ''
    asc = _strip_accents(raw.lower().strip())
    alnum = re.sub(r'[^a-z0-9\s]', ' ', asc)
    alnum = re.sub(r'\s+', ' ', alnum).strip()
    toks = [t for t in alnum.split() if t]
    postal_m = re.search(r'\b([0-9]{5,6})\b', raw)
    postal = postal_m.group(1) if postal_m else ''
    house_m = re.match(r'^(\d+)', alnum)
    house = house_m.group(1) if house_m else ''
    nums = [t for t in toks if re.fullmatch(r'\d+', t)]
    alphanum_house = re.findall(r'\b[a-z0-9]{1,3}[-/][a-z0-9]{1,5}\b', alnum)
    slash = re.findall(r'\b\d+/\d+\b', alnum)
    addr_distinct = [t for t in toks
                     if len(t) >= 4 and not re.fullmatch(r'\d+', t) and t not in GENERIC_STOPS]
    nums_sorted = '_'.join(sorted(nums[:6]))
    c3 = _char_ngrams(alnum[:80], 3)
    return {
        'alnum': alnum, 'tokens': toks, 'postal': postal, 'house_num': house,
        'numeric_tokens': nums, 'alphanum_house': alphanum_house,
        'slash_compounds': slash, 'addr_distinctive': addr_distinct,
        'nums_sorted': nums_sorted, 'c3grams': c3, 'n_toks': len(toks),
    }

# ── Blocker (self-contained, per-strategy tracking) ───────────────────────────
MAX_POSTING = 15_000

class TrackingBlocker:
    """
    Blocker that tracks which strategy recovers each true link.
    Used to produce per-strategy recall and cumulative union curve.
    """
    STRATEGIES = [
        'B01_exact_name','B02_no_suffix','B03_sorted_toks','B04_sorted_nosuf',
        'B05_first2','B06_tok0','B07_tok1','B08_initials','B09_pfx4','B10_pfx6',
        'B11_dist0','B11_dist1','B11_dist2','B11_dist3',
        'B12_dist_pair01','B13_skel0','B13_skel1','B13_skel2','B14_skel_pair01',
        'B15_3gram_sig','B16_4gram_sig','B17_nosuf_lenb',
        'B18_house_tok0','B19_postal_pfx4','B20_postal_house','B21_numsig_tok0',
        'B22_addr_dist0','B23_addr_dist01','B24_addr_name_joint',
        'B25_slash_compound','B26_alphanum_house','B27_addr_3gram','B28_addr_lead2',
        'B29_num_toks','B30_nosuf_postal','B31_skel_house','B32_dist0_postal',
        'B33_phonetic_dist0','B34_phonetic_tok0','B35_phonetic_pair01',
        'B36_sortnosuf_addr','B37_tok0_lenb','B38_tok2','B39_tok3',
    ]

    def __init__(self, max_posting=MAX_POSTING):
        self.max_posting = max_posting
        self.idx = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.n_indexed = 0

    def _add(self, s, country, key, cid):
        pl = self.idx[s][country][key]
        if len(pl) < self.max_posting:
            pl.append(cid)

    def index_record(self, cid, country, nn, na):
        cntry = country or 'UNKNOWN'
        a = lambda s, k: self._add(s, cntry, k, cid)
        self._apply(nn, na, cntry, a)
        self.n_indexed += 1

    def query_per_strategy(self, country, nn, na):
        """Returns {strategy -> set_of_candidate_ids}"""
        cntry = country or 'UNKNOWN'
        per_strat = defaultdict(set)

        def collect(s, key):
            for cid in self.idx[s][cntry].get(key, []):
                per_strat[s].add(cid)

        self._apply(nn, na, cntry, collect)
        return per_strat

    def _apply(self, nn, na, cntry, fn):
        alnum = nn['alnum']; nosuf = nn['no_suffix']; toks = nn['tokens']
        sorted_t = nn['sorted_tokens']; sorted_ns = nn['sorted_nosuf']
        dists = nn['distinctive']; cskel = nn['consonant_skeletons']
        initials = nn['initials']; c3 = nn['c3grams']; c4 = nn['c4grams']
        pfx4 = nn['pfx4']; pfx6 = nn['pfx6']
        na_postal = na['postal']; na_house = na['house_num']
        na_nums = na['numeric_tokens']; na_dist = na['addr_distinctive']
        na_slash = na['slash_compounds']; na_ah = na['alphanum_house']
        na_numsig = na['nums_sorted']; na_c3 = na['c3grams']; na_toks = na['tokens']

        if alnum: fn('B01_exact_name', alnum)
        if nosuf and nosuf != alnum: fn('B02_no_suffix', nosuf)
        if sorted_t: fn('B03_sorted_toks', sorted_t)
        if sorted_ns and sorted_ns != sorted_t: fn('B04_sorted_nosuf', sorted_ns)
        if len(toks) >= 2: fn('B05_first2', f"{toks[0]} {toks[1]}")
        if toks and len(toks[0]) >= 3: fn('B06_tok0', toks[0])
        if len(toks) >= 2 and len(toks[1]) >= 3: fn('B07_tok1', toks[1])
        if len(initials) >= 2: fn('B08_initials', initials)
        if pfx4: fn('B09_pfx4', pfx4)
        if pfx6: fn('B10_pfx6', pfx6)
        for i, dt in enumerate(dists[:4]): fn(f'B11_dist{i}', dt)
        if len(dists) >= 2: fn('B12_dist_pair01', f"{dists[0]}_{dists[1]}")
        for i, sk in enumerate(cskel[:3]):
            if sk: fn(f'B13_skel{i}', sk)
        if len(cskel) >= 2 and cskel[0] and cskel[1]:
            fn('B14_skel_pair01', f"{cskel[0]}_{cskel[1]}")
        if c3: fn('B15_3gram_sig', '_'.join(sorted(c3)[:3]))
        if c4: fn('B16_4gram_sig', '_'.join(sorted(c4)[:2]))
        if nosuf:
            lb = len(nosuf) // 5
            fn('B17_nosuf_lenb', f"{nosuf[:6]}_{lb}")
        if na_house and toks: fn('B18_house_tok0', f"{na_house}_{toks[0][:4]}")
        if na_postal and pfx4: fn('B19_postal_pfx4', f"{na_postal}_{pfx4}")
        if na_postal and na_house: fn('B20_postal_house', f"{na_postal}_{na_house}")
        if na_numsig and toks: fn('B21_numsig_tok0', f"{na_numsig}_{toks[0][:4]}")
        if len(na_dist) >= 1: fn('B22_addr_dist0', na_dist[0])
        if len(na_dist) >= 2: fn('B23_addr_dist01', f"{na_dist[0]}_{na_dist[1]}")
        if len(na_dist) >= 1 and pfx4: fn('B24_addr_name_joint', f"{na_dist[0]}_{pfx4}")
        for sc in na_slash[:2]: fn('B25_slash_compound', f"{sc}_{pfx4}")
        for ah in na_ah[:2]: fn('B26_alphanum_house', ah)
        if na_c3: fn('B27_addr_3gram', '_'.join(sorted(na_c3)[:3]))
        if len(na_toks) >= 2: fn('B28_addr_lead2', f"{na_toks[0]} {na_toks[1]}")
        if na_nums: fn('B29_num_toks', '_'.join(sorted(na_nums[:4])))
        if nosuf and na_postal: fn('B30_nosuf_postal', f"{nosuf[:8]}_{na_postal}")
        if cskel and cskel[0] and na_house: fn('B31_skel_house', f"{cskel[0]}_{na_house}")
        if dists and na_postal: fn('B32_dist0_postal', f"{dists[0]}_{na_postal}")
        if dists:
            ph = _consonant_skeleton(dists[0])
            if ph: fn('B33_phonetic_dist0', ph)
        if toks:
            ph0 = _consonant_skeleton(toks[0])
            if ph0: fn('B34_phonetic_tok0', ph0)
        if len(toks) >= 2:
            ph_p = f"{_consonant_skeleton(toks[0])}_{_consonant_skeleton(toks[1])}"
            if ph_p != '_': fn('B35_phonetic_pair01', ph_p)
        if sorted_ns and na_dist: fn('B36_sortnosuf_addr', f"{sorted_ns[:10]}_{na_dist[0]}")
        if toks:
            lb2 = len(alnum) // 4
            fn('B37_tok0_lenb', f"{toks[0][:5]}_{lb2}")
        if len(toks) >= 3: fn('B38_tok2', toks[2])
        if len(toks) >= 4: fn('B39_tok3', toks[3])


# ── Data loading helpers ───────────────────────────────────────────────────────
def load_tsv(path):
    """Load TSV as list of dicts."""
    rows = []
    with open(path, encoding='utf-8', errors='replace') as f:
        reader = __import__('csv').DictReader(f, delimiter='\t')
        for row in reader:
            rows.append(row)
    return rows


def find_dataset(data_dir_arg):
    """Try to locate the dataset directory."""
    candidates = []
    if data_dir_arg:
        candidates.append(Path(data_dir_arg))
    candidates += [
        Path('./student_resource/dataset'),
        Path('../student_resource/dataset'),
        Path('/kaggle/input/amazon-ml-challenge-2026/dataset'),
    ]
    # Also check env variable
    env_dir = os.environ.get('DATASET_DIR')
    if env_dir:
        candidates.insert(0, Path(env_dir))

    for p in candidates:
        if (p / 'train' / 'train_source1.tsv').exists():
            return p / 'train'
        if (p / 'train_source1.tsv').exists():
            return p
    return None


# ── Synthetic data generator ──────────────────────────────────────────────────
def make_synthetic_dataset(n_s1=2000, n_s2=10000, noise_level=0.3):
    """
    Generate synthetic entity resolution dataset.
    Each S1 entity has 1-4 matches in S2 with noise applied.
    Includes singletons (no matches).
    """
    random.seed(42)
    names_pool = [
        "Acme Global LLC", "Best Tech Solutions", "Sunrise Trading Co",
        "Global Ventures Pvt Ltd", "Star Industries Inc", "Premier Services Ltd",
        "Blue Ocean Corp", "Mountain Peak Enterprises", "Rapid Growth LLC",
        "Silver Lake Holdings", "Green Valley Industries", "North Star Consulting",
        "Pacific Rim Trading", "Eagle Eye Technologies", "Golden Gate Partners",
        "Red Rock Manufacturing", "Crystal Clear Solutions", "Iron Bridge Corp",
        "Maple Leaf Services", "Desert Wind Enterprises", "Summit Group",
        "Horizon Technologies Ltd", "Cascade Systems Inc", "Pinnacle Ventures",
        "Metro Services SARL", "Elite Solutions SAS", "Prestige Group SA",
        "Alliance Commerce", "Dynamic Consultants", "Apex Industries",
    ] * (n_s1 // 20 + 5)

    addresses_pool = [
        "123 Main Street, New York, NY 10001",
        "456 Oak Avenue, Los Angeles, CA 90001",
        "789 Elm Road, Chicago, IL 60601",
        "101 Maple Drive, Houston, TX 77001",
        "202 Pine Blvd, Phoenix, AZ 85001",
        "303 Cedar Lane, Philadelphia, PA 19101",
        "404 Birch Way, San Antonio, TX 78201",
        "505 Walnut St, San Diego, CA 92101",
        "606 Chestnut Ave, Dallas, TX 75201",
        "707 Spruce Court, San Jose, CA 95101",
        "12 Rue de la Paix, Paris 75001",
        "45 Boulevard Haussmann, Paris 75009",
        "28 Avenue des Champs, Lyon 69001",
        "MG Road, Bangalore 560001",
        "Connaught Place, New Delhi 110001",
        "Park Street, Kolkata 700001",
        "Jubilee Hills, Hyderabad 500033",
    ] * (n_s1 // 10 + 5)

    noise_variants = {
        'suffix': [(' LLC', ''), (' Inc', ''), (' Ltd', ''), (' Corp', ''), (' Co', '')],
        'abbrev': [('Street', 'St'), ('Avenue', 'Ave'), ('Boulevard', 'Blvd'), ('Road', 'Rd')],
        'typo':   [('i', 'y'), ('o', '0'), ('e', 'a')],
    }

    def apply_noise(text, level):
        if random.random() > level:
            return text
        choice = random.choice(['suffix', 'abbrev', 'typo'])
        variants = noise_variants[choice]
        for old, new in variants:
            if old in text:
                return text.replace(old, new, 1)
        return text

    countries = ['US', 'IN', 'FR']
    s1_data = []
    s2_data = []
    gt = {}
    s2_id_counter = [0]

    for i in range(n_s1):
        s1id = f'S1-{i:07d}'
        name = random.choice(names_pool[:len(names_pool)//2])
        addr = random.choice(addresses_pool)
        country = random.choice(countries)
        s1_data.append({'entity_id': s1id, 'business_name': name,
                        'business_address': addr, 'country': country})

        # 15% singletons
        if random.random() < 0.15:
            gt[s1id] = set()
            continue

        # 1-4 matches
        n_matches = random.randint(1, 4)
        matches = set()
        for _ in range(n_matches):
            s2id = f'S2-{s2_id_counter[0]:07d}'
            s2_id_counter[0] += 1
            n_name = apply_noise(name, noise_level)
            n_addr = apply_noise(addr, noise_level)
            s2_data.append({'entity_id': s2id, 'business_name': n_name,
                            'business_address': n_addr, 'country': country})
            matches.add(s2id)
        gt[s1id] = matches

    # Fill S2 with distractors up to n_s2
    while len(s2_data) < n_s2:
        s2id = f'S2-{s2_id_counter[0]:07d}'
        s2_id_counter[0] += 1
        s2_data.append({
            'entity_id': s2id,
            'business_name': random.choice(names_pool),
            'business_address': random.choice(addresses_pool),
            'country': random.choice(countries)
        })

    random.shuffle(s2_data)
    print(f"Synthetic dataset: {n_s1} S1, {len(s2_data)} S2, {sum(1 for v in gt.values() if v)} matched, {sum(1 for v in gt.values() if not v)} singletons")
    return s1_data, s2_data, gt


# ── Main benchmark ─────────────────────────────────────────────────────────────
def run_benchmark(s1_data, s2_data, gt, sample_size):
    print(f"\n{'='*70}")
    print(f"BLOCKING RECALL EFFICIENCY BENCHMARK")
    print(f"  S1 entities:      {len(s1_data):,}")
    print(f"  S2 candidates:    {len(s2_data):,}")
    print(f"  GT pairs:         {sum(len(v) for v in gt.values()):,}")
    print(f"  Singletons:       {sum(1 for v in gt.values() if not v):,}")
    print(f"  Sample size:      {sample_size:,}")
    print(f"{'='*70}\n")

    # Build blocker from full S2
    print("Building blocker from S2...")
    blocker = TrackingBlocker(max_posting=MAX_POSTING)
    t0 = time.time()
    for row in s2_data:
        cid = row['entity_id']
        nn  = normalize_name(row.get('business_name', ''))
        na  = normalize_address(row.get('business_address', ''))
        blocker.index_record(cid, row.get('country', 'UNKNOWN'), nn, na)
    print(f"  Indexed {blocker.n_indexed:,} candidates in {time.time()-t0:.1f}s")

    # Sample S1 for recall measurement
    random.seed(42)
    eval_s1 = random.sample(s1_data, min(sample_size, len(s1_data)))
    # Only those with at least one true match (singletons trivially "100%" recall)
    eval_with_gt = [r for r in eval_s1 if gt.get(r['entity_id'], set())]

    print(f"  Evaluation set:   {len(eval_with_gt):,} (non-singleton S1 entities)")

    # Per-strategy recall tracking
    strategy_hits   = defaultdict(int)   # how many true links each strategy recovers alone
    total_true_links = 0
    union_recovered  = 0
    cand_counts      = []

    t0 = time.time()
    for row in eval_with_gt:
        sid     = row['entity_id']
        country = row.get('country', 'UNKNOWN')
        nn      = normalize_name(row.get('business_name', ''))
        na      = normalize_address(row.get('business_address', ''))
        gt_set  = gt.get(sid, set())

        per_strat = blocker.query_per_strategy(country, nn, na)
        union_all = set().union(*per_strat.values()) if per_strat else set()

        total_true_links += len(gt_set)
        union_recovered  += len(gt_set & union_all)
        cand_counts.append(len(union_all))

        for strat, cand_set in per_strat.items():
            strategy_hits[strat] += len(gt_set & cand_set)

    elapsed = time.time() - t0

    # Overall metrics
    recall = union_recovered / max(total_true_links, 1)
    mean_cands = float(np.mean(cand_counts)) if cand_counts else 0
    p95_cands  = float(np.percentile(cand_counts, 95)) if cand_counts else 0
    p99_cands  = float(np.percentile(cand_counts, 99)) if cand_counts else 0

    # Per-strategy recall
    strat_recall = {s: strategy_hits[s] / max(total_true_links, 1) for s in TrackingBlocker.STRATEGIES}

    # Cumulative union recall (greedy, best-first)
    sorted_strats = sorted(strat_recall.items(), key=lambda x: x[1], reverse=True)

    # Print results
    print(f"\n{'='*70}")
    print(f"OVERALL RESULTS")
    print(f"  Union Blocking Recall:   {recall*100:.2f}%  ({'OK - TARGET MET' if recall >= 0.95 else 'BELOW 95% TARGET'})")
    print(f"  True links probed:       {total_true_links:,}")
    print(f"  True links recovered:    {union_recovered:,}")
    print(f"  Missed:                  {total_true_links - union_recovered:,}")
    print(f"  Mean candidates/S1:      {mean_cands:.0f}")
    print(f"  P95 candidates/S1:       {p95_cands:.0f}")
    print(f"  P99 candidates/S1:       {p99_cands:.0f}")
    print(f"  Elapsed:                 {elapsed:.1f}s")
    print(f"{'='*70}")

    print(f"\nPER-STRATEGY RECALL (top 20 by individual recall):")
    print(f"{'Strategy':<28}  {'Recall':>8}  {'Links':>8}")
    print(f"{'-'*28}  {'-'*8}  {'-'*8}")
    for s, r in sorted_strats[:20]:
        print(f"  {s:<26}  {r*100:>7.2f}%  {strategy_hits[s]:>7,}")

    print(f"\nCUMULATIVE UNION RECALL (greedy, best-first):")
    print(f"{'Strategies Added':<35}  {'Cumul Recall':>13}  {'Delta':>8}")
    print(f"{'-'*35}  {'-'*13}  {'-'*8}")
    seen = set()
    cumul = 0
    for i, (s, r) in enumerate(sorted_strats):
        prev = cumul
        cumul = r if i == 0 else cumul  # approximate with individual strat recall
        # For accurate cumulative, re-query union
    # Accurate cumulative: re-run union queries adding strategies one by one
    cumul_hits = set()
    prev_recall = 0.0
    cumul_per_entity = [set() for _ in eval_with_gt]
    gt_per_entity    = [gt.get(r['entity_id'], set()) for r in eval_with_gt]
    strat_per_entity = []
    # Pre-compute per-entity per-strategy
    print("  (Computing accurate cumulative union...)")
    all_per_entity = []
    for row in eval_with_gt:
        country = row.get('country', 'UNKNOWN')
        nn = normalize_name(row.get('business_name', ''))
        na = normalize_address(row.get('business_address', ''))
        all_per_entity.append(blocker.query_per_strategy(country, nn, na))

    cumul_recovered = [0] * len(eval_with_gt)
    union_sets = [set() for _ in eval_with_gt]

    print(f"\n{'Strategy added':<28}  {'Cumul Recall':>13}  {'Delta':>8}  {'New links':>10}")
    print(f"{'-'*28}  {'-'*13}  {'-'*8}  {'-'*10}")
    total_tl = sum(len(g) for g in gt_per_entity)
    running_total = 0
    for s, _ in sorted_strats:
        new_links = 0
        for i, (per_strat, gt_set) in enumerate(zip(all_per_entity, gt_per_entity)):
            new_cands = per_strat.get(s, set()) - union_sets[i]
            new_tp    = len(gt_set & new_cands)
            new_links += new_tp
            union_sets[i].update(per_strat.get(s, set()))
        running_total += new_links
        cumul_recall = running_total / max(total_tl, 1)
        delta = cumul_recall - prev_recall
        print(f"  +{s:<26}  {cumul_recall*100:>12.2f}%  {delta*100:>+7.2f}%  {new_links:>9,}")
        prev_recall = cumul_recall
        if cumul_recall >= 0.999:
            print(f"  ... (remaining strategies add <0.1% each)")
            break

    print(f"\n{'='*70}")
    if recall >= 0.95:
        print(f"TARGET MET: Blocking recall {recall*100:.2f}% >= 95%")
    elif recall >= 0.90:
        print(f"CLOSE: {recall*100:.2f}% (need {(0.95-recall)*total_true_links:.0f} more true links)")
        print(f"SUGGESTION: Add n-gram retrieval (BM25 or MinHash LSH) for missed cases")
    else:
        print(f"BELOW TARGET: {recall*100:.2f}% - significant coverage gap")
        print(f"SUGGESTIONS:")
        print(f"  1. Lower MAX_POSTING to 20,000 to allow more keys")
        print(f"  2. Add BM25/TF-IDF approximate retrieval as catch-all")
        print(f"  3. Add more phonetic strategies (Metaphone, Jaro-Winkler pre-filter)")
    print(f"{'='*70}\n")

    return recall


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_dir = find_dataset(args.data_dir)

    if train_dir is None or args.synthetic:
        if not args.synthetic:
            print("Dataset not found locally. Running SYNTHETIC smoke test.")
            print("Pass --data_dir <path> to use real data.\n")
        else:
            print("Running SYNTHETIC benchmark (--synthetic flag).\n")
        s1_data, s2_data, gt = make_synthetic_dataset(n_s1=2000, n_s2=15000, noise_level=0.25)
    else:
        print(f"Loading REAL dataset from: {train_dir}")
        s1_rows = load_tsv(train_dir / 'train_source1.tsv')
        s2_rows = load_tsv(train_dir / 'train_source2.tsv')
        gt_rows = load_tsv(train_dir / 'train_ground_truth.tsv')

        s1_data = [{'entity_id': r['entity_id'],
                    'business_name': r.get('business_name', ''),
                    'business_address': r.get('business_address', ''),
                    'country': r.get('country', 'UNKNOWN')} for r in s1_rows]

        s2_data = [{'entity_id': r['entity_id'],
                    'business_name': r.get('business_name', ''),
                    'business_address': r.get('business_address', ''),
                    'country': r.get('country', 'UNKNOWN')} for r in s2_rows]

        gt = {}
        for row in gt_rows:
            sid = row['entity_id']
            m   = row.get('match_ids', row.get('matched_entity_ids', ''))
            if m and str(m).strip() not in ('', 'nan', 'NULL'):
                gt[sid] = set(x.strip() for x in str(m).split(',') if x.strip())
            else:
                gt[sid] = set()
        print(f"Loaded: {len(s1_data):,} S1, {len(s2_data):,} S2, GT for {len(gt):,} S1 entities")

    run_benchmark(s1_data, s2_data, gt, sample_size=args.sample)
