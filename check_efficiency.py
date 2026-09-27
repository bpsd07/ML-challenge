"""
╔══════════════════════════════════════════════════════════════════════╗
║      BLOCKING RECALL EFFICIENCY CHECK  —  Amazon ML Challenge 2026  ║
║      DROP THIS FILE INTO YOUR KAGGLE NOTEBOOK AND RUN ALL CELLS     ║
║      Target: >95% blocking recall                                    ║
╚══════════════════════════════════════════════════════════════════════╝

HOW TO USE (Kaggle):
  1. Upload this file to your Kaggle notebook session (+ icon → Upload)
  2. Run:   exec(open('check_efficiency.py').read())

HOW TO USE (local):
  python check_efficiency.py --data_dir /path/to/dataset/train

WHAT IT CHECKS:
  - Blocking recall: what % of true matches does the blocker find?
  - Per-strategy breakdown (which of the 39 strategies help most)
  - Mean / P95 / P99 candidates per S1 entity
  - Whether the 95% efficiency target is MET or MISSED
"""

import sys, re, time, random, unicodedata, argparse
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    sys.exit("numpy required:  pip install numpy")

# ─── parse args (safe inside Jupyter too) ────────────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--data_dir', default=None)
parser.add_argument('--sample',   type=int, default=5000)
args, _ = parser.parse_known_args()

# ─── Auto-detect Kaggle dataset path ─────────────────────────────────────────
_SEARCH_PATHS = [
    args.data_dir,
    "/kaggle/input/amazon-ml-challenge-2026/dataset/train",
    "/kaggle/input/amazon-ml-challenge-2026/dataset",
    "/kaggle/input/amazon-ml-challenge-2026",
    "./student_resource/dataset/train",
    "./student_resource/dataset",
    "../student_resource/dataset/train",
]

DATA_DIR = None
for _p in _SEARCH_PATHS:
    if _p and Path(_p, "train_source1.tsv").exists():
        DATA_DIR = Path(_p)
        break
    if _p and Path(_p).exists():
        # check subdirs
        for sub in [Path(_p,"train"), Path(_p)]:
            if Path(sub,"train_source1.tsv").exists():
                DATA_DIR = Path(sub)
                break
    if DATA_DIR: break

if DATA_DIR:
    print(f"[OK] Dataset found at: {DATA_DIR}")
else:
    print("[WARN] Real dataset not found — running SYNTHETIC smoke test instead.")
    print("       Pass --data_dir /path/to/train  to use real data.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALISATION  (self-contained, no imports from project)
# ═══════════════════════════════════════════════════════════════════════════════
_SUFFIXES = [
    "PRIVATE LIMITED","PVT LTD","PVT LIMITED","LLC","INC","INCORPORATED",
    "CORP","CORPORATION","LTD","LIMITED","CO","COMPANY","LLP","PLC",
    "HOLDINGS","GROUP","ENTERPRISES","INDUSTRIES","PVT","PRIVATE",
    "SARL","SAS","SA","EURL","SNC","SCI","SCOP","GIE","GMBH","AG","SPA","SRL","SL",
]
_SUFFIX_RE = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in sorted(_SUFFIXES, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)
_STOPS = {
    'the','and','of','in','for','to','at','by','on','a','an','co','company',
    'services','service','enterprises','enterprise','group','holdings','solutions',
    'systems','associates','consultants','consulting','management','international',
    'national','india','us','usa','france','corporation','limited','private',
    'llc','inc','pvt','ltd','sa','sas','sarl',
}

def _strip_acc(t):
    return ''.join(c for c in unicodedata.normalize('NFKD', t) if not unicodedata.combining(c))

def _ng(text, n):
    t = re.sub(r'\s+',' ', text).strip()
    if len(t) < n: return frozenset([t]) if t else frozenset()
    return frozenset(t[i:i+n] for i in range(len(t)-n+1))

def nn(raw):
    raw = raw or ''
    asc   = _strip_acc(raw.lower().strip())
    alnum = re.sub(r'\s+',' ', re.sub(r'[^a-z0-9\s]',' ', asc)).strip()
    nosuf = re.sub(r'\s+',' ', _SUFFIX_RE.sub(' ', alnum)).strip()
    toks  = [t for t in alnum.split() if t]
    tns   = [t for t in nosuf.split() if t]
    dist  = [t for t in tns if t not in _STOPS and len(t) >= 3]
    return dict(
        alnum=alnum, nosuf=nosuf, toks=toks, tns=tns,
        sorted_t=' '.join(sorted(toks)), sorted_ns=' '.join(sorted(tns)),
        dist=dist, init=''.join(t[0] for t in toks),
        c3=_ng(nosuf,3), c4=_ng(nosuf,4),
        pfx4=alnum[:4], pfx6=alnum[:6], ntoks=len(toks),
    )

def na(raw):
    raw   = raw or ''
    asc   = _strip_acc(raw.lower().strip())
    alnum = re.sub(r'\s+',' ', re.sub(r'[^a-z0-9\s]',' ', asc)).strip()
    toks  = [t for t in alnum.split() if t]
    pm    = re.search(r'\b([0-9]{5,6})\b', raw)
    hm    = re.match(r'^(\d+)', alnum)
    nums  = [t for t in toks if re.fullmatch(r'\d+', t)]
    dist  = [t for t in toks if len(t)>=4 and not re.fullmatch(r'\d+',t) and t not in _STOPS]
    return dict(
        alnum=alnum, toks=toks,
        postal=pm.group(1) if pm else '',
        house=hm.group(1) if hm else '',
        nums=nums, dist=dist,
        numsig='_'.join(sorted(nums[:6])),
        slash=re.findall(r'\b\d+/\d+\b', alnum),
        ah=re.findall(r'\b[a-z0-9]{1,3}[-/][a-z0-9]{1,5}\b', alnum),
        c3=_ng(alnum[:80], 3),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 39-STRATEGY BLOCKER
# ═══════════════════════════════════════════════════════════════════════════════
MAX_POSTING = 15_000

STRATEGY_NAMES = [
    'B01_exact_name','B02_no_suffix','B03_sorted_toks','B04_sorted_nosuf',
    'B05_first2','B06_tok0','B07_tok1','B08_initials','B09_pfx4','B10_pfx6',
    'B11_dist0','B11_dist1','B11_dist2','B11_dist3',
    'B12_dist_pair01','B13_exact_tok0_postal','B14_exact_tok0_house',
    'B15_3gram_sig','B16_4gram_sig','B17_nosuf_lenb',
    'B18_house_tok0','B19_postal_pfx4','B20_postal_house','B21_numsig_tok0',
    'B22_addr_dist0','B23_addr_dist01','B24_addr_name_joint',
    'B25_slash','B26_ah','B27_addr_3gram','B28_addr_lead2','B29_num_toks',
    'B30_nosuf_postal','B31_exact_dist0_house','B32_dist0_postal',
    'B33_exact_tok0_tok1','B34_exact_sorted_first2','B35_exact_nosuf_house',
    'B36_sortns_addr','B37_tok0_lenb','B38_tok2','B39_tok3',
]

class Blocker:
    def __init__(self, max_p=MAX_POSTING):
        self.max_p = max_p
        self.idx   = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.n     = 0

    def _add(self, s, cntry, key, cid):
        L = self.idx[s][cntry][key]
        if len(L) < self.max_p: L.append(cid)

    def index(self, cid, country, N, A):
        c = country or 'UNK'
        a = lambda s, k: self._add(s, c, k, cid)
        self._apply(N, A, c, a)
        self.n += 1

    def query(self, country, N, A):
        """Returns {strategy_name: set_of_candidate_ids}"""
        c  = country or 'UNK'
        ps = defaultdict(set)
        fn = lambda s, k: [ps[s].add(cid) for cid in self.idx[s][c].get(k, [])]
        self._apply(N, A, c, fn)
        return ps

    def _apply(self, N, A, c, fn):
        al=N['alnum']; ns=N['nosuf']; toks=N['toks']; st=N['sorted_t']
        sns=N['sorted_ns']; dist=N['dist']
        ini=N['init']; c3=N['c3']; c4=N['c4']; p4=N['pfx4']; p6=N['pfx6']
        postal=A['postal']; house=A['house']; nums=A['nums']
        adist=A['dist']; sl=A['slash']; ah=A['ah']
        numsig=A['numsig']; ac3=A['c3']; atoks=A['toks']

        if al:                          fn('B01_exact_name', al)
        if ns and ns!=al:               fn('B02_no_suffix', ns)
        if st:                          fn('B03_sorted_toks', st)
        if sns and sns!=st:             fn('B04_sorted_nosuf', sns)
        if len(toks)>=2:                fn('B05_first2', f"{toks[0]} {toks[1]}")
        if toks and len(toks[0])>=3:    fn('B06_tok0', toks[0])
        if len(toks)>=2 and len(toks[1])>=3: fn('B07_tok1', toks[1])
        if len(ini)>=2:                 fn('B08_initials', ini)
        if p4:                          fn('B09_pfx4', p4)
        if p6:                          fn('B10_pfx6', p6)
        for i,d in enumerate(dist[:4]): fn(f'B11_dist{i}', d)
        if len(dist)>=2:                fn('B12_dist_pair01', f"{dist[0]}_{dist[1]}")
        if toks and postal:             fn('B13_exact_tok0_postal', f"{toks[0]}_{postal}")
        if toks and house:              fn('B14_exact_tok0_house', f"{toks[0]}_{house}")
        if c3: fn('B15_3gram_sig', '_'.join(sorted(c3)[:3]))
        if c4: fn('B16_4gram_sig', '_'.join(sorted(c4)[:2]))
        if ns: fn('B17_nosuf_lenb', f"{ns[:6]}_{len(ns)//5}")
        if house and toks: fn('B18_house_tok0', f"{house}_{toks[0][:4]}")
        if postal and p4:  fn('B19_postal_pfx4', f"{postal}_{p4}")
        if postal and house: fn('B20_postal_house', f"{postal}_{house}")
        if numsig and toks: fn('B21_numsig_tok0', f"{numsig}_{toks[0][:4]}")
        if len(adist)>=1: fn('B22_addr_dist0', adist[0])
        if len(adist)>=2: fn('B23_addr_dist01', f"{adist[0]}_{adist[1]}")
        if len(adist)>=1 and p4: fn('B24_addr_name_joint', f"{adist[0]}_{p4}")
        for s in sl[:2]: fn('B25_slash', f"{s}_{p4}")
        for h in ah[:2]: fn('B26_ah', h)
        if ac3: fn('B27_addr_3gram', '_'.join(sorted(ac3)[:3]))
        if len(atoks)>=2: fn('B28_addr_lead2', f"{atoks[0]} {atoks[1]}")
        if nums: fn('B29_num_toks', '_'.join(sorted(nums[:4])))
        if ns and postal: fn('B30_nosuf_postal', f"{ns[:8]}_{postal}")
        if dist and house: fn('B31_exact_dist0_house', f"{dist[0]}_{house}")
        if dist and postal: fn('B32_dist0_postal', f"{dist[0]}_{postal}")
        if len(toks)>=2:
            fn('B33_exact_tok0_tok1', f"{toks[0]}_{toks[1]}")
            fn('B34_exact_sorted_first2', f"{min(toks[0], toks[1])}_{max(toks[0], toks[1])}")
        if ns and house: fn('B35_exact_nosuf_house', f"{ns[:8]}_{house}")
        if sns and adist: fn('B36_sortns_addr', f"{sns[:10]}_{adist[0]}")
        if toks: fn('B37_tok0_lenb', f"{toks[0][:5]}_{len(al)//4}")
        if len(toks)>=3: fn('B38_tok2', toks[2])
        if len(toks)>=4: fn('B39_tok3', toks[3])


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════
def load_tsv(path):
    import csv
    rows = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            rows.append(row)
    return rows

def make_synthetic(n_s1=3000, n_s2=20000, noise=0.25):
    """Generate synthetic noisy dataset for smoke testing."""
    random.seed(42)
    name_pool = [
        "Acme Global LLC","Best Tech Solutions Inc","Sunrise Trading Co",
        "Global Ventures Pvt Ltd","Star Industries Corp","Premier Services Ltd",
        "Blue Ocean Enterprises","Mountain Peak LLC","Rapid Growth Inc",
        "Silver Lake Holdings","Green Valley Industries","North Star Consulting",
        "Pacific Rim Trading Co","Eagle Eye Technologies Ltd","Golden Gate Partners",
        "Red Rock Manufacturing","Crystal Solutions Corp","Iron Bridge LLC",
        "Maple Leaf Services","Desert Wind Enterprises","Summit Group Inc",
        "Horizon Technologies","Cascade Systems","Pinnacle Ventures Ltd",
        "Metro Services SARL","Elite Solutions SAS","Prestige Group SA",
        "Alliance Commerce Inc","Dynamic Consultants","Apex Industries Pvt",
    ] * 20
    addr_pool = [
        "123 Main St New York NY 10001","456 Oak Ave Los Angeles CA 90001",
        "789 Elm Rd Chicago IL 60601","101 Maple Dr Houston TX 77001",
        "202 Pine Blvd Phoenix AZ 85001","12 Rue de la Paix Paris 75001",
        "45 Boulevard Haussmann Paris 75009","MG Road Bangalore 560001",
        "Connaught Place New Delhi 110001","Park Street Kolkata 700001",
        "Jubilee Hills Hyderabad 500033","Anna Salai Chennai 600002",
        "303 Cedar Lane Philadelphia PA 19101","28 Avenue Lyon 69001",
        "1405/28 Linking Road Mumbai 400050","Plot 80/2 Sector 15 Noida 201301",
    ] * 10
    def noisy(t, rate):
        if random.random() > rate: return t
        ops = [
            lambda x: re.sub(r'\bLLC\b','',x), lambda x: re.sub(r'\bInc\b','',x),
            lambda x: re.sub(r'\bLtd\b','',x), lambda x: x.replace('Street','St'),
            lambda x: x.replace('Avenue','Ave'), lambda x: x.replace('Road','Rd'),
            lambda x: x.replace('ii','i'), lambda x: x.replace('oo','o'),
        ]
        return random.choice(ops)(t)

    s1, s2, gt, ctr = [], [], {}, [0]
    countries = ['US','IN','FR']
    for i in range(n_s1):
        sid = f'S1-{i:07d}'; nm = random.choice(name_pool); ad = random.choice(addr_pool)
        co  = random.choice(countries)
        s1.append({'entity_id':sid,'business_name':nm,'business_address':ad,'country':co})
        if random.random() < 0.15: gt[sid]=set(); continue
        matches = set()
        for _ in range(random.randint(1,4)):
            cid = f'S2-{ctr[0]:07d}'; ctr[0]+=1
            s2.append({'entity_id':cid,'business_name':noisy(nm,noise),
                       'business_address':noisy(ad,noise),'country':co})
            matches.add(cid)
        gt[sid]=matches
    while len(s2) < n_s2:
        cid=f'S2-{ctr[0]:07d}'; ctr[0]+=1
        s2.append({'entity_id':cid,'business_name':random.choice(name_pool),
                   'business_address':random.choice(addr_pool),'country':random.choice(countries)})
    random.shuffle(s2)
    return s1, s2, gt


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════
def run(s1_data, s2_data, gt, sample):
    W = 70
    sep = '='*W
    print(f"\n{sep}")
    print("  BLOCKING RECALL EFFICIENCY CHECK  —  Amazon ML Challenge 2026")
    print(sep)
    print(f"  S1 total:           {len(s1_data):>10,}")
    print(f"  S2 candidates:      {len(s2_data):>10,}")
    print(f"  GT pairs total:     {sum(len(v) for v in gt.values()):>10,}")
    print(f"  Singletons:         {sum(1 for v in gt.values() if not v):>10,}")
    print(f"  Eval sample (S1):   {sample:>10,}")
    print(sep)

    # Build blocker from ALL of S2
    print("\nStep 1/3  Building blocker from S2...")
    B = Blocker()
    t0 = time.time()
    for row in s2_data:
        B.index(row['entity_id'], row.get('country','UNK'),
                nn(row.get('business_name','')), na(row.get('business_address','')))
    print(f"          Indexed {B.n:,} candidates in {time.time()-t0:.1f}s\n")

    # Sample non-singleton S1
    random.seed(42)
    pool = [r for r in s1_data if gt.get(r['entity_id'])]
    eval_s1 = random.sample(pool, min(sample, len(pool)))
    print(f"Step 2/3  Scoring {len(eval_s1):,} non-singleton S1 entities...")

    strat_hits  = defaultdict(int)
    total_links = 0
    union_hits  = 0
    cand_counts = []
    t0 = time.time()

    # Pre-collect per-entity per-strategy hits (needed for accurate cumulative)
    all_ps = []
    for row in eval_s1:
        sid    = row['entity_id']
        N      = nn(row.get('business_name',''))
        A      = na(row.get('business_address',''))
        co     = row.get('country','UNK')
        gt_set = gt.get(sid, set())
        ps     = B.query(co, N, A)
        union  = set().union(*ps.values()) if ps else set()
        total_links += len(gt_set)
        union_hits  += len(gt_set & union)
        cand_counts.append(len(union))
        for s, cset in ps.items():
            strat_hits[s] += len(gt_set & cset)
        all_ps.append((gt_set, ps))

    recall     = union_hits  / max(total_links, 1)
    mean_c     = sum(cand_counts)/max(len(cand_counts),1)
    cand_arr   = sorted(cand_counts)
    p95        = cand_arr[int(0.95*len(cand_arr))] if cand_arr else 0
    p99        = cand_arr[int(0.99*len(cand_arr))] if cand_arr else 0

    print(f"          Done in {time.time()-t0:.1f}s\n")

    # Results
    TARGET = 0.95
    status = "TARGET MET" if recall >= TARGET else ("CLOSE" if recall >= 0.90 else "BELOW TARGET")
    banner = f"  {status}: {recall*100:.2f}%  (need >= {TARGET*100:.0f}%)"

    print(f"Step 3/3  Computing cumulative union recall curve...\n")

    # Accurate greedy cumulative
    sorted_strats = sorted(
        [(s, strat_hits.get(s,0)/max(total_links,1)) for s in STRATEGY_NAMES],
        key=lambda x: x[1], reverse=True
    )
    union_sets  = [set() for _ in all_ps]
    running     = 0
    prev_recall = 0.0
    cum_rows    = []
    for s, _ in sorted_strats:
        added = 0
        for i, (gt_set, ps) in enumerate(all_ps):
            new = ps.get(s, set()) - union_sets[i]
            added += len(gt_set & new)
            union_sets[i].update(ps.get(s, set()))
        running += added
        cr   = running / max(total_links, 1)
        delta = cr - prev_recall
        cum_rows.append((s, cr, delta, added))
        prev_recall = cr
        if cr >= 0.9999: break

    # Print summary
    print(sep)
    print(banner)
    print(sep)
    print(f"  Union blocking recall:  {recall*100:.2f}%")
    print(f"  True links found:       {union_hits:,} / {total_links:,}")
    print(f"  Links missed:           {total_links-union_hits:,}")
    print(f"  Mean candidates/S1:     {mean_c:.0f}")
    print(f"  P95  candidates/S1:     {p95:,}")
    print(f"  P99  candidates/S1:     {p99:,}")
    print(sep)

    print(f"\n{'Strategy':<28} {'IndivRecall':>12} {'Links':>8}")
    print(f"{'-'*28} {'-'*12} {'-'*8}")
    for s, r in sorted_strats[:20]:
        hits = strat_hits.get(s,0)
        print(f"  {s:<26} {r*100:>10.2f}%  {hits:>7,}")

    print(f"\n{'Strategy added':<28} {'CumulRecall':>12} {'Delta':>8} {'NewLinks':>9}")
    print(f"{'-'*28} {'-'*12} {'-'*8} {'-'*9}")
    for s, cr, delta, added in cum_rows:
        marker = " <-- 95% crossed!" if abs(cr - TARGET) < 0.005 and delta > 0 else ""
        print(f"  +{s:<26} {cr*100:>10.2f}%  {delta*100:>+6.2f}%  {added:>8,}{marker}")

    print(f"\n{sep}")
    if recall >= TARGET:
        print(f"  RESULT:  OK  Blocking efficiency {recall*100:.2f}% >= 95%  TARGET MET")
        needed_extra = 0
    else:
        needed_extra = int((TARGET - recall) * total_links)
        print(f"  RESULT:  BELOW TARGET  ({recall*100:.2f}% < 95%)")
        print(f"           Need {needed_extra:,} more true links covered")
        print(f"  FIX:  Add BM25/TF-IDF approximate retrieval as catch-all strategy")
    print(sep)

    return recall, mean_c, p95, p99


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════
if DATA_DIR:
    print("Loading real dataset...")
    import csv

    def _load(fname):
        rows = []
        with open(DATA_DIR/fname, encoding='utf-8', errors='replace') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                rows.append(row)
        return rows

    s1_raw = _load('train_source1.tsv')
    s2_raw = _load('train_source2.tsv')
    gt_raw = _load('train_ground_truth.tsv')

    s1_data = [{'entity_id':r['entity_id'],
                'business_name':r.get('business_name',''),
                'business_address':r.get('business_address',''),
                'country':r.get('country','UNK')} for r in s1_raw]
    s2_data = [{'entity_id':r['entity_id'],
                'business_name':r.get('business_name',''),
                'business_address':r.get('business_address',''),
                'country':r.get('country','UNK')} for r in s2_raw]

    gt = {}
    for row in gt_raw:
        sid = row.get('entity_id') or row.get('source1_entity_id','')
        m   = row.get('match_ids') or row.get('matched_entity_ids','')
        if m and str(m).strip() not in ('','nan','NULL'):
            gt[sid] = set(x.strip() for x in str(m).split(',') if x.strip())
        else:
            gt[sid] = set()

    print(f"Loaded: {len(s1_data):,} S1  |  {len(s2_data):,} S2  |  GT: {len(gt):,} entities")

else:
    print("Using synthetic data (3000 S1, 20000 S2, 25% noise)...")
    s1_data, s2_data, gt = make_synthetic(n_s1=3000, n_s2=20000, noise=0.25)

recall, mean_c, p95, p99 = run(s1_data, s2_data, gt, sample=args.sample)
