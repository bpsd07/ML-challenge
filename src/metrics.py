"""
Stage 8: Official Competition Metric Implementation
Macro-averaged F_0.5 across S1 entities.
F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
Singleton rule:
- correct empty prediction: F_0.5 = 1.0
- false positive on singleton: F_0.5 = 0.0
- non-singleton with zero predictions: F_0.5 = 0.0
"""

from typing import Dict, Set, List, Tuple
import numpy as np


def compute_entity_f05(gt_ids: Set[str], pred_ids: Set[str]) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F_0.5 for a single S1 entity.
    Returns: (f05, precision, recall)
    """
    is_singleton = (len(gt_ids) == 0)
    has_preds = (len(pred_ids) > 0)

    if is_singleton:
        if not has_preds:
            # Correctly identified singleton
            return 1.0, 1.0, 1.0
        else:
            # False merge on singleton
            return 0.0, 0.0, 1.0

    if not has_preds:
        # Missed all matches on non-singleton
        return 0.0, 0.0, 0.0

    tp = len(gt_ids.intersection(pred_ids))
    fp = len(pred_ids - gt_ids)
    fn = len(gt_ids - pred_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision == 0.0 or recall == 0.0:
        return 0.0, precision, recall

    # F_beta with beta = 0.5: beta^2 = 0.25, (1 + beta^2) = 1.25
    # F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    denom = (0.25 * precision) + recall
    f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0
    return f05, precision, recall


def evaluate_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    s1_metadata: Dict[str, dict] = None
) -> dict:
    """
    Compute macro-averaged F_0.5 across all S1 entities in ground truth.
    Provides detailed stratified breakdown:
    - Overall macro F_0.5, macro Precision, macro Recall
    - Singleton F_0.5
    - Matched entity F_0.5
    - S2 only, S3 only, and combined metrics
    - Country-specific macro F_0.5
    - Match-count bucket breakdowns (0, 1, 2, 3, 4, 5+)
    """
    all_s1_ids = list(ground_truth.keys())
    n_entities = len(all_s1_ids)

    f05_scores = []
    precisions = []
    recalls = []

    singleton_scores = []
    matched_scores = []

    s2_precisions, s2_recalls = [], []
    s3_precisions, s3_recalls = [], []

    country_scores = {}
    bucket_scores = {0: [], 1: [], 2: [], 3: [], 4: [], "5+": []}

    for s1_id in all_s1_ids:
        gt_set = ground_truth[s1_id]
        pred_set = predictions.get(s1_id, set())

        f05, prec, rec = compute_entity_f05(gt_set, pred_set)
        f05_scores.append(f05)
        precisions.append(prec)
        recalls.append(rec)

        n_matches = len(gt_set)
        if n_matches == 0:
            singleton_scores.append(f05)
            bucket_scores[0].append(f05)
        else:
            matched_scores.append(f05)
            if n_matches in [1, 2, 3, 4]:
                bucket_scores[n_matches].append(f05)
            else:
                bucket_scores["5+"].append(f05)

        # Source-specific stats
        gt_s2 = {m for m in gt_set if m.startswith("S2-")}
        gt_s3 = {m for m in gt_set if m.startswith("S3-")}
        pred_s2 = {m for m in pred_set if m.startswith("S2-")}
        pred_s3 = {m for m in pred_set if m.startswith("S3-")}

        if len(gt_s2) > 0 or len(pred_s2) > 0:
            s2_f05, s2_p, s2_r = compute_entity_f05(gt_s2, pred_s2)
            s2_precisions.append(s2_p)
            s2_recalls.append(s2_r)

        if len(gt_s3) > 0 or len(pred_s3) > 0:
            s3_f05, s3_p, s3_r = compute_entity_f05(gt_s3, pred_s3)
            s3_precisions.append(s3_p)
            s3_recalls.append(s3_r)

        # Country breakdown if metadata available
        if s1_metadata and s1_id in s1_metadata:
            cntry = s1_metadata[s1_id].get("country", "Unknown")
            if cntry not in country_scores:
                country_scores[cntry] = []
            country_scores[cntry].append(f05)

    res = {
        "n_entities": n_entities,
        "macro_f05": round(float(np.mean(f05_scores)), 4) if f05_scores else 0.0,
        "macro_precision": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        "macro_recall": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "singleton_f05": round(float(np.mean(singleton_scores)), 4) if singleton_scores else 0.0,
        "matched_entity_f05": round(float(np.mean(matched_scores)), 4) if matched_scores else 0.0,
        "s2_precision": round(float(np.mean(s2_precisions)), 4) if s2_precisions else 0.0,
        "s2_recall": round(float(np.mean(s2_recalls)), 4) if s2_recalls else 0.0,
        "s3_precision": round(float(np.mean(s3_precisions)), 4) if s3_precisions else 0.0,
        "s3_recall": round(float(np.mean(s3_recalls)), 4) if s3_recalls else 0.0,
        "country_f05": {k: round(float(np.mean(v)), 4) for k, v in country_scores.items()},
        "bucket_f05": {str(k): round(float(np.mean(v)), 4) for k, v in bucket_scores.items() if v}
    }
    return res
