"""Shared accuracy metrics for all benchmark libraries (zer, splink, etc.).

Functions
---------
norm_id                 Normalise a record ID to string (handles int-valued floats).
avg_precision           Compute area under the PR curve (Average Precision).
best_threshold_metrics  Find F1-optimal threshold; return full P/R/F1/counts.
blocking_recall         Fraction of GT pairs that appear in the candidate set.
write_scored_pairs_csv  Write (score, is_match) pairs to CSV, sorted descending.
load_scored_pairs_csv   Load a scored_pairs CSV back as a list of dicts.
"""

import csv
from pathlib import Path


def norm_id(v):
    """Normalise a record ID to string; handles integer-valued float IDs.

    Pure-digit strings (e.g. "065625821") are returned as-is to preserve leading
    zeros.  Float-formatted integers (e.g. "123.0") are collapsed to "123".
    """
    s = str(v).strip()
    if s.isdigit():
        return s
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except (ValueError, OverflowError):
        pass
    return s


def avg_precision(labels, scores, n_total_pos=None):
    """Average precision (area under PR curve). labels/scores are parallel lists.

    Pass n_total_pos=len(gt_pairs) to include blocking false negatives in the recall
    denominator, making AP consistent with the scalar recall metric and the PR curve.
    """
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_blocked_pos = sum(labels)
    n_pos = n_total_pos if n_total_pos is not None else n_blocked_pos
    if n_pos == 0:
        return None
    tp = 0; fp = 0; area = 0.0; prev_r = 0.0
    for _, label in pairs:
        if label: tp += 1
        else:     fp += 1
        r = tp / n_pos
        p = tp / (tp + fp)
        area += (r - prev_r) * p
        prev_r = r
    return round(area, 4)


def best_threshold_metrics(labels, scores, n_total_pos=None):
    """Find threshold maximising F1; return (f1, precision, recall, threshold, tp, fp, fn).

    Pass n_total_pos=len(gt_pairs) to include blocking false negatives in the recall
    denominator, making metrics comparable across systems with different blocking recall.
    """
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = n_total_pos if n_total_pos is not None else sum(labels)
    if n_pos == 0:
        return None, None, None, None, 0, len(labels), 0
    tp = 0; fp = 0; fn = n_pos; best_f1 = -1.0
    best = (0.0, 0.0, 0.0, 1.0, 0, 0, n_pos)
    for score, label in pairs:
        if label: tp += 1; fn -= 1
        else:     fp += 1
        denom = 2 * tp + fp + fn
        if denom > 0:
            f1 = 2 * tp / denom
            if f1 > best_f1:
                best_f1 = f1
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / n_pos
                best = (round(f1, 4), round(p, 4), round(r, 4), round(score, 6), tp, fp, fn)
    return best


def blocking_recall(candidate_pair_ids, gt_pairs):
    """Fraction of GT pairs that appear in the candidate set.

    candidate_pair_ids: iterable of (id_a, id_b) raw values (str, int, or float).
    gt_pairs:           set of canonical (min_id, max_id) string-normalised tuples.
    """
    n_pos = len(gt_pairs)
    if n_pos == 0:
        return None
    candidate_set = set()
    for row in candidate_pair_ids:
        a = norm_id(row[0]); b = norm_id(row[1])
        candidate_set.add((min(a, b), max(a, b)))
    found = sum(1 for p in gt_pairs if p in candidate_set)
    return round(found / n_pos, 4)


def write_scored_pairs_csv(path, scores, labels):
    """Write (score, is_match) pairs to a CSV file sorted by score descending.

    Removes the 500K truncation — CSV can hold millions of rows without the
    memory and load-time cost of embedding them in a JSON file.
    """
    path = Path(path)
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["score", "is_match"])
        for score, label in pairs:
            w.writerow([score, int(bool(label))])


def load_scored_pairs_csv(path):
    """Load a scored_pairs CSV; returns list of {"score": float, "is_match": bool}."""
    result = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            result.append({
                "score":    float(row["score"]),
                "is_match": row["is_match"] in ("1", "True", "true"),
            })
    return result
