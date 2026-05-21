#!/usr/bin/env python3
"""
Plot zer-bench results from *_benchmark.json files.

Usage:
    python benchmarks/utils/plot_results.py --input bench_results/data/<run>/
    python benchmarks/utils/plot_results.py --input bench_results/data/<run>/ --output my_plots/

--input is required.  --output is optional; defaults to bench_results/plots/<run>/
where <run> is the folder name of the input path.

Discovers all *_benchmark.json files under --input (or reads a single file),
groups runs by scenario, and produces up to three figures (each saved as
.png, .svg, and .pdf):

  accuracy_comparison
      Per-scenario grid: precision/recall/F1 and PR-AUC grouped bars,
      with distinct colour per library.

  pr_auc_bars
      PR-AUC bar chart per scenario.  Skipped when no PR-AUC values are present.

  pr_curves      [written only when scored_pairs are present in the JSON]
      Per-scenario Precision–Recall curves, one line per library.

  strat_recall   [written only when strat data is present in the JSON]
      Per-scenario grouped bars showing recall broken down by match type
      (dedupe vs link vs cross_dedup, etc.).

  judge_impact   [written only when zer+judge_* entries are present]
      Per-scenario F1/Recall accuracy lift from enabling the neural judge,
      with +/- delta annotations relative to the baseline zer run.
"""

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UTILS_DIR = Path(__file__).resolve().parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from bench_metrics import load_scored_pairs_csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Visual style ───────────────────────────────────────────────────────────────
# Pastel fills (ACM-paper style) + darker saturated borders of the same hue.

_KNOWN_FILL = {
    "zer":    "#A8C8F0",  # pastel blue
    "splink": "#F5AAAA",  # pastel red
}
_KNOWN_BORDER = {
    "zer":    "#1E88E5",
    "splink": "#E53935",
}


def library_color(name, palette_name="tab10"):
    """Return the base (border-weight) color for a library name."""
    return _KNOWN_BORDER.get(name) or plt.get_cmap(palette_name)(hash(name) % 10)
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]

# Stage palette — fixed per concept so the same stage always has the same color across
# all libraries and plot types. "total" is also used as the single-class color for
# external libs (splink, dedupe) that expose only one pipeline stage.
_STAGE_PALETTE = {
    "block":    "#B8C8EC",  # pastel slate blue
    "compare":  "#B8DCC8",  # pastel teal
    "predict":  "#B8DCC8",  # splink alias
    "match":    "#B8DCC8",  # alias
    "classify": "#B8DCC8",  # alias
    "em":       "#F0BCBC",  # pastel red
    "score":    "#D0C8EC",  # pastel purple
    "judge":    "#EEE0A8",  # pastel yellow
    "total":    "#C0C0C0",  # pastel gray
    "load":     "#E8E8E8",  # very light gray (init, not part of pipeline)
    "train":    "#E8E8E8",
    "index":    "#E8E8E8",
    "setup":    "#F5E6C8",  # pastel amber — u-sampling setup (splink only, excluded from total)
}
_STAGE_BORDER = {
    "block":    "#4C72B0",
    "compare":  "#55A868",
    "predict":  "#55A868",
    "match":    "#55A868",
    "classify": "#55A868",
    "em":       "#C44E52",
    "score":    "#8172B3",
    "judge":    "#CCB974",
    "total":    "#606060",
    "load":     "#AAAAAA",
    "train":    "#AAAAAA",
    "index":    "#AAAAAA",
    "setup":    "#D4982A",  # amber border for setup stage
}


def _apply_paper_style() -> None:
    """Flat scientific paper style: white background, no grid, minimal spines, pastel fills."""
    plt.rcParams.update({
        "figure.facecolor":      "white",
        "axes.facecolor":        "white",
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.spines.left":      True,
        "axes.spines.bottom":    True,
        "axes.edgecolor":        "#AAAAAA",
        "axes.linewidth":        0.8,
        "axes.grid":             False,
        "font.family":           "sans-serif",
        "font.size":             9,
        "axes.titlesize":        10,
        "axes.titleweight":      "normal",
        "axes.labelsize":        9,
        "xtick.labelsize":       8,
        "ytick.labelsize":       8,
        "xtick.color":           "#444444",
        "ytick.color":           "#444444",
        "axes.labelcolor":       "#222222",
        "text.color":            "#222222",
        "legend.fontsize":       8,
        "legend.framealpha":     0.9,
        "legend.edgecolor":      "#CCCCCC",
        "legend.borderpad":      0.5,
        "figure.dpi":            150,
    })


_SAVE_FORMATS = ("png", "svg", "pdf")


def _save_fig(fig, out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for fmt in _SAVE_FORMATS:
        path = os.path.join(out_dir, f"{stem}.{fmt}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [plot] {path}")


def _color(lib: str, ordered_libs: list) -> str:
    if lib in _KNOWN_FILL:
        return _KNOWN_FILL[lib]
    if lib.startswith("zer"):
        return _KNOWN_FILL["zer"]
    idx = ordered_libs.index(lib) if lib in ordered_libs else 0
    c = plt.get_cmap("tab10")(idx % 10)
    # Lighten the tab10 colour to match the pastel-fill style
    return tuple(min(1.0, v * 0.5 + 0.5) for v in c[:3]) + (1.0,)


def _border_color(lib: str, ordered_libs: list) -> str:
    if lib in _KNOWN_BORDER:
        return _KNOWN_BORDER[lib]
    if lib.startswith("zer"):
        return _KNOWN_BORDER["zer"]
    idx = ordered_libs.index(lib) if lib in ordered_libs else 0
    return plt.get_cmap("tab10")(idx % 10)


def _marker(lib: str, ordered_libs: list) -> str:
    idx = ordered_libs.index(lib) if lib in ordered_libs else 0
    return _MARKERS[idx % len(_MARKERS)]


# ── Data loading ───────────────────────────────────────────────────────────────

def _resolve_scored_pairs(data: dict, path: str):
    """Load scored pairs from a sidecar CSV if present, otherwise fall back to inline JSON."""
    files = data.get("files", {})
    sp_csv = files.get("scored_pairs_csv") if files else None
    if sp_csv:
        sp_path = Path(path).parent / sp_csv
        if sp_path.exists():
            return load_scored_pairs_csv(str(sp_path))
    return data.get("scored_pairs")


def _load_accuracy_record(data: dict, path: str) -> dict:
    m = data.get("metrics", {})
    rec = {
        "run_id":         data.get("run_id", ""),
        "library":        data.get("library", "zer"),
        "scenario":       data.get("scenario"),
        "mode":           data.get("mode", ""),
        "dataset":        data.get("dataset", ""),
        "target":         data.get("target", "cpu"),
        "timestamp_unix": data.get("timestamp_unix", 0),
        "total_records":  m.get("total_records"),
        "candidate_pairs":m.get("candidate_pairs"),
        "auto_matched":   m.get("auto_matched"),
        "borderline":     m.get("borderline"),
        "auto_rejected":  m.get("auto_rejected"),
        "precision":      m.get("precision"),
        "recall":         m.get("recall"),
        "f1":             m.get("f1"),
        "pr_auc":         m.get("pr_auc"),
        "true_pos":       m.get("true_pos"),
        "false_pos":      m.get("false_pos"),
        "false_neg":      m.get("false_neg"),
        "strat":          data.get("strat", []),
        "scored_pairs":   _resolve_scored_pairs(data, path),
        "_source_file":   path,
    }
    rec["_group"] = rec["scenario"] or f"{rec['mode']}/{rec['dataset']}"
    return rec


def _load_throughput_record(data: dict, path: str) -> dict:
    lib      = data.get("library", "")
    pipeline = data.get("pipeline", {})
    raw      = data.get("raw", {})

    # Requires results produced after refactor (pipeline key required).
    block_ms    = pipeline.get("block_ms")
    compare_ms  = pipeline.get("compare_ms")
    em_ms       = pipeline.get("em_ms")
    score_ms    = pipeline.get("score_ms")
    judge_ms    = pipeline.get("judge_ms")
    u_sample_ms = pipeline.get("u_sample_ms", 0) or 0
    total_ms    = pipeline.get("total_ms")

    thr = data.get("throughput", {})
    pairs_per_s = (thr.get("pairs_per_s") or thr.get("compare_pairs_per_s")
                   or thr.get("predict_pairs_per_s") or thr.get("match_pairs_per_s"))
    em_vectors_per_s = thr.get("em_vectors_per_s")

    mem      = data.get("memory_mb", {})
    mem_peak = mem.get("peak_mb") or mem.get("peak")

    # Detailed per-stage memory lives under raw.memory_mb (new) or top-level (old zer)
    raw_mem    = raw.get("memory_mb", {}) if raw else {}
    mem_detail = raw_mem if raw_mem else mem

    # Fall back to max of stage readings when no explicit peak is provided (e.g. zer)
    if mem_peak is None and raw_mem:
        numeric_vals = [v for v in raw_mem.values() if isinstance(v, (int, float)) and v > 0]
        if numeric_vals:
            mem_peak = max(numeric_vals)

    bands = data.get("match_bands", {})
    rec = {
        "run_id":            data.get("run_id", ""),
        "library":           lib,
        "scenario":          None,
        "mode":              "throughput",
        "dataset":           data.get("dataset", ""),
        "target":            data.get("backend", "cpu"),
        "timestamp_unix":    0,
        "total_records":     data.get("total_records"),
        "candidate_pairs":   data.get("candidate_pairs"),
        "block_ms":          block_ms,
        "compare_ms":        compare_ms,
        "em_ms":             em_ms,
        "score_ms":          score_ms,
        "judge_ms":          judge_ms,
        "u_sample_ms":       u_sample_ms,
        "total_ms":          total_ms,
        "elapsed_ms":        total_ms,
        "pairs_per_s":       pairs_per_s,
        "em_vectors_per_s":  em_vectors_per_s,
        "mem_peak_mb":       mem_peak,
        "mem_detail":        mem_detail,
        "auto_matched":      bands.get("auto_matched"),
        "borderline":        bands.get("borderline"),
        "auto_rejected":     bands.get("auto_rejected"),
        "precision": None, "recall": None, "f1": None, "pr_auc": None,
        "true_pos": None, "false_pos": None, "false_neg": None,
        "strat": [],
        "_source_file": path,
    }
    rec["_group"]     = data.get("dataset", "throughput")
    rec["raw_stages"] = raw.get("stages", {}) if raw else {}
    return rec


def load_benchmark_jsons(input_path: str) -> list:
    """Load all *_benchmark.json files under input_path."""
    if os.path.isfile(input_path):
        paths = [input_path]
    else:
        paths = sorted(glob.glob(
            os.path.join(input_path, "**", "*_benchmark.json"), recursive=True
        ))
    records = []
    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
            if data.get("mode") == "throughput":
                rec = _load_throughput_record(data, p)
            else:
                rec = _load_accuracy_record(data, p)
            records.append(rec)
        except Exception as exc:
            print(f"  [warn] skipping {p}: {exc}", file=sys.stderr)
    return records


# ── Figure 1: accuracy + runtime grid ────────────────────────────────────────

def plot_accuracy_grid(records: list, out_dir: str) -> None:
    if not any(r["precision"] is not None for r in records):
        return

    groups    = sorted({r["_group"] for r in records})
    libraries = sorted({r["library"] for r in records})
    n_groups  = len(groups)
    n_libs    = len(libraries)
    bar_w     = 0.72 / max(n_libs, 1)
    metric_xs = [0.0, 1.0, 2.0, 3.0]  # precision, recall, f1, pr_auc

    n_cols = min(2, n_groups)
    n_rows = math.ceil(n_groups / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7.5 * n_cols, 3.5 * n_rows + 0.8),
        squeeze=False,
    )
    fig.suptitle("zer-bench, accuracy comparison",
                 fontsize=13, fontweight="bold")

    for r_idx, group in enumerate(groups):
        ax  = axes[r_idx // n_cols][r_idx % n_cols]
        grp = [r for r in records if r["_group"] == group]

        for li, lib in enumerate(libraries):
            row = next((r for r in grp if r["library"] == lib), None)
            if row is None:
                continue
            color  = _color(lib, libraries)
            border = _border_color(lib, libraries)
            x_off  = (li - n_libs / 2 + 0.5) * bar_w
            vals   = [row["precision"], row["recall"], row["f1"], row["pr_auc"]]

            for xi, val in enumerate(vals):
                if val is None:
                    continue
                ax.bar(
                    metric_xs[xi] + x_off, val,
                    width=bar_w * 0.88,
                    color=color,
                    edgecolor=border, linewidth=1.0,
                    label=lib if xi == 0 else None,
                )
                ax.text(
                    metric_xs[xi] + x_off, val + 0.005, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=6,
                )

        ax.set_xticks(metric_xs)
        ax.set_xticklabels(["Precision", "Recall", "F1", "PR-AUC"])
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score")
        ax.set_title(group, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc="lower right",
                      framealpha=0.75, ncol=max(1, n_libs // 4))

    for r_idx in range(n_groups, n_rows * n_cols):
        axes[r_idx // n_cols][r_idx % n_cols].set_visible(False)

    fig.tight_layout(pad=1.5)
    _save_fig(fig, out_dir, "accuracy_comparison")
    plt.close(fig)


# ── Figure 2: PR-AUC bar chart ────────────────────────────────────────────────

def plot_pr_auc_bars(records: list, out_dir: str) -> None:
    if not any(r["pr_auc"] is not None for r in records):
        return

    groups    = sorted({r["_group"] for r in records})
    libraries = sorted({r["library"] for r in records})
    n_groups  = len(groups)
    n_cols    = min(3, n_groups)
    n_rows    = math.ceil(n_groups / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.8 * n_cols, 4.0 * n_rows + 0.6),
        squeeze=False,
    )
    fig.suptitle("zer-bench, PR-AUC comparison",
                 fontsize=13, fontweight="bold")

    for g_idx, group in enumerate(groups):
        ax   = axes[g_idx // n_cols][g_idx % n_cols]
        grp  = [r for r in records if r["_group"] == group]

        for li, lib in enumerate(libraries):
            row = next((r for r in grp if r["library"] == lib), None)
            if row is None or row["pr_auc"] is None:
                continue
            val = row["pr_auc"]
            ax.bar(
                li, val,
                width=0.72,
                color=_color(lib, libraries),
                edgecolor=_border_color(lib, libraries), linewidth=1.0,
                label=lib,
            )
            ax.text(li, val + 0.008, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7)

        ax.set_xticks(list(range(len(libraries))))
        ax.set_xticklabels(libraries, rotation=20, ha="right", fontsize=7)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("PR-AUC")
        ax.set_title(group, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    for g_idx in range(n_groups, n_rows * n_cols):
        axes[g_idx // n_cols][g_idx % n_cols].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, out_dir, "pr_auc_bars")
    plt.close(fig)


# ── Figure 3: PR curves ────────────────────────────────────────────────────

def _compute_pr_curve(scored_pairs: list, n_total_gt: int = None):
    """Return (recalls, precisions) lists from a scored_pairs list.

    Pairs are grouped by score value so that all pairs sharing the same score
    contribute a single (recall, precision) point. This eliminates zigzag
    artifacts that arise from arbitrary ordering within same-score groups (a
    common occurrence in splink and any system that discretises its scores).

    Null/NaN entries are filtered. Groups where tp==0 (pure false-positive
    score tiers before the first true match) are skipped, so the curve
    naturally starts at high precision rather than near the origin.

    n_total_gt: total ground-truth positive count including pairs missed by
    blocking (= true_pos + false_neg from the benchmark JSON).  When provided,
    recall is expressed as a fraction of ALL ground-truth pairs rather than
    only those generated by the blocker.  This makes the curve consistent with
    the scalar recall metric and reveals the blocking-recall ceiling.

    The curve is always extended to (1.0, 0.0) so that every plot terminates at
    the same corner regardless of blocking quality:
      - blocking recall < 1.0: vertical drop at recall=blocking_recall, then
        horizontal segment to (1.0, 0.0).
      - blocking recall = 1.0 but clean candidate set (precision > 0 at
        recall=1): terminal (1.0, 0.0) anchor added conventionally.
    """
    clean = []
    for p in scored_pairs:
        score    = p.get("score")
        is_match = p.get("is_match")
        if score is None or is_match is None:
            continue
        try:
            s = float(score)
        except (TypeError, ValueError):
            continue
        if math.isnan(s) or math.isinf(s):
            continue
        clean.append((s, bool(is_match)))

    if not clean:
        return [], []

    clean.sort(key=lambda x: -x[0])
    n_pos_blocked = sum(m for _, m in clean)
    if n_pos_blocked == 0:
        return [], []

    # Use global GT count as denominator when available.
    n_pos = n_total_gt if (n_total_gt and n_total_gt > n_pos_blocked) else n_pos_blocked

    tp = 0; fp = 0
    recalls = [0.0]; precisions = [1.0]  # anchor: at threshold=∞, recall=0, precision=1
    i = 0
    while i < len(clean):
        # Consume all pairs with the same score as one group.
        group_score = clean[i][0]
        while i < len(clean) and clean[i][0] == group_score:
            if clean[i][1]:
                tp += 1
            else:
                fp += 1
            i += 1
        # Emit one point per score group, skipping pure-FP leading groups.
        if tp > 0:
            recalls.append(tp / n_pos)
            precisions.append(tp / (tp + fp))

    # Extend the curve to (1.0, 0.0) so every plot reaches the same corner.
    if recalls:
        last_r = recalls[-1]
        if last_r < 1.0 - 1e-9:
            # Blocking recall < 1.0: drop precision to 0 at the blocking ceiling,
            # then extend horizontally to recall=1.0.
            recalls.extend([last_r, 1.0])
            precisions.extend([0.0, 0.0])
        elif precisions[-1] > 1e-9:
            # Blocking recall is 1.0 but candidate set is clean (precision > 0):
            # add conventional terminal anchor at (1.0, 0.0).
            recalls.append(1.0)
            precisions.append(0.0)

    return recalls, precisions


def plot_pr_curves(records: list, out_dir: str) -> None:
    valid = [r for r in records if r.get("scored_pairs")]
    if not valid:
        return

    groups    = sorted({r["_group"] for r in valid})
    libraries = sorted({r["library"] for r in records})
    n_groups  = len(groups)
    n_cols    = min(3, n_groups)
    n_rows    = math.ceil(n_groups / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.5 * n_rows + 0.6),
        squeeze=False,
    )
    fig.suptitle("zer-bench, Precision–Recall curves",
                 fontsize=13, fontweight="bold")

    for g_idx, group in enumerate(groups):
        ax  = axes[g_idx // n_cols][g_idx % n_cols]
        grp = [r for r in valid if r["_group"] == group]

        for lib in libraries:
            row = next((r for r in grp if r["library"] == lib), None)
            if row is None or not row.get("scored_pairs"):
                continue
            tp = row.get("true_pos"); fn = row.get("false_neg")
            n_total_gt = (tp + fn) if (tp is not None and fn is not None) else None
            recalls, precisions = _compute_pr_curve(row["scored_pairs"], n_total_gt=n_total_gt)
            if not recalls:
                continue
            color  = _color(lib, libraries)
            border = _border_color(lib, libraries)
            marker = _marker(lib, libraries)
            pr_auc = row.get("pr_auc")
            label  = f"{lib} (AP={pr_auc:.3f})" if pr_auc is not None else lib
            # Downsample for SVG/PDF compactness — keep at most 2000 points
            step = max(1, len(recalls) // 2000)
            ax.plot(recalls[::step], precisions[::step],
                    color=border, linewidth=1.4,
                    marker=marker, markersize=0,
                    label=label)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(group, fontsize=10)
        ax.legend(fontsize=7, loc="lower left")

    for g_idx in range(n_groups, n_rows * n_cols):
        axes[g_idx // n_cols][g_idx % n_cols].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, out_dir, "pr_curves")
    plt.close(fig)


# ── Figure 4: stratified recall breakdown ────────────────────────────────────

def plot_strat_grid(records: list, out_dir: str) -> None:
    valid = [
        r for r in records
        if r["strat"] and any(s.get("match_type", "").strip() for s in r["strat"])
    ]
    if not valid:
        return

    groups    = sorted({r["_group"] for r in valid})
    libraries = sorted({r["library"] for r in records})
    n_groups  = len(groups)
    n_cols    = min(2, n_groups)
    n_rows    = math.ceil(n_groups / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5 * n_cols, 4.2 * n_rows + 0.6),
        squeeze=False,
    )
    fig.suptitle("zer-bench, stratified recall by match type",
                 fontsize=13, fontweight="bold")

    for s_idx, group in enumerate(groups):
        ax   = axes[s_idx // n_cols][s_idx % n_cols]
        grp  = [r for r in valid if r["_group"] == group]

        all_types = sorted({
            s["match_type"].strip()
            for r in grp
            for s in r["strat"]
            if s.get("match_type", "").strip()
        })
        if not all_types:
            ax.set_visible(False)
            continue

        n_types = len(all_types)
        n_libs  = len(grp)
        bar_w   = 0.72 / max(n_libs, 1)

        for li, row in enumerate(sorted(grp, key=lambda r: r["library"])):
            lib    = row["library"]
            color  = _color(lib, libraries)
            border = _border_color(lib, libraries)
            x_off  = (li - n_libs / 2 + 0.5) * bar_w
            strat_by_type = {s["match_type"].strip(): s for s in row["strat"]}

            for ti, mt in enumerate(all_types):
                s      = strat_by_type.get(mt, {})
                recall = float(s.get("recall", 0.0))
                count  = int(s.get("count_gt", 0))
                ax.bar(
                    ti + x_off, recall,
                    width=bar_w * 0.88,
                    color=color,
                    edgecolor=border, linewidth=1.0,
                    label=lib if ti == 0 else None,
                )
                if count:
                    ax.text(
                        ti + x_off, min(recall + 0.01, 1.0),
                        f"n={count}", ha="center", va="bottom",
                        fontsize=5.5,
                    )

        ax.set_xticks(range(n_types))
        ax.set_xticklabels(all_types, rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Recall")
        ax.set_title(group, fontsize=10)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc="lower right", framealpha=0.75)

    for s_idx in range(n_groups, n_rows * n_cols):
        axes[s_idx // n_cols][s_idx % n_cols].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, out_dir, "strat_recall")
    plt.close(fig)


# ── Figure 4: judge accuracy / runtime tradeoff ──────────────────────────────

def plot_judge_impact(records: list, out_dir: str) -> None:
    """Per-scenario comparison of zer (no judge) vs zer+judge_* variants.

    F1 and Recall bars side-by-side with delta annotations showing the
    accuracy lift (or drop) each judge variant provides over the baseline.

    Skipped entirely when no zer+judge_* records are present.
    """
    zer_records  = [r for r in records if r["library"] == "zer" or r["library"].startswith("zer+judge")]
    judge_records = [r for r in zer_records if r["library"].startswith("zer+judge")]
    if not judge_records:
        return

    groups = sorted({r["_group"] for r in zer_records})
    valid_groups = [
        g for g in groups
        if any(r["library"] == "zer"              for r in zer_records if r["_group"] == g)
        and any(r["library"].startswith("zer+judge") for r in zer_records if r["_group"] == g)
    ]
    if not valid_groups:
        return

    zer_libs = sorted({r["library"] for r in zer_records
                       if any(r2["_group"] in valid_groups and r2["library"] == r["library"]
                              for r2 in zer_records)})
    n_groups  = len(valid_groups)
    bar_w     = 0.72 / max(len(zer_libs), 1)
    metric_xs = [0.0, 1.0]  # F1, Recall

    fig, axes = plt.subplots(
        n_groups, 1,
        figsize=(7, 3.5 * n_groups + 0.8),
        squeeze=False,
    )
    fig.suptitle("zer-bench, judge impact: accuracy comparison",
                 fontsize=13, fontweight="bold", y=1.01)

    for r_idx, group in enumerate(valid_groups):
        ax  = axes[r_idx][0]
        grp = [r for r in zer_records if r["_group"] == group]

        for li, lib in enumerate(zer_libs):
            row = next((r for r in grp if r["library"] == lib), None)
            if row is None:
                continue
            color  = _color(lib, zer_libs)
            border = _border_color(lib, zer_libs)
            x_off  = (li - len(zer_libs) / 2 + 0.5) * bar_w

            for xi, metric in enumerate(["f1", "recall"]):
                val = row.get(metric)
                if val is None:
                    continue
                ax.bar(
                    metric_xs[xi] + x_off, val,
                    width=bar_w * 0.88,
                    color=color,
                    edgecolor=border, linewidth=1.0,
                    label=lib if xi == 0 else None,
                )
                ax.text(
                    metric_xs[xi] + x_off, val + 0.005, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=6.5,
                )

        # Delta annotations: lift over zer baseline
        zer_row = next((r for r in grp if r["library"] == "zer"), None)
        if zer_row:
            for lib in zer_libs:
                if not lib.startswith("zer+judge"):
                    continue
                judge_row = next((r for r in grp if r["library"] == lib), None)
                if judge_row is None:
                    continue
                li = zer_libs.index(lib)
                x_off = (li - len(zer_libs) / 2 + 0.5) * bar_w
                for xi, metric in enumerate(["f1", "recall"]):
                    base = zer_row.get(metric)
                    new  = judge_row.get(metric)
                    if base is None or new is None:
                        continue
                    delta = new - base
                    sign  = "+" if delta >= 0 else ""
                    col   = "#1a237e" if delta >= 0 else "#b71c1c"
                    ax.annotate(
                        f"{sign}{delta:.3f}",
                        xy=(metric_xs[xi] + x_off, new + 0.11),
                        ha="center", va="bottom", fontsize=6, color=col,
                        fontweight="bold",
                    )

        ax.set_xticks(metric_xs)
        ax.set_xticklabels(["F1", "Recall"])
        ax.set_ylim(0, 1.22)
        ax.set_ylabel("Score")
        ax.set_title(group, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, loc="lower right", framealpha=0.75)

    fig.tight_layout()
    _save_fig(fig, out_dir, "judge_impact")
    plt.close(fig)


# ── Figure 5: throughput ──────────────────────────────────────────────────────

def _stage_color(stage: str) -> str:
    return _STAGE_PALETTE.get(stage.lower(), "#C8C8C8")


def _stage_border_color(stage: str) -> str:
    return _STAGE_BORDER.get(stage.lower(), "#888888")


def _pipeline_stages(rec: dict) -> list:
    """Return [(label, ms)] for all non-null, non-zero canonical pipeline stages.

    u_sample_ms is shown as a distinct 'setup' bar when present (splink only).
    It is excluded from pipeline.total_ms for fair comparison with zer.
    """
    entries = [
        ("setup",   rec.get("u_sample_ms")),   # u-sampling: splink setup, no zer equivalent
        ("block",   rec.get("block_ms")),
        ("compare", rec.get("compare_ms")),
        ("em",      rec.get("em_ms")),
        ("score",   rec.get("score_ms")),
        ("judge",   rec.get("judge_ms")),
    ]
    return [(l, v) for l, v in entries if v is not None and v > 0]


def _plot_stage_pie(rec: dict, lib_out_dir: str) -> None:
    """Stage duration pie chart. Only called when ≥2 valid pipeline stages exist."""
    stages = _pipeline_stages(rec)
    if len(stages) < 2:
        return

    total_ms = rec.get("total_ms") or sum(v for _, v in stages)
    labels   = [f"{l}\n{v} ms" for l, v in stages]
    vals     = [v for _, v in stages]
    colors   = [_stage_color(l) for l, _ in stages]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.suptitle(
        f"{rec['library']}  ·  {rec['dataset']}  ·  stage breakdown",
        fontsize=10,
    )

    wedges, _, autotexts = ax.pie(
        vals,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"linewidth": 1.2},
        textprops={"fontsize": 8},
    )
    for wedge, (label, _) in zip(wedges, stages):
        wedge.set_edgecolor(_stage_border_color(label))
    for at in autotexts:
        at.set_fontsize(7.5)
        at.set_color("#222222")

    ax.set_title(f"total  {total_ms} ms", fontsize=8, pad=6)

    pie_dir = os.path.join(lib_out_dir, "stage_pie")
    os.makedirs(pie_dir, exist_ok=True)
    fig.tight_layout()
    _save_fig(fig, pie_dir, "stage_pie")
    plt.close(fig)


def _memory_timeline(rec: dict) -> list:
    """Return [(stage_label, cumulative_ms, rss_mb)] from raw stage durations + snapshots.

    Cumulative time is computed from raw.stages so that load/train/init stages
    are reflected on the x-axis even though they are excluded from the pipeline total.
    """
    lib  = rec.get("library", "")
    rs   = rec.get("raw_stages", {})   # raw stage durations
    mem  = rec.get("mem_detail", {})   # per-stage RSS readings

    def ms(key):  return rs.get(key) or 0
    def mb(key):  return mem.get(key)

    pts = []

    if lib.startswith("zer"):
        t = ms("compare_ms")
        if mb("after_compare") is not None:
            pts.append(("compare", t, mb("after_compare")))
        t += ms("em_ms")
        if mb("after_em") is not None:
            pts.append(("em", t, mb("after_em")))
        t += ms("score_ms")
        if mb("after_score") is not None:
            pts.append(("score", t, mb("after_score")))

    elif lib == "splink":
        t = ms("load_ms")
        if mb("after_load") is not None:
            pts.append(("load", t, mb("after_load")))
        t += ms("u_sample_ms")   # u-sampling (setup stage, excluded from pipeline total)
        t += ms("train_ms")
        if mb("after_train") is not None:
            pts.append(("train", t, mb("after_train")))
        t += ms("predict_ms")
        if mb("after_predict") is not None:
            pts.append(("predict", t, mb("after_predict")))

    return [(l, t, m) for l, t, m in pts if m is not None]


def _plot_memory_timeline(rec: dict, lib_out_dir: str) -> None:
    """Line chart: RSS memory (MB) at each pipeline stage boundary over wall time."""
    timeline = _memory_timeline(rec)
    if not timeline:
        return

    lib     = rec["library"]
    dataset = rec["dataset"]

    stage_labels = [p[0] for p in timeline]
    times        = [p[1] for p in timeline]
    mems         = [p[2] for p in timeline]
    peak         = rec.get("mem_peak_mb")

    line_color = _border_color(lib, [lib])

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    fig.suptitle(f"{lib}  ·  {dataset}  ·  memory usage", fontsize=10)

    ax.plot(times, mems, color=line_color, linewidth=1.8,
            marker="o", markersize=6, zorder=3,
            markerfacecolor=_color(lib, [lib]),
            markeredgecolor=line_color, markeredgewidth=1.2)

    # Annotate each measurement point
    y_max = max(mems + ([peak] if peak else []))
    for t, m, label in zip(times, mems, stage_labels):
        ax.annotate(
            f"{label}\n{m:.0f} MB",
            xy=(t, m),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=7.5, color="#333333",
        )

    # Optional peak line when peak exceeds all measured points
    if peak and peak > max(mems, default=0) * 1.01:
        ax.axhline(peak, color="#C44E52", linewidth=0.9,
                   linestyle="--", alpha=0.75, zorder=2)
        ax.text(times[-1], peak + y_max * 0.02,
                f"peak  {peak:.0f} MB",
                ha="right", va="bottom", fontsize=7, color="#C44E52")

    ax.set_xlabel("wall time (ms)")
    ax.set_ylabel("RSS memory (MB)")
    ax.set_ylim(0, y_max * 1.35)
    ax.spines["left"].set_visible(True)

    tl_dir = os.path.join(lib_out_dir, "memory_timeline")
    os.makedirs(tl_dir, exist_ok=True)
    fig.tight_layout()
    _save_fig(fig, tl_dir, "memory_timeline")
    plt.close(fig)


def _plot_throughput_comparison_bars(thr: list, out_dir: str, title_suffix: str = "") -> None:
    """Multi-library side-by-side comparison: pipeline time, throughput, peak memory."""
    libraries = [r["library"] for r in thr]
    xs        = list(range(len(libraries)))
    colors    = [_color(lib, libraries) for lib in libraries]
    borders   = [_border_color(lib, libraries) for lib in libraries]

    def _bar_panel(ax, vals, ylabel, title, label_fn):
        ax.bar(xs, vals, color=colors, edgecolor=borders, linewidth=1.0, width=0.55)
        max_v = max(vals, default=1) or 1
        for xi, val in zip(xs, vals):
            ax.text(xi, val + max_v * 0.02, label_fn(val),
                    ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(libraries, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    suptitle = "zer-bench  ·  throughput comparison"
    if title_suffix:
        suptitle = f"{suptitle}  ·  {title_suffix}"

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), squeeze=False)
    fig.suptitle(suptitle, fontsize=11)

    _bar_panel(axes[0][0],
               [r.get("total_ms") or 0 for r in thr],
               "ms", "pipeline time (ms)",
               lambda v: f"{int(v):,} ms")

    _bar_panel(axes[0][1],
               [(r.get("pairs_per_s") or 0) / 1e6 for r in thr],
               "M pairs / s", "throughput (M pairs/s)",
               lambda v: f"{v:.2f} M")

    _bar_panel(axes[0][2],
               [r.get("mem_peak_mb") or 0 for r in thr],
               "MB", "peak memory (MB)",
               lambda v: f"{v:.0f} MB")

    fig.tight_layout()
    _save_fig(fig, out_dir, "throughput_comparison")
    plt.close(fig)


def _plot_throughput_grid(thr: list, datasets: list, out_dir: str) -> None:
    """Multi-scenario grid: 3 metric rows × n_scenario columns.

    Each column is one scenario (titled at the top); each row is one metric
    (pipeline time, throughput, peak memory) with its own y-axis and raw values,
    mirroring the layout of the per-scenario _plot_throughput_comparison_bars.
    """
    libraries = sorted({r["library"] for r in thr})
    n_cols    = len(datasets)   # one column per scenario

    # (row title, y-label, value extractor, bar annotation formatter)
    metric_rows = [
        ("pipeline time (ms)",    "ms",        lambda r: r.get("total_ms") or 0,              lambda v: f"{int(v):,} ms"),
        ("throughput (M pairs/s)", "M pairs/s", lambda r: (r.get("pairs_per_s") or 0) / 1e6,  lambda v: f"{v:.2f} M"),
        ("peak memory (MB)",       "MB",        lambda r: r.get("mem_peak_mb") or 0,           lambda v: f"{v:.0f} MB"),
    ]
    n_rows = len(metric_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.0 * n_rows + 0.8),
        squeeze=False,
    )
    fig.suptitle("zer-bench  ·  throughput comparison", fontsize=13, fontweight="bold")

    for d_idx, dataset in enumerate(datasets):
        grp      = [r for r in thr if r["_group"] == dataset]
        grp_libs = [r["library"] for r in grp]
        xs       = list(range(len(grp)))
        colors   = [_color(lib, libraries) for lib in grp_libs]
        borders  = [_border_color(lib, libraries) for lib in grp_libs]

        for m_idx, (row_title, ylabel, val_fn, label_fn) in enumerate(metric_rows):
            ax   = axes[m_idx][d_idx]
            vals = [val_fn(r) for r in grp]

            ax.bar(xs, vals, color=colors, edgecolor=borders, linewidth=1.0, width=0.55)
            max_v = max(vals, default=1) or 1
            for xi, val in zip(xs, vals):
                ax.text(xi, val + max_v * 0.02, label_fn(val),
                        ha="center", va="bottom", fontsize=7.5)
            ax.set_xticks(xs)
            ax.set_xticklabels(grp_libs, rotation=20, ha="right")
            ax.set_ylabel(ylabel)

            # Scenario name as column header on the top row only
            if m_idx == 0:
                ax.set_title(dataset, fontsize=10)
            # Metric label on the left column only
            if d_idx == 0:
                ax.set_ylabel(f"{row_title}\n{ylabel}")

    fig.tight_layout(pad=1.5)
    _save_fig(fig, out_dir, "throughput_comparison")
    plt.close(fig)


def plot_throughput(records: list, out_dir: str) -> None:
    thr = [r for r in records if r["mode"] == "throughput"]
    if not thr:
        return

    datasets    = sorted({r["_group"] for r in thr})
    n_datasets  = len(datasets)

    if n_datasets > 1:
        # Multi-scenario: emit one overview grid covering all scenarios.
        _plot_throughput_grid(thr, datasets, out_dir)
    elif len(thr) > 1:
        # Single scenario, multiple libraries: flat comparison bars.
        _plot_throughput_comparison_bars(thr, out_dir)

    for dataset in datasets:
        grp     = [r for r in thr if r["_group"] == dataset]
        # Each scenario lives in its own subdirectory when there are multiple scenarios.
        grp_dir = os.path.join(out_dir, dataset) if n_datasets > 1 else out_dir

        if n_datasets > 1 and len(grp) > 1:
            # Per-scenario detail comparison (in addition to the overview grid).
            _plot_throughput_comparison_bars(grp, grp_dir, title_suffix=dataset)

        for rec in grp:
            lib_dir = os.path.join(grp_dir, rec["library"])
            _plot_stage_pie(rec, lib_dir)          # skipped automatically if < 2 stages
            _plot_memory_timeline(rec, lib_dir)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True,
                    help="Directory containing *_benchmark.json files, or a single JSON file")
    ap.add_argument("--output", default=None,
                    help="Output directory for plots (default: <input>/plots/)")
    args = ap.parse_args()

    if args.output:
        out_dir = args.output
    else:
        folder_name = os.path.basename(os.path.abspath(args.input).rstrip("/"))
        out_dir = str(_REPO_ROOT / "bench_results" / "plots" / folder_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading benchmark results from: {args.input}")
    records = load_benchmark_jsons(args.input)
    if not records:
        print("No *_benchmark.json files found.",
              file=sys.stderr)
        sys.exit(1)

    thr_records = [r for r in records if r["mode"] == "throughput"]
    acc_records = [r for r in records if r["mode"] != "throughput"]

    if acc_records:
        groups    = sorted({r["_group"] for r in acc_records})
        libraries = sorted({r["library"] for r in acc_records})
        print(f"  {len(acc_records)} accuracy run(s) | {len(libraries)} library/libraries "
              f"| {len(groups)} scenario(s)")
        for g in groups:
            libs_in_group = sorted({r["library"] for r in acc_records if r["_group"] == g})
            print(f"    {g}: {', '.join(libs_in_group)}")

    if thr_records:
        print(f"  {len(thr_records)} throughput run(s): "
              f"{', '.join(sorted({r['library'] for r in thr_records}))}")

    _apply_paper_style()

    def _acc_subdir(name: str) -> str:
        return os.path.join(out_dir, name)

    if acc_records:
        plot_accuracy_grid(acc_records, _acc_subdir("accuracy_comparison"))
        plot_pr_auc_bars(acc_records,   _acc_subdir("pr_auc_bars"))
        plot_pr_curves(acc_records,     _acc_subdir("pr_curves"))
        plot_strat_grid(acc_records,    _acc_subdir("strat_recall"))
        plot_judge_impact(acc_records,  _acc_subdir("judge_impact"))
    plot_throughput(thr_records,    out_dir)

    print(f"\nDone. Output: {out_dir}/")


if __name__ == "__main__":
    main()
