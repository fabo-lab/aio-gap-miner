"""Evaluation for the AIO Gap-Miner.

Primary metric is **PR-AUC (average precision)** because citation is a rare,
query-relative positive: only a handful of the candidate URLs per query get
cited, so ROC-AUC and accuracy flatter the model. Every model score is reported
against two baselines:

* **prevalence** -- what a constant/random ranker scores (the positive rate);
* **rank-only** -- predicting citation from organic position alone
  (``1 / organic_rank``). This is the "beat the heuristic" bar: if the learned
  model doesn't clear rank-only, it isn't earning its keep.

A domain-specific metric, ``precision_at_true_k``, asks the question a
practitioner actually cares about: if we surface the top-k predicted URLs per
query (k = how many that query actually cites), how many true citations do we catch?
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from . import config

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Return PR-AUC and ROC-AUC for a set of scores."""
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def rank_only_score(df: pd.DataFrame) -> np.ndarray:
    """The rank-only heuristic baseline: propensity = 1 / organic_rank."""
    return (1.0 / df["organic_rank"].clip(lower=1)).to_numpy()


def prevalence(y_true: np.ndarray) -> float:
    """Positive rate -- the PR-AUC a random ranker achieves in expectation."""
    return float(np.mean(y_true))


def precision_at_true_k(
    df: pd.DataFrame,
    y_score: np.ndarray,
    group_col: str = config.GROUP_COL,
    target: str = config.TARGET,
) -> float:
    """Mean over queries of hits-in-top-k / k, with k = citations in that query.

    Because we take the top-k and k equals the number of true citations,
    precision@k and recall@k coincide -- one clean number for "how well do we
    surface the right sources per query".
    """
    tmp = df[[group_col, target]].copy()
    tmp["_score"] = np.asarray(y_score)

    scores: list[float] = []
    for _, grp in tmp.groupby(group_col):
        k = int(grp[target].sum())
        if k == 0:
            continue
        topk = grp.sort_values("_score", ascending=False).head(k)
        scores.append(float(topk[target].sum()) / k)
    return float(np.mean(scores)) if scores else float("nan")


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Find the probability threshold that maximises F1; return (threshold, f1)."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one more prec/rec point than thresholds.
    f1 = np.divide(
        2 * prec[:-1] * rec[:-1],
        (prec[:-1] + rec[:-1]),
        out=np.zeros_like(prec[:-1]),
        where=(prec[:-1] + rec[:-1]) > 0,
    )
    best = int(np.argmax(f1))
    return float(thr[best]), float(f1[best])


def evaluation_summary(
    df: pd.DataFrame,
    oof_pred: np.ndarray,
) -> dict:
    """Assemble the full comparison: model vs rank-only vs prevalence."""
    y_true = df[config.TARGET].to_numpy()

    model_metrics = compute_metrics(y_true, oof_pred)
    rank_metrics = compute_metrics(y_true, rank_only_score(df))

    thr, f1 = best_f1_threshold(y_true, oof_pred)

    return {
        "n_rows": int(len(df)),
        "n_queries": int(df[config.GROUP_COL].nunique()),
        "positive_rate": prevalence(y_true),
        "model_pr_auc": model_metrics["pr_auc"],
        "model_roc_auc": model_metrics["roc_auc"],
        "rank_only_pr_auc": rank_metrics["pr_auc"],
        "rank_only_roc_auc": rank_metrics["roc_auc"],
        "lift_over_rank": model_metrics["pr_auc"] - rank_metrics["pr_auc"],
        "lift_over_prevalence": model_metrics["pr_auc"] - prevalence(y_true),
        "precision_at_true_k_model": precision_at_true_k(df, oof_pred),
        "precision_at_true_k_rank": precision_at_true_k(df, rank_only_score(df)),
        "best_f1_threshold": thr,
        "best_f1": f1,
    }


def per_fold_scores(
    df: pd.DataFrame,
    score: np.ndarray,
    groups: pd.Series,
    n_splits: int = config.N_SPLITS,
) -> tuple[list[float], list[float]]:
    """Per-fold PR-AUC and ROC-AUC for an OOF score array, on GroupKFold splits.

    Scoring each validation fold separately avoids the pooling artifact where
    fold-to-fold probability-scale differences distort a single pooled AUC.
    """
    from sklearn.model_selection import GroupKFold

    y = df[config.TARGET].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    aps, rocs = [], []
    for _, va in gkf.split(df, y, groups):
        aps.append(float(average_precision_score(y[va], score[va])))
        rocs.append(float(roc_auc_score(y[va], score[va])))
    return aps, rocs


def compare_models(
    df: pd.DataFrame,
    scores: dict[str, np.ndarray],
    groups: pd.Series,
    n_splits: int = config.N_SPLITS,
) -> pd.DataFrame:
    """Per-fold model comparison (PR-AUC mean +/- std, ROC-AUC, precision@k).

    ``scores`` maps a model name to its out-of-fold prediction array. The
    rank-only heuristic and prevalence floor are appended for reference. All
    models are scored on the *same* GroupKFold splits, so the comparison is
    apples-to-apples and free of the pooled-OOF scale artifact.
    """
    y = df[config.TARGET].to_numpy()
    all_scores = dict(scores)
    all_scores["Rank-only heuristic"] = rank_only_score(df)

    rows = []
    for name, score in all_scores.items():
        aps, rocs = per_fold_scores(df, score, groups, n_splits)
        rows.append(
            {
                "model": name,
                "pr_auc": round(float(np.mean(aps)), 4),
                "pr_auc_std": round(float(np.std(aps)), 4),
                "roc_auc": round(float(np.mean(rocs)), 4),
                "precision_at_k": round(precision_at_true_k(df, score), 4),
            }
        )
    rows.append(
        {"model": "Random / prevalence", "pr_auc": round(prevalence(y), 4),
         "pr_auc_std": 0.0, "roc_auc": 0.5, "precision_at_k": float("nan")}
    )
    return pd.DataFrame(rows).set_index("model")


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_pr_curves(
    df: pd.DataFrame,
    oof_pred: np.ndarray,
    save_path: str | Path | None = None,
):
    """Precision-Recall curves: model vs rank-only, with the prevalence floor."""
    y_true = df[config.TARGET].to_numpy()
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for label, score in [("Gap-Miner (LightGBM)", oof_pred),
                         ("Rank-only baseline", rank_only_score(df))]:
        prec, rec, _ = precision_recall_curve(y_true, score)
        ap = average_precision_score(y_true, score)
        ax.plot(rec, prec, linewidth=2, label=f"{label} — AP={ap:.3f}")

    base = prevalence(y_true)
    ax.axhline(base, linestyle="--", color="grey", linewidth=1,
               label=f"Prevalence floor — {base:.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall: predicting AI Overview citations")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig, ax


def plot_confusion(
    df: pd.DataFrame,
    oof_pred: np.ndarray,
    threshold: float | None = None,
    save_path: str | Path | None = None,
):
    """Confusion matrix at the F1-optimal threshold (or a supplied one)."""
    y_true = df[config.TARGET].to_numpy()
    if threshold is None:
        threshold, _ = best_f1_threshold(y_true, oof_pred)
    y_pred = (oof_pred >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xticks([0, 1], ["Not cited", "Cited"])
    ax.set_yticks([0, 1], ["Not cited", "Cited"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix @ threshold {threshold:.3f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="black", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig, ax
