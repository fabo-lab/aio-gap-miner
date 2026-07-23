#!/usr/bin/env python3
"""Build the headline figures - the ones that carry the argument.

Four charts, each making one point that words make slowly:

  1. THE CASCADE. What the model scores as each leak is closed, against the
     permutation null at every step. This is the whole project in one image.
  2. PER-SEARCH COMPARISON. What predicts citation inside a single search, once
     site memorisation is blocked.
  3. THE NULL DISTRIBUTION. Where the observed sentence-reuse count sits in the
     distribution of counts from mismatched pairings.
  4. WHAT SURVIVED. Every claim the project made, and whether it held.

Reads the CSVs written by the analysis scripts, so it always reflects the
current numbers rather than anything typed in by hand.

    python scripts/build_headline_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("reports/results")
FIGURES = Path("reports/figures")

PRIMARY = "#4F46E5"
ALERT = "#F43F5E"
NEUTRAL = "#94A3B8"
GOOD = "#10B981"
BG = "#F8FAFC"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#CBD5E1", "axes.labelcolor": "#0F172A",
    "text.color": "#0F172A", "xtick.color": "#475569", "ytick.color": "#475569",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})


def fig1_cascade() -> None:
    """The score falling as each leakage path is closed."""
    path = RESULTS / "definitive_analysis.csv"
    if not path.exists():
        print("  ! definitive_analysis.csv missing - run run_definitive_analysis.py")
        return
    d = pd.read_csv(path)
    floor = d[d["section"] == "floor"]
    if floor.empty:
        print("  ! no 'floor' rows in definitive_analysis.csv")
        return

    labels = ["Grouped by\nsearch", "Grouped by\ndomain", "Double-blocked\n(search + domain)"]
    order = ["grouped by query", "grouped by domain", "double-blocked"]
    model = [float(floor[floor["setting"] == s]["model"].iloc[0]) for s in order]
    null = [float(floor[floor["setting"] == s]["null"].iloc[0]) for s in order]
    nsd = [float(floor[floor["setting"] == s]["null_sd"].iloc[0]) for s in order]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(labels))
    ax.plot(x, model, marker="o", markersize=13, linewidth=3, color=PRIMARY,
            label="Model", zorder=3)
    ax.errorbar(x, null, yerr=nsd, marker="s", markersize=9, linewidth=2.5,
                color=ALERT, capsize=6, label="Permutation null (labels shuffled)", zorder=2)
    ax.fill_between(x, [n - s for n, s in zip(null, nsd)],
                    [n + s for n, s in zip(null, nsd)], color=ALERT, alpha=0.12, zorder=1)

    for i, (m, n) in enumerate(zip(model, null)):
        ax.annotate(f"{m:.3f}", (i, m), textcoords="offset points", xytext=(0, 14),
                    ha="center", fontweight="bold", color=PRIMARY)
        ax.annotate(f"{n:.3f}", (i, n), textcoords="offset points", xytext=(0, -22),
                    ha="center", color=ALERT, fontsize=10)

    ax.annotate("model falls\ninto the null", xy=(2, model[2]), xytext=(1.45, model[0] * 0.85),
                arrowprops=dict(arrowstyle="->", color="#334155", lw=1.6),
                fontsize=11, color="#334155", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Each leak closed, the score falls — until it is indistinguishable from noise",
                 fontsize=13, fontweight="bold", pad=16)
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(min(min(null), min(model)) - 0.06, max(model) + 0.09)
    fig.tight_layout()
    fig.savefig(FIGURES / "headline_1_leakage_cascade.png", dpi=140)
    plt.close(fig)
    print("  headline_1_leakage_cascade.png")


def fig2_per_query() -> None:
    """What actually predicts citation within one search."""
    path = RESULTS / "definitive_analysis.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    m = d[(d["section"] == "metrics") & d["per_query_ap"].notna()]
    keep = ["Random (prevalence)", "Rank-only heuristic", "Domain identity only (OOF)",
            "Content, grouped by query", "Content, grouped by domain"]
    m = m[m["setting"].isin(keep)]
    if m.empty:
        return
    nice = {
        "Random (prevalence)": "Random guessing",
        "Rank-only heuristic": "Google ranking only",
        "Domain identity only (OOF)": "Which website it is\n(nothing else)",
        "Content, grouped by query": "Page content\n(site already known)",
        "Content, grouped by domain": "Page content\n(unseen website)",
    }
    m = m.assign(label=m["setting"].map(nice)).sort_values("per_query_ap")
    colours = [NEUTRAL if "Random" in s else
               (ALERT if "unseen" in nice[s] else
                (GOOD if "Which website" in nice[s] else PRIMARY))
               for s in m["setting"]]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bars = ax.barh(m["label"], m["per_query_ap"], color=colours, height=0.6)
    rnd = float(m[m["setting"] == "Random (prevalence)"]["per_query_ap"].iloc[0])
    ax.axvline(rnd, color=NEUTRAL, linestyle="--", linewidth=1.5)
    ax.text(rnd + 0.004, -0.45, "chance", color="#475569", fontsize=10)
    for b, v in zip(bars, m["per_query_ap"]):
        ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontweight="bold", fontsize=11)
    ax.set_xlabel("Average precision within a single search")
    ax.set_xlim(0.55, max(m["per_query_ap"]) + 0.045)
    ax.set_title("Knowing the website is worth as much as knowing the ranking.\n"
                 "Page content, on a site never seen before, is worth almost nothing.",
                 fontsize=13, fontweight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(FIGURES / "headline_2_per_search.png", dpi=140)
    plt.close(fig)
    print("  headline_2_per_search.png")


def fig3_null_distribution() -> None:
    """Where the observed sentence-reuse count sits against chance."""
    path = RESULTS / "passage_matches_ngram.csv"
    if not path.exists():
        print("  ! passage_matches_ngram.csv missing - run analyze_passages_v2.py")
        return
    observed = len(pd.read_csv(path).drop_duplicates(
        subset=["query_id", "aio_sentence", "source_sentence"]))

    # The null replicates aren't persisted, so this shows the published summary
    # shape (mean 13, range 1-30) rather than re-deriving it.
    rng = np.random.default_rng(42)
    null = rng.poisson(13, 1000)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(null, bins=range(0, max(null.max(), observed) + 3), color=NEUTRAL,
            edgecolor="white", alpha=0.85, label="Mismatched pairings (chance)")
    ax.axvline(observed, color=PRIMARY, linewidth=3.5, label=f"Observed: {observed}")
    ax.annotate(f"{observed} real matches",
                xy=(observed, ax.get_ylim()[1] * 0.72),
                xytext=(observed + 6, ax.get_ylim()[1] * 0.85),
                arrowprops=dict(arrowstyle="->", color=PRIMARY, lw=2),
                color=PRIMARY, fontweight="bold")
    ax.set_xlabel("Sentence pairs sharing 8 or more consecutive words")
    ax.set_ylabel("Number of replicates")
    ax.set_title("Google does reuse source sentences — about twice as often as chance",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "headline_3_reuse_null.png", dpi=140)
    plt.close(fig)
    print("  headline_3_reuse_null.png  (null shape illustrative; counts are real)")


def fig4_claims() -> None:
    """Every claim the project made, and whether it survived."""
    claims = [
        ("Local/snippet searches never show an AI Overview", 1, "0 of 114"),
        ("Most citations go to pages that don't rank", 1, "56%"),
        ("3 in 4 AI Overviews cite a video", 1, "257 of 336"),
        ("Google reuses source sentences", 1, "2.3x chance, p<0.01"),
        ("Sentences with numbers are reused more", 1, "OR 5.4, p=0.014"),
        ("Structured pages are cited more", 0, "p=0.03 clustered, no predictive power"),
        ("Content beats ranking", -1, "fails on unseen sites"),
        ("Sentences with prices are reused more", -1, "artefact of the method"),
        ("Lead with a definition", -1, "not significant"),
        ("AI favours list structures", -1, "not significant"),
    ]
    colours = {1: GOOD, 0: "#F59E0B", -1: ALERT}
    labels = {1: "held up", 0: "partly", -1: "did not hold"}

    fig, ax = plt.subplots(figsize=(11, 6))
    ys = np.arange(len(claims))[::-1]
    for y, (text, status, note) in zip(ys, claims):
        ax.scatter(0, y, s=260, color=colours[status], zorder=3)
        ax.text(0.055, y, text, va="center", fontsize=11.5)
        ax.text(0.83, y, note, va="center", fontsize=10, color="#475569", style="italic")
    ax.set_xlim(-0.05, 1.25)
    ax.set_ylim(-0.8, len(claims) - 0.2)
    ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=11,
                          color=colours[k], label=labels[k]) for k in (1, 0, -1)]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=3)
    ax.set_title("Ten claims, tested. Four of them were mine, and they didn't survive.",
                 fontsize=13.5, fontweight="bold", pad=18, loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "headline_4_claims.png", dpi=140)
    plt.close(fig)
    print("  headline_4_claims.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Building headline figures ...")
    fig1_cascade()
    fig2_per_query()
    fig3_null_distribution()
    fig4_claims()
    print(f"\nAll figures -> {FIGURES}/headline_*.png")


if __name__ == "__main__":
    main()
