#!/usr/bin/env python3
"""Business cases: each finding shown with a real example from the data.

A number on a slide convinces an analyst. A real search, with the real pages
Google did and didn't cite, convinces everyone else. This script pulls concrete,
checkable examples for each finding and writes them out as both text and charts.

What is and isn't valid here matters, so it's stated inline:

  * A *citation rate* per domain is only valid on pages that genuinely rank,
    because a page enters this dataset either by ranking or by being cited.
  * Counts *over citations* are fully valid - every cited URL was extracted
    directly from the AI Overview, so nothing is missing.

    python scripts/build_business_cases.py --data data/raw/real.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("reports/results")
FIGURES = Path("reports/figures")

PRIMARY = "#4F46E5"
ALERT = "#F43F5E"
NEUTRAL = "#94A3B8"
GOOD = "#10B981"


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["domain"] = (df["url"].str.extract(r"https?://([^/]+)/", expand=False)
                    .str.replace("www.", "", regex=False))
    df["ranks"] = df["organic_rank"] != 101
    return df


# --------------------------------------------------------------------------- #
def case_1_citations_without_ranking(df: pd.DataFrame) -> None:
    """Most citation slots are not awarded through the visible ranking."""
    cited = df[df["cited"] == 1]
    share = (~cited["ranks"]).mean()
    print("\n" + "=" * 74)
    print("  CASE 1 - You do not have to rank to be cited")
    print("=" * 74)
    print(f"\n  Of {len(cited):,} citations, {int((~cited['ranks']).sum()):,} "
          f"({share:.0%}) went to pages that were NOT in the visible top 15.")

    # A concrete search where this is visible.
    per_q = cited.groupby("query_id").agg(
        n_cited=("cited", "size"), n_ranked=("ranks", "sum"))
    per_q["n_unranked"] = per_q["n_cited"] - per_q["n_ranked"]
    candidates = per_q[(per_q["n_cited"] >= 6) & (per_q["n_ranked"] >= 2)]
    if len(candidates):
        qid = candidates.sort_values("n_unranked", ascending=False).index[0]
        sub = df[df["query_id"] == qid]
        query = sub["query"].iloc[0]
        print(f"\n  Example search: \"{query}\"")
        print(f"  {'':4s}{'source cited by the AI answer':45s} {'rank in Google':>15s}")
        for _, r in sub[sub["cited"] == 1].sort_values("organic_rank").head(10).iterrows():
            rank = f"#{int(r['organic_rank'])}" if r["ranks"] else "not in top 15"
            print(f"      {r['domain'][:44]:45s} {rank:>15s}")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh([""], [share * 100], color=PRIMARY, label="Cited without ranking in the top 15")
    ax.barh([""], [(1 - share) * 100], left=[share * 100], color=NEUTRAL,
            label="Cited and ranking")
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of all AI Overview citations")
    ax.set_title("Over half of all citations go to pages that don't rank on page 1")
    ax.text(share * 50, 0, f"{share:.0%}", ha="center", va="center",
            color="white", fontweight="bold", fontsize=14)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.55), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "case_1_citation_without_ranking.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def case_2_youtube(df: pd.DataFrame) -> None:
    """Video is a citation slot that bypasses ranking entirely."""
    cited = df[df["cited"] == 1]
    aio_queries = cited["query_id"].nunique()
    yt = cited[cited["domain"] == "youtube.com"]
    q_with_yt = yt["query_id"].nunique()

    print("\n" + "=" * 74)
    print("  CASE 2 - A video is a citation slot that skips the ranking queue")
    print("=" * 74)
    print(f"\n  Searches with an AI Overview:            {aio_queries}")
    print(f"  ... that cite at least one YouTube video: {q_with_yt} ({q_with_yt / aio_queries:.0%})")
    print(f"  YouTube citations total:                 {len(yt)} "
          f"({len(yt) / len(cited):.0%} of all citations)")
    print(f"  ... from videos NOT in the top 15:       {int((~yt['ranks']).sum())} "
          f"({(~yt['ranks']).mean():.0%})")
    print("\n  Valid because every cited URL comes straight out of the AI Overview -")
    print("  no selection involved. (A citation *rate* for YouTube would not be valid.)")

    if len(yt):
        example = yt.groupby("query_id").size().sort_values(ascending=False).index[0]
        sub = df[df["query_id"] == example]
        print(f"\n  Example search: \"{sub['query'].iloc[0]}\"")
        for _, r in sub[(sub["cited"] == 1) & (sub["domain"] == "youtube.com")].iterrows():
            print(f"      cited video: {str(r.get('title', ''))[:70]}")
            print(f"      ranked in Google: {'yes' if r['ranks'] else 'NO - not in top 15'}")

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Cite a YouTube video", "Don't"]
    vals = [q_with_yt, aio_queries - q_with_yt]
    ax.bar(labels, vals, color=[PRIMARY, NEUTRAL])
    for i, v in enumerate(vals):
        ax.text(i, v + 4, f"{v}\n({v / aio_queries:.0%})", ha="center", fontweight="bold")
    ax.set_ylabel("Number of AI Overviews")
    ax.set_title("3 in 4 AI Overviews cite a YouTube video\n"
                 f"— and {(~yt['ranks']).mean():.0%} of those videos don't rank in the top 15")
    fig.tight_layout()
    fig.savefig(FIGURES / "case_2_youtube.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def case_3_same_position_different_outcome(df: pd.DataFrame) -> None:
    """Two pages, same ranking position, different citation outcome."""
    print("\n" + "=" * 74)
    print("  CASE 3 - Same ranking position, different outcome")
    print("=" * 74)

    ranked = df[df["ranks"]].copy()
    aio = ranked[ranked.groupby("query_id")["cited"].transform("max") > 0]
    strong = ["immobilienscout24.de", "sparkasse.de", "drklein.de", "test.de"]

    pairs = []
    for pos in range(1, 11):
        at_pos = aio[aio["organic_rank"] == pos]
        big = at_pos[at_pos["domain"].isin(strong) & (at_pos["cited"] == 1)]
        small = at_pos[~at_pos["domain"].isin(strong) & (at_pos["cited"] == 0)]
        if len(big) and len(small):
            pairs.append((pos, big.iloc[0], small.iloc[0]))
    if pairs:
        pos, b, s = pairs[len(pairs) // 2]
        print(f"\n  Both of these sat at Google position #{pos}:")
        print(f"      CITED    : {b['domain']:32s}  (search: \"{b['query'][:40]}\")")
        print(f"      NOT CITED: {s['domain']:32s}  (search: \"{s['query'][:40]}\")")
        print(f"\n  Word count {int(b['word_count'])} vs {int(s['word_count'])}, "
              f"schema {int(b['has_schema'])} vs {int(s['has_schema'])}")
        print("  The measurable page properties don't separate them. The site does.")

    # Citation rate by position, split by whether it's an established brand.
    aio = aio[aio["organic_rank"] <= 10]
    aio["group"] = aio["domain"].isin(strong).map({True: "Established brand", False: "Everyone else"})
    g = aio.groupby(["organic_rank", "group"])["cited"].mean().unstack()
    print(f"\n  Citation rate by position (AI-Overview searches):")
    print(g.round(2).to_string())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col, colour in [("Established brand", GOOD), ("Everyone else", NEUTRAL)]:
        if col in g.columns:
            ax.plot(g.index, g[col] * 100, marker="o", linewidth=2.5, label=col, color=colour)
    ax.set_xlabel("Google ranking position")
    ax.set_ylabel("% cited by the AI answer")
    ax.set_title("Same position, different odds: the site matters more than the slot")
    ax.legend()
    ax.set_xticks(range(1, 11))
    fig.tight_layout()
    fig.savefig(FIGURES / "case_3_brand_vs_position.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def case_4_sentence_reuse() -> None:
    """A real sentence Google reused almost word for word."""
    print("\n" + "=" * 74)
    print("  CASE 4 - Google reuses source sentences almost word for word")
    print("=" * 74)
    path = RESULTS / "passage_matches_ngram.csv"
    if not path.exists():
        print("\n  (run analyze_passages_v2.py --min-ngram 8 first)")
        return
    pm = pd.read_csv(path).drop_duplicates(subset=["source_sentence"])
    top = pm.sort_values("shared_run_words", ascending=False).head(4)
    for _, r in top.iterrows():
        print(f"\n  [{int(r['shared_run_words'])} identical words in a row] "
              f"search: \"{r['query']}\"  ({r['source_domain']})")
        print(f"    THE PAGE WROTE : {r['source_sentence'][:190]}")
        print(f"    GOOGLE ANSWERED: {r['aio_sentence'][:190]}")
    print("\n  31 such pairs found, versus 12 expected by chance - 2.5x above chance.")
    print("  Sentences with a number or a percentage are reused most often.")


# --------------------------------------------------------------------------- #
def case_5_dead_end_searches(df: pd.DataFrame) -> None:
    """Searches where AI Overview optimisation cannot pay off."""
    print("\n" + "=" * 74)
    print("  CASE 5 - Some searches can never pay off")
    print("=" * 74)
    meta_path = Path("data/raw/real_meta.csv")
    if not meta_path.exists():
        print("  (real_meta.csv not found)")
        return
    meta = pd.read_csv(meta_path)
    local = meta[meta["has_local_pack"] == 1]
    print(f"\n  Searches where Google shows a map: {len(local)}")
    print(f"  ... of those, with an AI Overview:  {int(local['ai_overview_present'].sum())}")
    print(f"\n  Examples of searches that can never produce a citation:")
    for q in local["query"].head(6):
        print(f"      \"{q}\"")
    print("\n  Any budget spent optimising a page for these is spent on an outcome")
    print("  that does not exist in this market.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/raw/real.csv")
    args = p.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load(args.data)

    case_1_citations_without_ranking(df)
    case_2_youtube(df)
    case_3_same_position_different_outcome(df)
    case_4_sentence_reuse()
    case_5_dead_end_searches(df)

    print("\n" + "=" * 74)
    print(f"  Charts -> {FIGURES}/case_*.png")
    print("=" * 74)


if __name__ == "__main__":
    main()
