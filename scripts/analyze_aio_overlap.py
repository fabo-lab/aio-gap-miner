#!/usr/bin/env python3
"""Analysis 2 - which cited page actually shaped the AI Overview answer?

The real Gap-Miner question, at content level: for each query with an AI
Overview, we have (a) the full AIO answer text and (b) the actual page content
of every URL Google cited (from the HTML cache). This measures how much each
cited page's content overlaps the AIO answer - i.e. which source the answer was
really built from, not just which pages were listed.

Run AFTER extract_from_cache.py:
    python scripts/analyze_aio_overlap.py

Outputs to reports/results/:
    aio_source_overlap.csv   - per (query, cited URL): similarity to the AIO text,
                               best-matching sentence, and its similarity score
    aio_overlap_summary.csv  - per query: how concentrated the answer is
And a chart: reports/figures/insight_6_source_overlap.png

Note: overlap is evidence, not proof. High overlap means "this page and the
answer share a lot of wording" - a strong indicator the page was the source,
not a demonstration that Google copied from it.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RESULTS = Path("reports/results")
FIGURES = Path("reports/figures")
HTML_DIR = Path("data/raw/_cache") / "html"


def clean_markdown(text: str) -> str:
    """Strip markdown links, bold, reference markers, and bare URLs from AIO text."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]*)\*", r"\1", text)
    text = re.sub(r"\[+\d*\]+", " ", text)
    text = re.sub(r"\(https?://[^)]*\)", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _norm_url(u: str) -> str:
    """Normalise for matching: drop scheme, leading www., trailing slash."""
    u = re.sub(r"^https?://", "", str(u).strip().lower())
    u = re.sub(r"^www\.", "", u)
    return u.rstrip("/")


def _page_text(query_id: str, idx: int) -> str:
    hf = HTML_DIR / f"{query_id}__{idx:02d}.html"
    if not hf.exists():
        return ""
    try:
        soup = BeautifulSoup(hf.read_text(encoding="utf-8", errors="replace"), "lxml")
    except OSError:
        return ""
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.split()) >= 5]


def main() -> None:
    aio_text_path = RESULTS / "aio_text.csv"
    cite_path = RESULTS / "aio_citations_detail.csv"
    if not aio_text_path.exists() or not cite_path.exists():
        raise SystemExit("Run scripts/extract_from_cache.py first.")

    aio_df = pd.read_csv(aio_text_path)
    cite_df = pd.read_csv(cite_path)
    real = pd.read_csv("data/raw/real.csv")

    idx_map: dict[str, dict[str, int]] = {}
    for qid, grp in real.groupby("query_id"):
        idx_map[qid] = {_norm_url(u): i for i, u in enumerate(grp["url"].tolist())}

    overlap_rows = []
    for _, arow in aio_df.iterrows():
        qid = arow["query_id"]
        aio_clean = clean_markdown(str(arow["aio_text"]))
        if len(aio_clean.split()) < 20:
            continue
        aio_sentences = _split_sentences(aio_clean)
        if not aio_sentences:
            continue

        for _, crow in cite_df[cite_df["query_id"] == qid].iterrows():
            url = crow["cited_url"]
            idx = idx_map.get(qid, {}).get(_norm_url(url))
            page = _page_text(qid, idx) if idx is not None else ""
            source_text = page if len(page.split()) >= 30 else str(crow.get("cited_snippet", "") or "")
            if len(source_text.split()) < 10:
                continue

            try:
                tfidf = TfidfVectorizer().fit([aio_clean, source_text])
                whole_sim = float(cosine_similarity(tfidf.transform([aio_clean]),
                                                    tfidf.transform([source_text]))[0][0])
                sent_sims = cosine_similarity(tfidf.transform(aio_sentences),
                                              tfidf.transform([source_text])).ravel()
                best_i = int(sent_sims.argmax())
            except ValueError:
                continue

            overlap_rows.append({
                "query_id": qid,
                "query": arow["query"],
                "cited_domain": crow["cited_domain"],
                "cited_url": url,
                "overlap_with_aio": round(whole_sim, 3),
                "best_sentence_sim": round(float(sent_sims[best_i]), 3),
                "best_matching_aio_sentence": aio_sentences[best_i][:300],
                "had_full_html": int(len(page.split()) >= 30),
            })

    overlap = pd.DataFrame(overlap_rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(RESULTS / "aio_source_overlap.csv", index=False)

    if overlap.empty:
        print("  aio_source_overlap.csv   (0 pairs - no scorable source text found)")
        return

    summary_rows = []
    for qid, grp in overlap.groupby("query_id"):
        srt = grp.sort_values("overlap_with_aio", ascending=False)
        summary_rows.append({
            "query_id": qid,
            "query": grp["query"].iloc[0],
            "num_sources": len(grp),
            "top_source_overlap": srt["overlap_with_aio"].iloc[0],
            "mean_source_overlap": round(grp["overlap_with_aio"].mean(), 3),
            "top_domain": srt["cited_domain"].iloc[0],
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "aio_overlap_summary.csv", index=False)

    print(f"  aio_source_overlap.csv   ({len(overlap)} query-source pairs)")
    print(f"  aio_overlap_summary.csv  ({len(summary)} queries)\n")
    print(f"Pairs scored with full page HTML: {int(overlap['had_full_html'].sum())} / {len(overlap)}")
    print(f"Median overlap (page vs AIO answer): {overlap['overlap_with_aio'].median():.3f}")
    print(f"Median best-sentence similarity:     {overlap['best_sentence_sim'].median():.3f}\n")
    print("Highest-overlap examples (a source that clearly shaped the answer):")
    for _, r in overlap.sort_values("overlap_with_aio", ascending=False).head(5).iterrows():
        print(f"  [{r['overlap_with_aio']:.2f}] {r['query'][:40]:40s} <- {r['cited_domain']}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(summary["top_source_overlap"], bins=20, color="#17becf", edgecolor="white")
    ax.set_xlabel("Overlap of the strongest source with the AI Overview answer")
    ax.set_ylabel("Number of queries")
    ax.set_title("How much the top source shapes the AI Overview answer")
    fig.tight_layout()
    fig.savefig(FIGURES / "insight_6_source_overlap.png", dpi=130)
    plt.close(fig)
    print(f"\nChart -> {FIGURES}/insight_6_source_overlap.png")


if __name__ == "__main__":
    main()
