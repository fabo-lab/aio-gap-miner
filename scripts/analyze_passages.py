#!/usr/bin/env python3
"""Analysis 4 - the "smoking gun": which exact sentences did the AI Overview lift?

The overlap analysis showed *which source* shaped each AI Overview. This goes one
level deeper: for the top sources, it lines up each sentence of the source page
against each sentence of the AI Overview answer and finds the near-identical
pairs. The result is concrete evidence - "this exact sentence on this page became
that sentence in Google's answer" - which is the most persuasive thing you can
show about how AI Overviews are built.

Run AFTER extract_from_cache.py and analyze_aio_overlap.py:
    python scripts/analyze_passages.py

Outputs to reports/results/:
    passage_matches.csv   - source sentence <-> AIO sentence pairs, with similarity,
                            sorted so the clearest near-verbatim lifts are on top.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RESULTS = Path("reports/results")
HTML_DIR = Path("data/raw/_cache") / "html"

# Only look at the strongest sources per query, and only keep sentence pairs above
# this similarity - below it, "matches" are coincidental shared words, not lifts.
TOP_SOURCES_PER_QUERY = 3
MIN_PAIR_SIM = 0.45


def clean_markdown(text: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]*)\*", r"\1", text)
    text = re.sub(r"\[+\d*\]+", " ", text)
    text = re.sub(r"\(https?://[^)]*\)", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _norm_url(u: str) -> str:
    u = re.sub(r"^https?://", "", str(u).strip().lower())
    u = re.sub(r"^www\.", "", u)
    return u.rstrip("/")


def _split_sentences(text: str, min_words: int = 6) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.split()) >= min_words]


def _page_sentences(query_id: str, idx: int) -> list[str]:
    hf = HTML_DIR / f"{query_id}__{idx:02d}.html"
    if not hf.exists():
        return []
    try:
        soup = BeautifulSoup(hf.read_text(encoding="utf-8", errors="replace"), "lxml")
    except OSError:
        return []
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" "))
    return _split_sentences(text)


def main() -> None:
    overlap_path = RESULTS / "aio_source_overlap.csv"
    aio_path = RESULTS / "aio_text.csv"
    if not overlap_path.exists() or not aio_path.exists():
        raise SystemExit("Run scripts/extract_from_cache.py and analyze_aio_overlap.py first.")

    overlap = pd.read_csv(overlap_path)
    aio = pd.read_csv(aio_path).set_index("query_id")
    real = pd.read_csv("data/raw/real.csv")

    idx_map: dict[str, dict[str, int]] = {}
    for qid, grp in real.groupby("query_id"):
        idx_map[qid] = {_norm_url(u): i for i, u in enumerate(grp["url"].tolist())}

    pair_rows = []
    for qid, grp in overlap.groupby("query_id"):
        if qid not in aio.index:
            continue
        aio_text = clean_markdown(str(aio.loc[qid, "aio_text"]))
        aio_sents = _split_sentences(aio_text)
        if not aio_sents:
            continue
        query = grp["query"].iloc[0]

        top_sources = grp.sort_values("overlap_with_aio", ascending=False).head(TOP_SOURCES_PER_QUERY)
        for _, srow in top_sources.iterrows():
            idx = idx_map.get(qid, {}).get(_norm_url(srow["cited_url"]))
            if idx is None:
                continue
            src_sents = _page_sentences(qid, idx)
            if not src_sents:
                continue

            corpus = aio_sents + src_sents
            try:
                tfidf = TfidfVectorizer().fit(corpus)
                aio_vecs = tfidf.transform(aio_sents)
                src_vecs = tfidf.transform(src_sents)
            except ValueError:
                continue
            sims = cosine_similarity(aio_vecs, src_vecs)

            for ai_i, ai_sent in enumerate(aio_sents):
                j = int(sims[ai_i].argmax())
                sim = float(sims[ai_i][j])
                if sim >= MIN_PAIR_SIM:
                    pair_rows.append({
                        "query": query,
                        "source_domain": srow["cited_domain"],
                        "similarity": round(sim, 3),
                        "source_sentence": src_sents[j][:400],
                        "ai_overview_sentence": ai_sent[:400],
                    })

    pairs = pd.DataFrame(pair_rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if pairs.empty:
        pairs.to_csv(RESULTS / "passage_matches.csv", index=False)
        print("  passage_matches.csv   (0 pairs found - try lowering MIN_PAIR_SIM)")
        return

    pairs = pairs.sort_values("similarity", ascending=False)
    pairs = pairs.drop_duplicates(subset=["ai_overview_sentence", "source_domain"]).reset_index(drop=True)
    pairs.to_csv(RESULTS / "passage_matches.csv", index=False)

    print(f"  passage_matches.csv   ({len(pairs)} near-verbatim sentence pairs)")
    print(f"\n  Queries covered: {pairs['query'].nunique()}")
    print(f"  Median similarity: {pairs['similarity'].median():.2f}")
    print(f"\nStrongest lifts (similarity >= {MIN_PAIR_SIM}):\n")
    for _, r in pairs.head(8).iterrows():
        print(f"  [{r['similarity']:.2f}] {r['query']}  (source: {r['source_domain']})")
        print(f"     PAGE : {r['source_sentence'][:160]}")
        print(f"     AIO  : {r['ai_overview_sentence'][:160]}\n")


if __name__ == "__main__":
    main()
