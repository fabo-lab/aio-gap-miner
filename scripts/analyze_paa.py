#!/usr/bin/env python3
"""Analysis 1 — cluster the 'People Also Ask' questions into content themes.

Every PAA question is a real question users ask that Google surfaces — a direct
map of content gaps to fill. With ~2,000 of them, clustering reveals the main
themes and which single questions recur most across queries. This turns raw
questions into a prioritised content plan.

Run AFTER extract_from_cache.py:
    python scripts/analyze_paa.py

Outputs to reports/results/:
    paa_clusters.csv          — every question with its assigned theme cluster
    paa_cluster_summary.csv   — per cluster: size + representative questions
    paa_top_questions.csv     — the most frequently recurring questions
And a chart: reports/figures/insight_7_paa_themes.png
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

RESULTS = Path("reports/results")
FIGURES = Path("reports/figures")

# German stopwords (lightweight list — enough for question clustering).
GERMAN_STOP = [
    "der", "die", "das", "und", "ist", "ich", "für", "fur", "wie", "was", "wo",
    "wann", "warum", "welche", "welcher", "welches", "ein", "eine", "einen",
    "mein", "meine", "meiner", "man", "kann", "muss", "sich", "auf", "mit",
    "von", "zu", "im", "in", "den", "dem", "des", "es", "am", "an", "bei",
    "wird", "werden", "sind", "hat", "haben", "oder", "auch", "als", "nach",
    "bis", "aus", "um", "so", "wieviel", "viel", "gibt",
]


def main() -> None:
    paa_path = RESULTS / "paa_questions.csv"
    if not paa_path.exists():
        raise SystemExit("Run scripts/extract_from_cache.py first.")

    paa = pd.read_csv(paa_path)
    paa["q_norm"] = paa["paa_question"].str.strip().str.lower()

    # Most frequently recurring exact questions across all queries.
    top_q = (
        paa.groupby("paa_question").size().sort_values(ascending=False)
        .head(30).rename("count").reset_index()
    )
    top_q.to_csv(RESULTS / "paa_top_questions.csv", index=False)

    # Cluster unique questions by TF-IDF + KMeans.
    unique_q = paa["paa_question"].dropna().drop_duplicates().tolist()
    n_clusters = min(10, max(2, len(unique_q) // 40))

    vec = TfidfVectorizer(stop_words=GERMAN_STOP, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(unique_q)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    qdf = pd.DataFrame({"paa_question": unique_q, "cluster": labels})
    # Attach frequency so we can show high-impact clusters.
    freq = paa.groupby("paa_question").size().rename("count")
    qdf = qdf.merge(freq, on="paa_question", how="left")
    qdf.to_csv(RESULTS / "paa_clusters.csv", index=False)

    # Per-cluster summary: label each cluster by its top TF-IDF terms.
    terms = vec.get_feature_names_out()
    summary_rows = []
    for c in range(n_clusters):
        centroid = km.cluster_centers_[c]
        top_terms = [terms[i] for i in centroid.argsort()[::-1][:4]]
        members = qdf[qdf["cluster"] == c].sort_values("count", ascending=False)
        summary_rows.append({
            "cluster": c,
            "size": len(members),
            "total_occurrences": int(members["count"].sum()),
            "top_terms": ", ".join(top_terms),
            "example_questions": " | ".join(members["paa_question"].head(3).tolist()),
        })
    summary = pd.DataFrame(summary_rows).sort_values("total_occurrences", ascending=False)
    summary.to_csv(RESULTS / "paa_cluster_summary.csv", index=False)

    print(f"  paa_clusters.csv          ({len(qdf)} unique questions, {n_clusters} clusters)")
    print(f"  paa_cluster_summary.csv   ({n_clusters} themes)")
    print(f"  paa_top_questions.csv     (top 30 recurring questions)")
    print()
    print("Question themes, by how often they appear:")
    for _, r in summary.iterrows():
        print(f"  [{r['total_occurrences']:4d}x] {r['top_terms']}")
        print(f"          e.g. {r['example_questions'][:90]}")
    print()
    print("Most recurring individual questions:")
    for _, r in top_q.head(8).iterrows():
        print(f"  {r['count']:3d}x  {r['paa_question']}")

    # Chart: theme sizes.
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    s = summary.sort_values("total_occurrences")
    labels_short = [t[:35] for t in s["top_terms"]]
    ax.barh(labels_short, s["total_occurrences"], color="#8c564b")
    ax.set_xlabel("Total question occurrences in this theme")
    ax.set_title("What people ask about real-estate valuation (PAA themes)")
    fig.tight_layout()
    fig.savefig(FIGURES / "insight_7_paa_themes.png", dpi=130)
    plt.close(fig)
    print(f"\nChart -> {FIGURES}/insight_7_paa_themes.png")


if __name__ == "__main__":
    main()
