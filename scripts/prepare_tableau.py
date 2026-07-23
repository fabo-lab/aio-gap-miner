#!/usr/bin/env python3
"""Build tidy, Tableau-ready tables for the AIO Gap-Miner dashboard.

Tableau works best with flat, "tidy" tables: one row per thing, one column per
attribute, no merged headers, no wide pivots. This produces five such tables,
each backing one dashboard sheet of the story:

  1. tableau_queries.csv    - one row per search query: intent segment, whether an
                              AI Overview appeared, citation rate  -> "which searches
                              are even worth optimising?"
  2. tableau_pages.csv      - one row per (query, page): all content features, the
                              model's predicted probability, whether it was cited,
                              and its single strongest SHAP driver -> "why this page?"
  3. tableau_features.csv   - long-format feature importance for both variants
                              -> "what drives citation overall?"
  4. tableau_domains.csv    - per domain: appearances, citations, citation rate
                              -> "who is winning the AI Overview game?"
  5. tableau_questions.csv  - PAA questions with theme + frequency (if available)
                              -> "what should I actually write about?"

    python scripts/prepare_tableau.py --data data/raw/real.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

RESULTS = Path("reports/results")
OUT = Path("tableau")

CONTENT_NUMERIC = [
    "word_count", "has_schema", "num_lists_tables", "has_faq",
    "query_url_similarity", "passage_match_score", "content_freshness_days",
    "num_entities_matched", "readability_score", "is_https", "is_forum", "is_video",
]
CATEGORICAL = ["content_type"]
LGBM_PARAMS = dict(
    objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
    is_unbalance=True, random_state=42, verbose=-1,
)

# Plain-English names so the dashboard is readable by a non-technical audience.
FEATURE_LABELS = {
    "word_count": "Content depth (word count)",
    "query_url_similarity": "Topic match to the search",
    "readability_score": "Readability",
    "passage_match_score": "Best passage match",
    "content_freshness_days": "Content freshness",
    "num_lists_tables": "Lists & tables",
    "num_entities_matched": "Key terms covered",
    "structure_score": "Structured markup",
    "has_schema": "Has schema markup",
    "has_faq": "Has FAQ section",
    "is_video": "Is a video page",
    "is_forum": "Is a forum page",
    "is_https": "Uses HTTPS",
    "content_type": "Page type",
    "organic_rank": "Google rank",
    "rank_reciprocal": "Google rank (inverted)",
}


def segment_of(row: pd.Series) -> str:
    if row.get("has_local_pack", 0) == 1:
        return "Local (map)"
    if row.get("has_featured_snippet", 0) == 1:
        return "Featured snippet"
    return "Informational"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/raw/real.csv")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)

    # ---------------------------------------------------------------- queries
    enriched = RESULTS / "serp_features_enriched.csv"
    meta_path = enriched if enriched.exists() else Path(args.data).with_name("real_meta.csv")
    meta = pd.read_csv(meta_path)
    # Citation rate per query from the page-level data (robust across meta sources).
    rate = df.groupby("query_id")["cited"].mean().rename("citation_rate")
    meta = meta.merge(rate, on="query_id", how="left")
    meta["citation_rate"] = meta["citation_rate"].fillna(0.0).round(4)
    meta["intent_segment"] = meta.apply(segment_of, axis=1)
    meta["aio_shown"] = meta["ai_overview_present"].map({1: "Yes", 0: "No"})
    qcols = ["query_id", "query", "intent_segment", "aio_shown", "ai_overview_present"]
    for c in ["citation_rate", "num_candidates", "num_cited", "num_organic_results",
              "ai_overview_num_references", "has_local_pack", "has_people_also_ask",
              "has_featured_snippet", "has_video", "has_knowledge_graph"]:
        if c in meta.columns:
            qcols.append(c)
    meta[qcols].to_csv(OUT / "tableau_queries.csv", index=False)
    print(f"  tableau_queries.csv    ({len(meta)} queries)")

    # ------------------------------------------------------------------ pages
    work = df.copy()
    work["structure_score"] = (
        work["has_schema"].fillna(0) + work["has_faq"].fillna(0)
        + (work["num_lists_tables"].fillna(0) > 3).astype(int)
    )
    X = work[CONTENT_NUMERIC + ["structure_score"] + CATEGORICAL].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    y = work["cited"].astype(int)

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X, y)
    work["predicted_probability"] = model.predict_proba(X)[:, 1].round(4)

    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    top_idx = np.abs(sv).argmax(axis=1)
    work["top_driver"] = [FEATURE_LABELS.get(X.columns[i], X.columns[i]) for i in top_idx]
    work["top_driver_effect"] = [
        "increases chance" if sv[r, i] > 0 else "decreases chance"
        for r, i in enumerate(top_idx)
    ]
    work["cited_label"] = work["cited"].map({1: "Cited", 0: "Not cited"})
    work["domain"] = work["url"].str.extract(r"https?://([^/]+)/?", expand=False).str.replace("www.", "", regex=False)

    # A "gap" = model says likely, reality says not cited -> the actionable list.
    thr = work["predicted_probability"].quantile(0.75)
    work["gap_flag"] = np.where(
        (work["predicted_probability"] >= thr) & (work["cited"] == 0),
        "Missed opportunity", "—",
    )

    page_cols = [
        "query_id", "query", "url", "domain", "cited", "cited_label",
        "predicted_probability", "top_driver", "top_driver_effect", "gap_flag",
        "organic_rank", "word_count", "query_url_similarity", "readability_score",
        "passage_match_score", "num_lists_tables", "has_schema", "has_faq",
        "content_freshness_days", "num_entities_matched", "structure_score",
        "is_video", "is_forum", "content_type",
    ]
    page_cols = [c for c in page_cols if c in work.columns]
    # Attach the query's intent segment so the dashboard can filter pages by intent.
    work = work.merge(meta[["query_id", "intent_segment"]], on="query_id", how="left")
    page_cols.append("intent_segment")
    work[page_cols].to_csv(OUT / "tableau_pages.csv", index=False)
    print(f"  tableau_pages.csv      ({len(work)} pages, incl. predictions + SHAP driver)")

    # --------------------------------------------------------------- features
    rows = []
    for variant in ("A", "B"):
        f = RESULTS / f"shap_importance_variant_{variant}.csv"
        if not f.exists():
            continue
        imp = pd.read_csv(f)
        col = "mean_abs_shap" if "mean_abs_shap" in imp.columns else imp.columns[1]
        for _, r in imp.iterrows():
            rows.append({
                "variant": f"Variant {variant}",
                "feature": r["feature"],
                "feature_label": FEATURE_LABELS.get(r["feature"], r["feature"]),
                "importance": round(float(r[col]), 4),
            })
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "tableau_features.csv", index=False)
        print(f"  tableau_features.csv   ({len(rows)} feature-importance rows)")
    else:
        print("  tableau_features.csv   (skipped - run run_variants.py first)")

    # ---------------------------------------------------------------- domains
    dom = (
        work.groupby("domain")
        .agg(appearances=("cited", "size"), citations=("cited", "sum"))
        .reset_index()
    )
    dom["citation_rate"] = (dom["citations"] / dom["appearances"]).round(3)
    dom = dom[dom["appearances"] >= 5].sort_values("citations", ascending=False)
    dom.to_csv(OUT / "tableau_domains.csv", index=False)
    print(f"  tableau_domains.csv    ({len(dom)} domains with 5+ appearances)")

    # -------------------------------------------------------------- questions
    paa_path = RESULTS / "paa_questions.csv"
    clusters_path = RESULTS / "paa_clusters.csv"
    if paa_path.exists():
        paa = pd.read_csv(paa_path)
        counts = paa.groupby("paa_question").size().rename("times_asked").reset_index()
        if clusters_path.exists():
            cl = pd.read_csv(clusters_path)[["paa_question", "cluster"]]
            counts = counts.merge(cl, on="paa_question", how="left")
        counts = counts.sort_values("times_asked", ascending=False)
        counts.to_csv(OUT / "tableau_questions.csv", index=False)
        print(f"  tableau_questions.csv  ({len(counts)} unique questions)")
    else:
        print("  tableau_questions.csv  (skipped - run extract_from_cache.py first)")

    print(f"\nAll Tableau tables written to {OUT}/")
    print("Note: these contain your real query data -> keep private (already git-ignored "
          "via tableau/real_*.csv? rename if you want them ignored).")


if __name__ == "__main__":
    main()
