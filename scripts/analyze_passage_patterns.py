#!/usr/bin/env python3
"""Analysis 6 - what KIND of sentence gets lifted into an AI Overview?

analyze_passages.py found the near-verbatim sentence pairs. This asks the more
useful question: what do those sentences have in common? It profiles the lifted
source sentences (do they contain numbers? a formula? a definition? how long are
they?) and compares them against sentences from the same pages that were NOT
lifted. The comparison is what makes it a finding rather than a list of examples:
if lifted sentences contain numbers twice as often as non-lifted ones, that's a
concrete, testable writing rule.

Run AFTER analyze_passages.py:
    python scripts/analyze_passage_patterns.py

Outputs to reports/results/:
    passage_patterns.csv        - per lifted sentence: its characteristics
    passage_pattern_summary.csv - lifted vs not-lifted comparison
And a chart: reports/figures/insight_9_passage_patterns.png
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup

RESULTS = Path("reports/results")
FIGURES = Path("reports/figures")
HTML_DIR = Path("data/raw/_cache") / "html"
SEED = 42

# Sentence characteristics worth testing, each a concrete writing choice.
DEFINITION_CUES = (
    " ist ein", " ist eine", " ist der", " ist die", " ist das",
    " bedeutet", " bezeichnet", " versteht man", " definiert",
)
LIST_CUES = ("erstens", "zweitens", "folgende", "zum einen", "zum anderen", ":")


def profile_sentence(s: str) -> dict:
    """Turn one sentence into measurable characteristics."""
    low = s.lower()
    return {
        "n_words": len(s.split()),
        "has_number": int(bool(re.search(r"\d", s))),
        "has_currency": int(bool(re.search(r"(euro|€|\beur\b)", low))),
        "has_percent": int("%" in s or "prozent" in low),
        "has_formula": int(bool(re.search(r"=|×|\*|\+", s))),
        "has_definition_cue": int(any(c in low for c in DEFINITION_CUES)),
        "has_list_cue": int(any(c in low for c in LIST_CUES)),
        "starts_capital": int(bool(s[:1].isupper())),
    }


def _page_sentences(query_id: str, idx: int, min_words: int = 6) -> list[str]:
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
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.split()) >= min_words]


def _norm_url(u: str) -> str:
    u = re.sub(r"^https?://", "", str(u).strip().lower())
    u = re.sub(r"^www\.", "", u)
    return u.rstrip("/")


def main() -> None:
    pm_path = RESULTS / "passage_matches.csv"
    if not pm_path.exists():
        raise SystemExit("Run scripts/analyze_passages.py first.")

    pm = pd.read_csv(pm_path)
    if pm.empty:
        raise SystemExit("passage_matches.csv is empty.")

    # ---- Profile the LIFTED sentences.
    lifted = pd.DataFrame([profile_sentence(str(s)) for s in pm["source_sentence"]])
    lifted["group"] = "Lifted into AI Overview"
    lifted["sentence"] = pm["source_sentence"].values
    lifted["similarity"] = pm["similarity"].values
    lifted["query"] = pm["query"].values
    lifted["source_domain"] = pm["source_domain"].values

    # ---- Build a comparison set: sentences from the SAME pages that were not lifted.
    overlap_path = RESULTS / "aio_source_overlap.csv"
    real_path = Path("data/raw/real.csv")
    control_rows = []
    if overlap_path.exists() and real_path.exists():
        overlap = pd.read_csv(overlap_path)
        real = pd.read_csv(real_path)
        idx_map: dict[str, dict[str, int]] = {}
        for qid, grp in real.groupby("query_id"):
            idx_map[qid] = {_norm_url(u): i for i, u in enumerate(grp["url"].tolist())}

        lifted_set = set(pm["source_sentence"].astype(str))
        rng = random.Random(SEED)
        # Sample from the same top sources the lifted sentences came from.
        pairs = overlap.sort_values("overlap_with_aio", ascending=False).groupby("query_id").head(3)
        for _, r in pairs.iterrows():
            idx = idx_map.get(r["query_id"], {}).get(_norm_url(r["cited_url"]))
            if idx is None:
                continue
            sents = [s for s in _page_sentences(r["query_id"], idx) if s not in lifted_set]
            if not sents:
                continue
            for s in rng.sample(sents, k=min(3, len(sents))):
                control_rows.append(s)

    control = pd.DataFrame([profile_sentence(s) for s in control_rows]) if control_rows else pd.DataFrame()
    if not control.empty:
        control["group"] = "Not lifted (same pages)"

    RESULTS.mkdir(parents=True, exist_ok=True)
    lifted.to_csv(RESULTS / "passage_patterns.csv", index=False)

    traits = ["has_number", "has_currency", "has_percent", "has_formula",
              "has_definition_cue", "has_list_cue"]

    print(f"Profiled {len(lifted)} lifted sentences", end="")
    if not control.empty:
        print(f" vs {len(control)} not-lifted sentences from the same pages.\n")
    else:
        print(" (no control set available).\n")

    print(f"  Median length of a lifted sentence: {lifted['n_words'].median():.0f} words", end="")
    if not control.empty:
        print(f"   (not lifted: {control['n_words'].median():.0f} words)")
    else:
        print()
    print()

    rows = []
    print(f"  {'characteristic':22s} {'lifted':>9s} {'not lifted':>12s} {'ratio':>8s}")
    for t in traits:
        l_rate = lifted[t].mean()
        c_rate = control[t].mean() if not control.empty else float("nan")
        ratio = (l_rate / c_rate) if (not control.empty and c_rate > 0) else float("nan")
        rows.append({"characteristic": t, "lifted_rate": round(l_rate, 3),
                     "not_lifted_rate": round(c_rate, 3) if c_rate == c_rate else None,
                     "ratio": round(ratio, 2) if ratio == ratio else None})
        c_txt = f"{c_rate:11.1%}" if c_rate == c_rate else f"{'n/a':>11s}"
        r_txt = f"{ratio:7.2f}x" if ratio == ratio else f"{'n/a':>8s}"
        print(f"  {t:22s} {l_rate:8.1%} {c_txt} {r_txt}")

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "passage_pattern_summary.csv", index=False)

    # Which domains get lifted most often - who is "feeding" the AI?
    print("\n  Sources most often lifted from:")
    for dom, n in lifted["source_domain"].value_counts().head(6).items():
        print(f"    {n:3d}x  {dom}")

    # Chart: lifted vs not-lifted trait rates.
    if not control.empty:
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        y = range(len(traits))
        h = 0.38
        labels = {
            "has_number": "Contains a number",
            "has_currency": "Contains a price (€)",
            "has_percent": "Contains a percentage",
            "has_formula": "Contains a formula",
            "has_definition_cue": "Defines something",
            "has_list_cue": "Introduces a list",
        }
        ax.barh([i + h / 2 for i in y], [lifted[t].mean() * 100 for t in traits],
                height=h, label="Lifted into AI Overview", color="#2ca02c")
        ax.barh([i - h / 2 for i in y], [control[t].mean() * 100 for t in traits],
                height=h, label="Not lifted (same pages)", color="#c7c7c7")
        ax.set_yticks(list(y))
        ax.set_yticklabels([labels[t] for t in traits])
        ax.set_xlabel("% of sentences with this characteristic")
        ax.set_title("What kind of sentence does an AI Overview lift?")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "insight_9_passage_patterns.png", dpi=130)
        plt.close(fig)
        print(f"\nChart -> {FIGURES}/insight_9_passage_patterns.png")

    print(f"\nTables -> {RESULTS}/passage_patterns.csv, passage_pattern_summary.csv")


if __name__ == "__main__":
    main()
