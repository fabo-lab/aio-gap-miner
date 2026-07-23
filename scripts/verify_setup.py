#!/usr/bin/env python3
"""Verify the project is complete, current, and internally consistent.

Run this after moving folders, restoring a backup, or any time you want to be
sure the results on disk actually match the code that produced them.

It checks four things:
  1. Inputs      - is the raw data and the cache present, with the expected sizes?
  2. Outputs     - is every expected result file and chart there?
  3. Freshness   - was anything generated BEFORE the script that produces it was
                   last changed? (that means it's stale and needs a re-run)
  4. Consistency - do the headline numbers recompute from the data on disk?

    python scripts/verify_setup.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

OK = "  OK  "
WARN = " WARN "
FAIL = " FAIL "

issues: list[str] = []


def check(condition: bool, label: str, detail: str = "", warn_only: bool = False) -> bool:
    tag = OK if condition else (WARN if warn_only else FAIL)
    print(f"[{tag}] {label}" + (f"  — {detail}" if detail else ""))
    if not condition and not warn_only:
        issues.append(label)
    return condition


def main() -> None:
    print("=" * 74)
    print("  PROJECT VERIFICATION")
    print(f"  Working directory: {Path.cwd()}")
    print("=" * 74)

    # ---------------------------------------------------------------- 1. inputs
    print("\n--- 1. Inputs -------------------------------------------------------")
    real = Path("data/raw/real.csv")
    meta = Path("data/raw/real_meta.csv")
    cache_json = Path("data/raw/_cache/serp_json")
    cache_html = Path("data/raw/_cache/html")

    if check(real.exists(), "data/raw/real.csv exists"):
        df = pd.read_csv(real)
        check(len(df) > 6000, "real.csv row count", f"{len(df):,} rows, "
              f"{df['query_id'].nunique()} queries, {df['cited'].mean():.1%} cited")
        check("organic_rank" in df.columns and "cited" in df.columns,
              "real.csv has the expected columns")
    else:
        df = None

    check(meta.exists(), "data/raw/real_meta.csv exists")

    n_json = len(list(cache_json.glob("*.json"))) if cache_json.exists() else 0
    n_html = len(list(cache_html.glob("*.html"))) if cache_html.exists() else 0
    check(n_json > 500, "raw SERP JSON cache", f"{n_json} files", warn_only=True)
    check(n_html > 5000, "raw HTML cache", f"{n_html} files", warn_only=True)
    if n_html == 0:
        print("         (without the HTML cache, analyze_passages.py and")
        print("          analyze_aio_overlap.py cannot be re-run)")

    check(Path(".env").exists(), "credentials file (.env) present", warn_only=True)

    # --------------------------------------------------------------- 2. outputs
    print("\n--- 2. Expected outputs ---------------------------------------------")
    expected_results = [
        "headline_comparison.csv", "artifact_audit.csv",
        "domain_citation_ranked_only.csv", "query_segment_summary.csv",
        "passage_pattern_summary.csv",
    ]
    for f in expected_results:
        check(Path("reports/results", f).exists(), f"reports/results/{f}")

    expected_figures = [
        "insight_3_top_domains.png", "insight_5_wordcount_curve.png",
        "insight_8_intent_segments.png", "insight_9_passage_patterns.png",
    ]
    for f in expected_figures:
        check(Path("reports/figures", f).exists(), f"reports/figures/{f}")

    # ------------------------------------------------------------- 3. freshness
    print("\n--- 3. Freshness (is any output older than the script that makes it?) --")
    pairs = [
        ("scripts/analyze_domains.py", "reports/figures/insight_3_top_domains.png"),
        ("scripts/audit_artifacts.py", "reports/figures/insight_5_wordcount_curve.png"),
        ("scripts/run_headline_comparison.py", "reports/results/headline_comparison.csv"),
        ("scripts/analyze_query_intent.py", "reports/results/query_segment_summary.csv"),
        ("scripts/analyze_passage_patterns.py", "reports/results/passage_pattern_summary.csv"),
    ]
    for script, output in pairs:
        sp, op = Path(script), Path(output)
        if not sp.exists() or not op.exists():
            check(False, f"{op.name} vs {sp.name}", "one of them is missing", warn_only=True)
            continue
        fresh = op.stat().st_mtime >= sp.stat().st_mtime
        check(fresh, f"{op.name} is newer than {sp.name}",
              "" if fresh else "STALE — re-run that script", warn_only=not fresh)

    # ----------------------------------------------------------- 4. consistency
    print("\n--- 4. Do the headline numbers recompute? ---------------------------")
    if df is not None:
        ranked = df[df["organic_rank"] != 101]
        check(abs(ranked["cited"].mean() - 0.291) < 0.01,
              "citation rate on ranked pages ≈ 29.1%",
              f"{ranked['cited'].mean():.1%}")
        check(int((df["organic_rank"] == 101).sum()) == 1789,
              "rank-101 sentinel rows = 1,789",
              f"{int((df['organic_rank'] == 101).sum()):,}")
        check(df[df["organic_rank"] == 101]["cited"].mean() == 1.0,
              "all rank-101 rows are cited (the known artefact)")

        vid = ranked[ranked["is_video"] == 1]["cited"].mean()
        check(vid < ranked["cited"].mean(),
              "video pages cited LESS than average on ranked pages",
              f"{vid:.1%} vs {ranked['cited'].mean():.1%} — the corrected finding")

    hc = Path("reports/results/headline_comparison.csv")
    if hc.exists():
        h = pd.read_csv(hc)
        content = h[h["model"].str.contains("content signals only", case=False)]
        rank = h[h["model"].str.contains("Rank-only", case=False)]
        if len(content) and len(rank):
            c, r = float(content["pr_auc"].iloc[0]), float(rank["pr_auc"].iloc[0])
            check(c > r, "content-only model beats rank-only baseline",
                  f"{c:.3f} vs {r:.3f}")

    # ------------------------------------------------------------------ 5. git
    print("\n--- 5. Git safety ---------------------------------------------------")
    try:
        tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout
        leaks = [ln for ln in tracked.splitlines()
                 if any(k in ln for k in ("tableau_", "paa_questions", "aio_text",
                                          "passage_matches", "query_segments",
                                          "queries_538", ".env"))
                 and not ln.endswith(".env.example")]
        check(not leaks, "no private files tracked by git",
              "" if not leaks else f"FOUND: {leaks}")
        status = subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip()
        check(True, "uncommitted changes",
              f"{len(status.splitlines())} file(s)" if status else "working tree clean",
              warn_only=True)
    except FileNotFoundError:
        check(False, "git available", "git not found", warn_only=True)

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 74)
    if issues:
        print(f"  {len(issues)} problem(s) found:")
        for i in issues:
            print(f"    - {i}")
    else:
        print("  All critical checks passed.")
    print("=" * 74)


if __name__ == "__main__":
    main()
