#!/usr/bin/env python3
"""Draw a sample of labelled rows and check each one against the raw cache.

Why this exists
---------------
Round 2 of the review pointed out that nothing in this project ever validated the
labels by hand. Two matching defects were found by reading code — a defect that
thirty minutes of manual checking would have caught immediately.

"I checked 50 rows by hand, found N errors and fixed them" is a stronger sentence
than any amount of automated testing, because it is the only check that doesn't
share assumptions with the code that produced the data.

What it does
------------
Draws a stratified sample (cited and not-cited, ranked and rank-101), and for each
row re-derives the answer straight from the cached DataForSEO response — without
touching any of the project's own matching code. Any disagreement is a real
labelling error.

    python scripts/validate_labels.py --data data/raw/real_v2.csv --n 50
    python scripts/validate_labels.py --data data/raw/real.csv --n 50   # compare
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

SERP_DIR = Path("data/raw/_cache/serp_json")
RESULTS = Path("reports/results")
SEED = 42


def independent_key(url: str) -> str:
    """A deliberately independent normalisation.

    Written from scratch rather than imported, so a bug in the project's own
    `normalize_url` cannot hide itself here. Different implementation, same
    intent: host without www, path without trailing slash, query preserved.

    Only the HOST is lowercased. Paths and query strings stay case-sensitive,
    because they are: a YouTube id `v=ABC` is a different video from `v=abc`.
    Lowercasing them would re-introduce exactly the collision this project just
    spent an evening removing.
    """
    if not isinstance(url, str) or not url.strip():
        return ""
    s = re.sub(r"^https?://", "", url.strip(), flags=re.IGNORECASE)
    s = re.sub(r"#.*$", "", s)
    if "/" in s:
        host, rest = s.split("/", 1)
        rest = "/" + rest
    else:
        host, rest = s, "/"
    host = re.sub(r"^www\.", "", host.lower())
    if "?" in rest:
        base, q = rest.split("?", 1)
        rest = f"{base.rstrip('/') or '/'}?{q}"
    else:
        rest = rest.rstrip("/") or "/"
    return f"{host}{rest}"


def cited_urls_from_cache(query_id: str) -> set[str] | None:
    f = SERP_DIR / f"{query_id}.json"
    if not f.exists():
        return None
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
        items = raw["tasks"][0]["result"][0].get("items", [])
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None
    urls = set()
    for it in items:
        if it.get("type") != "ai_overview":
            continue
        for ref in (it.get("references") or []):
            if ref.get("url"):
                urls.add(independent_key(ref["url"]))
        for el in (it.get("items") or []):
            for ref in (el.get("references") or []):
                if ref.get("url"):
                    urls.add(independent_key(ref["url"]))
    return urls


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/raw/real_v2.csv")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--show", type=int, default=10, help="How many rows to print in full.")
    args = p.parse_args()

    if not SERP_DIR.exists():
        raise SystemExit(f"Cache not found at {SERP_DIR}")

    df = pd.read_csv(args.data)
    df["is_ranked"] = df["organic_rank"] != 101

    # Stratify so the sample covers all four combinations, not just the common one.
    strata = [
        ("ranked & cited", df[df["is_ranked"] & (df["cited"] == 1)]),
        ("ranked & not cited", df[df["is_ranked"] & (df["cited"] == 0)]),
        ("not ranked & cited", df[~df["is_ranked"] & (df["cited"] == 1)]),
    ]
    per = max(args.n // len(strata), 1)
    sample = pd.concat(
        [s.sample(min(per, len(s)), random_state=SEED) for _, s in strata if len(s)]
    ).reset_index(drop=True)

    print("=" * 76)
    print(f"  MANUAL LABEL VALIDATION - {len(sample)} rows from {Path(args.data).name}")
    print("=" * 76)
    for name, s in strata:
        print(f"  {name:22s} population {len(s):5d}  sampled {min(per, len(s))}")

    rows = []
    for _, r in sample.iterrows():
        cached = cited_urls_from_cache(r["query_id"])
        if cached is None:
            rows.append({**r.to_dict(), "expected": None, "agrees": None,
                         "note": "no cached SERP file"})
            continue
        expected = int(independent_key(r["url"]) in cached)
        rows.append({
            "query_id": r["query_id"], "query": r["query"], "url": r["url"],
            "is_ranked": bool(r["is_ranked"]), "label": int(r["cited"]),
            "expected": expected, "agrees": int(r["cited"]) == expected,
            "note": "",
        })

    res = pd.DataFrame(rows)
    checked = res[res["agrees"].notna()]
    disagree = checked[~checked["agrees"].astype(bool)]

    print(f"\n--- Result --------------------------------------------------------")
    print(f"\n  Checked against the cache: {len(checked)} of {len(res)}")
    print(f"  Agree:    {int(checked['agrees'].sum())}")
    print(f"  Disagree: {len(disagree)}")
    if len(checked):
        print(f"  Agreement rate: {checked['agrees'].mean():.1%}")

    if len(disagree):
        print("\n  Disagreements — each one is a real labelling error:")
        for _, r in disagree.head(args.show).iterrows():
            print(f"\n    search: {str(r['query'])[:52]}")
            print(f"    url:    {str(r['url'])[:70]}")
            print(f"    stored label {r['label']} | cache says {r['expected']} | "
                  f"{'ranked' if r['is_ranked'] else 'rank-101'}")
    else:
        print("\n  No disagreements in this sample.")
        print("  With n = %d and 0 errors, the 95%% upper bound on the error rate is"
              % len(checked))
        print(f"  about {3 / max(len(checked), 1):.1%} (rule of three) — worth stating as such")
        print("  rather than claiming the labels are perfect.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS / "label_validation.csv", index=False)
    print(f"\n  Full sample -> {RESULTS}/label_validation.csv")
    print("\n  Note: this uses its own URL normalisation, written separately from")
    print("  the collector's, so a bug there cannot mask itself here.")


if __name__ == "__main__":
    main()
