#!/usr/bin/env python3
"""Passage analysis, rebuilt - n-gram overlap with a permutation null.

Why this replaces analyze_passages.py and analyze_passage_patterns.py
--------------------------------------------------------------------
The independent review found four problems with the first version:

1. NO NULL. Similarity was compared against nothing. With enough sentence pairs,
   some will match by chance - and the threshold (0.45) sat *below* the 99th
   percentile of a rough null distribution. "245 matches" had no reference point.

2. "NEAR-VERBATIM" WAS WRONG. Across those 245 pairs the median longest shared
   word sequence was 3 words. That is shared vocabulary, not reuse. Only ~6%
   shared 8 or more consecutive words.

3. THE NUMBER EFFECT WAS PARTLY THE MEASURE. TF-IDF weights rare tokens heavily,
   and digits are rare. So a similarity-based matcher preferentially selects
   number-bearing sentences - inflating the "sentences with prices are 10x more
   likely" result.

4. THE CONTROL GROUP WAS WRONG. Lifted sentences came from a small set of pages;
   the control was sampled from a much larger set. Page composition and sentence
   properties were confounded.

What this version does instead
------------------------------
* Defines reuse as a **shared sequence of >= MIN_NGRAM consecutive words**. That
  is what "lifted" is supposed to mean, and it is not driven by token rarity.
* Runs a **permutation test**: the same pipeline with AI Overviews paired against
  pages from *other* searches. The observed count is only meaningful relative to
  that null.
* Draws the control sentences from **exactly the same (search, page) pairs** that
  produced a match, so page composition is held constant.
* Applies **Holm correction** across the sentence characteristics tested.

    python scripts/analyze_passages_v2.py
    python scripts/analyze_passages_v2.py --min-ngram 6   # looser definition
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

RESULTS = Path("reports/results")
HTML_DIR = Path("data/raw/_cache") / "html"
SEED = 42

DEFINITION_CUES = (" ist ein", " ist eine", " ist der", " ist die", " ist das",
                   " bedeutet", " bezeichnet", " versteht man", " definiert")


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


def normalise_words(text: str) -> list[str]:
    """Lowercase word tokens, punctuation stripped - the unit for n-gram matching."""
    return re.findall(r"[a-zäöüßA-ZÄÖÜ0-9€%.,-]+", text.lower())


def ngrams(words: list[str], n: int) -> set[tuple]:
    if len(words) < n:
        return set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def longest_common_run(a: list[str], b: list[str]) -> int:
    """Length of the longest run of consecutive words appearing in both."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def split_sentences(text: str, min_words: int = 6) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.split()) >= min_words]


def page_sentences(query_id: str, idx: int) -> list[str]:
    hf = HTML_DIR / f"{query_id}__{idx:02d}.html"
    if not hf.exists():
        return []
    try:
        soup = BeautifulSoup(hf.read_text(encoding="utf-8", errors="replace"), "lxml")
    except OSError:
        return []
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return split_sentences(re.sub(r"\s+", " ", soup.get_text(" ")))


def find_matches(aio_sents: list[str], src_sents: list[str], min_n: int) -> list[dict]:
    """Sentence pairs sharing at least min_n consecutive words."""
    src_tokens = [normalise_words(s) for s in src_sents]
    src_grams = [ngrams(t, min_n) for t in src_tokens]
    out = []
    for a in aio_sents:
        at = normalise_words(a)
        ag = ngrams(at, min_n)
        if not ag:
            continue
        for j, sg in enumerate(src_grams):
            if ag & sg:
                run = longest_common_run(at, src_tokens[j])
                out.append({"aio_sentence": a[:400], "source_sentence": src_sents[j][:400],
                            "shared_run_words": run})
                break  # one match per AIO sentence is enough
    return out


def profile(sentence: str) -> dict:
    low = sentence.lower()
    return {
        "n_words": len(sentence.split()),
        "has_number": int(bool(re.search(r"\d", sentence))),
        "has_currency": int(bool(re.search(r"(euro|€|\beur\b)", low))),
        "has_percent": int("%" in sentence or "prozent" in low),
        "has_definition_cue": int(any(c in low for c in DEFINITION_CUES)),
    }


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-ngram", type=int, default=8,
                   help="Consecutive words that must be shared to count as reuse.")
    p.add_argument("--permutations", type=int, default=5,
                   help="Null replicates (each pairs every AI Overview with another search's pages).")
    args = p.parse_args()
    MIN_N = args.min_ngram

    aio_path = RESULTS / "aio_text.csv"
    cite_path = RESULTS / "aio_citations_detail.csv"
    if not aio_path.exists() or not cite_path.exists():
        raise SystemExit("Run scripts/extract_from_cache.py first.")
    if not HTML_DIR.exists():
        raise SystemExit(f"HTML cache not found at {HTML_DIR}.")

    aio_df = pd.read_csv(aio_path)
    cite_df = pd.read_csv(cite_path)
    real = pd.read_csv("data/raw/real.csv")

    idx_map: dict[str, dict[str, int]] = {}
    for qid, grp in real.groupby("query_id"):
        idx_map[qid] = {_norm_url(u): i for i, u in enumerate(grp["url"].tolist())}

    # Pre-load sentences for every (query, cited page) pair, capped for runtime.
    print(f"Loading cached pages (n-gram threshold: {MIN_N} consecutive words) ...")
    pairs: list[tuple] = []          # (query_id, query, domain, src_sentences)
    aio_by_qid: dict[str, list[str]] = {}
    for _, arow in aio_df.iterrows():
        qid = arow["query_id"]
        sents = split_sentences(clean_markdown(str(arow["aio_text"])))
        if not sents:
            continue
        aio_by_qid[qid] = sents
        # Deduplicate: the citation table can list the same URL for a query more
        # than once, which would process the same page repeatedly and inflate
        # every count downstream.
        seen_idx = set()
        for _, c in cite_df[cite_df["query_id"] == qid].drop_duplicates(
                subset=["cited_url"]).iterrows():
            idx = idx_map.get(qid, {}).get(_norm_url(c["cited_url"]))
            if idx is None or idx in seen_idx:
                continue
            seen_idx.add(idx)
            src = page_sentences(qid, idx)
            if src:
                pairs.append((qid, arow["query"], c["cited_domain"], src))
    print(f"  {len(aio_by_qid)} AI Overviews, {len(pairs)} (search, cited page) pairs with cached HTML\n")

    # ---- Observed matches
    obs_rows = []
    matched_pairs = set()
    for qid, query, domain, src in pairs:
        for m in find_matches(aio_by_qid[qid], src, MIN_N):
            m.update({"query_id": qid, "query": query, "source_domain": domain})
            obs_rows.append(m)
            matched_pairs.add((qid, domain))
    observed = pd.DataFrame(obs_rows)
    if len(observed):
        observed = observed.drop_duplicates(subset=["query_id", "aio_sentence", "source_sentence"])
    n_obs = len(observed)
    print(f"OBSERVED: {n_obs} sentence pairs share >= {MIN_N} consecutive words")
    if n_obs:
        print(f"  across {observed['query_id'].nunique()} searches, "
              f"{observed['source_domain'].nunique()} domains")
        print(f"  median shared run: {observed['shared_run_words'].median():.0f} words | "
              f"max: {observed['shared_run_words'].max():.0f}")

    # ---- Permutation null: same pipeline, mismatched pairings
    rng = random.Random(SEED)
    qids = list(aio_by_qid.keys())
    null_counts = []
    for rep in range(args.permutations):
        count = 0
        for qid, _, _, src in pairs:
            others = [q for q in qids if q != qid]
            fake_qid = rng.choice(others)
            count += len(find_matches(aio_by_qid[fake_qid], src, MIN_N))
        null_counts.append(count)
        print(f"NULL replicate {rep + 1}: {count} pairs from mismatched pairings")

    null_mean = sum(null_counts) / len(null_counts) if null_counts else 0
    print(f"\n  Observed {n_obs} vs null mean {null_mean:.1f}"
          + (f"  ->  {n_obs / null_mean:.1f}x above chance" if null_mean else "  ->  null is zero"))
    if null_counts:
        import statistics
        n_ge = sum(1 for c in null_counts if c >= n_obs)
        # (n_ge + 1) / (B + 1) is the standard estimator - it can never report p = 0,
        # which would be a claim the number of replicates cannot support.
        pval = (n_ge + 1) / (len(null_counts) + 1)
        lo, hi = sorted(null_counts)[int(0.025 * len(null_counts))], \
            sorted(null_counts)[min(int(0.975 * len(null_counts)), len(null_counts) - 1)]
        print(f"    null distribution: median {statistics.median(null_counts):.0f}, "
              f"95% range [{lo}, {hi}], max {max(null_counts)}")
        print(f"    replicates reaching {n_obs} or more: {n_ge} of {len(null_counts)}")
        print(f"    permutation p = {pval:.4f}"
              + ("  (significant)" if pval < 0.05 else "  (not significant)"))
        if len(null_counts) < 100:
            print(f"    NOTE: with {len(null_counts)} replicates the smallest attainable")
            print(f"          p-value is {1 / (len(null_counts) + 1):.3f} - run more for a real one.")
    if null_mean == 0 and n_obs > 0:
        print("  A shared 8-word sequence essentially never happens by chance here,")
        print("  which is what makes this criterion defensible.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    if n_obs:
        observed.sort_values("shared_run_words", ascending=False).to_csv(
            RESULTS / "passage_matches_ngram.csv", index=False)
        print(f"\n  Table -> {RESULTS}/passage_matches_ngram.csv")
        print("\n  Strongest reuse examples:")
        for _, r in observed.sort_values("shared_run_words", ascending=False).head(5).iterrows():
            print(f"\n    [{int(r['shared_run_words'])} shared words] {r['query'][:45]}  ({r['source_domain']})")
            print(f"      PAGE: {r['source_sentence'][:130]}")
            print(f"      AIO : {r['aio_sentence'][:130]}")

    # ---- Trait comparison with a matched control group
    if n_obs >= 20:
        print("\n" + "-" * 74)
        print("  Sentence characteristics - control drawn from the SAME matched pages")
        print("-" * 74)
        lifted_set = set(observed["source_sentence"])
        # A SEPARATE generator: the permutation loop above consumes random numbers,
        # so sharing one would make the control group depend on --permutations.
        # It did, and it changed the conclusions between runs.
        ctrl_rng = random.Random(SEED + 1)
        control = []
        for qid, _, domain, src in pairs:
            if (qid, domain) not in matched_pairs:
                continue
            pool = [s for s in src if s[:400] not in lifted_set]
            control.extend(ctrl_rng.sample(pool, k=min(5, len(pool))))
        lift_prof = pd.DataFrame([profile(s) for s in observed["source_sentence"]])
        ctrl_prof = pd.DataFrame([profile(s) for s in control])
        print(f"  {len(lift_prof)} reused sentences vs {len(ctrl_prof)} control sentences "
              f"from the same pages")

        # Length is a confound: an n-gram criterion mechanically favours longer
        # sentences, and longer sentences carry more numbers. Match the control
        # group on sentence length so the comparison isolates the characteristic.
        med_l, med_c = lift_prof["n_words"].median(), ctrl_prof["n_words"].median()
        print(f"  Median length before matching: reused {med_l:.0f} vs control {med_c:.0f} words")
        if abs(med_l - med_c) > 2:
            lo, hi = lift_prof["n_words"].quantile([0.1, 0.9])
            keep = (ctrl_prof["n_words"] >= lo) & (ctrl_prof["n_words"] <= hi)
            keep_l = (lift_prof["n_words"] >= lo) & (lift_prof["n_words"] <= hi)
            ctrl_prof = ctrl_prof[keep].reset_index(drop=True)
            lift_prof = lift_prof[keep_l].reset_index(drop=True)
            print(f"  -> length-matched to [{lo:.0f}, {hi:.0f}] words: "
                  f"{len(lift_prof)} reused vs {len(ctrl_prof)} control "
                  f"(medians {lift_prof['n_words'].median():.0f} / "
                  f"{ctrl_prof['n_words'].median():.0f})")
        print()

        try:
            from scipy.stats import fisher_exact
            traits = ["has_number", "has_currency", "has_percent", "has_definition_cue"]
            raw_p, stats = [], []
            for t in traits:
                a, b = int(lift_prof[t].sum()), int(len(lift_prof) - lift_prof[t].sum())
                c, d = int(ctrl_prof[t].sum()), int(len(ctrl_prof) - ctrl_prof[t].sum())
                odds, pv = fisher_exact([[a, b], [c, d]])
                raw_p.append(pv)
                stats.append((t, lift_prof[t].mean(), ctrl_prof[t].mean(), odds))
            adj = holm(raw_p)
            print(f"  {'characteristic':22s} {'reused':>8s} {'control':>9s} {'OR':>7s} {'p (Holm)':>11s}")
            out_rows = []
            for (t, lm, cm, odds), pv in zip(stats, adj):
                verdict = "holds" if pv < 0.05 else "not significant"
                print(f"  {t:22s} {lm:7.1%} {cm:8.1%} {odds:7.2f} {pv:11.2g}  {verdict}")
                out_rows.append({"characteristic": t, "reused_rate": round(float(lm), 3),
                                 "control_rate": round(float(cm), 3), "odds_ratio": round(float(odds), 2),
                                 "p_holm": float(pv), "significant": bool(pv < 0.05)})
            pd.DataFrame(out_rows).to_csv(RESULTS / "passage_traits_ngram.csv", index=False)
            print(f"\n  Table -> {RESULTS}/passage_traits_ngram.csv")
        except ImportError:
            print("  (scipy not installed - skipping significance tests)")
        print(f"\n  Median length: reused {lift_prof['n_words'].median():.0f} words | "
              f"control {ctrl_prof['n_words'].median():.0f} words")
    else:
        print(f"\n  Only {n_obs} matches - too few for a trait comparison. Report the")
        print("  examples as qualitative evidence, not as a rate.")


if __name__ == "__main__":
    main()
