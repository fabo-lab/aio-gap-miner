# AIO Gap-Miner

**Predicting which pages Google cites in its AI Overviews — and finding out, the hard way, what actually drives it.**

When you search Google, you often get an **AI Overview**: a written answer at the
top with a handful of source links. If your page isn't one of those sources, most
people never see you — no matter how well you rank.

This project asks what decides who gets picked. It ended up being as much about
**how to test that honestly** as about the answer, because the first four
versions of the analysis were all fooling themselves in different ways.

---

## The short version

- **Citation is driven mostly by which website a page is on**, not by measurable
  properties of the page. On websites the model has never seen, it does not beat
  "just trust the Google ranking".
- **22% of searches never show an AI Overview at all** — knowing which ones is
  the most directly actionable finding here.
- **Google does reuse source sentences near-verbatim**, about 2.5× more often
  than chance — a real but modest effect, and smaller than it first appeared.
- Four separate forms of leakage were found and removed. Three by me, one by an
  independent reviewer I asked to attack the project.

---

## The data

| | |
|---|---|
| Searches attempted | 538 (3 failed during collection) |
| Searches in the modelling data | **533** · SERP snapshots: 536 |
| (search, page) pairs | 6,646 → **4,857** after removing a leaking subset |
| Distinct URLs in those rows | **1,361** (this turns out to matter a lot) |
| Distinct domains | 752 |
| Searches with an AI Overview | 336 · with ≥1 citation: 333 |
| Pages crawled | 6,198 |
| "People also ask" questions | 1,970 (714 unique) |

German real-estate searches, collected via the DataForSEO API. The run also
cached every raw API response and every page's HTML, which is what made the later
analyses possible without re-querying anything.

---

## Finding 1 — Not every search is a game you can play

| Search type | Searches | AI Overview shown | Citation rate (ranked pages) |
|---|---|---|---|
| Local (Google shows a map) | 89 | **0%** | 0% |
| Featured snippet shown | 25 | **0%** | 0% |
| Informational | 422 | 79.6% | 38.2% |

![Intent segments](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/insight_8_intent_segments.png)

**0 of 114 searches** that showed a map or a featured snippet had an AI Overview.
Google answers those a different way. Optimising a page for them is effort spent
on something that cannot happen.

This is the finding I'd act on first — it's cheap to check and it saves budget.

*Method note: KMeans clustering was tried first and produced overlapping,
hard-to-explain groups. A transparent rule on Google's own SERP signals separates
the outcome almost perfectly. The segmentation is effectively binary, so the
count (0 of 114) is the honest way to state it, not a percentage.*

---

## Finding 2 — The model works, until you ask it to generalise

Every model below is evaluated on the same 4,857 ranked pages, with 5-fold
cross-validation, reported as mean ± standard deviation across folds.

The **grouping** column is the important one. Grouping by search stops the same
*search* leaking across the split. Grouping by domain stops the same *website*
leaking — which is the test that matters if you want to say anything about
content.

| Features | Grouped by | PR-AUC | ROC-AUC |
|---|---|---|---|
| Content only | search | 0.527 ± 0.033 | 0.727 |
| Content only | **domain** | **0.350 ± 0.052** | **0.552** |
| Content + rank | search | 0.577 ± 0.035 | 0.757 |
| Content + rank | **domain** | **0.376 ± 0.027** | **0.600** |
| *Rank-only heuristic* | — | *0.368* | *0.616* |
| *Random (prevalence)* | — | *0.291* | *0.500* |

**On websites it has never seen, the model does not beat the ranking heuristic
it was built to beat.** The impressive-looking 0.527 depends on already knowing
the site.

### Why: the model was memorising websites

| | |
|---|---|
| Rows | 4,857 |
| Distinct URLs | 1,361 |
| Rows from URLs appearing in more than one search | **82%** |
| Content features that never change for a given page | **10 of 13** |

The same page appears for many searches with an identical feature vector, so
`GroupKFold(query_id)` puts it in training *and* test. The proof:

| Feature set | Grouped by search | What it can possibly know |
|---|---|---|
| **Page-constant features only** | **0.522 ± 0.030** | nothing about query-page fit |
| Query-dependent features only | 0.402 ± 0.039 | only about query-page fit |

Features that by construction cannot say whether a page matches a search
reproduce nearly the entire score. That is site memorisation, measured directly.

---

## Finding 3 — So what does decide it? The site.

On searches that actually have an AI Overview (baseline citation rate 48.7%):

| Domain | Times ranked | Cited | 95% CI | vs baseline |
|---|---|---|---|---|
| immobilienscout24.de | 91 | 89% | [0.81, 0.94] | **1.83×** |
| drklein.de | 91 | 76% | [0.66, 0.83] | 1.56× |
| sparkasse.de | 167 | 71% | [0.63, 0.77] | 1.45× |
| test.de | 36 | 67% | [0.50, 0.80] | 1.37× |
| heid-immobilienbewertung.de | 183 | 61% | [0.53, 0.67] | 1.25× |
| check24.de | 129 | 48% | [0.40, 0.57] | 0.99× |

![Domain leaderboard](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/insight_3_top_domains.png)

Established brands are cited noticeably more often *from the same ranking
positions*. 425 domains with fewer than 30 appearances are excluded — their rates
are noise.

This is the same finding as Finding 2, seen from the other side: the signal lives
in the site, not in the page.

---

## Finding 4 — Content depth still matters, modestly

| Word count | Cited (ranked pages) | n |
|---|---|---|
| <500 | 18.7% | 455 |
| 500–1k | 24.5% | 691 |
| 1k–2k | 25.9% | 1,730 |
| 2k–4k | **36.4%** | 1,571 |
| >4k | 33.8% | 408 |

![Word count](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/insight_5_wordcount_curve.png)

Citation rate rises with depth and plateaus above ~2,000 words. An earlier
version showed a U-shape with a peak at short pages; that peak was made entirely
of cited-but-not-ranked rows. The chart plots both lines so the distortion is
visible.

**Structural features** (schema markup, FAQ sections) are genuinely associated
with citation and the association survives conditioning:

| Feature | Ranked pages | AI-Overview searches only |
|---|---|---|
| `has_schema` | OR 1.82 (p≈5e-14) | OR 1.60 (p≈7e-07) |
| `has_faq` | OR 1.89 (p≈2e-23) | OR 1.61 (p≈2e-10) |

**But permutation importance gives both ≈ 0** — the model gains no predictive
power from them. The honest phrasing: *structured pages are cited more often, but
structure adds nothing once other signals are known.* Given Finding 2, that is
plausibly another site effect.

---

## Finding 5 — Google does reuse source sentences, modestly

Comparing each AI Overview's text against the real text of its cited pages, and
counting sentence pairs that share **8 or more consecutive words**:

| | |
|---|---|
| Observed pairs | **31** (24 searches, 12 domains) |
| Expected by chance (5 permutation replicates) | 12 (range 9–20) |
| **Ratio** | **2.5× above chance** |

A real example:

> **Page (sparkasse.de):** "Ein vollständiges Verkehrswertgutachten kostet in der Regel zwischen 0,5 und 1,5 Prozent des Verkehrswerts der Immobilie."
>
> **Google's answer:** "Ein rechtssicheres Verkehrswertgutachten kostet in der Regel zwischen 0,5 und 1,5 Prozent des Immobilienwerts."

Eleven consecutive words identical.

**What survives testing**, comparing the reused sentences against length-matched
sentences from the same pages, with Holm correction:

| Sentence contains | Reused | Control | OR | p (Holm) | |
|---|---|---|---|---|---|
| a number | 68% | 19% | 9.2 | 0.0012 | holds |
| a percentage | 32% | 0% | ∞ | 0.002 | holds |
| a price in € | 20% | 9% | 2.4 | 0.50 | **not significant** |
| a definition cue | 0% | 9% | — | 0.50 | **not significant** |

**What an earlier version got wrong, and why it's worth saying:**
the first pass used TF-IDF cosine similarity, reported 245 "near-verbatim" pairs,
and claimed sentences with prices were 10× more likely to be reused. All three
were wrong. TF-IDF weights rare tokens heavily and digits are rare, so the
matcher was *selecting* number-bearing sentences. The pairs shared a median of
3 words, which is shared vocabulary, not reuse. And the same pages were counted
multiple times. After deduplicating, switching to an n-gram criterion, matching
the control group on sentence length, and running a permutation null, the price
effect disappears entirely.

The surviving claim is narrow: **quantitative sentences are reused more often,
about 2.5× above chance, on a sample of 31 pairs.** Exploratory, not settled.

---

## Finding 6 — Two common SEO beliefs this data doesn't support

| Claim | Result |
|---|---|
| "Lead with a definition" | Not significant after correction (p = 0.50) |
| "AI loves lists" | Never significant |

Both were reported as findings in an earlier version of this project. Neither
survived a multiple-comparison correction.

---

## The leakage audit — the actual point of this project

Four ways the model could cheat. Three found by me, the fourth by an independent
reviewer.

**1. A placeholder that encoded the answer.** Cited pages absent from Google's
visible results got a placeholder rank of 101. Such pages are in the data *only
because* they were cited, so all 1,789 of those rows are `cited = 1` by
construction — 27% of the dataset.

**2. A feature computed from the answer.** `domain_citation_rate` was calculated
from the same citations being predicted. For 572 single-occurrence domains it
*was* the label.

**3. Indirect leakage through a subgroup.** Keeping the rank-101 rows while
dropping `organic_rank` still leaked: those rows are ~9× more likely to be video
pages, so the model could identify the always-cited subgroup through other
features. It scored 0.773 overall but 0.479 on genuinely ranked pages.

**4. Leakage in the cross-validation itself.** Documented in Finding 2 — pages
repeat across searches with identical features, so grouping by search was not
enough. This is the largest of the four and it inverted the project's original
headline claim.

**And a sweep of the descriptive statistics.** After leak 3, the same artefact
turned out to have distorted non-model results too:

- "YouTube is cited 95% of the time" — 92% of its rows were rank-101. Among
  ranked video pages the rate is **15.8%** versus **29.3%** for non-video: the
  opposite conclusion.
- "Citation rate is U-shaped in word count" — the short-page peak was entirely
  rank-101 rows.
- The intent-segment citation rate was 53% over all rows, **38%** on ranked pages.

`scripts/audit_artifacts.py` now recomputes every descriptive finding both ways.

---

## Methodological corrections worth naming

Beyond the leaks, an independent review caught these:

**The population answers two questions at once.** 205 of 533 searches never show
an AI Overview, making 40% of ranked rows `cited = 0` by construction. Conditional
on searches that do have one, prevalence rises from 0.291 to 0.487, and the
model's lift over random drops from **1.81× to 1.51×**. The conditional number is
the honest headline.

**The calibration error was a hyperparameter.** `is_unbalance=True` deliberately
inflates positive probabilities. Switching it off drops the calibration error
from **0.109 to 0.049**. All results here use `is_unbalance=False`.

**One holdout split is a lottery draw.** Over 50 random splits the held-out score
is **0.578 ± 0.041, range [0.463, 0.669]**. A single number from that range means
little; the previously reported 0.535 was a low draw.

**Model differences need a test, not a comparison of point estimates.** Paired
bootstrap over searches: content+rank beats content-only by **+0.050, 95% CI
[+0.034, +0.066]**. Real, but small. Across 5 seeds the model varies by only
0.007, so the difference is not seed noise.

**Robustness:** 357 rows had a failed crawl, and their semantic features come
from the SERP snippet — text Google chose *because* it matches the query.
Excluding them moves PR-AUC from 0.527 to 0.546. Stable.

**Features too sparse to support a claim:** `is_forum` (4 rows) and `is_video`
(95 rows). Read as "no conclusion possible", not as evidence.

---

## What you'd actually do with this

1. **Check the search type first** — 22% can never show an AI Overview
2. **Be realistic about brand** — site identity dominates page-level tweaks
3. **Use the model where it works** — scoring pages on sites you already know is
   a real workflow; scoring an unknown site is not
4. **Write quantitatively** — numbers and percentages are reused more often, on a
   small sample
5. **Test the advice you're given** — two widely repeated tips didn't survive

---

## How the pipeline works

```
Collect searches (DataForSEO: SERP + AI Overview citations)
        ↓
Crawl every candidate page, extract content features
        ↓
Audit for leakage · restrict to one clean population
        ↓
Train LightGBM + Logistic Regression, grouped by search AND by domain
        ↓
Explain with SHAP · harden with holdout, calibration, permutation, bootstrap
        ↓
Mine the cached raw data: AI Overview text, PAA questions, reused sentences
        ↓
Export tidy tables for Tableau
```

**Why PR-AUC, not accuracy?** Only 29% of ranked pages get cited. A model that
always says "not cited" would look 71% accurate and be useless.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

python scripts/generate_sample_data.py   # synthetic demo data, no API key needed
python scripts/run_pipeline.py
pytest -q
```

### Collect your own data

```bash
export DATAFORSEO_LOGIN="your_login"
export DATAFORSEO_PASSWORD="your_password"

python scripts/collect_real_data.py --queries your_queries.txt --out data/raw/real.csv
```

Credentials live in environment variables or a local `.env` — never in code, never
committed (see `.env.example`).

### The analysis

All of it runs off the local cache written during collection
(`data/raw/_cache/`), so no step needs new API calls.

```bash
python scripts/extract_from_cache.py           # AI Overview text, PAA questions, citations
python scripts/run_final_analysis.py           # Findings 2 + all robustness checks
python scripts/fix_descriptive_stats.py        # Findings 1, 3, 4 on the clean population
python scripts/audit_artifacts.py              # the artefact sweep
python scripts/analyze_passages_v2.py --min-ngram 8   # Finding 5 with a permutation null
python scripts/analyze_paa.py                  # question themes
python scripts/prepare_tableau.py              # tidy tables for the dashboard
python scripts/verify_setup.py                 # check everything is present and current
```

**Privacy:** real query lists, collected data, the cache, and per-row analysis
outputs are git-ignored — they're business research. Charts and aggregate
summaries in `reports/` are public: those are the findings.

---

## Repo structure

```
aio-gap-miner/
├── src/aio_gap_miner/     # package: config, data, features, model, evaluate, explain, collect/
├── scripts/               # one script per question, each documenting what it corrects
├── reports/
│   ├── figures/           # all charts (public)
│   └── results/           # aggregate summaries (public), row-level data (git-ignored)
├── notebooks/             # methodology walkthrough on synthetic data
├── tableau/               # dashboard spec + data source
└── tests/                 # automated checks
```

---

## Limitations

- **One vertical, one country, one snapshot.** SERPs change daily.
- **Correlation, not causation.** SHAP explains the model, not Google. Proving
  cause needs an intervention: change a page, see whether the citation follows.
- **Restricting to ranked pages drops 56% of all citations** (1,789 of 3,201).
  The model answers a narrower question than the project started with, and the
  excluded rows differ systematically — it is a real selection, not a random one.
- **The model does not generalise to unseen websites.** Stated as a finding
  above, but it is also the main limitation.
- **The passage analysis rests on 31 pairs.** Directional, not settled.
- **Two known measurement flaws** in the collected features, documented rather
  than fixed because they'd require re-collection: the TF-IDF similarity used an
  English stopword list on German text, and the vectoriser was refit per page,
  so `query_url_similarity` and `passage_match_score` are not strictly comparable
  across rows.
- **`structure_score` is defined two ways** in the codebase (weighted in
  `features.py`, additive in the analysis scripts). All reported results use the
  additive definition.

---

## Tech stack

Python · pandas · scikit-learn · LightGBM · SHAP · scipy · SQLAlchemy/SQLite ·
seaborn · matplotlib · DataForSEO API · BeautifulSoup · Tableau · Git · pytest

## License

MIT — see [LICENSE](LICENSE).
