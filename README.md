# AIO Gap-Miner

**Predicting which pages Google cites in its AI Overviews — and finding out, five corrections later, that the honest answer is "not much better than chance".**

When you search Google, you often get an **AI Overview**: a written answer at the
top with a handful of source links. If your page isn't one of those sources, most
people never see you — no matter how well you rank.

This project set out to model what decides that. It became a project about **how
to test such a model honestly**, because five separate versions of the analysis
were fooling themselves in different ways. The final answer is smaller than the
first one, and it is the only one that survives scrutiny.

*Data collected 16–17 July 2026.*

---

## The short version

- **Under the strictest honest test — unseen search *and* unseen website — the
  model performs inside the permutation null.** It does not predict citation
  better than chance.
- **Knowing only the domain is about as informative as knowing only the ranking**
  (per-search AP 0.728 vs 0.719, against 0.609 for random). Measurable page
  properties add little once the site is unknown.
- **56% of all citations go to pages that don't rank** in the visible results.
- **3 of 4 AI Overviews cite a YouTube video**, averaging 2.4 videos each.
- **22% of searches never show an AI Overview** — the one cheaply actionable finding.
- **Five kinds of leakage found and removed.** Three by me, two by independent
  reviewers I asked to attack the project.

---

## The data

| | |
|---|---|
| Searches attempted | 538 (3 failed during collection) |
| Searches in the modelling data | **533** · SERP snapshots: 536 |
| (search, page) pairs | 6,646 → **4,857** after removing a leaking subset |
| Distinct URLs in those rows | **1,361** (this turns out to matter enormously) |
| Distinct domains | 752 |
| AI Overviews captured | 336 |
| Pages crawled | 6,198 |
| "People also ask" questions | 1,970 (714 unique) |

German real-estate searches via the DataForSEO API. Every raw API response and
every page's HTML was cached during collection, which is what allowed every later
correction to be made without re-querying anything.

---

## Finding 1 — The model does not generalise. That is the result.

Every model on the same 4,857 ranked pages. Three cross-validation schemes, each
stricter than the last, each with its own **permutation null**: labels shuffled
*within* each search, everything else identical. Prevalence (0.291) is not the
floor — a model can beat it simply by detecting which searches have an AI
Overview at all.

| Cross-validation | Model | Permutation null | Margin | |
|---|---|---|---|---|
| Grouped by search | 0.525 | 0.382 ± 0.006 | +0.142 | above the null |
| Grouped by domain | 0.347 | 0.310 ± 0.016 | +0.037 | barely above |
| **Double-blocked** (unseen search **and** unseen domain) | **0.310** | **0.312 ± 0.040** | **−0.002** | **inside the null** |

*Rank-only heuristic: 0.368. Random: 0.291.*

Under the only scheme that blocks both leakage paths, the content model scores
**inside its own null distribution**.

### Why: it was memorising websites

| | |
|---|---|
| Rows | 4,857 |
| Distinct URLs | 1,361 |
| Rows from URLs appearing in more than one search | **82%** |
| Content features that never change for a given page | **10 of 13** |

The same page appears for many searches with an identical feature vector, so
grouping by search puts it in training *and* test. The proof: features that by
construction cannot say whether a page fits a search (`page-constant only`)
reproduce nearly the full query-grouped score, 0.522 against 0.527.

---

## Finding 2 — The site carries the signal, not the page

The label is query-relative: *which of these ~9 candidates does Google cite?*
Pooled PR-AUC mixes that with *which searches have citations at all*, so the
honest metric is **average precision computed within each search**, then averaged.
**Precision@3** answers what a practitioner would ask.

| Predictor | Per-search AP | P@3 |
|---|---|---|
| Random | 0.609 | 0.493 |
| Rank-only heuristic | 0.719 | 0.647 |
| **Domain identity only** (out-of-fold encoded) | **0.728** | 0.650 |
| Content, grouped by search *(leaky)* | 0.745 | 0.661 |
| **Content, grouped by domain** | **0.634** | 0.539 |

**Knowing only which website a page is on is as informative as knowing only its
ranking position.** Content signals evaluated on unseen websites (0.634) sit far
closer to random (0.609) than to either.

The domain-only baseline uses out-of-fold target encoding, so unlike the original
`domain_citation_rate` feature it does not leak.

---

## Finding 3 — But the ranking isn't the gate

**1,789 of 3,201 citations (56%) went to pages not in the visible results.**
A real example — *"verkehrswertermittlung eigentumswohnung"*:

| Source Google cited | Its rank |
|---|---|
| sparkasse.de | #3 |
| immobilienscout24.de | #5 |
| check24.de | **not ranked** |
| mcmakler.de | **not ranked** |

![Citations without ranking](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/case_1_citation_without_ranking.png)

---

## Finding 4 — Not every search is a game you can play

| Search type | Searches | AI Overview shown |
|---|---|---|
| Local (Google shows a map) | 89 | **0%** |
| Featured snippet shown | 25 | **0%** |
| Informational | 422 | 79.6% |

**0 of 114** searches showing a map or snippet had an AI Overview (95% CI [0, 3.2%]
by the rule of three). Citation rate among informational searches, on ranked
pages: **38.2%**.

This is the one finding that is both robust and free to act on.

---

## Finding 5 — Video is heavily cited

**These numbers were recounted from the raw cache after a matching bug was found
(see the leakage audit, item 5); the earlier published figures were wrong.**

| | |
|---|---|
| AI Overviews citing at least one YouTube video | **257 of 336 (76%)** |
| YouTube citations in total | **607** |
| Distinct videos | **96** |
| Searches citing more than one video | **182 (71%)**, mean 2.36, max 5 |
| Citations carrying a `&t=` offset | 56% (63% among non-Shorts) |
| Top 10 videos' share of all video citations | 52% |

**What this does *not* show:** among the 38 videos cited more than once with a
timestamp, **24 (63%) always carry the same timestamp**. Where it never varies it
is a property of the video — a chapter or most-replayed marker — not of the
answer. So "Google links a moment, not a video" is not supported.

**Do AI Overviews use what the video says?** Not established. Transcripts were
collected for the cited videos (51% coverage; YouTube rate-limits). Exact 6-word
reuse: 5 matches against a null of ~2 — a non-result, and 28 of 30 transcripts
are auto-generated, so exact matching is the wrong instrument. A fuzzy alignment
test (does the best-matching passage sit near the linked moment?) gives 48% within
60 s versus 39% for an unrelated AI Overview and 28% for a random point — but the
117 cases come from only 19 videos, so the margin's confidence interval touches
zero. Reported as an open question.

---

## Finding 6 — Content depth and structure: real association, no predictive power

| Word count | Cited (ranked pages) | n |
|---|---|---|
| <500 | 18.7% | 455 |
| 500–1k | 24.5% | 691 |
| 1k–2k | 25.9% | 1,730 |
| 2k–4k | **36.4%** | 1,571 |
| >4k | 33.8% | 408 |

![Word count](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/insight_5_wordcount_curve.png)

Structural features, on searches that have an AI Overview, **with standard errors
clustered by page** (82% of rows are repeated pages, so row-level p-values are far
too small):

| Feature | OR | p naive | **p clustered by URL** | **p clustered by domain** |
|---|---|---|---|---|
| `has_schema` | 1.60 | 6.7e-07 | **0.037** | **0.040** |
| `has_faq` | 1.61 | 1.8e-10 | **0.023** | **0.016** |

Clustering inflates the standard errors roughly 2.5×. Both survive, but "highly
significant" does not. And permutation importance gives both ≈ 0 — the model
gains nothing from them. Structured pages are cited more often; structure adds no
predictive power once other signals are known.

---

## Finding 7 — Google does reuse source sentences

Sentence pairs sharing **8+ consecutive words** between an AI Overview and a page
it cited: **31 observed** against a permutation null of **12** (5 replicates,
range 9–20) — **2.5× above chance**.

> **Page (sparkasse.de):** "Ein vollständiges Verkehrswertgutachten kostet in der Regel zwischen 0,5 und 1,5 Prozent des Verkehrswerts der Immobilie."
>
> **Google's answer:** "Ein rechtssicheres Verkehrswertgutachten kostet in der Regel zwischen 0,5 und 1,5 Prozent des Immobilienwerts."

Against length-matched controls from the same pages (25 vs 32 sentences), Fisher
exact with Holm correction: a number OR 9.2 (p = 0.0012) and a percentage
(p = 0.002) hold; a price in € (p = 0.50) and a definition cue (p = 0.50) do not.

**An earlier version of this analysis was wrong on all three counts:** it used
TF-IDF cosine, reported 245 "near-verbatim" pairs, and claimed price sentences
were 10× likelier to be reused. TF-IDF weights rare tokens heavily and digits are
rare, so the matcher was *selecting* number-bearing sentences; the pairs shared a
median of 3 words; and duplicate pages were counted repeatedly. After
deduplicating, switching to n-grams, length-matching the control and adding a
permutation null, the price effect disappears.

*With 5 permutation replicates the smallest attainable p-value is 1/6 ≈ 0.17, so
"2.5× above chance" has no interval yet. 1,000 replicates is a documented next step.*

---

## Finding 8 — Two common SEO beliefs this data doesn't support

| Claim | Result |
|---|---|
| "Lead with a definition" | Not significant after correction (p = 0.50) |
| "AI loves lists" | Never significant |

Both were reported as findings in earlier versions of this project.

---

## The leakage audit — the actual point

**1. A placeholder that encoded the answer.** Cited pages absent from the visible
results got rank 101; they are in the data *only because* they were cited, so all
1,789 are `cited = 1` by construction (27% of rows).

**2. A feature computed from the answer.** `domain_citation_rate` was calculated
from the citations being predicted — correlation 1.000 with the domain's observed
rate. For 572 single-occurrence domains it *was* the label.

**3. Indirect leakage through a subgroup.** Keeping those rows while dropping
`organic_rank` still leaked: they are ~9× more likely to be video pages. That
model scored 0.773 overall but 0.479 on genuinely ranked pages.

**4. Leakage in the cross-validation.** Pages repeat across searches with
identical features, so grouping by search wasn't enough.

**5. Leakage in the *fix* for #4.** Grouping by domain lets the same search into
both sides — and since ~38% of searches have no AI Overview, "does this search
produce citations?" leaks straight back. Only double-blocked CV closes both, and
under it the model scores inside its null.

**6. A matching bug that undercounted citations.** `normalize_url()` dropped the
query string, so every `youtube.com/watch?v=…` collapsed to one dictionary key
and only the last reference per search survived. Evidence: of 249 searches with a
`watch?v=` citation, **zero** had more than one — while Shorts, which carry the id
in the path, showed 3 of 62 with several. Recounting from the raw cache gives 607
citations across 96 videos, not 314 across 59. The share of AI Overviews citing
*at least one* video (76%) is unaffected, since one reference always survived.

**And a sweep of the descriptive statistics.** The rank-101 artefact had also
distorted non-model results: "YouTube is cited 95% of the time" (92% of its rows
were rank-101; among ranked video pages it's 15.8% vs 29.3% for non-video), the
U-shaped word-count curve, and the intent-segment citation rate (53% over all
rows, 38% on ranked pages). `scripts/audit_artifacts.py` recomputes every
descriptive finding both ways.

---

## Other methodological corrections

**Calibration and holdout, at the honest grouping.** Both were previously reported
from the leaky setting:

| | Grouped by search | Grouped by domain |
|---|---|---|
| Calibration error | 0.049 | **0.118** (under-estimates) |
| Holdout, 30 splits | 0.577 ± 0.042 | **0.364 ± 0.053** |

**A single holdout split is a lottery draw** — the range across splits is
[0.463, 0.669] by search and [0.275, 0.466] by domain.

**Robustness:** 357 rows had a failed crawl, and their semantic features come from
the SERP snippet — text Google chose *because* it matches the query. Excluding
them moves PR-AUC 0.527 → 0.546. Stable.

**Too sparse for a claim:** `is_forum` (4 rows), `is_video` (95 rows).

**`config.FEATURES` cleaned.** It still contained the leaking columns, and three
scripts fell back to it — producing PR-AUC 0.93 with `domain_citation_rate` among
the top features. `build_xy` now raises rather than silently using that fallback.

---

## What you'd actually do with this

1. **Check the search type first** — 22% can never show an AI Overview
2. **Don't expect on-page tweaks to carry you** — on an unfamiliar site, the
   measurable content signals here predict citation about as well as chance
3. **Chase the non-ranking slots** — 56% of citations go to pages outside the
   visible results, and 76% of AI Overviews cite video
4. **Write quantitatively** — numbers and percentages are reused more often

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q
```

### Collect your own data

```bash
export DATAFORSEO_LOGIN="your_login"
export DATAFORSEO_PASSWORD="your_password"
python scripts/collect_real_data.py --queries your_queries.txt --out data/raw/real.csv
```

### The analysis

Everything runs off the local cache written during collection — no step needs new
API calls.

```bash
python scripts/extract_from_cache.py           # AI Overview text, PAA questions, citations
python scripts/run_definitive_analysis.py      # Findings 1 + 2 + all robustness checks
python scripts/fix_descriptive_stats.py        # Findings 4, 6 on the clean population
python scripts/audit_artifacts.py              # the artefact sweep
python scripts/recount_youtube.py              # Finding 5, corrected for the matching bug
python scripts/analyze_passages_v2.py --min-ngram 8   # Finding 7 with a permutation null
python scripts/build_business_cases.py         # every finding with a real example
python scripts/verify_setup.py                 # check everything is present and current
```

**Privacy:** real query lists, collected data, the cache, and per-row analysis
outputs are git-ignored — they're business research. Charts and aggregate
summaries in `reports/` are public: those are the findings.

---

## Limitations

- **One vertical, one country, one snapshot** (16–17 July 2026). SERPs change daily.
- **Correlation, not causation.** Proving cause needs an intervention: change a
  page, see whether the citation follows.
- **Ranking is a collider.** Restricting to ranked pages conditions on a common
  effect of the same content factors that drive citation, which induces Berkson
  bias. Associations *within* the ranked stratum are distorted, and at least one
  reverses sign against the full population (`passage_match_score` tertiles:
  0.567/0.468/0.408 over all rows, 0.275/0.296/0.302 on ranked pages). The
  restriction is valid for the estimand "among pages Google already ranks"; it is
  not valid for "what drives citation".
- **No true holdout.** Every number comes from cross-validation on the same 533
  searches, across five leakage rounds and a metric change. A set of untouched
  searches would be the single most convincing addition.
- **`organic_rank` is DataForSEO's `rank_absolute`**, which counts all SERP blocks
  including the AI Overview itself, so it partly encodes SERP composition.
  `rank_group` is in the cache; re-extracting it is a documented next step.
- **11 false-negative labels** from the same matching bug as leak 6 (http/https
  variants), enumerable and left in place because rebuilding the training data
  would invalidate every downstream result.
- **Imputation used a global median** across all rows, including the leaking
  subset. `content_freshness_days` is that constant for 63% of ranked rows.
- **SHAP comes from a model fitted on all rows**, so the explanations partly
  describe memorised sites. Descriptive for this dataset, not a causal account.
- **Video alignment rests on 19 videos**; transcript coverage is 51%.
- **The passage analysis rests on 31 pairs** with a 5-replicate null.
- **Two measurement flaws** in the collected features: an English stopword list on
  German text, and a TF-IDF vectoriser refit per page, so `query_url_similarity`
  and `passage_match_score` are not strictly comparable across rows.
- **`structure_score` is defined two ways** in the codebase. All reported results
  use the additive definition.

---

## Tech stack

Python · pandas · scikit-learn · LightGBM · SHAP · scipy · statsmodels ·
SQLAlchemy/SQLite · seaborn · matplotlib · DataForSEO API · BeautifulSoup ·
youtube-transcript-api · Tableau · Git · pytest

## License

MIT — see [LICENSE](LICENSE).
