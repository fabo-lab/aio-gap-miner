# AIO Gap-Miner

**Predicting which pages Google cites in its AI Overviews — and finding out, the hard way, what actually drives it.**

When you search Google, you often get an **AI Overview**: a written answer at the
top with a handful of source links. If your page isn't one of those sources, most
people never see you — no matter how well you rank.

This project asks what decides who gets picked. It ended up being as much about
**how to test that honestly** as about the answer, because four separate versions
of the analysis were fooling themselves in different ways.

---

## The short version

- **Citation is driven mostly by which website a page is on.** On sites the model
  has never seen, it does not beat "just trust the Google ranking".
- **But more than half of all citations go to pages that don't rank** in the
  visible top 15 — so the ranking is not the gate it appears to be.
- **3 of 4 AI Overviews cite a YouTube video**, and 97% of those videos don't
  rank. That's the clearest opening for a small site.
- **22% of searches never show an AI Overview at all.**
- **Google reuses source sentences near-verbatim**, ~2.5× above chance. Real, but
  smaller than it first looked.
- **Four kinds of leakage found and removed.** Three by me, the largest by an
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
| AI Overviews captured with full text | 336 |
| Pages crawled | 6,198 |
| "People also ask" questions | 1,970 (714 unique) |

German real-estate searches via the DataForSEO API. Every raw API response and
every page's HTML was cached during collection, which is what made all the later
analyses possible without re-querying anything.

---

## Finding 1 — Not every search is a game you can play

| Search type | Searches | AI Overview shown | Citation rate (ranked pages) |
|---|---|---|---|
| Local (Google shows a map) | 89 | **0%** | 0% |
| Featured snippet shown | 25 | **0%** | 0% |
| Informational | 422 | 79.6% | 38.2% |

![Intent segments](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/insight_8_intent_segments.png)

**0 of 114** searches showing a map or a snippet had an AI Overview. Optimising a
page for those is effort spent on an outcome that doesn't exist.

---

## Finding 2 — The model works, until you ask it to generalise

All models on the same 4,857 ranked pages, 5-fold CV, mean ± sd across folds.
The **grouping** column is the point: grouping by search stops the same *search*
leaking; grouping by domain stops the same *website* leaking.

| Features | Grouped by | PR-AUC | ROC-AUC |
|---|---|---|---|
| Content only | search | 0.527 ± 0.033 | 0.727 |
| Content only | **domain** | **0.350 ± 0.052** | **0.552** |
| Content + rank | search | 0.577 ± 0.035 | 0.757 |
| Content + rank | **domain** | **0.376 ± 0.027** | **0.600** |
| *Rank-only heuristic* | — | *0.368* | *0.616* |
| *Random (prevalence)* | — | *0.291* | *0.500* |

**On websites it has never seen, the model does not beat the ranking heuristic it
was built to beat.**

### Why: it was memorising websites

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

## Finding 3 — So what does decide it? Mostly the site.

On searches that actually have an AI Overview (baseline 48.7%):

| Domain | Times ranked | Cited | 95% CI | vs baseline |
|---|---|---|---|---|
| immobilienscout24.de | 91 | 89% | [0.81, 0.94] | **1.83×** |
| drklein.de | 91 | 76% | [0.66, 0.83] | 1.56× |
| sparkasse.de | 167 | 71% | [0.63, 0.77] | 1.45× |
| test.de | 36 | 67% | [0.50, 0.80] | 1.37× |
| check24.de | 129 | 48% | [0.40, 0.57] | 0.99× |

![Brand vs position](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/case_3_brand_vs_position.png)

At positions 9–10, established brands are cited 65–69% of the time; everyone else
37–38%. Same slot, roughly double the odds. 425 domains with fewer than 30
appearances are excluded — their rates are noise.

---

## Finding 4 — But the ranking isn't the gate

![Citations without ranking](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/case_1_citation_without_ranking.png)

**1,789 of 3,201 citations (56%) went to pages that were not in the visible top
15.** A real example — search: *"verkehrswertermittlung eigentumswohnung"*:

| Source Google cited | Its rank |
|---|---|
| sparkasse.de | #3 |
| immobilienscout24.de | #5 |
| immoverkauf24.de | #8 |
| check24.de | **not in top 15** |
| mcmakler.de | **not in top 15** |

More than half the citation slots aren't awarded through the ranking you can see.
For a small site that can't win page one, that is the opening.

---

## Finding 5 — Video is the clearest opening

![YouTube](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/case_2_youtube.png)

| | |
|---|---|
| AI Overviews citing at least one YouTube video | **254 of 333 (76%)** |
| YouTube citations in total | 314 (9.8% of all citations) |
| ... from videos **not** in the top 15 | **305 (97%)** |
| Distinct videos behind those 314 citations | **59** (one cited 39 times) |
| Cited video URLs carrying a `&t=` timestamp | **65%** (median 103 s) |

Google doesn't just link the video — in two thirds of cases it links a *moment*
inside it. And only 14% of these searches show a video block at all, so this
isn't Google surfacing an existing video carousel.

**This is the most actionable finding in the project:** a video is a citation slot
that bypasses the ranking queue, and a very small number of videos currently
covers the whole market.

### Does Google actually use what the video says? Not established.

Transcripts were collected for the cited videos (51% coverage — YouTube
rate-limits). Two tests:

- **Exact reuse** (shared 6-word sequences): 5 matches vs a null of ~2. A
  non-result. 28 of 30 transcripts are auto-generated, so they carry no
  punctuation and regular transcription errors — exact matching is the wrong
  instrument for that text.
- **Alignment** (does the best-matching passage sit near the linked moment?):
  within 60 s in **48%** of cases, versus 39% for an unrelated AI Overview and
  28% for a random point in the video. Right direction, 9-point margin.

**Honest verdict: suggestive, not conclusive.** The 117 cases come from only 19
videos, so they are far from independent. Reported as an open question rather
than a finding.

---

## Finding 6 — Content depth matters, modestly

| Word count | Cited (ranked pages) | n |
|---|---|---|
| <500 | 18.7% | 455 |
| 500–1k | 24.5% | 691 |
| 1k–2k | 25.9% | 1,730 |
| 2k–4k | **36.4%** | 1,571 |
| >4k | 33.8% | 408 |

![Word count](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/insight_5_wordcount_curve.png)

Rises with depth, plateaus above ~2,000 words. An earlier version showed a
U-shape with a peak at short pages; that peak was entirely cited-but-not-ranked
rows. The chart plots both lines so the distortion stays visible.

**Structural features** are genuinely associated with citation — `has_schema`
OR 1.60, `has_faq` OR 1.61 (both p < 1e-6, conditional on AI-Overview searches) —
**but permutation importance gives both ≈ 0.** Structured pages get cited more
often; structure adds no predictive power once other signals are known. Given
Finding 2, that is plausibly another site effect.

---

## Finding 7 — Google does reuse source sentences

Counting sentence pairs sharing **8+ consecutive words** between an AI Overview
and the pages it cited:

| | |
|---|---|
| Observed pairs | **31** (24 searches, 12 domains) |
| Expected by chance (5 permutation replicates) | 12 (range 9–20) |
| Ratio | **2.5× above chance** |

> **Page (sparkasse.de):** "Ein vollständiges Verkehrswertgutachten kostet in der Regel zwischen 0,5 und 1,5 Prozent des Verkehrswerts der Immobilie."
>
> **Google's answer:** "Ein rechtssicheres Verkehrswertgutachten kostet in der Regel zwischen 0,5 und 1,5 Prozent des Immobilienwerts."

Against length-matched control sentences from the same pages, with Holm
correction:

| Sentence contains | Reused | Control | OR | p (Holm) | |
|---|---|---|---|---|---|
| a number | 68% | 19% | 9.2 | 0.0012 | holds |
| a percentage | 32% | 0% | ∞ | 0.002 | holds |
| a price in € | 20% | 9% | 2.4 | 0.50 | **not significant** |
| a definition cue | 0% | 9% | — | 0.50 | **not significant** |

**What an earlier version got wrong:** it used TF-IDF cosine similarity, reported
245 "near-verbatim" pairs, and claimed sentences with prices were 10× more likely
to be reused. TF-IDF weights rare tokens heavily and digits are rare, so the
matcher was *selecting* number-bearing sentences. The pairs shared a median of 3
words. And duplicate pages were counted repeatedly. After deduplicating, using an
n-gram criterion, length-matching the control, and running a permutation null,
the price effect disappears entirely.

---

## Finding 8 — Two common SEO beliefs this data doesn't support

| Claim | Result |
|---|---|
| "Lead with a definition" | Not significant after correction (p = 0.50) |
| "AI loves lists" | Never significant |

Both were reported as findings in an earlier version of this project.

---

## The leakage audit — the actual point

**1. A placeholder that encoded the answer.** Cited pages absent from Google's
visible results got rank 101. Such pages are in the data *only because* they were
cited, so all 1,789 rows are `cited = 1` by construction — 27% of the dataset.

**2. A feature computed from the answer.** `domain_citation_rate` was calculated
from the citations being predicted. For 572 single-occurrence domains it *was*
the label.

**3. Indirect leakage through a subgroup.** Keeping those rows while dropping
`organic_rank` still leaked: they're ~9× more likely to be video pages, so the
model could identify the always-cited subgroup another way. It scored 0.773
overall but 0.479 on genuinely ranked pages.

**4. Leakage in the cross-validation itself.** Finding 2 — pages repeat across
searches with identical features, so grouping by search wasn't enough. The
largest of the four, and it inverted the project's original headline.

**And a sweep of the descriptive statistics.** The same artefact had distorted
non-model results too: the "YouTube is cited 95%" rate (92% of its rows were
rank-101 — among ranked video pages it's 15.8% vs 29.3% for non-video), the
U-shaped word-count curve, and the intent-segment citation rate (53% over all
rows, 38% on ranked pages). `scripts/audit_artifacts.py` now recomputes every
descriptive finding both ways.

---

## Methodological corrections

**The population answers two questions at once.** 205 of 533 searches never show
an AI Overview, making 40% of ranked rows `cited = 0` by construction. Conditional
on searches that do, prevalence rises 0.291 → 0.487 and the model's lift over
random drops **1.81× → 1.51×**.

**The calibration error was a hyperparameter.** `is_unbalance=True` inflates
positive probabilities; switching it off drops calibration error **0.109 → 0.049**.
All results use `is_unbalance=False`.

**One holdout split is a lottery draw.** Over 50 random splits: **0.578 ± 0.041,
range [0.463, 0.669]**.

**Model differences need a test.** Paired bootstrap over searches: content+rank
beats content-only by **+0.050, 95% CI [+0.034, +0.066]**. Real, but small.
Across 5 seeds the model varies by only 0.007.

**Robustness:** 357 rows had a failed crawl, and their semantic features come from
the SERP snippet — text Google chose *because* it matches the query. Excluding
them: PR-AUC 0.527 → 0.546. Stable.

**Too sparse for a claim:** `is_forum` (4 rows), `is_video` (95 rows).

---

## What you'd actually do with this

1. **Check the search type first** — 22% can never show an AI Overview
2. **Make a video** — 3 in 4 AI Overviews cite one, 97% of those don't rank
3. **Chase the non-ranking slots** — 56% of citations go to pages outside the top 15
4. **Write quantitatively** — numbers and percentages are reused more often
5. **Be realistic about brand** — site identity dominates page-level tweaks

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
committed.

### The analysis

Everything runs off the local cache written during collection, so no step needs
new API calls.

```bash
python scripts/extract_from_cache.py           # AI Overview text, PAA questions, citations
python scripts/run_final_analysis.py           # Finding 2 + all robustness checks
python scripts/fix_descriptive_stats.py        # Findings 1, 3, 6 on the clean population
python scripts/audit_artifacts.py              # the artefact sweep
python scripts/analyze_passages_v2.py --min-ngram 8   # Finding 7 with a permutation null
python scripts/build_business_cases.py         # every finding with a real example
python scripts/fetch_youtube_transcripts.py    # Finding 5 (rate-limited; resumable)
python scripts/analyze_video_alignment.py      # the timestamp alignment test
python scripts/verify_setup.py                 # check everything is present and current
```

**Privacy:** real query lists, collected data, the cache, and per-row analysis
outputs are git-ignored — they're business research. Charts and aggregate
summaries in `reports/` are public: those are the findings.

---

## Limitations

- **One vertical, one country, one snapshot.** SERPs change daily.
- **Correlation, not causation.** Proving cause needs an intervention: change a
  page, see whether the citation follows.
- **The model doesn't generalise to unseen websites.** A finding, and the main limit.
- **Restricting to ranked pages drops 56% of all citations.** A real selection,
  not a random sample — the excluded rows differ systematically.
- **The video alignment result rests on 19 videos.** Not independent observations.
- **Transcript coverage is 51%** and 28 of 30 are auto-generated.
- **The passage analysis rests on 31 pairs.**
- **Two known measurement flaws**, documented rather than fixed because they'd
  require re-collection: the TF-IDF similarity used an English stopword list on
  German text, and the vectoriser was refit per page, so `query_url_similarity`
  and `passage_match_score` aren't strictly comparable across rows.
- **`structure_score` is defined two ways** in the codebase (weighted in
  `features.py`, additive in the analysis scripts). All reported results use the
  additive definition.

---

## Tech stack

Python · pandas · scikit-learn · LightGBM · SHAP · scipy · SQLAlchemy/SQLite ·
seaborn · matplotlib · DataForSEO API · BeautifulSoup · youtube-transcript-api ·
Tableau · Git · pytest

## License

MIT — see [LICENSE](LICENSE).
