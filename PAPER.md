# Predicting AI Overview Citations: Six Ways I Was Wrong

**A methodological account of a data analytics capstone project.**

Fabian Pott · July 2026 · [github.com/fabo-lab/aio-gap-miner](https://github.com/fabo-lab/aio-gap-miner)

---

## Abstract

Google's AI Overviews answer a search directly and cite a handful of sources.
Pages that aren't cited become invisible regardless of ranking, so predicting
citation is commercially interesting — and, as it turns out, methodologically
treacherous.

I collected 533 real German real-estate searches with their AI Overview
citations, crawled the 6,646 candidate pages, and trained a gradient-boosting
model. The first version scored PR-AUC 0.93. The final honest figure is **0.302
against a permutation null of 0.313** — statistically indistinguishable from no
signal at all.

Getting from the first number to the last took six separate corrections. Three I
found myself; three came from independent reviews I commissioned specifically to
attack the work. This paper documents all of them, because the corrections are
the substance of the project.

The finding that survives is not the one I set out to demonstrate: **which
website a page belongs to predicts citation about as well as its Google ranking,
while its measurable content properties — evaluated on a site the model has never
seen — predict it barely better than chance.**

---

## 1 · Data

| | |
|---|---|
| Searches attempted | 538 (3 failed during collection) |
| Searches in the modelling data | 533 · SERP snapshots: 536 |
| (search, page) pairs | 6,646 → **4,857** after removing a leaking subset |
| Distinct URLs among ranked rows | **1,361** |
| Distinct domains | 752 |
| AI Overviews captured with full text | 336 |
| Pages crawled | 6,198 |
| "People also ask" questions | 1,970 (714 unique) |
| Collection window | 16–17 July 2026 |

Source: DataForSEO SERP API (`/v3/serp/google/organic/live/advanced` with
`load_async_ai_overview`), location code 2276 (Germany), language `de`.

Every raw API response and every page's HTML was cached at collection time. That
decision is what made all six later corrections possible without re-querying —
and since SERPs change daily, re-querying would not have reproduced the same data.

**Unit of analysis:** one row per (search, candidate page), labelled `cited = 1`
if the page appears among the AI Overview's references.

**Features:** page length, readability (Flesch), schema markup, FAQ presence,
list/table count, content freshness, HTTPS, forum/video flags, content type, and
two semantic scores (query↔page and query↔best-passage similarity, TF-IDF).

---

## 2 · The six corrections

### 2.1 · A placeholder that encoded the answer

Pages cited by the AI Overview but absent from the visible organic results were
given a sentinel rank of 101. But such a page is in the dataset **only because it
was cited** — so all 1,789 sentinel rows carry `cited = 1` by construction, 27%
of the data.

A model given `organic_rank` learns "101 → cited" and stops. *Removed by
restricting all analysis to genuinely ranked pages.*

### 2.2 · A feature computed from the label

`domain_citation_rate` measured how often a domain gets cited — calculated from
the very citations being predicted. Correlation with the domain's observed
citation rate: **1.000**. For the 572 domains appearing once, it *was* the label.

*Removed.* Notably, this feature had been introduced deliberately as a
"better than backlink-based authority" alternative. It was better only at cheating.

### 2.3 · Indirect leakage through a subgroup

An intermediate fix kept the sentinel rows but dropped `organic_rank` as a
feature. This still leaked: sentinel rows differ systematically — they are about
**9× more likely to be video pages** — so the model could identify the
always-cited subgroup through other features.

Evidence: that configuration scored **0.773** overall but **0.479** when
evaluated only on genuinely ranked pages, worse than a model trained on ranked
pages alone.

*This one also distorted descriptive statistics*, which I only checked
afterwards:

| Statistic | As first reported | Corrected |
|---|---|---|
| "YouTube is cited 95% of the time" | 95% | **15.8%** for ranked video pages (vs 29.3% non-video) |
| Word count vs citation rate | U-shaped, peak at short pages | monotone rise, plateau above 2k words |
| Citation rate, informational searches | 53% | **38.2%** |

`scripts/audit_artifacts.py` now recomputes every descriptive figure both ways.

### 2.4 · Leakage in the cross-validation

**Found by the first independent review.** `GroupKFold(query_id)` stops the same
*search* appearing on both sides of a split. It does not stop the same *page*.

| | |
|---|---|
| Ranked rows | 4,857 |
| Distinct URLs among them | 1,361 |
| Rows from URLs appearing in several searches | **82%** |
| Content features that never vary for a given page | **10 of 13** |

So the same page sat in training and test with an identical feature vector. The
proof is direct: features that by construction cannot say whether a page fits a
search — everything except the three query-dependent ones — score **0.522**
against the full set's 0.527. Almost the entire result was site memorisation.

| Features | Grouped by search | Grouped by domain |
|---|---|---|
| Content only | 0.527 | **0.350** |
| Content + rank | 0.577 | **0.376** |
| *Rank-only heuristic* | *0.368* | *0.368* |

### 2.5 · Leakage in the fix for 2.4

**Found by the second review.** Grouping by domain holds domains back but lets
the same *search* into both sides — the very thing query-grouping existed for.
And since ~38% of searches have no AI Overview at all, "does this search produce
citations?" is a strong query-level signal that leaks straight back in.

The correct floor is therefore not prevalence but a **permutation null**: labels
shuffled *within* each search, everything else identical.

![Cascade](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/headline_1_leakage_cascade.png)

| Cross-validation | Model | Permutation null | Margin |
|---|---|---|---|
| Grouped by search | 0.527 | 0.382 ± 0.007 | +0.145 |
| Grouped by domain | 0.345 | 0.311 ± 0.015 | +0.033 |
| **Double-blocked** | **0.302** | **0.313 ± 0.043** | **−0.011** |

Under double-blocked CV — a test row needs both an unseen search and an unseen
domain — the model scores **inside its own null distribution**.

*Caveat on that scheme:* it uses only the diagonal of the query × domain fold
grid, so 80% of rows go unscored and the ones that remain average ~2 candidates
per search instead of ~9. Pooled AP stays interpretable; per-search AP on two
candidates does not, and is excluded from comparison.

### 2.6 · A matching bug that undercounted citations

`normalize_url()` built its key as `scheme://host/path`, discarding the query
string. Two consequences:

- every `youtube.com/watch?v=…` collapsed to the same key, and since references
  went into a dictionary, only the last one per search survived;
- an `http://` organic result never matched an `https://` reference.

The evidence was unambiguous: of 249 searches with a `watch?v=` citation,
**zero** had more than one — while Shorts, which carry the id in the path and so
don't collide, showed 3 of 62 with several.

| | Before | After |
|---|---|---|
| YouTube citations | 314 | **607** |
| Distinct videos | 59 | **96** |
| Searches citing more than one video | 0 | **182 (71%)** |
| Mean videos per citing search | 1.0 | **2.36** |
| Citations with a timestamp | "65%" | **56%** |

Labels were also affected: 20 rows changed after rebuilding from the raw cache
(11 false negatives, 9 false positives — the false positives all YouTube URLs
that had inherited another video's citation). Every one of the 20 was verified
against the cached response: **20/20 confirm the rebuilt version, 0/20 the
original.**

The same rebuild replaced `rank_absolute` with `rank_group`. The former counts
every SERP block including the AI Overview itself, so it partly encoded which
blocks were present. 91% of rows changed rank, median shift 2 positions.

**One reviewer hypothesis did not survive:** the rank-only baseline was expected
to fall once the true organic rank was used. It moved from 0.368 to **0.373** —
the artefact component was negligible.

---

## 3 · What the corrected analysis shows

### 3.1 · The right metric

The label is query-relative: *which of these ~9 candidates does Google cite?*
Pooled PR-AUC mixes that with *which searches have citations at all*. Mean
per-search average precision answers the question the label poses; Precision@3
answers the one a practitioner asks.

![Per-search comparison](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/headline_2_per_search.png)

| Predictor | Pooled AP | Per-search AP | P@3 |
|---|---|---|---|
| Random | 0.290 | 0.610 | 0.495 |
| Rank-only heuristic | 0.373 | 0.721 | 0.647 |
| **Domain identity only** (out-of-fold encoded) | 0.465 | **0.726** | 0.650 |
| Content, grouped by search *(leaky)* | 0.527 | 0.744 | 0.655 |
| **Content, grouped by domain** | 0.345 | **0.631** | 0.517 |

The domain-only baseline knows nothing but which website a page is on, encoded
out-of-fold so it cannot leak. It performs on par with the ranking heuristic.
Content evaluated on unseen domains lands closer to random than to either.

### 3.2 · Calibration and holdout, at the honest grouping

Both had previously been reported from the leaky setting:

| | Grouped by search | Grouped by domain |
|---|---|---|
| Calibration error | 0.048 | **0.118** (under-estimates) |
| Holdout, 30 random splits | 0.552 ± 0.043 | **0.349 ± 0.048** |

A single holdout split is a lottery draw — the range is [0.449, 0.643] by search
and [0.273, 0.441] by domain.

An earlier report of calibration error 0.109 → 0.049 was attributed to fixing a
hyperparameter (`is_unbalance`). That was half the story: on unseen domains the
error returns to 0.118, in the opposite direction.

### 3.3 · Clustered inference

82% of rows are repeated pages, so row-level p-values are far too small:

| Feature | OR | p naive | p clustered by URL | p clustered by domain |
|---|---|---|---|---|
| `has_schema` | 1.63 | 2.1e-07 | **0.029** | **0.031** |
| `has_faq` | 1.59 | 7.2e-10 | **0.028** | **0.020** |

Standard errors inflate roughly 2.5×. Both survive, but "highly significant" does
not — and permutation importance gave both ≈ 0 all along, so the clustered
values are the ones that fit the rest of the evidence. **Structured pages are
cited more often; structure adds no predictive power once other signals are known.**

### 3.4 · Sentence reuse

Sentence pairs sharing **8 or more consecutive words** between an AI Overview and
a page it cited: **31 observed**, against a permutation null (1,000 replicates,
AI Overviews paired with pages from unrelated searches) with **mean 13** —
**2.3× above chance**.

> **Page (sparkasse.de):** "Ein vollständiges Verkehrswertgutachten kostet in der Regel zwischen 0,5 und 1,5 Prozent des Verkehrswerts der Immobilie."
>
> **Google's answer:** "Ein rechtssicheres Verkehrswertgutachten kostet in der Regel zwischen 0,5 und 1,5 Prozent des Immobilienwerts."

Against length-matched control sentences from the same pages (25 vs 32), Fisher
exact with Holm correction:

| Sentence contains | Reused | Control | OR | p (Holm) | |
|---|---|---|---|---|---|
| a number | 68% | 28% | 5.4 | 0.014 | holds |
| a percentage | 32% | 9% | 4.6 | 0.13 | not significant |
| a price in € | 20% | 9% | 2.4 | 0.56 | not significant |

**An earlier version claimed 245 "near-verbatim" pairs and a 10× effect for
prices.** It used TF-IDF cosine, which weights rare tokens heavily — and digits
are rare, so the matcher was *selecting* number-bearing sentences. The pairs
shared a median of 3 words. Pages were counted repeatedly. After deduplication,
an n-gram criterion, length-matched controls and a permutation null, only the
number effect survives.

**And a reproducibility bug of my own:** the control group was drawn with the
same random generator the permutation loop consumed, so the control set depended
on `--permutations`. It did, and it changed conclusions between runs — the
percentage effect was p = 0.002 with 5 replicates and p = 0.13 with 1,000. Fixed
with a separate generator. **The instability is itself the finding: at n = 25
versus 32, only the number effect is robust to redrawing the control.**

### 3.5 · Video

| | |
|---|---|
| AI Overviews citing at least one video | **257 of 336 (76%)** |
| Total video citations | 607 across 96 distinct videos |
| Searches citing more than one | 182 (71%), mean 2.36 |
| Citations from videos not in the visible results | **97%** |
| Top 10 videos' share of all video citations | 52% |

The "at least one" figure is the only one the matching bug left untouched — with
a "≥ 1" statement it doesn't matter which video won the dictionary slot.

**What this does not show:** among 38 videos cited repeatedly with a timestamp,
**24 (63%) always carry the same one**. Where it never varies it is a property of
the video — a chapter or most-replayed marker — not of the answer.

Transcripts were collected (51% coverage; YouTube rate-limits). Exact 6-word
reuse found 5 matches against a null of ~2: a non-result, and with 28 of 30
transcripts auto-generated — no punctuation, transcription errors — exact
matching is the wrong instrument. A fuzzy alignment test (does the best-matching
passage sit near the linked moment?) gave 48% within 60 s versus 39% for an
unrelated AI Overview and 28% for a random point. The 117 cases come from only 19
videos; clustered, the margin's confidence interval touches zero.

**Open question, not a finding.**

### 3.6 · Label validation

A stratified sample of 60 rows was re-derived from the cached responses using a
**separately written** URL normaliser, so a bug in the collector's version could
not mask itself. Agreement: **60/60**.

That bounds the error rate below roughly 5% (rule of three) — it does not
demonstrate correctness. The known 20 errors are 0.3% of rows, and a 60-row
sample has only a ~16% chance of catching one. The targeted check on the 20 known
changes is the stronger evidence: **20/20 confirm the rebuild**.

---

## 4 · Limitations

**Ranking is a collider.** It is a common effect of the same content factors that
drive citation, so conditioning on ranked pages induces Berkson bias.
Associations within that stratum are distorted, and at least one reverses sign:

| `passage_match_score` tertile | low | mid | high |
|---|---|---|---|
| All rows | 0.567 | 0.468 | **0.408** ↓ |
| Ranked only | 0.275 | 0.296 | **0.302** ↑ |

The restriction is valid for the estimand *"among pages Google already ranks"*.
It is not valid for *"what drives citation"*.

**No true holdout.** Every figure comes from cross-validation on the same 533
searches, across six correction rounds and a metric change. A set of untouched
searches would be the single most convincing addition.

**Restricting to ranked pages discards 56% of all citations** — a real selection,
not a random sample.

**SHAP was computed from a model fitted on all rows**, so its explanations partly
describe memorised sites. Descriptive for this dataset, not a causal account.

**Measurement flaws left documented rather than fixed**, since they would require
re-collection: the TF-IDF similarity used an English stopword list on German
text, and the vectoriser was refit per page, so `query_url_similarity` and
`passage_match_score` are not strictly comparable across rows.

**Imputation used a global median** across all rows including the leaking subset;
`content_freshness_days` is that constant for 63% of ranked rows.

**Single vertical, single country, single snapshot.** German real estate,
July 2026.

**Correlation throughout.** Establishing cause needs an intervention: change a
page, measure whether citation follows.

---

## 5 · What I'd do next

1. **A held-out set of searches** untouched by any of this analysis.
2. **Out-of-fold domain encoding as a first-class feature**, so site identity is
   modelled explicitly rather than leaking.
3. **Re-collection with German stopwords** and a globally fitted vectoriser.
4. **An intervention experiment** — the only route from correlation to cause.

---

## 6 · Conclusion

The model I set out to build does not work in the way I intended. On a website it
has not seen, it does not predict citation better than chance.

What the data does show is that **site identity carries most of the signal**, and
that a large share of citation slots — 56% — are not awarded through the visible
ranking at all. Both are useful, and neither was the hypothesis.

The six corrections in section 2 are the actual output of this project. Each one
made the headline number smaller. The final number is the first one I would
defend.

---

*All code, data-processing scripts and figures: [github.com/fabo-lab/aio-gap-miner](https://github.com/fabo-lab/aio-gap-miner).
Raw collected data is not published — it is commercial keyword research — but
every aggregate result and every figure is reproducible from the scripts.*
