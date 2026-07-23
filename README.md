# AIO Gap-Miner

**A model that predicts which web pages Google picks as sources for its AI Overview answers — and explains why, down to the sentence.**

When you search Google, you often get an **AI Overview**: a written answer at the
top of the page with a handful of source links. If your page isn't one of those
sources, most people never see you — no matter how well you rank.

This project builds a machine learning model on real Google data that answers:

1. **Is this search even worth optimising for?** (not all of them are)
2. **Will this page get picked as a source?**
3. **Why?** — which content signals matter, and which exact sentences get lifted

---

## The data

| | |
|---|---|
| Real searches collected | **533** (German real-estate topics, via DataForSEO API) |
| (search, page) pairs | **6,646** |
| AI Overviews with full answer text | **336** |
| Pages crawled and measured | 6,198 |
| "People also ask" questions collected | 1,970 |

Everything below comes from that dataset. No demo numbers.

---

## Finding 1 — Not every search is a game you can play

Before optimising anything, you should know whether Google even shows an AI
Overview for that kind of search. It usually doesn't.

![Intent segments](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/insight_8_intent_segments.png)

| Search type | Share | AI Overview appears | Citation rate |
|---|---|---|---|
| Local (Google shows a map) | 17% | **0%** | 0% |
| Featured snippet shown | 5% | **0%** | 0% |
| Informational | 79% | **80%** | 53% |

**22% of the searches in this dataset will never show an AI Overview.** Google
answers those with a map or a snippet instead. Optimising a page for those
searches is spending effort on something that can't happen.

*Method note: I tried KMeans clustering first. On these sparse feature flags it
produced overlapping, hard-to-explain groups. A transparent rule on Google's own
signals separates the outcome almost perfectly and anyone can follow it — so I
used that instead.*

---

## Finding 2 — The model works, and it beats the obvious answer

All models below are evaluated on **one population** — the 4,857 pages Google
actually ranks — so the numbers are directly comparable with no caveats.

| Model | PR-AUC | ROC-AUC | vs. random |
|---|---|---|---|
| Random guessing | 0.291 | 0.500 | 1.00x |
| Rank-only heuristic ("just trust the ranking") | 0.368 | 0.616 | 1.27x |
| Logistic Regression (content + rank) | 0.437 | 0.669 | 1.50x |
| **LightGBM — content signals only** | **0.523** | 0.724 | **1.80x** |
| **LightGBM — content + rank** | **0.568** | 0.754 | **1.95x** |

The standard SEO assumption is "rank well and you get cited". That scores 0.368.
**Content signals alone — with the ranking removed entirely — score 0.523.**
On identical pages, what a page says predicts citation better than where it sits.

*Note on why one population: an earlier framing kept "cited but not ranked" pages
(given a placeholder rank of 101) and just dropped the rank feature. That looked
safe but wasn't — all such rows are cited by construction, and they differ
systematically (≈9x more likely to be video pages), so the model could identify
that always-cited subgroup through other features. Evaluated on genuinely ranked
pages, that variant scored 0.479 — worse than a model trained only on ranked
pages. It was dropped. See `scripts/run_headline_comparison.py`.*

![Feature importance](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/shap_importance_variant_B.png)

Top drivers: **content depth, topic match to the search, readability, and best
passage match.**

### Content length isn't linear

![Word count curve](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/insight_5_wordcount_curve.png)

Citation rate is U-shaped: short precise pages (<500 words, 56%) and long
thorough ones (>4k words, 63%) both do well; the 500–1,000 word middle is worst
at 35%. This is also *why* the tree model beats logistic regression — a straight
line can't represent this.

---

## Finding 3 — It holds up under pressure

| Check | Result |
|---|---|
| Held-out test (107 searches never seen in training) | PR-AUC **0.535** vs 0.240 prevalence → **2.23x** |
| Top-4 features shared by all 5 CV folds | 3 of 4, importances varying under 13% |
| SHAP vs permutation importance | **5 of 5** top features agree |
| Calibration error | 0.131 — see below |

![SHAP stability](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/hardening_shap_stability.png)

Two independent importance methods — SHAP and permutation importance — agree on
all five top features. The story doesn't change depending on how the data is split.

**One honest weakness:** the model is well *ranked* but not well *calibrated*. It
ranks pages correctly, but its raw probabilities are optimistic (it says 88% where
reality is 62%). So the scores should be used to prioritise pages, not read as
literal probabilities. Calibrating it (e.g. isotonic regression on a held-out set)
is the obvious next step.

## Finding 4 — What Google actually lifts, sentence by sentence

This is where the project answers its original question. For every AI Overview,
I compared its answer text against the real content of every page it cited, and
found the near-identical sentence pairs. **245 of them across 129 searches.**

Real example:

> **Page (immoverkauf24.de):**
> "Die Formel lautet: Sachwert = (Bodenwert + Gebäudesachwert) × Marktanpassungsfaktor."
>
> **Google's AI answer:**
> "Die Formel lautet vereinfacht: Sachwert = (Bodenwert + Gebäudesachwert) × Marktanpassungsfaktor."

Another:

> **Page (haus.de):** "Bei einem ermittelten Wert von circa 400.000 Euro wären das also zwischen 2.000 und 6.000 Euro."
>
> **Google's AI answer:** "Bei einem Haus im Wert von 400.000 Euro liegen die Kosten somit meist zwischen 2.000 und 6.000 Euro."

### The pattern behind the examples

Comparing the 245 lifted sentences against 2,994 sentences from the *same pages*
that weren't lifted:

![Passage patterns](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/insight_9_passage_patterns.png)

| A sentence containing… | Lifted | Not lifted | How much likelier |
|---|---|---|---|
| a price in € | 27.8% | 2.7% | **10.1×** |
| a percentage | 13.9% | 1.5% | **9.2×** |
| any number | 48.6% | 15.1% | **3.2×** |
| a formula | 3.3% | 1.2% | 2.8× |

Sentence length was essentially identical (16 words vs 15). **It's not about
writing more — it's about writing something concrete enough to be quoted.**

---

## Finding 5 — Two common SEO beliefs this data doesn't support

Practitioner guides claim that AI systems favour list structures, and that you
should lead with a definition. In this dataset:

| Claim | What the data shows |
|---|---|
| "LLMs love lists" | List-introducing sentences: **0.90×** — no advantage |
| "Lead with a definition" | Definition sentences: **0.53×** — actually *less* likely to be lifted |

Structural features (`has_schema`, `has_faq`, `structure_score`) also ranked low
in both models. That doesn't mean structure is worthless — it means that in this
data, **concrete content beat structural markup.**

---

## The leakage audit — why there are two variants

Before training anything, I checked whether the model could cheat. It could, twice:

**1. A placeholder that encoded the answer.** Cited pages that didn't appear in
Google's visible results were given a placeholder rank of 101. But such pages are
in the data *only because* they were cited — so every rank-101 row was `cited = 1`
(27% of all rows). The model would have learned "rank 101 → cited" and nothing else.

**2. A feature computed from the answer itself.** `domain_citation_rate` measured
how often a domain gets cited — calculated from the very citations being predicted.
For the 572 domains appearing once, it *was* the answer.

**Both were removed**, which is exactly why two variants exist: A drops the
placeholder rows, B drops the rank feature entirely. Both report lower numbers
than the leaky version would have. That's the point — a smaller true number beats
a bigger fake one.

`domain_rating` and `page_authority` were also dropped: they're a constant
placeholder. Backlink-based authority scores are estimates whose crawl coverage
varies a lot by country and vertical, so rather than present a third-party
heuristic as ground truth, the model uses only measured content signals.

---

## What you'd do with this

- **Skip searches that can't win** — local and featured-snippet queries never show an AI Overview
- **Write for the four signals** — depth, topic match, readability, one strong matching passage
- **Put concrete numbers in your sentences** — prices, percentages, formulas
- **Answer the real questions** — 714 unique questions surfaced from the data, led by "Ist es sinnvoll, ein Wertgutachten zu machen?" (in 62 searches)
- **Work the gap list** — 59 pages the model scores as likely but that aren't cited yet

---

## How the pipeline works

```
Collect real searches (DataForSEO: SERP + AI Overview citations)
        ↓
Crawl every candidate page, extract content features
        ↓
Audit for leakage, build two clean variants
        ↓
Train LightGBM + Logistic Regression, GroupKFold by search query
        ↓
Explain with SHAP · harden with held-out + calibration + permutation
        ↓
Mine the cached raw data: AI Overview text, PAA questions, lifted sentences
        ↓
Export tidy tables for Tableau
```

**Why GroupKFold?** Pages compete within one search. If rows from the same search
were split across training and test, the model could see the answer. Grouping by
query prevents that.

**Why PR-AUC, not accuracy?** Only a minority of pages get cited. A model that
always says "not cited" would look accurate and be useless. PR-AUC measures how
well the true positives are actually found.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Try it on synthetic sample data first (no API key needed)
python scripts/generate_sample_data.py
python scripts/run_pipeline.py
pytest -q
```

### Collect your own real data

```bash
export DATAFORSEO_LOGIN="your_login"
export DATAFORSEO_PASSWORD="your_password"

python scripts/collect_real_data.py --queries your_queries.txt --out data/raw/real.csv
python scripts/run_variants.py --data data/raw/real.csv
```

Credentials live in environment variables or a local `.env` — never in the code,
never committed (see `.env.example`).

### The deep-dive analyses

Everything below runs off the **local cache** the collection run writes
(`data/raw/_cache/`: raw SERP JSON + raw page HTML). No new API calls, no
re-crawling — and it means new questions can be asked of the same snapshot later.

```bash
python scripts/extract_from_cache.py           # AI Overview text, PAA questions, citations
python scripts/analyze_query_intent.py         # Finding 1 — intent segments
python scripts/analyze_insights.py             # exploratory charts
python scripts/analyze_aio_overlap.py          # which source shaped each answer
python scripts/analyze_passages.py             # Finding 4 — the lifted sentences
python scripts/analyze_passage_patterns.py     # Finding 4b — what kind of sentence
python scripts/analyze_paa.py                  # question themes
python scripts/harden_model.py --variant B     # Finding 3 — robustness checks
python scripts/prepare_tableau.py              # tidy tables for the dashboard
```

**Privacy:** real query lists, collected data, the cache, and per-row analysis
outputs are git-ignored on purpose — they're business research. The charts and
aggregate summaries in `reports/` are public: those are the findings.

---

## Repo structure

```
aio-gap-miner/
├── src/aio_gap_miner/
│   ├── config.py            # settings and feature lists
│   ├── data.py              # loading + synthetic sample generator
│   ├── feature_sets.py      # the leakage fix: variants A and B
│   ├── features.py          # raw columns → model input
│   ├── database.py          # SQL / SQLAlchemy layer
│   ├── stats.py             # significance tests
│   ├── model.py             # LightGBM + Logistic Regression, GroupKFold
│   ├── evaluate.py          # PR-AUC, baselines, comparison tables
│   ├── explain.py           # SHAP
│   └── collect/             # DataForSEO client, crawler, feature assembly
├── scripts/                 # every analysis, one file per question
├── reports/
│   ├── figures/             # all charts (public)
│   └── results/             # aggregate summaries (public), row-level data (ignored)
├── notebooks/               # the analysis walked through step by step
├── tableau/                 # dashboard spec + data source
└── tests/                   # automated checks
```

---

## Limitations

Worth stating plainly:

- **One vertical, one country, one point in time.** SERPs change daily; this is a snapshot.
- **Correlation, not causation.** SHAP explains *the model*, not Google's internals. Proving cause would need an intervention experiment: change a page, measure whether citation follows.
- **The crawler reads HTML**, so JavaScript-rendered content is under-measured.
- **Text overlap is evidence, not proof.** High similarity strongly suggests a page was the source; it doesn't demonstrate copying.
- **245 lifted sentence pairs** is enough for a stable pattern, but a bigger and more varied sample would make it generalisable.

---

## Tech stack

Python · pandas · scikit-learn · LightGBM · SHAP · SQLAlchemy/SQLite ·
seaborn · matplotlib · scipy · DataForSEO API · BeautifulSoup · Tableau · Git · pytest

## License

MIT — see [LICENSE](LICENSE).
