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
| Searches collected | **538** (German real-estate topics, via DataForSEO API) |
| Searches in the modelling dataset | **533** (3 failed during collection, 2 SERP files unparseable) |
| (search, page) pairs | **6,646** |
| Pages crawled and measured | 6,198 |
| AI Overviews with full answer text | **336** |
| "People also ask" questions collected | 1,970 (714 unique) |
| Median sources cited per AI Overview | **21** (range 5–49) |

Everything below comes from that dataset. No demo numbers.

The collection run also cached every raw API response and every page's raw HTML
locally, which is what made the deeper analyses possible later without re-querying
or re-crawling anything.

---

## Finding 1 — Not every search is a game you can play

Before optimising anything, you should know whether Google even shows an AI
Overview for that kind of search. Often it doesn't.

![Intent segments](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/insight_8_intent_segments.png)

| Search type | Share | AI Overview appears | Citation rate |
|---|---|---|---|
| Local (Google shows a map) | 17% (89) | **0%** | 0% |
| Featured snippet shown | 5% (25) | **0%** | 0% |
| Informational | 79% (422) | **80%** | 53% |

**22% of the searches in this dataset never show an AI Overview.** Google answers
those with a map or a snippet instead. Optimising a page for those searches is
effort spent on something that cannot happen.

*Method note: I tried KMeans clustering first. On these sparse feature flags it
produced overlapping, hard-to-explain groups. A transparent rule on Google's own
signals separates the outcome almost perfectly and anyone can follow it — so I
used that instead. Choosing the interpretable tool over the fancy one was the
right call here.*

---

## Finding 2 — The model works, and it beats the obvious answer

All five rows below are measured on the **same population**: the 4,857 pages that
genuinely appear in Google's results. That makes them directly comparable, with
no caveats. (Why that restriction matters is explained in the leakage audit.)

| Model | PR-AUC | ROC-AUC | vs random |
|---|---|---|---|
| Random guessing (prevalence) | 0.291 | 0.500 | 1.00× |
| Rank-only heuristic | 0.368 | 0.616 | 1.27× |
| Logistic Regression (content + rank) | 0.437 | 0.669 | 1.50× |
| **LightGBM — content signals only** | **0.523** | 0.724 | **1.80×** |
| **LightGBM — content + rank** | **0.568** | 0.754 | **1.95×** |

The standard SEO assumption is "rank well and you'll get cited". That heuristic
alone scores 0.368. **Content signals alone — ignoring rank entirely — score
0.523.** Content carries more information about citation than ranking does.
Together they're best.

![Feature importance](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/shap_importance_variant_A.png)

The consistent top drivers are **content depth, topic match to the search,
readability, and best passage match**.

*Reproducibility note: LightGBM's histogram building depends on thread count, so
numbers can move by ±0.02 between machines even with a fixed seed. The ordering
and the conclusions are stable.*

---

## Finding 3 — Content depth matters, and the relationship is simple

![Word count](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/insight_5_wordcount_curve.png)

| Word count | Cited (ranked pages) | n |
|---|---|---|
| <500 | 18.7% | 455 |
| 500–1k | 24.5% | 691 |
| 1k–2k | 25.9% | 1,730 |
| 2k–4k | **36.4%** | 1,571 |
| >4k | 33.8% | 408 |

Citation rate rises with content depth and plateaus above roughly 2,000 words.

**An earlier version of this chart showed a U-shape** with a peak at very short
pages. That peak was made entirely of cited-but-not-ranked rows — a selection
artefact, not a finding. The chart above plots both lines so the distortion is
visible. `scripts/audit_artifacts.py` recomputes every descriptive statistic both
ways for exactly this reason.

---

## Finding 4 — It holds up under pressure

Robustness checks on the content + rank model:

| Check | Result |
|---|---|
| Held-out test (107 searches never seen in training) | PR-AUC **0.535** vs 0.240 prevalence — **2.23× random** |
| Top-4 features shared by all 5 CV folds | 3 of 4, importances vary under 13% |
| SHAP vs permutation importance (independent methods) | **5 of 5 top features agree** |
| Calibration error | 0.131 — see caveat below |

![SHAP stability](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/hardening_shap_stability.png)

**Honest caveat on calibration:** the model *ranks* pages reliably, but its raw
probabilities are optimistic — when it predicts 88%, the real rate is about 62%.
Use it to prioritise pages, not to quote a probability. Fixing this would mean
adding a calibration layer (Platt scaling or isotonic regression), which is a
clear next step rather than something to hide.

---

## Finding 5 — What Google actually lifts, sentence by sentence

This is where the project answers its original question. For every AI Overview,
I compared its answer text against the real content of every page it cited, and
found the near-identical sentence pairs. **245 of them across 129 searches.**

> **Page (immoverkauf24.de):**
> "Die Formel lautet: Sachwert = (Bodenwert + Gebäudesachwert) × Marktanpassungsfaktor."
>
> **Google's AI answer:**
> "Die Formel lautet vereinfacht: Sachwert = (Bodenwert + Gebäudesachwert) × Marktanpassungsfaktor."

> **Page (haus.de):**
> "Bei einem ermittelten Wert von circa 400.000 Euro wären das also zwischen 2.000 und 6.000 Euro."
>
> **Google's AI answer:**
> "Bei einem Haus im Wert von 400.000 Euro liegen die Kosten somit meist zwischen 2.000 und 6.000 Euro."

### The pattern behind the examples

Comparing the 245 lifted sentences against 2,994 sentences from the **same pages**
that weren't lifted:

![Passage patterns](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/insight_9_passage_patterns.png)

| A sentence containing… | Lifted | Not lifted | How much likelier |
|---|---|---|---|
| a price in € | 27.8% | 2.7% | **10.1×** |
| a percentage | 13.9% | 1.5% | **9.2×** |
| any number | 48.6% | 15.1% | **3.2×** |
| a formula | 3.3% | 1.2% | 2.8× |

Sentence length was essentially identical — 16 words versus 15. **It's not about
writing more. It's about writing something concrete enough to be quoted.**

Because the control group is drawn from the same pages, this comparison is not
affected by which pages entered the dataset — the selection cancels out.

---

## Finding 6 — Two common SEO beliefs this data doesn't support

Practitioner guides claim that AI systems favour list structures, and that you
should lead with a definition. In this dataset:

| Claim | What the data shows |
|---|---|
| "LLMs love lists" | List-introducing sentences: **0.90×** — no advantage |
| "Lead with a definition" | Definition sentences: **0.53×** — actually *less* likely to be lifted |

Structural page features also ranked low in the model. Checked against the
population baseline, schema markup gives **1.09×** and an FAQ section **1.26×** —
real but modest, and far below the content signals. In this data, **concrete
content beat structural markup.**

---

## Finding 7 — Which sites get cited when they actually rank

![Domain leaderboard](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/insight_3_top_domains.png)

| Domain | Times ranked | Times cited | Rate | vs average |
|---|---|---|---|---|
| immobilienscout24.de | 132 | 81 | 61% | **2.11×** |
| test.de | 42 | 24 | 57% | 1.96× |
| immoverkauf24.de | 37 | 21 | 57% | 1.95× |
| sparkasse.de | 216 | 118 | 55% | **1.88×** |
| drklein.de | 142 | 69 | 49% | 1.67× |
| ksk-immobilien.de | 187 | 79 | 42% | 1.45× |
| *(overall average)* | | | *29%* | *1.00×* |

Established, trusted brands get cited roughly **twice as often as average from
the same ranking positions**. This is an authority effect measured directly
against the outcome — which is also why the model doesn't use third-party
backlink scores (see below).

**Correction worth stating plainly:** an earlier version of this table reported
"YouTube is cited 95% of the time it appears". That was a selection artefact —
92% of YouTube's rows were cited-but-not-ranked pages, which are in the dataset
*only because* they were cited. Among video pages that genuinely rank, the
citation rate is **15.8%** versus **29.3%** for non-video pages: the opposite
conclusion. Video content does reach AI Overviews, but mostly from *outside* the
visible results rather than by ranking well.

---

## The leakage audit — the most important part

Before trusting any result, I checked whether the model could cheat. It could —
three times. All three are documented here because catching them is the point.

**1. A placeholder that encoded the answer.**
Cited pages that don't appear in Google's visible results were given a
placeholder rank of 101. But such pages are in the data *only because* they were
cited — so all 1,789 of those rows are `cited = 1` by construction (27% of the
dataset). A model would learn "rank 101 → cited" and nothing useful.

**2. A feature computed from the answer itself.**
`domain_citation_rate` measured how often a domain gets cited — calculated from
the very citations being predicted. For the 572 domains appearing once, it *was*
the label.

**3. Indirect leakage through a subgroup.**
A later framing kept the rank-101 rows and simply dropped `organic_rank` as a
feature. That looked safe but wasn't: those rows differ systematically (about
**9× more likely to be video pages**), so the model could identify the
always-cited subgroup through other features. Evidence: that model scored 0.773
overall but only **0.479** when evaluated on genuinely ranked pages — worse than
a model trained on ranked pages alone.

**The fix:** everything is now measured on ranked pages only, where nothing
encodes the label.

**And then a systematic sweep.** After finding that the third leak had also
distorted *descriptive* statistics (the YouTube rate, the word-count curve),
`scripts/audit_artifacts.py` recomputes every descriptive finding both ways.
Result: only the video finding was genuinely reversed; schema, FAQ and HTTPS held
up. Query-level findings — intent segments, SERP-feature predictors, sources per
answer — are unaffected, since the artefact is page-level.

**Also dropped:** `domain_rating` and `page_authority` were a constant
placeholder. Backlink-based authority scores are estimates whose crawl coverage
varies a lot by country and vertical, so rather than present a third-party
heuristic as ground truth, the model uses only measured content signals — and
Finding 7 shows an authority effect measured directly instead.

---

## What you'd do with this

- **Skip searches that can't win** — local and featured-snippet queries never showed an AI Overview here
- **Write for the four signals** — depth, topic match, readability, one strong matching passage
- **Put concrete numbers in your sentences** — prices, percentages, formulas
- **Answer the real questions** — 714 unique questions surfaced from the data, led by *"Ist es sinnvoll, ein Wertgutachten zu machen?"* (in 62 searches)
- **Work the gap list** — 59 pages the model scores as likely but that aren't cited yet

---

## How the pipeline works

```
Collect real searches (DataForSEO: SERP + AI Overview citations)
        ↓
Crawl every candidate page, extract content features
        ↓
Audit for leakage · restrict to one clean population
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

**Why PR-AUC, not accuracy?** Only 29% of ranked pages get cited. A model that
always says "not cited" would look 71% accurate and be useless. PR-AUC measures
how well the true positives are actually found.

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
python scripts/run_headline_comparison.py --data data/raw/real.csv
```

Credentials live in environment variables or a local `.env` — never in the code,
never committed (see `.env.example`).

### The full analysis

Everything below runs off the **local cache** the collection run writes
(`data/raw/_cache/`: raw SERP JSON + raw page HTML). No new API calls, no
re-crawling — which is what allows new questions to be asked of the same snapshot
later.

```bash
python scripts/extract_from_cache.py           # AI Overview text, PAA questions, citations
python scripts/run_headline_comparison.py      # Finding 2 — the model comparison
python scripts/analyze_query_intent.py         # Finding 1 — intent segments
python scripts/audit_artifacts.py              # Finding 3 + the artefact sweep
python scripts/harden_model.py --variant A     # Finding 4 — robustness checks
python scripts/analyze_passages.py             # Finding 5 — the lifted sentences
python scripts/analyze_passage_patterns.py     # Finding 5b — what kind of sentence
python scripts/analyze_domains.py              # Finding 7 — domain leaderboard
python scripts/analyze_paa.py                  # question themes
python scripts/prepare_tableau.py              # tidy tables for the dashboard
python scripts/verify_setup.py                 # check everything is present and current
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
│   ├── feature_sets.py      # leakage-safe feature definitions
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
├── notebooks/               # methodology walkthrough
├── tableau/                 # dashboard spec + data source
└── tests/                   # automated checks
```

---

## Limitations

Worth stating plainly:

- **One vertical, one country, one point in time.** SERPs change daily; this is a snapshot.
- **Correlation, not causation.** SHAP explains *the model*, not Google's internals. Proving cause needs an intervention experiment: change a page, measure whether citation follows.
- **Probabilities are not calibrated** (error 0.131) — use the model to rank pages, not to quote a likelihood.
- **The crawler reads HTML**, so JavaScript-rendered content is under-measured.
- **Text overlap is evidence, not proof.** High similarity strongly suggests a page was the source; it doesn't demonstrate copying. There is also no random-pairing baseline for the 245 matches — a useful addition.
- **245 lifted sentence pairs** is enough for a stable pattern, but a bigger, more varied sample would be needed to generalise.

---

## Tech stack

Python · pandas · scikit-learn · LightGBM · SHAP · SQLAlchemy/SQLite ·
seaborn · matplotlib · scipy · DataForSEO API · BeautifulSoup · Tableau · Git · pytest

## License

MIT — see [LICENSE](LICENSE).
