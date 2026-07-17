# AIO Gap-Miner

**A model that predicts which web pages Google picks for its AI Overview answers — and explains why.**

When you search Google, you often see an **AI Overview** box at the top with a short answer and 2-5 source links. If your page is not one of those links, it doesn't matter how well you rank — most people never scroll down to click your result.

This project builds a machine learning model that answers two questions for any (search query, web page) pair:

1. **Will this page get picked as a source?** (yes / no prediction)
2. **Why?** (which page features actually matter — using SHAP)

The goal: turn "make my content AI-friendly" from a guess into something you can actually measure.

---

## The real result (this is the important part)

I collected **533 real search queries** (German real-estate topics) through the DataForSEO API, checked which pages Google's AI Overview actually cited, and trained the model on it. This is not a demo with made-up numbers — this is real Google data from real searches.

**Before training, I checked the data for problems and found two.** Both are explained below. After fixing them, here are the honest results:

### Variant A — "Among pages that already rank, which get picked?"

(Only pages that show up in Google's normal top results, 4,857 rows)

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| **LightGBM (my model)** | **0.583** | 0.760 |
| Logistic Regression (simple baseline) | 0.442 | 0.670 |
| Rank-only guess ("just trust the ranking") | 0.369 | 0.617 |
| Random guessing | 0.291 | 0.500 |

My model beats "just look at the ranking" by **+21 percentage points** of PR-AUC. That means: knowing about the *content* of a page adds real, measurable value on top of knowing its Google rank.

![SHAP - what drives citation, ranked pages](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/shap_importance_variant_A.png)

### Variant B — "What content makes a page get picked, ignoring rank?"

(Every collected page, including ones that got cited without ranking in the top results, 6,646 rows)

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| **LightGBM (my model)** | **0.770** | 0.770 |
| Logistic Regression (simple baseline) | 0.658 | 0.666 |
| Rank-only guess | 0.433 | 0.272 |
| Random guessing | 0.482 | 0.500 |

![SHAP - what drives citation, content only](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/shap_importance_variant_B.png)

### The one finding that shows up in both variants

Look at both SHAP charts above — **the same four features come out on top every time**: word count, how well the page matches the search query, readability, and how well specific passages match. That consistency across two different ways of slicing the data is what makes this a real finding, not a fluke.

Full numbers: [`reports/results/`](reports/results). Reproduce it yourself:
```bash
python scripts/run_variants.py --data data/raw/real.csv
```

---

## The two problems I found in the data (and fixed) — before training anything

Before trusting any model, I checked the raw data for shortcuts the model could cheat with. I found two, and I think this is the most important part of the whole project — catching this *before* presenting fake-looking results.

**Problem 1: a label that leaked into a feature.**
Some cited pages didn't show up in Google's normal top results at all — they got picked by the AI Overview from further down, or from outside the visible list. I had marked those rows with a placeholder rank ("101"). But that placeholder rank only ever appears on pages that *are* cited — so the model could just learn "rank = 101 → cited" and call it a day. That's not a real pattern, that's leakage.

**Problem 2: a feature computed from the answer itself.**
I had a feature called "how often does this domain get cited" — but it was calculated *using the same citations I was trying to predict*. For domains that only appear once in the data, this feature was literally identical to the answer.

**The fix:** I removed both. That's exactly why the two variants above exist — Variant A drops the fake-ranked rows, Variant B drops the leaking rank feature entirely. Both report honest, lower numbers than before the fix — and that's the point. A smaller, true number beats a bigger, fake one.

---

## What does this mean for AI Overview optimization (AIO/AEO)?

The practical takeaway, in plain terms:

- **Ranking is not enough.** Pages get cited that don't even rank in the normal top 10-20 results — being well-written matters on its own.
- **The content features that matter most are consistent**: page length, how directly the text answers the query, readability, and passage-level match. These beat structural tricks like FAQ schema or table markup, which came out weaker in both variants.
- Because the model uses SHAP, it doesn't just say "this page will/won't get cited" — it can say **why**, feature by feature, for any single page. That's the difference between a prediction and something you can actually act on.

---

## How the whole pipeline works

```
Collect real search data (DataForSEO)
        ↓
Check page content (crawl + extract features)
        ↓
Clean the data, check for leakage
        ↓
Train two models: LightGBM (tree-based) + Logistic Regression (simple baseline)
        ↓
Evaluate with GroupKFold cross-validation (grouped by search query, so no cheating)
        ↓
Explain the winning model with SHAP
        ↓
Export results as plots, tables, and a Tableau file
```

**Why GroupKFold and not normal cross-validation?** Whether a page gets cited depends on which *other* pages are competing for the same query. If rows from the same query ended up split between training and test data, the model could accidentally see the answer. GroupKFold keeps every query's rows together, only ever on one side of the split.

**Why PR-AUC instead of accuracy?** Only a minority of pages get cited (~29-48% depending on the variant). A model that just guesses "not cited" every time would already look "accurate" by doing nothing useful. PR-AUC actually measures how well the model finds the true positives.

---

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Try it on a small, made-up example first (no API key needed)
python scripts/generate_sample_data.py
python scripts/run_pipeline.py

# 3. Run the tests
pip install pytest && pytest -q
```

### Collecting your own real data

```bash
export DATAFORSEO_LOGIN="your_login"
export DATAFORSEO_PASSWORD="your_password"

# Put your search queries in a text file, one per line, then:
python scripts/collect_real_data.py --queries your_queries.txt --out data/raw/real.csv

# Train + evaluate on it, leakage-safe:
python scripts/run_variants.py --data data/raw/real.csv
```

Credentials go in environment variables or a local `.env` file — never in the code, never committed. See `.env.example`.

**A note on privacy:** real query lists and real collected data are git-ignored on purpose (see `.gitignore`) — they're someone's actual business research, not generic demo data. Only the synthetic example and the code are meant to be public.

---

## Project folder structure

```
aio-gap-miner/
├── src/aio_gap_miner/
│   ├── config.py          # all settings and feature lists in one place
│   ├── data.py             # loads data, builds the synthetic example
│   ├── feature_sets.py     # the leakage fix: defines Variant A and Variant B
│   ├── features.py         # turns raw columns into the model's input
│   ├── database.py         # SQL / SQLAlchemy version of the data
│   ├── stats.py            # statistical tests (is the difference real?)
│   ├── model.py            # trains LightGBM + Logistic Regression
│   ├── evaluate.py         # scores models, builds comparison tables
│   ├── explain.py          # SHAP — explains individual predictions
│   └── collect/            # pulls real data from DataForSEO + crawls pages
├── scripts/
│   ├── run_pipeline.py       # run everything on one dataset
│   ├── run_variants.py       # run both leakage-safe variants (the real analysis)
│   ├── collect_real_data.py  # build a real dataset from a query list
│   └── export_tableau.py     # export results for Tableau
├── reports/
│   ├── figures/             # all charts, including the SHAP charts above
│   └── results/             # comparison tables as plain CSV
├── notebooks/                # the same analysis, walked through step by step
├── tableau/                  # dashboard data + spec
└── tests/                    # automated checks (17+ tests, all passing)
```

---

## Tech stack

Python · pandas · scikit-learn · LightGBM · SHAP · SQLAlchemy/SQLite · seaborn · scipy · DataForSEO API · BeautifulSoup · Tableau · Git · pytest

## What's left to do

- Cluster the queries by intent (local vs. informational vs. transactional) using the SERP-feature data already collected
- Add a real domain-authority signal from DataForSEO's own backlink data (currently a neutral placeholder — explained in the code)
- Connect the collected data to a proper SQL database with dbt for repeatable, tested transformations
- Turn per-page SHAP into a "gap report": for any page that *isn't* cited yet, list the specific things to fix

## License

MIT — see [LICENSE](LICENSE).
