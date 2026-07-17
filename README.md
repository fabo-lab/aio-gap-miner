# AIO Gap-Miner

**Predicting — and explaining — which URLs get cited in Google AI Overviews.**

Google's AI Overviews (and ChatGPT, Perplexity, Claude) answer informational
queries at the top of the page and cite a *handful* of sources. If your URL
isn't in that citation set, your ranking position barely matters — the click
never happens. This project trains supervised models over **(query, URL)** pairs
to predict which candidate URLs get cited, and uses **SHAP** to explain *why* —
turning "GEO" from folklore into a measurable, auditable signal.

> Data Analytics & AI capstone. The committed sample data is **synthetic** so the
> whole pipeline runs for anyone with zero data access; real labelled AI Overview
> citation data plugs into the same schema.

**End-to-end pipeline:** `SQLite / SQLAlchemy ETL → EDA (seaborn) → inferential
statistics → feature engineering → GroupKFold CV (Logistic Regression + LightGBM)
→ evaluation → TreeSHAP → Tableau hand-off.`

---

## The framing (and why each choice)

The unit of observation is one **(query, URL)** pair — one row per candidate URL
for a query, labelled `cited = 1` if that URL appeared in the AI Overview
citation set, else `0`.

| Decision | Choice | Why |
|---|---|---|
| Task | Binary classification over (query, URL) pairs | Citation is a per-candidate yes/no |
| Models | **Logistic Regression** + **LightGBM** | Transparent linear baseline vs non-linear gradient-boosted trees |
| Validation | **GroupKFold** grouped by `query_id` | Labels are *query-relative* → no query may leak across the train/test split |
| Metric | **PR-AUC** (average precision) | Positives are rare (~17%) and query-relative; ROC-AUC and accuracy flatter the model |
| Baseline | **rank-only** heuristic (`1/organic_rank`) | The bar to beat: does learning add anything over "just trust the ranking"? |
| Explainability | **TreeSHAP** | Exact per-feature attribution — the "why" a black box or a rules engine can't give |

The single most important methodological point is **grouped cross-validation**.
Because whether a URL is cited depends on the *other* candidates for the same
query, rows from one query must never sit on both sides of the split. GroupKFold
on `query_id` guarantees leakage-safe, out-of-fold evaluation, and both models
are scored on the *same* folds for an apples-to-apples comparison.

---

## Do cited and non-cited URLs actually differ? (inferential statistics)

Before modelling, the A/B-testing question on observational data: treat *cited*
vs *not cited* as two groups and test where they diverge. Features are skewed and
non-normal, so we use the **Mann-Whitney U** test and report a rank-biserial
**effect size** (not just a p-value).

| Signal | Median (cited) | Median (not) | p-value | Effect size |
|---|---|---|---|---|
| `organic_rank` | 6.0 | 12.0 | ~1e-182 | 0.51 (large) |
| `query_url_similarity` | 0.70 | 0.54 | ~1e-176 | 0.51 (large) |
| `passage_match_score` | 0.73 | 0.55 | ~1e-144 | 0.46 (medium) |
| `domain_rating` | 70.4 | 61.4 | ~1e-79 | 0.34 (medium) |
| `structure_score` | 0.56 | 0.48 | ~1e-76 | 0.33 (medium) |

Cited URLs rank higher, match the query more closely, and are more structured —
all significant with meaningful effect sizes.

---

## Results (synthetic demonstration data)

**Read this as a pipeline demonstration, not a finding.** The committed data is
synthetic, and its label is *generated from the same features the model then
uses* (a weighted score plus noise — see `_citation_propensity` in `data.py`). So
the model recovering those signals is true *by construction*: it shows the
plumbing is correct — leakage-safe CV, honest baselines, working SHAP — not that
these signals drive real AI Overview citations. The substantive, data-driven
result lives in the [real-dataset variants](#data-leakage-audit-and-the-two-analysis-variants)
below.

`400 queries · 7,361 (query, URL) pairs · 17.1% cited` — PR-AUC reported as
per-fold **mean ± std** on the shared GroupKFold splits. LightGBM's stopping
iteration is chosen on a *nested*, query-grouped hold-out carved from inside each
fold, so the outer validation fold is never used to tune the tree count (see
`model.py`).

| Model | PR-AUC | ROC-AUC | Precision@k |
|---|---|---|---|
| **Gap-Miner (LightGBM)** | 0.567 ± 0.034 | 0.816 | 0.582 |
| **Logistic Regression** | 0.585 ± 0.033 | 0.828 | 0.616 |
| Rank-only heuristic | 0.481 ± 0.023 | 0.757 | 0.526 |
| Random / prevalence | 0.171 | 0.500 | — |

Both learned models clear a *strong* rank-only heuristic — by ~9 PR-AUC points
for LightGBM, ~10 for logistic regression — and lift per-query precision@k.
Because the synthetic label is close to linear in the engineered features,
logistic regression is very competitive here; gradient boosting's edge typically
grows with the non-linear interactions present in real citation data. LightGBM is
carried forward for SHAP because tree attributions are exact.

### SHAP: does the explainer recover the known structure?

![SHAP summary](reports/figures/shap_summary.png)

On synthetic data this is a *sanity check on the explainer*, not evidence about
the world: TreeSHAP should surface the signals the generator actually used, and
it does — **query↔passage semantic match**, **content structure** (schema / FAQ /
lists & tables), and **domain citation history**, on top of ranking position. The
same SHAP machinery applied to the real-data variants is what turns the GEO
thesis — *structured, on-topic pages get cited beyond what their SERP position
predicts* — from folklore into a measurable, auditable claim.

![Precision-Recall](reports/figures/pr_curve.png)

*(All figures are regenerated by `scripts/run_pipeline.py`.)*

---

## Quickstart

```bash
# 1. Install (editable, src layout)
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Generate the synthetic sample dataset
python scripts/generate_sample_data.py

# 3a. Run the full pipeline (ETL -> stats -> CV -> SHAP; figures to reports/figures/)
python scripts/run_pipeline.py

# 3b. …or open the narrative notebook
jupyter lab notebooks/01_gap_miner_baseline.ipynb

# 4. Export the Tableau data source
python scripts/export_tableau.py
```

Run the tests:

```bash
pip install pytest && pytest -q
```

### Collect real data (DataForSEO + on-page crawl)

The `collect` module turns a list of queries into labelled training rows in the
exact schema above. For each query it pulls the Google SERP **and its AI Overview
citations** from DataForSEO, crawls each candidate URL for content features, and
scores query↔page similarity. A `(query, URL)` pair is labelled `cited = 1` if
the URL appears in the AI Overview references.

Two files are written per real run: `<out>` (the (query,URL) ML training set,
including `title`/`snippet`/`crawl_ok` reference columns the model doesn't use)
and `<out>_meta.csv` (one row per query: SERP feature flags — `has_local_pack`,
`has_people_also_ask`, etc. — for clustering queries by local/informational/
transactional intent). Rows are written incrementally as each query completes
(crash-safe for long batches), and a query whose fetch fails is logged and
skipped rather than aborting the run. By default, the raw DataForSEO JSON and
raw HTML of every crawled page are also cached to `<out's folder>/_cache/`
(git-ignored) — a permanent snapshot, since SERPs change over time and a later
re-analysis can't otherwise reproduce today's exact data. Disable with
`--no-cache` if disk space is a concern.

```bash
# One-time setup: your own DataForSEO account (pay-per-use, ~$0.003/query)
export DATAFORSEO_LOGIN="your_login"
export DATAFORSEO_PASSWORD="your_password"

# Prove the pipeline offline first (no credentials, no network):
python scripts/collect_real_data.py --dry-run --out data/raw/dryrun.csv

# Collect a real dataset (Germany by default; edit queries.example.txt):
python scripts/collect_real_data.py --queries queries.example.txt --out data/raw/real.csv

# Train on it — no code changes:
python scripts/run_pipeline.py --data data/raw/real.csv
```

Semantic similarity uses TF-IDF out of the box; install the optional
`sentence-transformers` extra (`pip install -e ".[embeddings]"`) for true
embeddings. `domain_rating`/`page_authority` stay a neutral placeholder by design
(see "Why no backlink-based authority score" below); `domain_citation_rate` is
the real, empirically-grounded authority signal.

### Credentials & security

**Credentials never live in the code.** The collector reads them from
environment variables (`DATAFORSEO_LOGIN`, `DATAFORSEO_PASSWORD`), so nothing
secret is ever committed. Two safe ways to provide them:

- **Shell profile** (most leak-proof): put the `export` lines in `~/.zshrc` /
  `~/.bashrc`. They live in your home directory, outside the repo — impossible
  to commit.
- **`.env` file**: `cp .env.example .env`, fill in your values. `.env` is
  git-ignored; the script auto-loads it via `python-dotenv`. Only `.env.example`
  (placeholders) is committed.

Verify before pushing: `git check-ignore .env` should print `.env`, and
`git status` should never list it. If a key is ever exposed, rotate it in the
DataForSEO dashboard — git history is permanent.

**Real collected data stays private, by the same discipline.** Once you collect
on your own real query list, that list *and* everything derived from it are
competitive research, not a generic demo — they're git-ignored, not deleted:

- Real query lists: name them `queries_*.txt` (anything else, e.g.
  `queries.example.txt`, stays public). `data/raw/*` (including the raw HTML /
  SERP-JSON cache) is already git-ignored.
- Real Tableau exports: `python scripts/export_tableau.py --data data/raw/real.csv
  --out tableau/real_<name>.csv` — the `real_` prefix keeps it private and never
  collides with the tracked synthetic demo file at the default path.
- The committed notebook and `reports/figures/*.png` are safe to update with
  real results and commit: they show aggregate feature importance and metrics,
  not per-row query/URL detail.

Same verify-before-push habit: `git check-ignore queries_538.txt` (or whatever
you named it) should print the path.

### Bring your own CSV

Alternatively, drop any CSV with the columns in `EXPECTED_COLUMNS`
(see `src/aio_gap_miner/data.py`) into `data/raw/` and point the pipeline at it:

```bash
python scripts/run_pipeline.py --data data/raw/your_citation_data.csv
```

---

## Repository layout

```
aio-gap-miner/
├── src/aio_gap_miner/         # installable package
│   ├── config.py              # paths, feature lists, LightGBM params (single source of truth)
│   ├── data.py                # (query, URL) schema, synthetic generator, loaders
│   ├── database.py            # SQLAlchemy/SQLite ETL + analytical SQL
│   ├── features.py            # engineered features + model-matrix builder
│   ├── stats.py               # descriptive + inferential statistics, seaborn plots
│   ├── model.py               # GroupKFold CV: LightGBM + Logistic Regression
│   ├── evaluate.py            # PR-AUC, baselines, precision@k, model comparison, plots
│   ├── explain.py             # TreeSHAP values + plots
│   └── collect/               # real-data collection (DataForSEO + crawl)
│       ├── serp.py            #   SERP + AI Overview citations
│       ├── crawl.py           #   on-page feature extraction
│       └── pipeline.py        #   semantic/authority features + schema assembly
├── notebooks/
│   └── 01_gap_miner_baseline.ipynb   # the narrative, with embedded outputs
├── scripts/
│   ├── generate_sample_data.py       # build the committed sample
│   ├── build_database.py             # standalone ETL step
│   ├── run_pipeline.py               # end-to-end CLI
│   ├── collect_real_data.py          # DataForSEO + crawl -> labelled dataset
│   ├── export_tableau.py             # write the Tableau data source
│   └── build_notebook.py             # regenerate the notebook from source
├── tableau/                   # Tableau data source + dashboard spec
├── tests/                     # schema / leakage / ETL / stats / comparison guards
├── data/sample/               # synthetic sample dataset (committed)
└── reports/figures/           # generated plots
```

---

## Tech stack

`Python · pandas · SQL (SQLAlchemy / SQLite) · seaborn · matplotlib · scipy ·
scikit-learn · LightGBM · SHAP · DataForSEO · BeautifulSoup · Tableau · Git`

## Code quality

Linting, formatting, and dead-code / dependency checks are wired in and pass
clean:

```bash
pip install -e ".[dev]"
ruff check .        # lint: E/F/I/UP/B — passes (notebooks exempt cell idioms, see config)
ruff format .       # formatting
deptry .            # unused/missing dependencies — passes
pre-commit install  # run all checks automatically on every commit
```

Config lives in `pyproject.toml` (`[tool.ruff]`, plus
`[tool.ruff.lint.per-file-ignores]` for notebook cell idioms, and
`[tool.deptry]`) and `.pre-commit-config.yaml`.

## Feature set

Signals per (query, URL) pair — SERP + on-page crawl:

- **Ranking:** `organic_rank`, `rank_reciprocal` *(engineered)*
- **Authority:** `domain_rating`, `page_authority` *(neutral placeholder by design — see below)*, `domain_citation_rate` *(real signal)*
- **Relevance:** `query_url_similarity`, `passage_match_score`, `num_entities_matched`
- **Structure / extractability:** `has_schema`, `has_faq`, `num_lists_tables`, `structure_score` *(engineered)*
- **Content:** `word_count`, `readability_score`, `content_freshness_days`, `content_type`
- **Source type:** `is_forum`, `is_video`, `is_https`

## Data-leakage audit and the two analysis variants

Auditing the first real dataset before modelling surfaced label leakage that
would have made a naive model look great and mean nothing. Documented honestly
because catching and handling it is the point:

- **`organic_rank == 101`** was a sentinel for cited URLs not in the organic
  block. Since those URLs are in the data *only because* they were cited, every
  rank-101 row is `cited = 1` — the feature encodes the label (~27% of rows).
- **`domain_citation_rate`** is computed from the `cited` label itself; for the
  572 single-occurrence domains it equals the label exactly. Dropped from the
  model (reconstructable leakage-free later via out-of-fold encoding).
- **`domain_rating` / `page_authority`** are a constant placeholder (no signal);
  dropped for cleanliness.

`scripts/run_variants.py` reports two defensible framings side by side:

| Variant | Question | Setup |
|---|---|---|
| **A** | Among pages Google already ranks, which get cited? | rank-101 rows removed, `organic_rank` kept |
| **B** | What on-page signals distinguish cited pages, independent of rank? | all rows kept, `organic_rank` dropped as a feature |

On the real dataset (533 queries, 6,646 rows), both beat their baselines on
per-fold GroupKFold PR-AUC, and both surface the *same* top content drivers —
`word_count`, `query_url_similarity`, `readability_score`, `passage_match_score`
— which is the robust, honest finding the project set out to test.

```bash
python scripts/run_variants.py --data data/raw/real.csv        # both variants
```

## Why no backlink-based authority score

`domain_rating` / `page_authority` are deliberately left as a neutral
placeholder rather than wired to a real backlink-based score (Moz, Ahrefs,
DataForSEO Backlinks `bulk_rank`). These proxies estimate authority from
backlink-crawl coverage, and that coverage is well documented to vary
significantly by country and vertical — for a DACH/niche market they'd be a
noisy approximation at best, presenting a third-party heuristic as if it were
ground truth. Instead, the model leans on **`domain_citation_rate`**: the
domain's own empirical citation track record *within the collected data* —
measured directly against the actual target rather than approximated via a
generic link-graph score.

A constant placeholder costs nothing: with zero variance across rows, the
model and SHAP correctly assign it ~0 importance — it doesn't bias the results,
it simply contributes no signal, exactly as if the columns were omitted.

## Roadmap

1. **Real labels** — swap the synthetic sample for a labelled AI Overview
   citation set (same schema).
2. **Richer features** — real embeddings for query↔passage similarity, NER-based
   entity coverage, SERP-feature flags, Core Web Vitals.
3. **Probability calibration** — isotonic/Platt on top of the ranker for an
   absolute "citation likelihood" score.
4. **Gap reports** — per-page SHAP turns the model into a prescriptive tool: for a
   citation you *don't* hold, it names the specific levers (add FAQ schema,
   tighten the answer passage, raise topical coverage) to close the gap.

## License

MIT — see [LICENSE](LICENSE).
