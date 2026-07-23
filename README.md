# AIO Gap-Miner

**Can you predict which pages Google cites in its AI Overviews?**
I built a model that said yes. Then I found six reasons it was wrong.

---

## The one chart that matters

![The leakage cascade](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/headline_1_leakage_cascade.png)

Each step to the right closes one path by which the model could cheat. The red
line is a **permutation null** — the same pipeline with the labels shuffled, so
it shows what the score would be with no real signal at all.

By the third step, the model sits inside its own noise.

---

## So what does decide it?

![What predicts citation within a search](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/headline_2_per_search.png)

**Knowing which website a page belongs to is worth as much as knowing its Google
ranking. Knowing what's on the page — when the site is new to the model — is
worth almost nothing.**

---

## What held up, and what didn't

![Claims tested](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/headline_4_claims.png)

Four of the claims that failed were my own.

---

## The findings worth acting on

**1 · Some searches can never pay off.** Of 114 searches where Google showed a
map or a featured snippet, **zero** had an AI Overview. Check before you spend.

**2 · Ranking is not the gate.** **56%** of all citations went to pages that
never appeared in the visible top results.

![Citations without ranking](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/case_1_citation_without_ranking.png)

**3 · Video is heavily cited.** **257 of 336** AI Overviews (76%) cite at least
one YouTube video, averaging 2.4 videos each — and 97% of those videos don't rank.

**4 · Google does reuse sentences**, about twice as often as chance:

> **Page (sparkasse.de):** "Ein vollständiges Verkehrswertgutachten kostet in der Regel **zwischen 0,5 und 1,5 Prozent des** Verkehrswerts der Immobilie."
>
> **Google's answer:** "Ein rechtssicheres Verkehrswertgutachten kostet in der Regel **zwischen 0,5 und 1,5 Prozent des** Immobilienwerts."

**5 · Brand beats position.** At Google positions 9–10, established sites are
cited roughly twice as often as everyone else from the same slot.

![Brand vs position](https://github.com/fabo-lab/aio-gap-miner/raw/main/reports/figures/case_3_brand_vs_position.png)

---

## The data

533 real German real-estate searches · 6,646 (search, page) pairs · 336 AI
Overviews captured in full · 6,198 pages crawled · collected 16–17 July 2026 via
the DataForSEO API.

---

## 📄 The full account

**[PAPER.md](PAPER.md)** — the complete methodology: all six leaks, how each was
found, every correction and its effect, the statistics, and the limitations.

That document is the actual work. This page is the summary.

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q
```

```bash
# Collect your own data (needs a DataForSEO account)
export DATAFORSEO_LOGIN="..." DATAFORSEO_PASSWORD="..."
python scripts/collect_real_data.py --queries your_queries.txt --out data/raw/real.csv

# Then the analysis - all of it runs off the local cache, no further API calls
python scripts/extract_from_cache.py
python scripts/rebuild_labels.py                       # corrected labels + true organic rank
python scripts/run_definitive_analysis.py --data data/raw/real_v2.csv
python scripts/analyze_passages_v2.py --min-ngram 8 --permutations 1000
python scripts/build_headline_figures.py
python scripts/verify_setup.py
```

Credentials live in environment variables or a local `.env` — never in the code.
Real query lists and collected data are git-ignored; the charts and aggregate
results in `reports/` are the public part.

---

## Stack

Python · pandas · scikit-learn · LightGBM · SHAP · scipy · statsmodels ·
DataForSEO API · BeautifulSoup · youtube-transcript-api · Tableau · pytest

MIT licensed — see [LICENSE](LICENSE).
