# Tableau dashboard — AIO Gap-Miner

The Python pipeline produces the data source; the interactive dashboard is built
in Tableau on top of it. Run the export first:

```bash
python scripts/export_tableau.py     # writes tableau/aio_gap_miner_tableau.csv
```

Then connect Tableau Desktop to `aio_gap_miner_tableau.csv`.

## Data source

One row per **(query, URL)** candidate. Key fields:

| Field | Type | Use |
|---|---|---|
| `query_id`, `query`, `url` | dimension | grain / labels |
| `cited` | measure (0/1) | actual outcome |
| `pred_proba_lgbm` | measure | model citation probability |
| `pred_proba_logreg` | measure | baseline probability |
| `predicted_cited` | measure (0/1) | model call at the decision threshold |
| `citation_gap` | measure (0/1) | predicted-cited **but not** actually cited = the opportunity |
| `top_driver` | dimension | the strongest SHAP feature for that row |
| `top_driver_shap` | measure | signed strength of that driver |
| feature columns | measure/dimension | filters & drill-down |

## Views to build (maps to "Project 4: Interactive Dashboard in Tableau")

1. **KPI header** — BANs (big aggregate numbers): total candidates, overall
   citation rate (`AVG([cited])`), model PR-AUC (parameter/annotation), number of
   citation gaps.
2. **Citation rate by rank bucket** — bar chart, `organic_rank` bucketed
   (top 3 / 4–10 / 11+) on columns, `AVG([cited])` on rows. Shows the ranking
   effect (a **Level-of-Detail** calc for per-query rates is a nice touch).
3. **Driver breakdown** — horizontal bar of `top_driver` frequency, coloured by
   mean `top_driver_shap`. This is the "why do pages get cited" story.
4. **Gap finder (interactive)** — a filtered table of `citation_gap = 1` rows,
   sorted by `pred_proba_lgbm` desc, with `query`, `url`, and the driver columns.
   Add filters for `content_type` and a `pred_proba_lgbm` range slider. This is
   the actionable output: the pages closest to winning a citation.

## Calculated fields to demonstrate (Tableau competencies)

```
// Predicted citation likelihood band
IF [pred_proba_lgbm] >= 0.66 THEN "High"
ELSEIF [pred_proba_lgbm] >= 0.33 THEN "Medium"
ELSE "Low" END

// LOD: citation rate per query (fixed at query grain)
{ FIXED [query_id] : AVG([cited]) }

// Model vs baseline lift, per row
[pred_proba_lgbm] - [pred_proba_logreg]
```

> Certification note: this dashboard doubles as portfolio evidence for the
> Tableau Desktop Specialist / Data Analyst tracks (calculated fields, LOD,
> filters, dashboard actions).
