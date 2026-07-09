"""Tests for the SQL/ETL layer, statistics, and the model comparison.

Fast, small-N checks that the ETL round-trips cleanly, the hypothesis tests
return sane output, and the two-model comparison beats the rank-only baseline.
"""

from __future__ import annotations

from aio_gap_miner.data import EXPECTED_COLUMNS, generate_synthetic_dataset
from aio_gap_miner.database import build_database, load_candidates, read_sql
from aio_gap_miner.evaluate import compare_models
from aio_gap_miner.features import build_xy
from aio_gap_miner.model import run_group_kfold_cv, run_logreg_group_kfold_cv
from aio_gap_miner.stats import hypothesis_tests


def _df():
    return generate_synthetic_dataset(n_queries=120, seed=7)


def test_sqlite_roundtrip(tmp_path):
    df = _df()
    engine = build_database(df, db_path=tmp_path / "t.db")
    back = load_candidates(engine)
    assert len(back) == len(df)
    assert set(back.columns) == set(EXPECTED_COLUMNS)
    # A simple SQL aggregation returns the right total.
    total = read_sql("SELECT COUNT(*) AS n FROM candidates", engine)["n"].iloc[0]
    assert int(total) == len(df)


def test_hypothesis_tests_output():
    tests = hypothesis_tests(_df())
    assert {"feature", "p_value", "effect_size_r"}.issubset(tests.columns)
    assert (tests["p_value"] >= 0).all() and (tests["p_value"] <= 1).all()
    # The engineered structure_score is testable (features added internally).
    assert "structure_score" in set(tests["feature"])


def test_models_beat_rank_only():
    df = _df()
    X, y, groups = build_xy(df)
    cv = run_group_kfold_cv(X, y, groups)
    oof_lr = run_logreg_group_kfold_cv(X, y, groups)
    comp = compare_models(df, {"lgbm": cv.oof_pred, "logreg": oof_lr}, groups)
    rank_pr = comp.loc["Rank-only heuristic", "pr_auc"]
    assert comp.loc["lgbm", "pr_auc"] > rank_pr
    assert comp.loc["logreg", "pr_auc"] > rank_pr
    # And everything beats the prevalence floor.
    assert rank_pr > comp.loc["Random / prevalence", "pr_auc"]
