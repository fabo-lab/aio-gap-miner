"""Explainability for the AIO Gap-Miner (TreeSHAP).

The whole point of the project is not just *whether* a URL gets cited but
*why*. TreeSHAP attributes each prediction to its features exactly (for tree
models), turning the LightGBM model into an auditable statement about what
drives AI Overview citations -- the differentiator versus a black-box or a
hand-tuned heuristic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X: pd.DataFrame):
    """Return ``(explainer, shap_values)`` for the positive (cited) class.

    Handles the several shapes different SHAP/LightGBM versions can return
    (list of per-class arrays, or a 3-D array).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    return explainer, shap_values


def _numeric_display(X: pd.DataFrame) -> pd.DataFrame:
    """Category columns -> integer codes so SHAP colour mapping never chokes."""
    disp = X.copy()
    for col in disp.columns:
        if str(disp[col].dtype) == "category":
            disp[col] = disp[col].cat.codes
    return disp


def mean_abs_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Global feature importance = mean(|SHAP|), sorted descending."""
    imp = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def plot_beeswarm(shap_values: np.ndarray, X: pd.DataFrame,
                  save_path: str | Path | None = None):
    """SHAP beeswarm summary: direction and magnitude of every feature."""
    plt.figure()
    shap.summary_plot(shap_values, _numeric_display(X), show=False, plot_size=(8, 6))
    fig = plt.gcf()
    fig.suptitle("SHAP summary — drivers of AI Overview citation", y=1.02, fontsize=11)
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_importance_bar(shap_values: np.ndarray, X: pd.DataFrame,
                        save_path: str | Path | None = None):
    """Bar chart of mean(|SHAP|) global importance."""
    plt.figure()
    shap.summary_plot(shap_values, _numeric_display(X), plot_type="bar",
                      show=False, plot_size=(8, 6))
    fig = plt.gcf()
    fig.suptitle("Global feature importance (mean |SHAP|)", y=1.02, fontsize=11)
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_dependence(shap_values: np.ndarray, X: pd.DataFrame, feature: str,
                    save_path: str | Path | None = None):
    """SHAP dependence plot for a single feature (shows its response curve)."""
    disp = _numeric_display(X)
    plt.figure(figsize=(6.5, 5))
    shap.dependence_plot(feature, shap_values, disp, show=False,
                         interaction_index=None)
    fig = plt.gcf()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig
