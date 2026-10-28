"""Binary ROC/AUROC helpers for early-warning indicator ensembles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
from sklearn.metrics import roc_curve as sklearn_roc_curve


@dataclass
class BinaryRocCurveResult:
    """ROC outputs for one indicator."""

    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auroc: float
    n_valid: int
    n_missing: int


@dataclass
class EnsembleRocResult:
    """Trajectory-level scores and ROC curves for an ensemble."""

    y_true: np.ndarray
    scores: dict[str, np.ndarray]
    curves: dict[str, BinaryRocCurveResult]


def _indicator_colors(result: EnsembleRocResult) -> dict[str, str]:
    """Assign consistent colors across ROC and AUROC plots."""
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    return {
        name: cycle[idx % len(cycle)]
        for idx, name in enumerate(result.curves)
    }


def _validate_y_true(y_true: Any) -> np.ndarray:
    """Validate a binary trajectory label vector."""
    y = np.asarray(y_true)
    if y.ndim != 1:
        raise ValueError("y_true must be a 1D binary array")

    if np.issubdtype(y.dtype, np.bool_):
        y = y.astype(int)
    else:
        y = y.astype(float)
        unique = np.unique(y)
        if not np.all(np.isin(unique, [0.0, 1.0])):
            raise ValueError("y_true must contain only binary labels 0/1")
        y = y.astype(int)

    if np.unique(y).size < 2:
        raise ValueError("y_true must contain both classes 0 and 1")
    return y


def _safe_tau(series: Any) -> float:
    """Kendall tau reduction for one trajectory-level indicator series."""
    values = np.asarray(series, dtype=float).reshape(-1)
    finite_mask = np.isfinite(values)
    if finite_mask.sum() < 2:
        return np.nan

    time = np.arange(values.size, dtype=float)[finite_mask]
    tau, _ = kendalltau(time, values[finite_mask], nan_policy="omit")
    return float(tau)


def _reduce_trajectory_collection(values: Any, n_trajectories: int) -> np.ndarray:
    """
    Reduce notebook-style indicator collections to one score per trajectory.

    Accepted inputs:
    - 1D score vector of length n_trajectories
    - 2D array of shape (n_trajectories, n_time)
    - sequence of 1D arrays, one per trajectory
    """
    if isinstance(values, np.ndarray):
        array = np.asarray(values)
        if array.ndim == 1:
            if array.shape[0] != n_trajectories:
                raise ValueError("score vector length must match y_true")
            if array.size == 0:
                return np.asarray([], dtype=float)
            first = np.asarray(array[0])
            if first.ndim == 0:
                return array.astype(float)
            return np.asarray([_safe_tau(item) for item in array], dtype=float)
        if array.ndim == 2:
            if array.shape[0] != n_trajectories:
                raise ValueError("indicator array first dimension must match y_true")
            return np.asarray([_safe_tau(row) for row in array.astype(float)], dtype=float)
        raise ValueError("indicator arrays must be 1D scores or 2D trajectory-by-time arrays")

    if not isinstance(values, Sequence):
        raise ValueError("indicator values must be a numpy array or a sequence of arrays")
    if len(values) != n_trajectories:
        raise ValueError("indicator sequence length must match y_true")
    if len(values) == 0:
        return np.asarray([], dtype=float)

    first = np.asarray(values[0])
    if first.ndim == 0:
        return np.asarray(values, dtype=float)
    return np.asarray([_safe_tau(item) for item in values], dtype=float)


def ensemble_roc(y_true: Any, indicators: Mapping[str, Any]) -> EnsembleRocResult:
    """
    Compute binary ROC/AUROC for an ensemble of trajectory-level indicator series.

    Parameters
    ----------
    y_true:
        1D binary labels aligned with trajectories. Use 1=tipping, 0=non-tipping.
    indicators:
        Mapping from indicator name to either:
        - a 2D array (n_trajectories, n_time),
        - a sequence of 1D arrays, or
        - a 1D precomputed score vector of length n_trajectories.

    Returns
    -------
    EnsembleRocResult
        Contains the reduced score vectors and one ROC curve per valid indicator.
    """
    labels = _validate_y_true(y_true)
    if not indicators:
        raise ValueError("indicators must contain at least one named indicator")

    scores: dict[str, np.ndarray] = {}
    curves: dict[str, BinaryRocCurveResult] = {}

    for name, values in indicators.items():
        score_vector = _reduce_trajectory_collection(values, len(labels))
        scores[name] = score_vector

        valid_mask = np.isfinite(score_vector)
        n_missing = int((~valid_mask).sum())
        if valid_mask.sum() == 0:
            continue

        y_valid = labels[valid_mask]
        score_valid = score_vector[valid_mask]
        if np.unique(y_valid).size < 2:
            continue

        fpr, tpr, thresholds = sklearn_roc_curve(y_valid, score_valid)
        auroc = float(sklearn_roc_auc_score(y_valid, score_valid))
        curves[name] = BinaryRocCurveResult(
            fpr=np.asarray(fpr, dtype=float),
            tpr=np.asarray(tpr, dtype=float),
            thresholds=np.asarray(thresholds, dtype=float),
            auroc=auroc,
            n_valid=int(valid_mask.sum()),
            n_missing=n_missing,
        )

    return EnsembleRocResult(y_true=labels, scores=scores, curves=curves)


def plot_roc(
    result: EnsembleRocResult,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    chance: bool = True,
    show_auroc: bool = True,
    auroc_decimals: int = 2,
    linewidth: float = 2,
    bold_best: bool = False,
    bold_top_n: int = 0,
    label_fontsize: float | None = None,
    tick_labelsize: float | None = None,
    legend_fontsize: float | None = None,
) -> plt.Axes:
    """Plot one ROC curve per indicator."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    label_kwargs = {} if label_fontsize is None else {"fontsize": label_fontsize}
    legend_kwargs = {} if legend_fontsize is None else {"fontsize": legend_fontsize}

    colors = _indicator_colors(result)
    bold_count = int(bold_top_n)
    if bold_best:
        bold_count = max(1, bold_count)
    bold_names = set()
    if bold_count > 0 and result.curves:
        names_by_auroc = sorted(
            result.curves,
            key=lambda name: result.curves[name].auroc,
            reverse=True,
        )
        bold_names = set(names_by_auroc[:bold_count])

    for name, curve in result.curves.items():
        label = (
            name
            if not show_auroc
            else f"{name} (AUROC={curve.auroc:.{auroc_decimals}f})"
        )
        ax.plot(curve.fpr, curve.tpr, lw=linewidth, color=colors[name], label=label)

    if chance:
        ax.plot([0.0, 1.0], [0.0, 1.0], color="black", ls="--", lw=linewidth, zorder=5, label="Chance (AUROC=0.50)")

    ax.set_xlabel("False positive rate", **label_kwargs)
    ax.set_ylabel("True positive rate", **label_kwargs)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    if tick_labelsize is not None:
        ax.tick_params(labelsize=tick_labelsize)
    ax.grid(alpha=0.3)
    ax.set_title(title or "")
    if result.curves or chance:
        legend = ax.legend(**legend_kwargs)
        if bold_names:
            bold_labels = {
                name
                if not show_auroc
                else f"{name} (AUROC={result.curves[name].auroc:.{auroc_decimals}f})"
                for name in bold_names
            }
            for text in legend.get_texts():
                if text.get_text() in bold_labels:
                    text.set_fontweight("bold")
    return ax


def plot_auroc(
    result: EnsembleRocResult,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    label_fontsize: float | None = None,
    tick_labelsize: float | None = None,
) -> plt.Axes:
    """Plot a horizontal AUROC summary bar chart for each indicator."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    label_kwargs = {} if label_fontsize is None else {"fontsize": label_fontsize}

    names = sorted(result.curves, key=lambda name: result.curves[name].auroc, reverse=True)
    if not names:
        ax.set_xlabel("AUROC", **label_kwargs)
        if tick_labelsize is not None:
            ax.tick_params(labelsize=tick_labelsize)
        ax.set_title(title or "")
        return ax

    colors = _indicator_colors(result)
    positions = np.arange(len(names), dtype=float)
    values = [result.curves[name].auroc for name in names]
    bar_colors = [colors[name] for name in names]
    bar_height = 0.72

    ax.barh(positions, values, height=bar_height, color=bar_colors, edgecolor="none")
    ax.set_yticks(positions, names)
    ax.invert_yaxis()
    ax.margins(y=0.1)
    ax.set_xlabel("AUROC", **label_kwargs)
    if tick_labelsize is not None:
        ax.tick_params(labelsize=tick_labelsize)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(title or "")
    return ax


__all__ = [
    "BinaryRocCurveResult",
    "EnsembleRocResult",
    "ensemble_roc",
    "plot_auroc",
    "plot_roc",
]
