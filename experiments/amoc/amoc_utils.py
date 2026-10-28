"""Shared data loading and plotting for the coupled AMOC figures."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

PATTERN_OJL = re.compile(r"ANN(\d+)\.ojlE213SSP245\w+F40oQ40\.nc")

PATTERN_OIJ = re.compile(r"ANN(\d+)\.oijE213SSP245\w+F40oQ40\.nc")

DEFAULT_SCENARIOS: List[str] = ["a", "c", "g", "i"]

MEMBER_LABELS_10 = np.array(list("abcdefghij"))

C1_MEMBERS_DEFAULT = {"a", "i"}

def scenario_dir(data_dir: Path, scenario: str) -> Path:
    """Return the canonical annual-data directory for one SSP245 member label."""
    return data_dir / f"SSP245{scenario}" / "Annual"

def iter_annual_files(base: Path, grid: str = "ojl") -> Iterable[Path]:
    """Yield sorted annual NetCDF files for a given grid code (`ojl`, `oij`, ...)."""
    patt = f"ANN*.{grid}*.nc"
    for p in sorted(base.glob(patt)):
        yield p

def parse_annual_year(name: str, grid: str = "ojl") -> Optional[int]:
    """Parse year from an annual filename; return `None` when pattern does not match."""
    patt = PATTERN_OJL if grid == "ojl" else PATTERN_OIJ
    m = patt.match(name)
    return int(m.group(1)) if m else None

def preprocess_control(
    x: np.ndarray,
    *,
    mode: str = "none",
    rolling_window: int = 10,
) -> np.ndarray:
    """Control preprocessing for 1D or 2D arrays.

    1D shape: (time,)
    2D shape: (member, time)
    """
    arr = np.asarray(x, dtype=np.float32)

    if mode == "none":
        return arr

    if mode == "zscore":
        if arr.ndim == 1:
            return (arr - arr.mean()) / (arr.std() + 1e-8)
        if arr.ndim == 2:
            mu = np.mean(arr, axis=1, keepdims=True)
            sd = np.std(arr, axis=1, keepdims=True) + 1e-8
            return (arr - mu) / sd
        raise ValueError(f"Unsupported ndim for zscore: {arr.ndim}")

    if mode == "rolling_mean":
        w = int(max(1, rolling_window))
        if w <= 1:
            return arr
        kernel = np.ones(w, dtype=np.float32)

        def _roll_1d(v: np.ndarray) -> np.ndarray:
            """Centered moving average with edge normalization."""
            num = np.convolve(v, kernel, mode="same")
            den = np.convolve(np.ones_like(v, dtype=np.float32), kernel, mode="same")
            return num / np.maximum(den, 1e-8)

        if arr.ndim == 1:
            return _roll_1d(arr)
        if arr.ndim == 2:
            return np.vstack([_roll_1d(row) for row in arr]).astype(np.float32)
        raise ValueError(f"Unsupported ndim for rolling_mean: {arr.ndim}")

    raise ValueError(f"Unknown preprocess mode: {mode}")

def get_control_all(
    ds: xr.Dataset,
    control_var: str,
    *,
    preprocess: str = "none",
    rolling_window: int = 10,
) -> np.ndarray:
    """Get (member, time) control from dataset then preprocess."""
    arr = np.asarray(ds[control_var].values, dtype=np.float32)
    return preprocess_control(arr, mode=preprocess, rolling_window=rolling_window)

def safe_log(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Numerically safe `log` that clips tiny/non-positive values to `floor`."""
    return np.log(np.clip(np.asarray(x, dtype=float), floor, None))

def plot_grouped_on_axis(
    ax,
    years: Sequence[float],
    curves: np.ndarray,
    c1_mask: np.ndarray,
    *,
    ylabel: str = "",
    title: str = "",
    log_scale: bool = False,
    c1_label: str = "Tipping",
    c0_label: str = "Non-tipping",
    c1_color: str = "C3",
    c0_color: str = "C0",
    label_fontsize: int = 16,
    ylabel_fontsize: int = 12,
    tick_labelsize: int = 13,
    title_fontsize: int = 16,
) -> None:
    """Plot light dashed members + thick C1/C0 means on one axis."""
    y = np.asarray(curves, dtype=float)
    if log_scale:
        y = safe_log(y)

    for i in range(y.shape[0]):
        color = c1_color if bool(c1_mask[i]) else c0_color
        ax.plot(years, y[i], "--", color=color, lw=1.0, alpha=0.20)

    for is_c1, color, label in [
        (True, c1_color, c1_label),
        (False, c0_color, c0_label),
    ]:
        idx = np.where(c1_mask == is_c1)[0]
        if idx.size == 0:
            continue
        mean_curve = np.nanmean(y[idx], axis=0)
        ax.plot(years, mean_curve, "-", color=color, lw=3.0, label=label)

    ax.set_xlabel("Year", fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.tick_params(labelsize=tick_labelsize)
    ax.grid(True, alpha=0.3)

def resolve_field_and_coords(ds: xr.Dataset, preferred: str = "sf_Atl") -> Tuple[str, str, str]:
    """Resolve field and its latitude/depth dimension names from a dataset."""
    candidates = [preferred]
    if preferred == "sfl_Atl":
        candidates.append("sf_Atl")
    elif preferred == "sf_Atl":
        candidates.append("sfl_Atl")

    field = next((v for v in candidates if v in ds.data_vars), None)
    if field is None:
        raise KeyError(f"None of {candidates} found in dataset vars: {list(ds.data_vars)}")

    da = ds[field]
    lat_dim = "lato2" if "lato2" in da.dims else ("lato" if "lato" in da.dims else None)
    depth_dim = "zoce" if "zoce" in da.dims else ("zoc" if "zoc" in da.dims else None)
    if lat_dim is None or depth_dim is None:
        raise KeyError(f"Cannot infer lat/depth dims for {field} dims={da.dims}")
    return field, lat_dim, depth_dim

def load_lat_slice_time_depth(
    data_dir: Path,
    scenario: str,
    lat_target: float,
    *,
    preferred_field: str = "sf_Atl",
    year_min: int = 2015,
    year_max: int = 2400,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Load a nearest-latitude depth profile through time.

    Returns
    -------
    years : ndarray
        Available years in sorted order.
    depth_ref : ndarray
        Depth coordinate shared across years.
    data : ndarray
        Profile array with shape `(depth, time)`.
    lat_actual : float
        Actual grid latitude selected by nearest-neighbor lookup.
    field_used : str
        Variable name selected from `preferred_field` and fallbacks.
    """
    base = scenario_dir(data_dir, scenario)
    year_prof: List[Tuple[int, np.ndarray]] = []
    lat_actual: Optional[float] = None
    depth_ref: Optional[np.ndarray] = None
    field_used: Optional[str] = None

    for p in iter_annual_files(base, "ojl"):
        year = parse_annual_year(p.name, "ojl")
        if year is None or year < year_min or year > year_max:
            continue
        with xr.open_dataset(p) as ds:
            field, lat_dim, depth_dim = resolve_field_and_coords(ds, preferred_field)
            da = ds[field]
            prof_da = da.sel({lat_dim: lat_target}, method="nearest")
            if lat_actual is None:
                lat_actual = float(prof_da[lat_dim].values)
                field_used = field
            profile = prof_da.values.astype(float)
            depth = da[depth_dim].values.astype(float)

        if depth_ref is None:
            depth_ref = depth
        year_prof.append((year, profile))

    year_prof.sort(key=lambda x: x[0])
    if not year_prof or depth_ref is None or lat_actual is None or field_used is None:
        raise FileNotFoundError(
            f"No usable annual profile data in {base} for latitude {lat_target}. "
            "Download the NASA inputs as described in README.md."
        )

    years = np.asarray([y for y, _ in year_prof], dtype=int)
    data = np.stack([v for _, v in year_prof], axis=1)
    return years, depth_ref, data, lat_actual, field_used
