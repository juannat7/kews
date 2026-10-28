"""Helpers for the Science Advances empirical single-series notebook workflow."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from zipfile import ZipFile

import numpy as np
import requests


ARCHIVE_URL = "https://zenodo.org/records/7379358/files/DEV_Bifurcation-main.zip"
ARCHIVE_NAME = "DEV_Bifurcation-main.zip"
ARCHIVE_ROOT_NAME = "DEV_Bifurcation-main"
LOCAL_DATASET_DIR = "zenodo_7379358"
EMPIRICAL_SUBDIR = "figure_2_empirical_examples"


SYSTEM_SPECS: dict[str, dict[str, Any]] = {
    "microcosm": {
        "filename": "microcosm.txt",
        "display_name": "Microcosm",
        "time_label": "Time (day)",
        "state_label": r"Light attenuation (m$^{-1}$)",
        "time_column": 0,
        "state_column": 1,
        "trend_start": 0.0,
        "transition_time": 27.0,
        "window_size": 200,
        "step_size": 5,
        "analysis_segments": (
            (0.57092, 1.4539),
            (4.5496, 5.4326),
            (10.642, 11.525),
            (14.454, 15.337),
            (19.499, 20.382),
            (24.502, 25.385),
        ),
        "analysis_segment_labels": ("P1", "P2", "P3", "P4", "P5", "P6"),
    },
    "voice": {
        "filename": "voice.txt",
        "display_name": "Voice",
        "time_label": "Time (s)",
        "state_label": "Subglottal pressure (kPa)",
        "time_column": 0,
        "state_column": 1,
        "trend_start": 0.3,
        "transition_time": 0.49,
        "window_size": 200,
        "step_size": 50,
    },
    "mitochondria": {
        "filename": "mitochondria.txt",
        "display_name": "Mitochondria",
        "time_label": "Time (s)",
        "state_label": "Relative fluorescence ratio",
        "time_column": 0,
        "state_column": 1,
        "trend_start": 3.2,
        "transition_time": 4.7,
        "window_size": 125,
        "step_size": 1,
    },
    "climate": {
        "filename": "climate.txt",
        "display_name": "Climate",
        "time_label": r"Time ($10^6$ years ago)",
        "state_label": r"CaCO$_3$ (%)",
        "time_column": 1,
        "state_column": 0,
        "trend_start": -38.0,
        "transition_time": -34.0,
        "window_size": 150,
        "step_size": 5,
    },
    "blackout": {
        "filename": "blackout.txt",
        "display_name": "Blackout",
        "time_label": "Time (s)",
        "state_label": "Frequency (Hz)",
        "time_column": 0,
        "state_column": 1,
        "trend_start": 200.0,
        "transition_time": 550.0,
        "window_size": 1000,
        "step_size": 50,
    },
}


def supported_system_names() -> tuple[str, ...]:
    """Return the supported empirical system names."""
    return tuple(SYSTEM_SPECS)


def _dataset_root(data_dir: Path) -> Path:
    """Resolve the local archive/extract root inside the experiment data directory."""
    return data_dir.expanduser().resolve() / LOCAL_DATASET_DIR


def _archive_path(data_dir: Path) -> Path:
    """Return the local path to the downloaded archive file."""
    return _dataset_root(data_dir) / ARCHIVE_NAME


def _extracted_root(data_dir: Path) -> Path:
    """Return the local extracted archive root."""
    return _dataset_root(data_dir) / ARCHIVE_ROOT_NAME


def _expected_member_paths(data_dir: Path) -> list[Path]:
    """List the extracted empirical files that must exist after unpacking."""
    root = _extracted_root(data_dir) / EMPIRICAL_SUBDIR
    return [root / spec["filename"] for spec in SYSTEM_SPECS.values()]


def download_empirical_archive(data_dir: Path) -> Path:
    """Download and extract the Science Advances empirical archive if needed."""
    data_dir = Path(data_dir)
    dataset_root = _dataset_root(data_dir)
    archive_path = _archive_path(data_dir)
    extracted_root = _extracted_root(data_dir)
    expected_members = _expected_member_paths(data_dir)

    if all(path.exists() for path in expected_members):
        return extracted_root

    dataset_root.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        with NamedTemporaryFile(dir=dataset_root, suffix=".part", delete=False) as fh:
            partial = Path(fh.name)
            try:
                with requests.get(ARCHIVE_URL, stream=True, timeout=60) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
                fh.close()
                # Validate before caching so an interrupted response can be retried.
                with ZipFile(partial) as archive:
                    if archive.testzip() is not None:
                        raise ValueError("Corrupt empirical archive")
                partial.replace(archive_path)
            finally:
                partial.unlink(missing_ok=True)

    with ZipFile(archive_path) as archive:
        # Extract only the paper inputs and their upstream terms.
        members = [str(path.relative_to(dataset_root)) for path in expected_members]
        members += [f"{ARCHIVE_ROOT_NAME}/{name}" for name in ("README.txt", "README.md", "LICENSE")]
        for member in members:
            archive.extract(member, dataset_root)

    missing = [str(path) for path in expected_members if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing extracted empirical files: {missing}")
    return extracted_root


def load_empirical_system(system_name: str, data_dir: Path) -> dict[str, Any]:
    """Load one empirical Science Advances system into standardized arrays."""
    if system_name not in SYSTEM_SPECS:
        supported = ", ".join(supported_system_names())
        raise ValueError(f"Unknown system `{system_name}`. Supported: {supported}")

    extracted_root = download_empirical_archive(data_dir)
    spec = SYSTEM_SPECS[system_name]
    source_path = extracted_root / EMPIRICAL_SUBDIR / spec["filename"]

    raw = np.genfromtxt(source_path, dtype=float, missing_values="NA", filling_values=np.nan)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"Expected a two-column text file at {source_path}")

    time = np.asarray(raw[:, spec["time_column"]], dtype=float)
    state = np.asarray(raw[:, spec["state_column"]], dtype=float)
    finite_mask = np.isfinite(time) & np.isfinite(state)
    time = time[finite_mask]
    state = state[finite_mask]
    if time.size == 0 or state.size == 0:
        raise ValueError(f"No finite rows found in {source_path}")

    return {
        "system_name": system_name,
        "display_name": spec["display_name"],
        "time": time,
        "state": state,
        "time_label": spec["time_label"],
        "state_label": spec["state_label"],
        "trend_start": spec["trend_start"],
        "transition_time": spec["transition_time"],
        "window_size": spec["window_size"],
        "step_size": spec["step_size"],
        "analysis_segments": tuple(spec.get("analysis_segments", ())),
        "analysis_segment_labels": tuple(spec.get("analysis_segment_labels", ())),
        "source_file": spec["filename"],
        "source_path": source_path,
        "series_length": int(state.shape[0]),
    }


__all__ = [
    "ARCHIVE_URL",
    "SYSTEM_SPECS",
    "download_empirical_archive",
    "load_empirical_system",
    "supported_system_names",
]
