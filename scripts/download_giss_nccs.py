#!/usr/bin/env python3
"""
Download NASA GISS ModelE E213SSP245 ocean/atmosphere NetCDF data from NCCS Data Portal.

Data source: https://portal.nccs.nasa.gov/datashare/modelE_ocean/E213SSP245/
Reference:   https://journals.ametsoc.org/view/journals/clim/36/18/JCLI-D-22-0536.1.xml

Scenarios (SSP2-4.5 ensemble, JCLI-D-22-0536.1):
  - a, i: tipped AMOC simulations
  - c, g: non-tipped AMOC simulations

Variable types:
  - Oceanic: oij (2D lat-lon), oijl (3D lat-lon-depth), ojl (2D lat-depth)
  - Toto:    atmospheric variables; aij (2D atmospheric lat-lon)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests
from tqdm import tqdm

# Default run ID suffix used in filenames
RUN_SUFFIX = "F40oQ40"
BASE_URL_DEFAULT = "https://portal.nccs.nasa.gov/datashare/modelE_ocean/E213SSP245/"
MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
DEFAULT_YEAR_START = 2015
DEFAULT_YEAR_END = 2400


def build_filenames(
    scenario: str,
    frequency: str,
    variable_type: str,
    grid_types: tuple[str, ...],
    years: range,
    months: tuple[str, ...] | None,
) -> list[tuple[str, str]]:
    """
    Build (filename, subpath) pairs. subpath is '' for oceanic in Annual/Monthly,
    or 'Toto/' for toto. Returns list of (filename, subpath).
    """
    run_tag = f"E213SSP245{scenario}{RUN_SUFFIX}"
    out: list[tuple[str, str]] = []

    if variable_type == "toto":
        grid_list = ["aij"]
        # Monthly aij is under Monthly/; Annual aij is under Annual/Toto/
        subpath = "Toto/" if frequency == "annual" else ""
    elif variable_type == "oceanic":
        grid_list = list(grid_types)
        subpath = ""
    else:  # both
        # oceanic first (subpath '')
        for g in grid_types:
            if frequency == "annual":
                for y in years:
                    out.append((f"ANN{y}.{g}{run_tag}.nc", ""))
            else:
                for y in years:
                    for m in (months or MONTHS):
                        out.append((f"{m}{y}.{g}{run_tag}.nc", ""))
        # then toto: Annual in Toto/, Monthly in Monthly/
        subpath_toto = "Toto/" if frequency == "annual" else ""
        if frequency == "annual":
            for y in years:
                out.append((f"ANN{y}.aij{run_tag}.nc", subpath_toto))
        else:
            for y in years:
                for m in (months or MONTHS):
                    out.append((f"{m}{y}.aij{run_tag}.nc", subpath_toto))

    if variable_type != "both":
        if frequency == "annual":
            for g in grid_list:
                for y in years:
                    out.append((f"ANN{y}.{g}{run_tag}.nc", subpath))
        else:
            for g in grid_list:
                for y in years:
                    for m in (months or MONTHS):
                        out.append((f"{m}{y}.{g}{run_tag}.nc", subpath))

    return out


def build_url_and_local(
    base_url: str,
    scenario: str,
    frequency: str,
    filename: str,
    subpath: str,
    out_dir: Path,
) -> tuple[str, Path]:
    """Build full URL and local path for a file."""
    scenario_dir = f"SSP245{scenario}/"
    freq_dir = "Annual/" if frequency == "annual" else "Monthly/"
    url_path = scenario_dir + freq_dir + subpath + filename
    url = urljoin(base_url, url_path)
    parts = [out_dir, f"SSP245{scenario}", "Annual" if frequency == "annual" else "Monthly"]
    if subpath.strip("/"):
        parts.append("Toto")
    local = Path(*parts) / filename
    return url, local


def download_one(
    url: str,
    local_path: Path,
    skip_existing: bool,
    dry_run: bool,
    session: requests.Session,
    retries: int = 3,
) -> str | None:
    """Download one file. Returns None on success, error message on failure."""
    if skip_existing and local_path.exists():
        return None  # skipped
    if dry_run:
        return None
    local_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            r = session.get(url, stream=True, timeout=60)
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(
                dir=local_path.parent,
                prefix=".tmp_",
                suffix=local_path.suffix,
                delete=False,
            ) as f:
                tmp = Path(f.name)
            try:
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                tmp.rename(local_path)
                return None
            except Exception as e:
                tmp.unlink(missing_ok=True)
                raise e
        except requests.RequestException as e:
            if attempt == retries - 1:
                return f"{url}: {e}"
            continue
    return f"{url}: failed after {retries} attempts"


def main() -> int:
    """CLI entry point for bulk NCCS downloads.

    Returns
    -------
    int
        Exit code (`0` on success, `1` when one or more downloads fail).
    """
    parser = argparse.ArgumentParser(
        description="Download GISS ModelE E213SSP245 data from NCCS portal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL_DEFAULT,
        help="Base URL of the E213SSP245 directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Local root directory; scenario/frequency structure is mirrored under it",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["a", "c", "g", "i"],
        default=["a", "c", "g", "i"],
        help="Scenario(s): a,i = tipped AMOC; c,g = non-tipped",
    )
    parser.add_argument(
        "--frequency",
        choices=["annual", "monthly"],
        default="annual",
        help="Annual or monthly files",
    )
    parser.add_argument(
        "--variable-type",
        choices=["oceanic", "toto", "both"],
        default="oceanic",
        help="oceanic (oij/oijl/ojl), toto (atmospheric aij), or both",
    )
    parser.add_argument(
        "--grid-types",
        nargs="+",
        choices=["oij", "oijl", "ojl"],
        default=["oij", "oijl", "ojl"],
        help="Oceanic grid types (ignored when variable-type is toto)",
    )
    parser.add_argument(
        "--years",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help=f"Inclusive year range (default: {DEFAULT_YEAR_START} {DEFAULT_YEAR_END})",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        choices=list(MONTHS),
        default=None,
        help="Monthly only: month abbreviations (default: all 12)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip files that already exist locally",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-download even if file exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print URLs/paths, do not download",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of concurrent downloads (1 = sequential)",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/") + "/"
    year_start, year_end = (
        (args.years[0], args.years[1])
        if args.years
        else (DEFAULT_YEAR_START, DEFAULT_YEAR_END)
    )
    years = range(year_start, year_end + 1)
    months = tuple(args.months) if args.months else None

    # Build a deterministic task list first so dry-run and actual download share logic.
    tasks: list[tuple[str, Path]] = []
    for scenario in args.scenarios:
        files = build_filenames(
            scenario,
            args.frequency,
            args.variable_type,
            tuple(args.grid_types),
            years,
            months,
        )
        for filename, subpath in files:
            url, local = build_url_and_local(
                base_url, scenario, args.frequency, filename, subpath, args.out_dir
            )
            tasks.append((url, local))

    if args.dry_run:
        for url, local in tasks:
            print(url, "->", local)
        return 0

    session = requests.Session()
    session.headers.setdefault("User-Agent", "kews-download/0.1.0")
    errors: list[str] = []

    def do_one(item: tuple[str, Path]) -> str | None:
        """Execute a single task tuple `(url, local_path)`."""
        return download_one(
            item[0],
            item[1],
            args.skip_existing,
            args.dry_run,
            session,
        )

    if args.workers <= 1:
        for item in tqdm(tasks, desc="Downloading", unit="file"):
            err = do_one(item)
            if err:
                errors.append(err)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(do_one, item): item for item in tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="file"):
                err = fut.result()
                if err:
                    errors.append(err)

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
