"""
groundwater_ml_clusters.py
==========================
Generates ML-ready cluster Excel files for groundwater-level forecasting.

Pipeline
--------
1.  Load USGS groundwater-level (NWIS) records and site metadata
    (Well Depth, National Aquifer Code).
2.  Align all water-level records to the NAVD 88 vertical datum.
3.  For every *Anchor* well, find up to 6 neighbouring USGS wells within 50 km.
4.  Pull monthly pumping volumes from the U.S. Water Withdrawal Database (USWWD)
    for all pumping wells within 150 km of the anchor.
5.  Extract monthly GridMET climate variables (precipitation, Tmin, Tmax, ET,
    humidity) at each well's coordinates using a lazy-loading approach.
6.  Apply temporal QC:
      - Only accept 60-month (5-year) continuous windows.
      - Require ≥ 90 % (54/60) of raw water-level measurements; fill minor
        gaps with linear interpolation.
      - Encode the Month feature as sin/cos (cyclic).
7.  Save one Excel workbook per anchor cluster.

Dependencies (install via pip)
-------------------------------
    pip install pandas numpy scipy requests openpyxl dataretrieval
    # Optional for faster raster reads:
    pip install xarray netCDF4 rioxarray
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Optional heavy dependencies – imported lazily where needed
# ---------------------------------------------------------------------------
try:
    import dataretrieval.nwis as nwis  # type: ignore
    _HAS_NWIS = True
except ImportError:  # pragma: no cover
    _HAS_NWIS = False
    warnings.warn("dataretrieval not installed – NWIS fetching disabled.")

try:
    import xarray as xr  # type: ignore
    _HAS_XR = True
except ImportError:  # pragma: no cover
    _HAS_XR = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Radius (km) within which pumping wells are associated with an anchor well.
PUMPING_RADIUS_KM: float = 150.0

#: Radius (km) within which USGS neighbour wells are searched.
NEIGHBOR_RADIUS_KM: float = 50.0

#: Maximum number of neighbour wells per anchor cluster.
MAX_NEIGHBORS: int = 6

#: Required sequence length in months.
SEQ_LEN_MONTHS: int = 60

#: Minimum fraction of valid raw water-level measurements in a window.
MIN_VALID_FRACTION: float = 0.90  # 54 / 60

#: Earth radius used for Haversine calculations (km).
EARTH_RADIUS_KM: float = 6371.0

#: GridMET base URL template.
#:  Variables: pr, tmmn, tmmx, pet, rmax  (daily, ~4 km resolution)
GRIDMET_BASE_URL: str = (
    "https://www.northwestknowledge.net/metdata/data/{var}_{year}.nc"
)

#: USWWD (U.S. Water Withdrawal Database) – public CSV endpoint.
#:  The table is updated periodically; adjust the URL as needed.
USWWD_URL: str = (
    "https://waterdata.usgs.gov/nwis/water_use"
    "?format=rdb&period=&wu_area=County&wu_year=ALL"
    "&wu_county=ALL&wu_category=GW&wu_county_cd=ALL"
)

#: NAVD88 datum code used in the NWIS service.
NAVD88_CODE: str = "NAVD88"

#: Output directory for Excel cluster files.
OUTPUT_DIR: Path = Path("clusters_output")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometres between two points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def haversine_matrix(
    lats: np.ndarray, lons: np.ndarray
) -> np.ndarray:
    """Return an (N, N) symmetric distance matrix (km) for arrays of coords."""
    n = len(lats)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(lats[i], lons[i], lats[j], lons[j])
            dist[i, j] = dist[j, i] = d
    return dist


def cyclic_encode_month(months: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Return (sin, cos) cyclic encodings for a Series of month integers (1–12)."""
    angle = 2 * math.pi * months / 12
    return np.sin(angle), np.cos(angle)


def interpolate_gaps(series: pd.Series, limit: int = 6) -> pd.Series:
    """
    Apply linear interpolation to fill gaps up to *limit* consecutive NaN
    months.  Does NOT extrapolate beyond the series edges.
    """
    return series.interpolate(method="linear", limit=limit, limit_direction="forward")


# ---------------------------------------------------------------------------
# 1.  USGS NWIS – site metadata + groundwater levels
# ---------------------------------------------------------------------------


def fetch_usgs_sites(
    state_cd: Optional[str] = None,
    site_list: Optional[List[str]] = None,
    bounding_box: Optional[Tuple[float, float, float, float]] = None,
) -> pd.DataFrame:
    """
    Retrieve USGS groundwater monitoring site metadata.

    Parameters
    ----------
    state_cd:
        Two-letter state code, e.g. ``"CA"``.
    site_list:
        Explicit list of USGS site numbers.
    bounding_box:
        ``(west, south, east, north)`` decimal-degree bounding box.

    Returns
    -------
    DataFrame with columns: site_no, station_nm, dec_lat_va, dec_long_va,
    well_depth_va, nat_aqfr_cd, alt_datum_cd, alt_va.
    """
    if not _HAS_NWIS:
        raise RuntimeError("dataretrieval is required to fetch USGS data.")

    kwargs: Dict = {"siteType": "GW", "outputDataTypeCd": "gw"}
    if state_cd:
        kwargs["stateCd"] = state_cd
    if site_list:
        kwargs["sites"] = site_list
    if bounding_box:
        kwargs["bBox"] = ",".join(str(v) for v in bounding_box)

    log.info("Fetching USGS site metadata …")
    sites_df, _ = nwis.get_info(**kwargs)
    cols = [
        "site_no", "station_nm",
        "dec_lat_va", "dec_long_va",
        "well_depth_va", "nat_aqfr_cd",
        "alt_datum_cd", "alt_va",
    ]
    available = [c for c in cols if c in sites_df.columns]
    sites_df = sites_df[available].copy()
    sites_df["dec_lat_va"] = pd.to_numeric(sites_df["dec_lat_va"], errors="coerce")
    sites_df["dec_long_va"] = pd.to_numeric(sites_df["dec_long_va"], errors="coerce")
    sites_df["well_depth_va"] = pd.to_numeric(
        sites_df.get("well_depth_va", pd.Series(dtype=float)), errors="coerce"
    )
    sites_df = sites_df.dropna(subset=["dec_lat_va", "dec_long_va"])
    log.info("  → %d sites retrieved.", len(sites_df))
    return sites_df.reset_index(drop=True)


def fetch_groundwater_levels(
    site_no: str,
    start_date: str = "2000-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Fetch daily groundwater-level records for a single USGS site and
    resample to a monthly median.

    Returns
    -------
    DataFrame indexed by ``YearMonth`` (Period[M]) with columns:
    ``water_level``, ``datum``.
    """
    if not _HAS_NWIS:
        raise RuntimeError("dataretrieval is required to fetch USGS data.")

    try:
        df, _ = nwis.get_gwlevels(
            sites=site_no,
            startDT=start_date,
            endDT=end_date,
            parameterCd="72019",  # depth to water level below land surface
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("  Could not fetch levels for %s: %s", site_no, exc)
        return pd.DataFrame(columns=["water_level", "datum"])

    if df.empty:
        return pd.DataFrame(columns=["water_level", "datum"])

    df = df.reset_index()
    date_col = next(
        (c for c in df.columns if "dt" in c.lower() or "date" in c.lower()),
        None,
    )
    if date_col is None:
        return pd.DataFrame(columns=["water_level", "datum"])

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()

    val_col = next(
        (c for c in df.columns if "72019" in c or "lev_va" in c.lower()),
        df.columns[0],
    )
    df["water_level"] = pd.to_numeric(df[val_col], errors="coerce")

    datum_col = next(
        (c for c in df.columns if "datum" in c.lower() or "cd" in c.lower()),
        None,
    )
    df["datum"] = df[datum_col] if datum_col else "UNKNOWN"

    monthly = (
        df[["water_level", "datum"]]
        .resample("MS")
        .agg({"water_level": "median", "datum": "first"})
    )
    monthly.index = monthly.index.to_period("M")
    return monthly


# ---------------------------------------------------------------------------
# 2.  Datum alignment – convert to NAVD88
# ---------------------------------------------------------------------------

# Approximate NGVD29 → NAVD88 shift (ft).  A rigorous conversion requires
# the VERTCON service; this constant is a national average placeholder.
_NGVD29_TO_NAVD88_FT: float = -3.58  # To convert NGVD29 → NAVD88, add -3.58 ft (NAVD88 = NGVD29 - 3.58 ft, national average)


def align_to_navd88(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift water-level values so that they are all referenced to NAVD88.

    Applies a constant NGVD29 → NAVD88 offset for legacy records.
    Records already in NAVD88 pass through unchanged.  Unknown datums
    are left as-is with a warning.
    """
    df = df.copy()
    datums = df["datum"].unique() if "datum" in df.columns else []
    for datum in datums:
        mask = df["datum"] == datum
        if str(datum).upper() in (NAVD88_CODE, "NAVD 88"):
            pass  # already correct
        elif str(datum).upper() in ("NGVD29", "NGVD 29"):
            df.loc[mask, "water_level"] = (
                df.loc[mask, "water_level"] + _NGVD29_TO_NAVD88_FT
            )
        else:
            if datum and str(datum).upper() not in ("NAN", "UNKNOWN", ""):
                log.warning(
                    "Unknown datum '%s' – water levels passed through unchanged.",
                    datum,
                )
    df["datum"] = NAVD88_CODE
    return df


# ---------------------------------------------------------------------------
# 3.  GridMET lazy-loading climate data
# ---------------------------------------------------------------------------

# Map friendly names to GridMET NetCDF variable names
_GRIDMET_VARS: Dict[str, str] = {
    "precip_mm": "precipitation_amount",
    "tmin_c": "air_temperature",    # tmmn file
    "tmax_c": "air_temperature",    # tmmx file
    "et_mm": "potential_evapotranspiration",
    "rh_max_pct": "relative_humidity",
}

# Map friendly names to GridMET file prefixes
_GRIDMET_FILE_PREFIX: Dict[str, str] = {
    "precip_mm": "pr",
    "tmin_c": "tmmn",
    "tmax_c": "tmmx",
    "et_mm": "pet",
    "rh_max_pct": "rmax",
}

# Unit conversion: GridMET native → standard
#   pr  : mm/day  → monthly sum (mm)
#   tmmn: K       → °C
#   tmmx: K       → °C
#   pet : mm/day  → monthly sum (mm)
#   rmax: %       → % (no change, monthly mean)
_GRIDMET_UNIT_CONV = {
    "precip_mm": lambda x: x,        # already aggregated as sum
    "tmin_c": lambda x: x - 273.15,
    "tmax_c": lambda x: x - 273.15,
    "et_mm": lambda x: x,            # already aggregated as sum
    "rh_max_pct": lambda x: x,       # aggregated as mean
}

# Aggregation function per variable
_GRIDMET_AGG: Dict[str, str] = {
    "precip_mm": "sum",
    "tmin_c": "mean",
    "tmax_c": "mean",
    "et_mm": "sum",
    "rh_max_pct": "mean",
}


class GridMETLoader:
    """
    Lazy-loading wrapper for GridMET annual NetCDF files.

    Each file is only downloaded / opened once and cached in memory.
    Point extraction uses nearest-neighbour lookup on the 4-km grid.
    """

    def __init__(self, cache_dir: str | Path = "gridmet_cache") -> None:
        if not _HAS_XR:
            raise ImportError("xarray is required for GridMETLoader.")
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ds_cache: Dict[Tuple[str, int], "xr.Dataset"] = {}

    # ------------------------------------------------------------------
    def _local_path(self, var_prefix: str, year: int) -> Path:
        return self._cache_dir / f"{var_prefix}_{year}.nc"

    # ------------------------------------------------------------------
    def _download_if_needed(self, var_prefix: str, year: int) -> Path:
        fpath = self._local_path(var_prefix, year)
        if fpath.exists():
            return fpath
        url = GRIDMET_BASE_URL.format(var=var_prefix, year=year)
        log.info("  Downloading GridMET %s %d …", var_prefix, year)
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(fpath, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        return fpath

    # ------------------------------------------------------------------
    def _open_dataset(self, var_prefix: str, year: int) -> "xr.Dataset":
        key = (var_prefix, year)
        if key not in self._ds_cache:
            fpath = self._download_if_needed(var_prefix, year)
            self._ds_cache[key] = xr.open_dataset(fpath, engine="netcdf4")
        return self._ds_cache[key]

    # ------------------------------------------------------------------
    def extract_monthly(
        self,
        lat: float,
        lon: float,
        var_name: str,
        start_year: int,
        end_year: int,
    ) -> pd.Series:
        """
        Extract a monthly time-series of *var_name* for a given (lat, lon).

        Parameters
        ----------
        var_name:
            One of ``precip_mm``, ``tmin_c``, ``tmax_c``, ``et_mm``,
            ``rh_max_pct``.

        Returns
        -------
        Series indexed by Period[M].
        """
        prefix = _GRIDMET_FILE_PREFIX[var_name]
        xr_var = _GRIDMET_VARS[var_name]
        conv = _GRIDMET_UNIT_CONV[var_name]
        agg = _GRIDMET_AGG[var_name]

        monthly_records: List[Tuple] = []
        for year in range(start_year, end_year + 1):
            try:
                ds = self._open_dataset(prefix, year)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "  GridMET %s %d unavailable: %s", var_name, year, exc
                )
                continue

            # Nearest-neighbour point selection
            lat_vals = ds["lat"].values if "lat" in ds.coords else ds["y"].values
            lon_vals = ds["lon"].values if "lon" in ds.coords else ds["x"].values
            i_lat = int(np.argmin(np.abs(lat_vals - lat)))
            i_lon = int(np.argmin(np.abs(lon_vals - lon)))

            da = ds[xr_var]
            # Select nearest grid cell (handle various coordinate name styles)
            try:
                point_da = da.isel(lat=i_lat, lon=i_lon)
            except ValueError:
                try:
                    point_da = da.isel(y=i_lat, x=i_lon)
                except ValueError:
                    dims = {d: 0 for d in da.dims if d not in ("time",)}
                    point_da = da.isel(**dims)

            daily_series = (
                point_da.to_series()
                .dropna()
                .rename(var_name)
            )
            daily_series.index = pd.to_datetime(daily_series.index)

            # Resample to monthly
            if agg == "sum":
                monthly = daily_series.resample("MS").sum()
            else:
                monthly = daily_series.resample("MS").mean()

            # Unit conversion applied after aggregation
            monthly = monthly.apply(conv)
            for ts, val in monthly.items():
                monthly_records.append((ts.to_period("M"), val))

        if not monthly_records:
            return pd.Series(dtype=float, name=var_name)

        s = pd.Series(
            {p: v for p, v in monthly_records},
            name=var_name,
            dtype=float,
        )
        s.index.name = "YearMonth"
        return s


# ---------------------------------------------------------------------------
# 4.  USWWD pumping data
# ---------------------------------------------------------------------------


def fetch_uswwd_pumping(
    anchor_lat: float,
    anchor_lon: float,
    radius_km: float = PUMPING_RADIUS_KM,
) -> pd.DataFrame:
    """
    Retrieve monthly groundwater withdrawal volumes from the USWWD for all
    pumping wells within *radius_km* of the anchor well.

    The USGS Water Use data is available at county level; this function
    downloads the national table once, geocodes counties to centroids, and
    filters by distance.

    Returns
    -------
    DataFrame with columns: year, month, county_fips, pumping_mgal_per_day.
    Aggregated over all counties within the radius.
    """
    try:
        log.info("  Downloading USWWD pumping data …")
        resp = requests.get(USWWD_URL, timeout=60)
        resp.raise_for_status()
        lines = [ln for ln in resp.text.splitlines() if not ln.startswith("#")]
        from io import StringIO
        df = pd.read_csv(
            StringIO("\n".join(lines)),
            sep="\t",
            dtype=str,
            on_bad_lines="skip",
        )
        # Drop the RDB header line (contains data types like '5s', '10d', …)
        df = df[~df.iloc[:, 0].str.contains(r"^\d+[sd]$", na=False)]
    except Exception as exc:  # noqa: BLE001
        log.warning("  USWWD fetch failed (%s). Returning empty pumping table.", exc)
        return _empty_pumping_df()

    # ------------------------------------------------------------------
    # Normalise column names (case-insensitive match)
    # ------------------------------------------------------------------
    df.columns = df.columns.str.lower().str.strip()

    year_col = _find_col(df, ["year", "year_nu"])
    state_col = _find_col(df, ["state_cd", "statcd", "state"])
    county_col = _find_col(df, ["county_cd", "countycd", "county"])
    gw_col = _find_col(
        df,
        [
            "groundwater_withdrawals_mgal_d",
            "gw_with_mgal_d",
            "gw_withdrawal",
            "fresh_groundwater_withdrawals_mgal_d",
        ],
    )
    lat_col = _find_col(df, ["lat_va", "dec_lat_va", "latitude"])
    lon_col = _find_col(df, ["long_va", "dec_long_va", "longitude"])

    if not all([year_col, gw_col]):
        log.warning("  USWWD columns not found; returning empty pumping table.")
        return _empty_pumping_df()

    df["_year"] = pd.to_numeric(df[year_col], errors="coerce")
    df["_gw_mgd"] = pd.to_numeric(df[gw_col], errors="coerce")

    if lat_col and lon_col:
        df["_lat"] = pd.to_numeric(df[lat_col], errors="coerce")
        df["_lon"] = pd.to_numeric(df[lon_col], errors="coerce")
        df = df.dropna(subset=["_lat", "_lon"])
        df["_dist_km"] = df.apply(
            lambda r: haversine_km(anchor_lat, anchor_lon, r["_lat"], r["_lon"]),
            axis=1,
        )
        df = df[df["_dist_km"] <= radius_km]
    else:
        log.warning("  No lat/lon columns in USWWD data – using all records.")

    df = df.dropna(subset=["_year", "_gw_mgd"])
    # Annual data → expand to monthly (uniform distribution)
    records = []
    for _, row in df.iterrows():
        for month in range(1, 13):
            records.append(
                {
                    "year": int(row["_year"]),
                    "month": month,
                    "pumping_mgal_per_day": row["_gw_mgd"],
                }
            )
    if not records:
        return _empty_pumping_df()

    pumping = pd.DataFrame(records)
    pumping["YearMonth"] = pd.PeriodIndex(
        pd.to_datetime(
            {"year": pumping["year"], "month": pumping["month"], "day": 1}
        )
    ).asfreq("M")
    agg_pumping = (
        pumping.groupby("YearMonth")["pumping_mgal_per_day"].sum().rename("pumping_total_mgd")
    )
    return agg_pumping.reset_index()


def _empty_pumping_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["YearMonth", "pumping_total_mgd"])


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name from *candidates* found in *df*."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# 5.  Cluster building
# ---------------------------------------------------------------------------


def build_anchor_neighbor_clusters(
    sites_df: pd.DataFrame,
    neighbor_radius_km: float = NEIGHBOR_RADIUS_KM,
    max_neighbors: int = MAX_NEIGHBORS,
) -> List[Dict]:
    """
    For every well in *sites_df*, designate it as an Anchor and find up to
    *max_neighbors* USGS wells within *neighbor_radius_km*.

    Returns
    -------
    List of dicts, each with:
      ``anchor``    : row from sites_df
      ``neighbors`` : list of rows from sites_df (up to max_neighbors)
    """
    lats = sites_df["dec_lat_va"].values.astype(float)
    lons = sites_df["dec_long_va"].values.astype(float)

    # Convert to radians for cKDTree on a sphere approximation
    # (cKDTree uses Euclidean distance on lat/lon – fine for ≤ 50 km)
    coords = np.column_stack([lats, lons])
    tree = cKDTree(coords)

    # Approximate degree → km for query radius (1° ≈ 111 km at equator).
    # The 1.5× buffer accounts for latitude-dependent compression of longitude
    # degrees (cos(lat) factor) and ensures no candidates are missed before the
    # exact Haversine filter below.
    approx_deg = neighbor_radius_km / 111.0 * 1.5

    clusters: List[Dict] = []
    for idx in range(len(sites_df)):
        anchor_row = sites_df.iloc[idx]
        anchor_lat = float(anchor_row["dec_lat_va"])
        anchor_lon = float(anchor_row["dec_long_va"])

        candidate_idxs = tree.query_ball_point(
            [anchor_lat, anchor_lon], r=approx_deg
        )
        candidate_idxs = [i for i in candidate_idxs if i != idx]

        # Re-filter by exact Haversine distance
        neighbours: List[Dict] = []
        for ci in candidate_idxs:
            nb_row = sites_df.iloc[ci]
            dist = haversine_km(
                anchor_lat, anchor_lon,
                float(nb_row["dec_lat_va"]),
                float(nb_row["dec_long_va"]),
            )
            if dist <= neighbor_radius_km:
                neighbours.append({"row": nb_row, "dist_km": dist})

        # Sort by proximity, keep only the closest max_neighbors
        neighbours.sort(key=lambda x: x["dist_km"])
        neighbours = neighbours[:max_neighbors]

        clusters.append(
            {
                "anchor": anchor_row,
                "neighbors": [nb["row"] for nb in neighbours],
            }
        )

    log.info("Built %d anchor clusters.", len(clusters))
    return clusters


# ---------------------------------------------------------------------------
# 6.  Temporal QC helpers
# ---------------------------------------------------------------------------


def find_valid_windows(
    monthly_series: pd.Series,
    seq_len: int = SEQ_LEN_MONTHS,
    min_valid: float = MIN_VALID_FRACTION,
) -> List[pd.PeriodIndex]:
    """
    Scan *monthly_series* (indexed by Period[M]) for contiguous windows of
    *seq_len* months that satisfy the data-completeness threshold.

    Returns
    -------
    List of PeriodIndex objects (one per valid window).
    """
    if monthly_series.empty:
        return []

    # Build a complete monthly index spanning the series range
    full_index = pd.period_range(
        start=monthly_series.index.min(),
        end=monthly_series.index.max(),
        freq="M",
    )
    series = monthly_series.reindex(full_index)

    required = int(math.ceil(min_valid * seq_len))
    windows: List[pd.PeriodIndex] = []
    for start in range(len(series) - seq_len + 1):
        window = series.iloc[start : start + seq_len]
        if window.notna().sum() >= required:
            windows.append(full_index[start : start + seq_len])
    return windows


def build_monthly_features(
    anchor_site: pd.Series,
    anchor_levels: pd.DataFrame,
    neighbor_levels: Dict[str, pd.Series],
    gridmet_loader: Optional[GridMETLoader],
    pumping_series: pd.Series,
    window: pd.PeriodIndex,
) -> pd.DataFrame:
    """
    Assemble a single 60-row DataFrame for one anchor cluster window.

    Columns produced
    ----------------
    YearMonth, year, month, month_sin, month_cos,
    water_level_navd88,
    well_depth_ft, nat_aqfr_cd,
    precip_mm, tmin_c, tmax_c, et_mm, rh_max_pct,
    pumping_total_mgd,
    neighbor_<site_no>_level  (up to 6)
    """
    df = pd.DataFrame({"YearMonth": window})
    df["year"] = df["YearMonth"].dt.year
    df["month"] = df["YearMonth"].dt.month

    # Cyclic month encoding
    df["month_sin"], df["month_cos"] = cyclic_encode_month(df["month"])

    # ---- Anchor water level (interpolated, NAVD88) ----------------------
    # Normalise anchor_levels to a Series indexed by Period[M].
    if isinstance(anchor_levels, pd.DataFrame):
        al = anchor_levels.copy()
        if not isinstance(al.index, pd.PeriodIndex):
            if "YearMonth" in al.columns:
                al = al.set_index("YearMonth")
        al_series = al["water_level"] if "water_level" in al.columns else pd.Series(dtype=float)
    else:
        al_series = anchor_levels

    al_series = al_series.reindex(window)
    al_series = interpolate_gaps(al_series)
    df["water_level_navd88"] = al_series.values

    # ---- Static physical features ---------------------------------------
    df["well_depth_ft"] = pd.to_numeric(
        anchor_site.get("well_depth_va", np.nan), errors="coerce"
    )
    df["nat_aqfr_cd"] = str(anchor_site.get("nat_aqfr_cd", ""))

    # ---- GridMET climate features --------------------------------------
    if gridmet_loader is not None:
        lat = float(anchor_site["dec_lat_va"])
        lon = float(anchor_site["dec_long_va"])
        start_yr = window.min().year
        end_yr = window.max().year
        for var in ("precip_mm", "tmin_c", "tmax_c", "et_mm", "rh_max_pct"):
            try:
                clim = gridmet_loader.extract_monthly(lat, lon, var, start_yr, end_yr)
                clim = clim.reindex(window)
            except Exception as exc:  # noqa: BLE001
                log.warning("  GridMET %s extraction failed: %s", var, exc)
                clim = pd.Series(np.nan, index=window, name=var)
            df[var] = clim.values
    else:
        for var in ("precip_mm", "tmin_c", "tmax_c", "et_mm", "rh_max_pct"):
            df[var] = np.nan

    # ---- Pumping --------------------------------------------------------
    if not pumping_series.empty:
        pump_idx = pumping_series.set_index("YearMonth")["pumping_total_mgd"] if isinstance(pumping_series, pd.DataFrame) else pumping_series
        pump_reindexed = pump_idx.reindex(window)
        df["pumping_total_mgd"] = pump_reindexed.values
    else:
        df["pumping_total_mgd"] = np.nan

    # ---- Neighbour water levels ----------------------------------------
    for site_no, nb_series in neighbor_levels.items():
        col = f"neighbor_{site_no}_level"
        nb_reindexed = nb_series.reindex(window)
        nb_reindexed = interpolate_gaps(nb_reindexed)
        df[col] = nb_reindexed.values

    df["YearMonth"] = df["YearMonth"].astype(str)
    return df


# ---------------------------------------------------------------------------
# 7.  Main orchestration
# ---------------------------------------------------------------------------


def run_pipeline(
    state_cd: Optional[str] = None,
    site_list: Optional[List[str]] = None,
    bounding_box: Optional[Tuple[float, float, float, float]] = None,
    start_date: str = "2000-01-01",
    end_date: str = "2024-12-31",
    output_dir: str | Path = OUTPUT_DIR,
    use_gridmet: bool = False,
    use_uswwd: bool = False,
) -> None:
    """
    Full end-to-end pipeline.

    Parameters
    ----------
    state_cd:
        Two-letter USGS state code to query sites, e.g. ``"CA"``.
    site_list:
        Explicit list of USGS site numbers (alternative to state_cd).
    bounding_box:
        ``(west, south, east, north)`` bounding box (alternative).
    start_date / end_date:
        Date range for groundwater-level records.
    output_dir:
        Folder where Excel cluster files are written.
    use_gridmet:
        Set ``True`` to fetch real GridMET data (requires internet + xarray).
    use_uswwd:
        Set ``True`` to fetch real USWWD pumping data (requires internet).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1: Load site metadata
    # ------------------------------------------------------------------ #
    sites_df = fetch_usgs_sites(
        state_cd=state_cd, site_list=site_list, bounding_box=bounding_box
    )
    if sites_df.empty:
        log.error("No USGS sites found.  Aborting.")
        return

    # ------------------------------------------------------------------ #
    # Step 2: Build anchor–neighbour clusters
    # ------------------------------------------------------------------ #
    clusters = build_anchor_neighbor_clusters(sites_df)

    # ------------------------------------------------------------------ #
    # Step 3: Optionally initialise GridMET loader
    # ------------------------------------------------------------------ #
    gridmet_loader: Optional[GridMETLoader] = None
    if use_gridmet:
        if _HAS_XR:
            gridmet_loader = GridMETLoader()
        else:
            log.warning("xarray not available – GridMET data will be NaN.")

    # ------------------------------------------------------------------ #
    # Step 4: Process each cluster
    # ------------------------------------------------------------------ #
    start_year = int(pd.to_datetime(start_date).year)
    end_year = int(pd.to_datetime(end_date).year)

    for cluster_idx, cluster in enumerate(clusters):
        anchor = cluster["anchor"]
        anchor_site_no = str(anchor["site_no"])
        log.info(
            "Processing cluster %d/%d – anchor %s",
            cluster_idx + 1,
            len(clusters),
            anchor_site_no,
        )

        # ---- Fetch anchor water levels ---------------------------------
        anchor_levels_raw = fetch_groundwater_levels(
            anchor_site_no, start_date=start_date, end_date=end_date
        )
        if anchor_levels_raw.empty:
            log.info("  No water-level data for anchor %s – skipping.", anchor_site_no)
            continue

        anchor_levels_navd = align_to_navd88(anchor_levels_raw)
        # Convert Period index to a usable series for window finding
        if isinstance(anchor_levels_navd.index, pd.PeriodIndex):
            anchor_wl_series = anchor_levels_navd["water_level"]
        else:
            anchor_wl_series = anchor_levels_navd.set_index(
                anchor_levels_navd.index.to_period("M")
            )["water_level"]

        # ---- Find valid 60-month windows --------------------------------
        windows = find_valid_windows(anchor_wl_series)
        if not windows:
            log.info(
                "  No valid 60-month windows for anchor %s – skipping.",
                anchor_site_no,
            )
            continue

        # ---- Fetch neighbour levels ------------------------------------
        neighbor_levels: Dict[str, pd.Series] = {}
        for nb_row in cluster["neighbors"]:
            nb_site_no = str(nb_row["site_no"])
            nb_raw = fetch_groundwater_levels(
                nb_site_no, start_date=start_date, end_date=end_date
            )
            if not nb_raw.empty:
                nb_navd = align_to_navd88(nb_raw)
                if isinstance(nb_navd.index, pd.PeriodIndex):
                    neighbor_levels[nb_site_no] = nb_navd["water_level"]
                else:
                    neighbor_levels[nb_site_no] = nb_navd.set_index(
                        nb_navd.index.to_period("M")
                    )["water_level"]

        # ---- Fetch pumping data ----------------------------------------
        pumping_df = pd.DataFrame(columns=["YearMonth", "pumping_total_mgd"])
        if use_uswwd:
            pumping_df = fetch_uswwd_pumping(
                anchor_lat=float(anchor["dec_lat_va"]),
                anchor_lon=float(anchor["dec_long_va"]),
            )

        # ---- Build features for each window ----------------------------
        window_dfs: List[pd.DataFrame] = []
        for win_idx, window in enumerate(windows):
            feat_df = build_monthly_features(
                anchor_site=anchor,
                anchor_levels=anchor_levels_navd,
                neighbor_levels=neighbor_levels,
                gridmet_loader=gridmet_loader,
                pumping_series=pumping_df,
                window=window,
            )
            feat_df.insert(0, "window_id", win_idx)
            window_dfs.append(feat_df)

        if not window_dfs:
            continue

        cluster_df = pd.concat(window_dfs, ignore_index=True)

        # ---- Write Excel output ----------------------------------------
        out_path = output_dir / f"cluster_{anchor_site_no}.xlsx"
        cluster_df.to_excel(out_path, index=False, engine="openpyxl")
        log.info("  Saved → %s  (%d rows)", out_path, len(cluster_df))

    log.info("Pipeline complete.  Output in: %s", output_dir.resolve())


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ML-ready groundwater cluster Excel files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--state", metavar="XX", help="Two-letter state code, e.g. CA")
    group.add_argument(
        "--sites",
        nargs="+",
        metavar="SITE_NO",
        help="One or more USGS site numbers",
    )
    group.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("W", "S", "E", "N"),
        help="Bounding box: west south east north",
    )
    parser.add_argument("--start", default="2000-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output-dir", default="clusters_output", help="Output directory"
    )
    parser.add_argument(
        "--gridmet", action="store_true", help="Fetch live GridMET climate data"
    )
    parser.add_argument(
        "--uswwd", action="store_true", help="Fetch live USWWD pumping data"
    )

    args = parser.parse_args()

    run_pipeline(
        state_cd=args.state,
        site_list=args.sites,
        bounding_box=tuple(args.bbox) if args.bbox else None,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output_dir,
        use_gridmet=args.gridmet,
        use_uswwd=args.uswwd,
    )
