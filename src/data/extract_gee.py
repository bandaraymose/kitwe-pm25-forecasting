"""
Extract PM2.5 time series for Kitwe CBD from Google Earth Engine.

Requires: ``earthengine-api``, ``ee.Authenticate()`` once, and a **Google Cloud
project ID** registered with Earth Engine. Set ``GEE_PROJECT_ID`` in ``.env`` or
pass ``--project`` (see README).
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee
import pandas as pd
from dotenv import load_dotenv

from config import gee_config as gc
from config.settings import DATA_RAW, DEFAULT_RAW_CSV


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _init_ee(*, project: str | None = None) -> None:
    load_dotenv(_project_root() / ".env")
    gee_project = (project or os.environ.get("GEE_PROJECT_ID") or "").strip()
    if not gee_project:
        raise SystemExit(
            "Earth Engine requires a Google Cloud project ID.\n\n"
            "1. In Google Cloud Console, create or pick a project.\n"
            "2. Enable the Earth Engine API for that project.\n"
            "3. Link the project to Earth Engine (Code Editor → Register a project).\n"
            "4. Add to a .env file in this folder:\n"
            "     GEE_PROJECT_ID=your-project-id\n"
            "   Or run:\n"
            "     python scripts/extract_gee_data.py --project your-project-id\n\n"
            "Docs: https://developers.google.com/earth-engine/guides/access"
        )
    ee.Initialize(project=gee_project)


def _study_geometry() -> ee.Geometry:
    point = ee.Geometry.Point([gc.KITWE_CBD_LON, gc.KITWE_CBD_LAT])
    return point.buffer(gc.REGION_BUFFER_METERS)


def _utc_date_from_ms(ms: Any) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _reduce_pm25(
    image: ee.Image,
    geometry: ee.Geometry,
    band: str,
    scale_m: int,
) -> float | None:
    d = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=scale_m,
        maxPixels=1e13,
        bestEffort=True,
    )
    raw = d.get(band).getInfo()
    if raw is None:
        return None
    return float(raw)


def extract_acag_monthly(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    ACAG-style monthly satellite PM2.5 (sat-io GLOBAL-SATELLITE-PM25/MONTHLY).

    Raw band ``b1`` is scaled ×0.1 to µg/m³ per dataset documentation.
    """
    start = start or gc.DATE_START
    end = end or gc.DATE_END
    geometry = _study_geometry()

    col = (
        ee.ImageCollection(gc.ACAG_MONTHLY_COLLECTION)
        .filterDate(start, end)
        .sort("system:time_start")
    )

    def scale_img(img: ee.Image) -> ee.Image:
        return img.multiply(gc.ACAG_SCALE_FACTOR).copyProperties(
            img, ["system:time_start", "system:time_end"]
        )

    col = col.map(scale_img)
    n = col.size().getInfo()
    if n == 0:
        return pd.DataFrame(
            columns=[
                "date",
                "pm25_ug_m3",
                "longitude",
                "latitude",
                "source",
                "frequency",
            ]
        )

    feats = col.toList(n)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        img = ee.Image(feats.get(i))
        t0 = img.get("system:time_start").getInfo()
        dt = _utc_date_from_ms(t0)
        pm = _reduce_pm25(img, geometry, gc.ACAG_PM25_BAND, gc.REDUCE_SCALE_METERS)
        rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "pm25_ug_m3": pm,
                "longitude": gc.KITWE_CBD_LON,
                "latitude": gc.KITWE_CBD_LAT,
                "source": gc.ACAG_MONTHLY_COLLECTION,
                "frequency": "monthly",
            }
        )

    return pd.DataFrame(rows)


def extract_lghap_daily(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    LGHAP daily PM2.5 at ~1 km (2000–2020 coverage on GEE). Gap-free dataset.
    """
    start = start or "2000-01-01"
    end = end or "2021-01-01"
    geometry = _study_geometry()

    col = (
        ee.ImageCollection(gc.LGHAP_DAILY_COLLECTION)
        .filterDate(start, end)
        .sort("system:time_start")
    )

    # LGHAP data is already in µg/m³, no scaling needed
    n = col.size().getInfo()
    if n == 0:
        return pd.DataFrame(
            columns=[
                "date",
                "pm25_ug_m3",
                "longitude",
                "latitude",
                "source",
                "frequency",
            ]
        )

    feats = col.toList(n)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        img = ee.Image(feats.get(i))
        t0 = img.get("system:time_start").getInfo()
        dt = _utc_date_from_ms(t0)
        pm = _reduce_pm25(img, geometry, gc.LGHAP_PM25_BAND, gc.REDUCE_SCALE_METERS)
        rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "pm25_ug_m3": pm,
                "longitude": gc.KITWE_CBD_LON,
                "latitude": gc.KITWE_CBD_LAT,
                "source": gc.LGHAP_DAILY_COLLECTION,
                "frequency": "daily",
            }
        )

    return pd.DataFrame(rows)


def extract_ghap_daily(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    GHAP daily PM2.5 at ~1 km (2017–2022 coverage on GEE). Band scaled ×0.1 to µg/m³.
    """
    start = start or "2017-01-01"
    end = end or "2023-01-01"
    geometry = _study_geometry()

    col = (
        ee.ImageCollection(gc.GHAP_DAILY_COLLECTION)
        .filterDate(start, end)
        .sort("system:time_start")
    )

    def scale_img(img: ee.Image) -> ee.Image:
        return img.multiply(0.1).copyProperties(
            img, ["system:time_start", "system:time_end"]
        )

    col = col.map(scale_img)
    n = col.size().getInfo()
    if n == 0:
        return pd.DataFrame(
            columns=[
                "date",
                "pm25_ug_m3",
                "longitude",
                "latitude",
                "source",
                "frequency",
            ]
        )

    feats = col.toList(n)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        img = ee.Image(feats.get(i))
        t0 = img.get("system:time_start").getInfo()
        dt = _utc_date_from_ms(t0)
        pm = _reduce_pm25(img, geometry, gc.GHAP_PM25_BAND, gc.REDUCE_SCALE_METERS)
        rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "pm25_ug_m3": pm,
                "longitude": gc.KITWE_CBD_LON,
                "latitude": gc.KITWE_CBD_LAT,
                "source": gc.GHAP_DAILY_COLLECTION,
                "frequency": "daily",
            }
        )

    return pd.DataFrame(rows)


def extract_pm25(
    source: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    src = source or gc.PM25_SOURCE
    if src == "acag_monthly":
        return extract_acag_monthly(start=start, end=end)
    if src == "ghap_daily":
        return extract_ghap_daily(start=start, end=end)
    if src == "lghap_daily":
        return extract_lghap_daily(start=start, end=end)
    if src == "combined_daily":
        return extract_combined_daily(start=start, end=end)
    raise ValueError(f"Unknown PM25_SOURCE: {src!r}")


def extract_combined_daily(
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Combine LGHAP (2000-2020) and GHAP (2017-2022) for complete daily coverage.
    For 2015-2016: Use LGHAP only
    For 2017-2020: Use LGHAP prioritized, fall back to GHAP if needed
    For 2021-2022: Use GHAP only
    For 2023-2024: No data available yet
    """
    start = start or gc.DATE_START
    end = end or gc.DATE_END
    
    all_data = []
    
    # Extract LGHAP data (2000-2020)
    lghap_start = max(start, "2000-01-01")
    lghap_end = min(end, "2021-01-01")
    if lghap_start < lghap_end:
        print(f"Extracting LGHAP daily data from {lghap_start} to {lghap_end}")
        lghap_data = extract_lghap_daily(start=lghap_start, end=lghap_end)
        all_data.append(lghap_data)
    
    # Extract GHAP data (2017-2022)
    ghap_start = max(start, "2017-01-01")
    ghap_end = min(end, "2023-01-01")
    if ghap_start < ghap_end:
        print(f"Extracting GHAP daily data from {ghap_start} to {ghap_end}")
        ghap_data = extract_ghap_daily(start=ghap_start, end=ghap_end)
        all_data.append(ghap_data)
    
    if not all_data:
        return pd.DataFrame(
            columns=[
                "date",
                "pm25_ug_m3",
                "longitude",
                "latitude",
                "source",
                "frequency",
            ]
        )
    
    # Combine and deduplicate by date, prioritizing LGHAP over GHAP for overlap
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    
    # Sort by date and source priority (LGHAP over GHAP)
    combined_df['source_priority'] = combined_df['source'].apply(
        lambda x: 0 if 'LGHAP' in x else 1
    )
    combined_df = combined_df.sort_values(['date', 'source_priority'])
    
    # Keep first occurrence (highest priority) for each date
    combined_df = combined_df.drop_duplicates(subset=['date'], keep='first')
    combined_df = combined_df.drop('source_priority', axis=1)
    combined_df = combined_df.sort_values('date')
    
    return combined_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Kitwe CBD PM2.5 from Google Earth Engine."
    )
    parser.add_argument(
        "--source",
        choices=("acag_monthly", "ghap_daily", "lghap_daily", "combined_daily"),
        default=gc.PM25_SOURCE,
        help="Dataset (default from config.gee_config.PM25_SOURCE)",
    )
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (exclusive)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_RAW_CSV,
        help=f"Output CSV path (default: {DEFAULT_RAW_CSV})",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Google Cloud project ID for Earth Engine (overrides GEE_PROJECT_ID in .env)",
    )
    args = parser.parse_args()

    _init_ee(project=args.project)
    df = extract_pm25(source=args.source, start=args.start, end=args.end)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
