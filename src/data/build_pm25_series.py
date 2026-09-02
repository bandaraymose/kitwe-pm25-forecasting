"""
Assemble Jan 2015 – Dec 2024 monthly PM2.5 for Kitwe from multiple sources:

1. SatPM V6 monthly NetCDF on AWS S3 (2015–2023) when downloads succeed.
2. Google Earth Engine sat-io monthly (typically 2015–2022) for remaining gaps.
3. Optional CSV ``data/raw/pm25_2024_monthly_supplement.csv`` (date, pm25_ug_m3)
   for 2024 and any other gaps. V6 S3 + GEE do not provide all of 2024 here.

Output: 120 rows (monthly), 2015-01-01 through 2024-12-01.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from config import gee_config as gc
from config.settings import DATA_RAW, DEFAULT_RAW_CSV

from src.data.extract_acag_s3_monthly import extract_acag_s3_monthly_range
from src.data.extract_gee import _init_ee, extract_acag_monthly


def _month_starts_2015_2024() -> pd.DatetimeIndex:
    return pd.date_range("2015-01-01", "2024-12-01", freq="MS")


def _load_supplement(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["date", "pm25_ug_m3"])
    df = pd.read_csv(path)
    if "date" not in df.columns or "pm25_ug_m3" not in df.columns:
        raise ValueError(
            f"{path} must have columns: date, pm25_ug_m3 (ISO dates YYYY-MM-DD)"
        )
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[["date", "pm25_ug_m3"]]


def build_pm25_series(
    supplement_csv: Path | None = None,
    try_s3_full: bool = True,
    try_gee_fallback: bool = True,
) -> pd.DataFrame:
    supp_path = supplement_csv or (DATA_RAW / "pm25_2024_monthly_supplement.csv")
    supp = _load_supplement(supp_path)

    master = pd.DataFrame({"date": _month_starts_2015_2024().strftime("%Y-%m-%d")})
    master["pm25_ug_m3"] = pd.NA
    master["source"] = pd.NA
    master["frequency"] = "monthly"

    if try_s3_full:
        try:
            s3_df = extract_acag_s3_monthly_range(
                date(2015, 1, 1), date(2023, 12, 31)
            )
            s3_df = s3_df.set_index("date")
            m = master.set_index("date")
            for idx in s3_df.index:
                if idx in m.index:
                    m.loc[idx, "pm25_ug_m3"] = s3_df.loc[idx, "pm25_ug_m3"]
                    m.loc[idx, "source"] = s3_df.loc[idx, "source"]
            master = m.reset_index()
        except Exception:
            if not try_gee_fallback:
                raise

    if try_gee_fallback and master["pm25_ug_m3"].isna().any():
        _init_ee()
        gee_df = extract_acag_monthly("2015-01-01", "2025-01-01")
        gee_df = gee_df.set_index("date")
        m = master.set_index("date")
        for idx in gee_df.index:
            if idx in m.index and pd.isna(m.loc[idx, "pm25_ug_m3"]):
                m.loc[idx, "pm25_ug_m3"] = gee_df.loc[idx, "pm25_ug_m3"]
                m.loc[idx, "source"] = gee_df.loc[idx, "source"]
        master = m.reset_index()

    if not supp.empty:
        supp = supp.set_index("date")
        m = master.set_index("date")
        for idx in supp.index:
            if idx in m.index:
                m.loc[idx, "pm25_ug_m3"] = supp.loc[idx, "pm25_ug_m3"]
                m.loc[idx, "source"] = "supplement_csv"
        master = m.reset_index()

    master["longitude"] = gc.KITWE_CBD_LON
    master["latitude"] = gc.KITWE_CBD_LAT
    return master


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build full 2015–2024 monthly PM2.5 series for Kitwe."
    )
    parser.add_argument(
        "--supplement",
        type=Path,
        default=DATA_RAW / "pm25_2024_monthly_supplement.csv",
        help="CSV with columns date, pm25_ug_m3 (e.g. 2024 from SatPM V5)",
    )
    parser.add_argument(
        "--no-s3",
        action="store_true",
        help="Skip AWS S3 SatPM V6 download",
    )
    parser.add_argument(
        "--no-gee",
        action="store_true",
        help="Do not use Google Earth Engine fallback",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_RAW_CSV,
    )
    args = parser.parse_args()

    df = build_pm25_series(
        supplement_csv=args.supplement,
        try_s3_full=not args.no_s3,
        try_gee_fallback=not args.no_gee,
    )
    missing = int(df["pm25_ug_m3"].isna().sum())
    if missing:
        raise SystemExit(
            f"Series incomplete: {missing} months missing PM2.5. "
            f"Add rows to {args.supplement} (date, pm25_ug_m3) or fix AWS/GEE. "
            "See README."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
