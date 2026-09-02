"""
Sample monthly SatPM V6.GL.02.04 PM2.5 (0.1° global grids) from AWS S3.

The bucket ``v6.gl.02.04`` is listed in the Registry of Open Data on AWS; object
GET may require AWS credentials and ``RequestPayer='requester'`` (requester pays
for data transfer). Configure ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``
in ``.env`` if downloads fail with AccessDenied.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import xarray as xr
from botocore.config import Config

from config import gee_config as gc
from config.acag_s3_config import S3_BUCKET, S3_KEY_TEMPLATE, S3_REGION


def _s3_client() -> Any:
    return boto3.client(
        "s3",
        region_name=S3_REGION,
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
    )


def _month_key(year: int, month: int) -> str:
    ym = f"{year}{month:02d}"
    return S3_KEY_TEMPLATE.format(year=year, ym=ym)


def _pm25_data_array(ds: xr.Dataset) -> xr.DataArray:
    for name in ("PM25", "PM2_5_DRY", "CNNPM25", "pm25"):
        if name in ds.data_vars:
            return ds[name]
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars.values()))
    raise ValueError(
        "Could not find PM2.5 variable; data_vars="
        + str(list(ds.data_vars))
    )


def _lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    for lat, lon in (("lat", "lon"), ("latitude", "longitude"), ("y", "x")):
        if lat in ds.coords and lon in ds.coords:
            return lat, lon
    raise ValueError("Could not find lat/lon coordinates: " + str(ds.coords))


def _normalize_lon(lon: float, ds: xr.Dataset, lon_name: str) -> float:
    c = ds.coords[lon_name]
    lo = float(c.min())
    hi = float(c.max())
    if hi > 180 and lon < 0:
        return lon % 360
    if hi <= 180 and lo >= -180:
        if lon > 180:
            return lon - 360
    return lon


def sample_netcdf_at_point(
    local_path: str | Path,
    lat: float,
    lon: float,
) -> float:
    ds = xr.open_dataset(local_path, decode_times=True)
    try:
        da = _pm25_data_array(ds)
        latn, lonn = _lat_lon_names(ds)
        lon_adj = _normalize_lon(lon, ds, lonn)
        if "time" in da.dims:
            da = da.isel(time=0)
        sub = da.sel({latn: lat, lonn: lon_adj}, method="nearest")
        val = float(np.asarray(sub.values).reshape(-1)[0])
        return val
    finally:
        ds.close()


def download_month_nc(
    year: int,
    month: int,
    dest_dir: Path | None = None,
    request_payer: str | None = "requester",
) -> Path:
    """Download one monthly NetCDF to ``dest_dir`` (or temp) and return path."""
    key = _month_key(year, month)
    client = _s3_client()
    extra: dict[str, Any] = {}
    if request_payer:
        extra["RequestPayer"] = request_payer
    if dest_dir is None:
        dest_dir = Path(tempfile.mkdtemp(prefix="acag_s3_"))
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / Path(key).name
    if out.exists() and out.stat().st_size > 0:
        return out
    if extra:
        client.download_file(S3_BUCKET, key, str(out), ExtraArgs=extra)
    else:
        client.download_file(S3_BUCKET, key, str(out))
    return out


def extract_acag_s3_monthly_range(
    start: date,
    end: date,
    lat: float | None = None,
    lon: float | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    For each month in [start, end] (inclusive of month starts), download GL
    monthly NetCDF and sample PM2.5 at (lat, lon) in µg/m³.
    """
    lat = lat if lat is not None else gc.KITWE_CBD_LAT
    lon = lon if lon is not None else gc.KITWE_CBD_LON

    months = pd.period_range(start=start, end=end, freq="M")
    rows: list[dict[str, Any]] = []
    cache = cache_dir or Path(tempfile.mkdtemp(prefix="acag_nc_"))

    for p in months:
        y, m = p.year, p.month
        try:
            nc_path = download_month_nc(y, m, dest_dir=cache / f"{y}_{m:02d}")
            pm = sample_netcdf_at_point(nc_path, lat, lon)
        except Exception as e:
            raise RuntimeError(
                f"Failed S3 month {y}-{m:02d} (key {_month_key(y, m)}): {e}\n"
                "Ensure AWS credentials are configured and this bucket allows "
                "your account for requester-pays downloads. See README."
            ) from e
        rows.append(
            {
                "date": date(y, m, 1).isoformat(),
                "pm25_ug_m3": pm,
                "longitude": lon,
                "latitude": lat,
                "source": f"s3://{S3_BUCKET}/{_month_key(y, m)}",
                "frequency": "monthly",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract Kitwe PM2.5 from SatPM V6 monthly NetCDF on AWS S3."
    )
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/raw/pm25_acag_s3_monthly.csv"),
    )
    args = parser.parse_args()
    start = date.fromisoformat(args.start[:10])
    end = date.fromisoformat(args.end[:10])
    df = extract_acag_s3_monthly_range(start, end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
