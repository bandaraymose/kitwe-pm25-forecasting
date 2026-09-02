"""
Google Earth Engine settings for Kitwe CBD PM2.5 extraction.

The Atmospheric Composition Analysis Group (ACAG) SatPM2.5 product mirrored on GEE
(sat-io) provides monthly and annual means — not native daily grids. The thesis
pipeline targets daily series; monthly means are the correct GEE-native series
for this product. Optional daily extraction via GHAP is available for 2017–2022
only (see PM25_SOURCE).
"""

from __future__ import annotations

# Kitwe Central Business District — representative point (WGS84)
KITWE_CBD_LON: float = 28.2130
KITWE_CBD_LAT: float = -12.8167

# Buffer (meters) around the point for spatial averaging (~neighborhood scale)
REGION_BUFFER_METERS: float = 500.0

# Study window (inclusive start, exclusive end in Earth Engine filterDate)
DATE_START: str = "2015-01-01"
DATE_END: str = "2025-01-01"

# ACAG-style global satellite PM2.5 on GEE (monthly means, band b1, scale ×0.1)
ACAG_MONTHLY_COLLECTION: str = "projects/sat-io/open-datasets/GLOBAL-SATELLITE-PM25/MONTHLY"
ACAG_PM25_BAND: str = "b1"
ACAG_SCALE_FACTOR: float = 0.1

# LGHAP daily PM2.5 (2000-2020, gap-free, 1km resolution) - China focused
LGHAP_DAILY_COLLECTION: str = "projects/sat-io/open-datasets/LGHAP/PM25_daily"
LGHAP_PM25_BAND: str = "b1"

# Global High Air Pollutants — daily 1 km (2017-2022, global coverage)
GHAP_DAILY_COLLECTION: str = "projects/sat-io/open-datasets/GHAP/GHAP_D1K_PM25"
GHAP_PM25_BAND: str = "b1"

PM25_SOURCE: str = "ghap_daily"  # "acag_monthly" | "ghap_daily" | "lghap_daily" | "combined_daily"

# reduceRegion scale (meters); ~0.01° grid ≈ 1 km at this latitude
REDUCE_SCALE_METERS: int = 1000
