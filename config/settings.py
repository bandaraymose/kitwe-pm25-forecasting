"""Paths and global settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PREDICTIONS = PROJECT_ROOT / "data" / "predictions"

DEFAULT_RAW_CSV = DATA_RAW / "pm25_kitwe_2015_2024.csv"
