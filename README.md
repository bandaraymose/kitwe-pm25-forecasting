# Kitwe PM2.5 Forecasting

Comparative forecasting (ARIMA/SARIMA, Prophet, LSTM) for PM2.5 in Kitwe CBD, Zambia, with a Streamlit dashboard (later phases).

## Environment

Python 3.10+ recommended.

```bash
cd kitwe-pm25-forecasting
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Google Earth Engine

1. Register at [Google Earth Engine](https://earthengine.google.com/).
2. Authenticate locally:

```bash
earthengine authenticate
```

3. **Google Cloud project (required for current Earth Engine):**  
   - Create or select a project in [Google Cloud Console](https://console.cloud.google.com/).  
   - Enable the **Earth Engine API** for that project.  
   - In [Earth Engine Code Editor](https://code.earthengine.google.com/), register / link that project if prompted.  
   - Copy `.env.example` to `.env` and set `GEE_PROJECT_ID=your-project-id` (the short ID, e.g. `my-ee-project-123`).  
   - Or pass `--project your-project-id` when running the extractor.

## Extract PM2.5 (data extraction)

Default source is **ACAG-style monthly** satellite PM2.5 on GEE (`GLOBAL-SATELLITE-PM25/MONTHLY`). The public GEE mirror provides **monthly means**, aligned with Atmospheric Composition Analysis Group methodology (band `b1`, scale ×0.1).

```bash
python scripts/extract_gee_data.py
```

Output: `data/raw/pm25_kitwe_2015_2024.csv`

Options:

```bash
python scripts/extract_gee_data.py --source acag_monthly --start 2015-01-01 --end 2025-01-01 -o data/raw/pm25_monthly.csv
python scripts/extract_gee_data.py --source ghap_daily --start 2017-01-01 --end 2023-01-01 -o data/raw/pm25_ghap_daily.csv
```

`ghap_daily` uses GHAP daily 1 km PM2.5 (typical coverage 2017–2022 on GEE).

## Full 10 calendar years (Jan 2015 – Dec 2024, monthly)

The GEE sat-io collection above usually **stops around the end of 2022**, so it alone does **not** give 120 monthly rows through 2024. To build a **complete monthly series** for the thesis window:

1. **SatPM V6 on AWS (2015–2023)** — Optional but recommended for **nine full years** in one product (V6.GL.02.04, 0.1° global monthly grids). Configure AWS credentials in `.env` (see `.env.example`). Downloads may be **requester-pays**; you need an AWS account and billing enabled.
2. **Google Earth Engine** — Fills months still missing after S3 (typically 2015–2022).
3. **Supplement CSV for 2024** — Public V6 S3 data used here **does not include 2024**. Add `data/raw/pm25_2024_monthly_supplement.csv` with columns `date`, `pm25_ug_m3` (e.g. from [SatPM V5.GL.06](https://www.satpm.org/v5-gl-06) monthly NetCDF on Box or the [SatPM data portal](https://satpm25data.net)).

Generate a 2024 template (NaN values to replace):

```bash
python scripts/generate_supplement_template.py
```

Then build the merged file:

```bash
python scripts/build_pm25_series.py -o data/raw/pm25_kitwe_2015_2024.csv
```

Use `--no-s3` if you are not using AWS. If any month is still missing, the script exits with an error until you fix the supplement CSV or S3/GEE access.

Standalone S3-only extract (2015–2023):

```bash
python -m src.data.extract_acag_s3_monthly --start 2015-01-01 --end 2023-12-31 -o data/raw/pm25_acag_s3_monthly.csv
```

## Configuration

Edit `config/gee_config.py` for Kitwe coordinates, buffer size, date range, and `PM25_SOURCE`. S3 bucket paths are in `config/acag_s3_config.py`.
