"""Write data/raw/pm25_2024_monthly_supplement.csv with 2024 month rows (NaN pm25)."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DATA_RAW  # noqa: E402

out = DATA_RAW / "pm25_2024_monthly_supplement.csv"
dates = pd.date_range("2024-01-01", "2024-12-01", freq="MS")
df = pd.DataFrame(
    {"date": dates.strftime("%Y-%m-%d"), "pm25_ug_m3": float("nan")}
)
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote template with NaN pm25: {out}")
print("Replace pm25_ug_m3 with values from SatPM V5.GL.06 or satpm25data.net.")
