"""Build merged 2015–2024 monthly PM2.5 CSV (run from project root)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.build_pm25_series import main  # noqa: E402

if __name__ == "__main__":
    main()
