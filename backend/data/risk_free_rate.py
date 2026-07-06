"""
Risk-free rate from FRED (3-month T-bill, series DGS3MO).

Uses FRED's public CSV endpoint, which requires no API key. DGS3MO is
quoted as an annualized percentage on a discount basis (e.g. 5.23 means
5.23%); we convert to a decimal continuously-compounded-style rate for
use directly in BSM (the difference between discount/bond-equivalent/
continuous compounding on a 3-month bill is second-order for pricing
purposes here, and is noted in the README as a simplification).
"""

import io
import time
import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

_cache: dict[str, tuple[float, float]] = {}  # series_id -> (rate, fetched_at)
_CACHE_TTL_SECONDS = 3600  # rates don't move intraday; refetch hourly is plenty


class RiskFreeRateError(Exception):
    pass


def get_risk_free_rate(series_id: str = "DGS3MO") -> float:
    """
    Latest available risk-free rate from FRED, as a decimal (e.g. 0.0523).
    FRED CSVs sometimes carry a trailing '.' for non-trading days -- this
    walks backward to the most recent non-missing observation.
    """
    now = time.time()
    cached = _cache.get(series_id)
    if cached is not None and (now - cached[1]) < _CACHE_TTL_SECONDS:
        return cached[0]

    url = FRED_CSV_URL.format(series_id=series_id)
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise RiskFreeRateError(f"FRED request failed with status {resp.status_code}")

    lines = resp.text.strip().splitlines()
    if len(lines) < 2:
        raise RiskFreeRateError(f"Unexpected empty response from FRED for {series_id!r}")

    # CSV format: DATE,DGS3MO \n 2026-07-01,5.23 \n ...
    for line in reversed(lines[1:]):
        date_str, _, value_str = line.partition(",")
        value_str = value_str.strip()
        if value_str and value_str != ".":
            rate = float(value_str) / 100.0
            _cache[series_id] = (rate, now)
            return rate

    raise RiskFreeRateError(f"No valid observations found in FRED series {series_id!r}")
