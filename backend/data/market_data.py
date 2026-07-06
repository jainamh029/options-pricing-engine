"""
Live market data via yfinance: spot price, dividend yield, historical
prices for realized vol, and (later) options chains.

No hardcoded prices, vols, or chains -- everything here is a live pull.
"""

import time
from datetime import datetime, timezone

import yfinance as yf

# Cache each chain fetch briefly so repeated calls (e.g. an API serving
# several UI requests for the same expiry) don't hammer yfinance.
_CHAIN_CACHE_TTL_SECONDS = 45
_chain_cache: dict[tuple[str, str], tuple] = {}  # (ticker, expiry) -> (calls, puts, fetched_at)


class MarketDataError(Exception):
    pass


def get_spot_price(ticker: str) -> float:
    """Latest available spot price for the underlying."""
    t = yf.Ticker(ticker)
    hist = t.history(period="1d")
    if hist.empty:
        raise MarketDataError(f"No price history returned for {ticker!r}")
    return float(hist["Close"].iloc[-1])


def get_dividend_yield(ticker: str) -> float:
    """
    Continuous dividend yield estimate, as a fraction (e.g. 0.006 for 0.6%).

    yfinance exposes this two ways: `.info['trailingAnnualDividendYield']`
    is already a fraction; `.info['dividendYield']` is the same figure but
    expressed in percentage points (e.g. 0.35 meaning 0.35%, i.e. needs
    /100). We prefer the fraction field and fall back to the percentage
    field only if the fraction field is unavailable. Defaults to 0.0 for
    non-dividend payers.
    """
    t = yf.Ticker(ticker)
    info = t.info

    trailing = info.get("trailingAnnualDividendYield")
    if trailing is not None:
        return float(trailing)

    div_yield_pct = info.get("dividendYield")
    if div_yield_pct is None:
        return 0.0
    return float(div_yield_pct) / 100.0


def get_historical_prices(ticker: str, period: str = "1y"):
    """Daily close prices, used for realized volatility calculations."""
    t = yf.Ticker(ticker)
    hist = t.history(period=period)
    if hist.empty:
        raise MarketDataError(f"No historical data returned for {ticker!r}")
    return hist["Close"]


def get_realized_volatility(ticker: str, period: str = "1y", trading_days: int = 252) -> float:
    """Annualized realized volatility from daily log returns."""
    import numpy as np

    closes = get_historical_prices(ticker, period=period)
    log_returns = np.log(closes / closes.shift(1)).dropna()
    daily_std = log_returns.std()
    return float(daily_std * np.sqrt(trading_days))


def get_option_expiries(ticker: str) -> list[str]:
    t = yf.Ticker(ticker)
    expiries = t.options
    if not expiries:
        raise MarketDataError(f"No option expiries available for {ticker!r}")
    return list(expiries)


def years_to_expiry(expiry_str: str) -> float:
    """Approximates expiry as 4pm ET (21:00 UTC) on the listed date."""
    expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").replace(hour=21, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max((expiry_dt - now).total_seconds(), 0) / (365.0 * 24 * 3600)


def pick_default_expiry(ticker: str, min_days: int = 7, expiry_override: str | None = None) -> str:
    """
    Picks the nearest listed expiry at least `min_days` out, to avoid
    near-zero-T contracts with unstable Greeks/IV by default. Pass
    expiry_override to use a specific listed expiry instead (validated
    against the live list).
    """
    expiries = get_option_expiries(ticker)
    if expiry_override:
        if expiry_override not in expiries:
            # A caller-supplied expiry that isn't listed is a bad request, not
            # an upstream data failure -- ValueError (not MarketDataError) so
            # API callers get a 400, not a 502.
            raise ValueError(f"{expiry_override!r} is not a listed expiry for {ticker!r}. Available: {expiries}")
        return expiry_override
    return next((e for e in expiries if years_to_expiry(e) >= min_days / 365), expiries[0])


def get_option_chain(ticker: str, expiry: str):
    """
    Raw options chain for a given expiry: returns (calls_df, puts_df, fetched_at_epoch).
    fetched_at_epoch is stamped here so downstream staleness checks have a
    reference independent of any per-row exchange timestamp.

    Cached for _CHAIN_CACHE_TTL_SECONDS to avoid re-hitting yfinance on
    every call for the same (ticker, expiry) pair.
    """
    key = (ticker, expiry)
    now = time.time()
    cached = _chain_cache.get(key)
    if cached is not None and (now - cached[2]) < _CHAIN_CACHE_TTL_SECONDS:
        return cached

    t = yf.Ticker(ticker)
    chain = t.option_chain(expiry)
    fetched_at = time.time()
    result = (chain.calls, chain.puts, fetched_at)
    _chain_cache[key] = result
    return result
