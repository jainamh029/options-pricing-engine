"""
Validation checks for real options-chain data.

Real chains contain rows that are not real tradeable prices: illiquid
strikes with zero volume/OI, crossed or zero bid/ask, and stale last-trade
prints. These must be flagged and excluded from pricing/IV calculations
rather than silently fed in, or the IV solver will diverge or return
garbage.
"""

import math
import time
import pandas as pd

# yfinance chain data is itself ~15min delayed on the free tier; this adds
# a margin on top of that before flagging an individual quote as stale.
DEFAULT_STALENESS_MINUTES = 20


def compute_mid(bid, ask):
    if bid is None or ask is None or pd.isna(bid) or pd.isna(ask):
        return None
    if bid <= 0 or ask <= 0 or bid > ask:
        return None
    return (bid + ask) / 2.0


def is_crossed_or_zero(bid, ask) -> bool:
    if bid is None or ask is None or pd.isna(bid) or pd.isna(ask):
        return True
    return bid <= 0 or ask <= 0 or bid > ask


def is_illiquid(volume, open_interest) -> bool:
    volume = 0 if volume is None or pd.isna(volume) else volume
    open_interest = 0 if open_interest is None or pd.isna(open_interest) else open_interest
    return volume == 0 and open_interest == 0


def is_snapshot_stale(fetched_at_epoch: float, now_epoch=None, max_age_minutes: float = DEFAULT_STALENESS_MINUTES) -> bool:
    """
    Whether the *whole chain snapshot* (as returned by market_data.get_option_chain,
    stamped with its own fetch time) is older than the freshness window. This is
    the actual staleness gate -- it catches a caller pricing against a snapshot
    pulled from a long-lived cache or a stalled process, not a normal illiquid
    strike.
    """
    if now_epoch is None:
        now_epoch = time.time()
    return (now_epoch - fetched_at_epoch) > max_age_minutes * 60


def last_trade_age_days(last_trade_date, now=None):
    """
    Informational only -- days since the last actual trade printed on this
    strike. NOT a tradeability gate: a strike with no recent trade can still
    have a perfectly live, firm bid/ask from market makers. Returns None if
    no trade timestamp is available.
    """
    if last_trade_date is None or pd.isna(last_trade_date):
        return None
    if now is None:
        now = pd.Timestamp.now(tz=last_trade_date.tzinfo or "UTC")
    return (now - last_trade_date).total_seconds() / 86400.0


def annotate_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of `df` with added columns: mid, is_crossed, is_illiquid,
    last_trade_age_days (informational), is_tradeable (crossed/illiquid
    checks pass and mid is a real number).

    Snapshot-level staleness (is the whole chain pull too old?) is checked
    separately via is_snapshot_stale() against the fetched_at timestamp
    returned by market_data.get_option_chain -- it is not a per-row concept.
    """
    out = df.copy()
    out["mid"] = [compute_mid(b, a) for b, a in zip(out["bid"], out["ask"])]
    out["is_crossed"] = [is_crossed_or_zero(b, a) for b, a in zip(out["bid"], out["ask"])]
    out["is_illiquid"] = [
        is_illiquid(v, oi) for v, oi in zip(out.get("volume"), out.get("openInterest"))
    ]
    out["last_trade_age_days"] = [last_trade_age_days(ltd) for ltd in out.get("lastTradeDate")]
    out["is_tradeable"] = (~out["is_crossed"]) & (~out["is_illiquid"]) & out["mid"].notna()
    return out


def check_put_call_parity(calls: pd.DataFrame, puts: pd.DataFrame, S: float, r: float, q: float,
                           T: float, tolerance: float = 0.10) -> pd.DataFrame:
    """
    For each strike present in both (already-annotated) calls and puts,
    restricted to tradeable rows, verify:

        C - P ~= S*e^(-qT) - K*e^(-rT)

    Real chains will show a handful of violations near the edges (deep
    ITM/OTM, illiquid strikes) -- flagging them correctly is the point,
    not a sign the solver is broken.
    """
    for name, df in (("calls", calls), ("puts", puts)):
        if "mid" not in df.columns or "is_tradeable" not in df.columns:
            raise ValueError(f"{name} dataframe must be annotate_chain()'d first (missing mid/is_tradeable)")

    merged = pd.merge(
        calls[["strike", "mid", "is_tradeable"]].rename(columns={"mid": "call_mid", "is_tradeable": "call_tradeable"}),
        puts[["strike", "mid", "is_tradeable"]].rename(columns={"mid": "put_mid", "is_tradeable": "put_tradeable"}),
        on="strike", how="inner",
    )
    merged = merged[merged["call_tradeable"] & merged["put_tradeable"]].copy()
    merged["lhs"] = merged["call_mid"] - merged["put_mid"]
    merged["rhs"] = S * math.exp(-q * T) - merged["strike"] * math.exp(-r * T)
    merged["diff"] = merged["lhs"] - merged["rhs"]
    merged["violated"] = merged["diff"].abs() > tolerance
    return merged[["strike", "call_mid", "put_mid", "lhs", "rhs", "diff", "violated"]].reset_index(drop=True)
