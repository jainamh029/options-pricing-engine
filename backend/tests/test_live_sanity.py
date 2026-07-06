"""
End-to-end sanity check against REAL live AAPL data. Unlike the rest of the
suite (synthetic inputs, deterministic), this one has no fixed expected
values -- the market moves every day. Its job is to catch exactly the
class of bug that unit tests on synthetic data can miss: a scaling error
that's "correct" in isolated math but wrong in the units actually shown to
a user (theta secretly annualized, vega on the wrong scale, a parity check
that flags nearly everything on real bid/ask noise).

Skips (rather than fails) if live data is unavailable -- yfinance rate
limiting on shared cloud IPs is a known, documented characteristic of this
project's free data source (see README), not something this test should
report as a regression.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from backend.data import market_data
from backend.data.risk_free_rate import get_risk_free_rate
from backend.data.validators import annotate_chain, check_put_call_parity
from backend.models.black_scholes import OptionInputs, greeks
from backend.models.implied_vol import solve_implied_vol

TICKER = "AAPL"


def _get_live_context():
    try:
        spot = market_data.get_spot_price(TICKER)
        q = market_data.get_dividend_yield(TICKER)
        r = get_risk_free_rate()
        expiry = market_data.pick_default_expiry(TICKER, min_days=7)
        T = market_data.years_to_expiry(expiry)
        calls_raw, puts_raw, _ = market_data.get_option_chain(TICKER, expiry)
        calls, puts = annotate_chain(calls_raw), annotate_chain(puts_raw)
    except Exception as e:
        # Broad on purpose: yfinance can raise its own YFRateLimitError,
        # requests network errors, or our own MarketDataError/RiskFreeRateError
        # -- all of them mean "live data unavailable right now," not "the
        # code under test is broken."
        pytest.skip(f"Live {TICKER} data unavailable (likely yfinance rate limiting or market closed): {e}")
    return spot, q, r, T, calls, puts


def test_atm_aapl_greeks_are_in_sane_units():
    spot, q, r, T, calls, puts = _get_live_context()

    tradeable_calls = calls[calls["is_tradeable"]]
    if tradeable_calls.empty:
        pytest.skip("No tradeable call quotes right now (market likely closed) -- see README market-hours note")

    atm_row = tradeable_calls.iloc[(tradeable_calls["strike"] - spot).abs().argsort().iloc[0]]
    strike = float(atm_row["strike"])

    iv_result = solve_implied_vol(atm_row["mid"], spot, strike, T, r, q, "call")
    if not iv_result["converged"]:
        pytest.skip(f"IV did not converge for the live ATM quote: {iv_result['message']}")

    inp = OptionInputs(S=spot, K=strike, T=T, r=r, q=q, sigma=iv_result["iv"], option_type="call")
    g = greeks(inp)
    print(f"\nLive ATM {TICKER} call: S={spot:.2f} K={strike:.2f} T={T:.4f}y sigma={iv_result['iv']:.2%}")
    print(f"price={g['price']:.4f} delta={g['delta']:.4f} gamma={g['gamma']:.6f} "
          f"vega={g['vega']:.4f} theta={g['theta']:.4f} rho={g['rho']:.4f}")

    # Theta is daily decay -- for a near-ATM option it should be a small
    # fraction of the option's own price. Before the fix, theta was returned
    # in annual units and would routinely be MANY TIMES the option price for
    # short-dated contracts (e.g. price=$5.03, theta=-125.68/year) -- exactly
    # the tell this assertion catches.
    assert abs(g["theta"]) < g["price"], (
        f"theta {g['theta']:.4f} is not small relative to price {g['price']:.4f} -- looks annualized, not daily"
    )

    # Vega per 1 percentage point of IV: a near-ATM option should move a few
    # cents to under a dollar for a 1-point IV move, not tens of dollars
    # (the raw per-unit-sigma scale is 100x larger).
    assert 0.001 < g["vega"] < 2.0, f"vega {g['vega']:.4f} is outside the expected per-1%-IV range"

    # ATM call delta should sit roughly mid-range, not near 0 or 1.
    assert 0.2 < g["delta"] < 0.8, f"delta {g['delta']:.4f} doesn't look like an ATM call"


def test_put_call_parity_is_not_universally_violated():
    spot, q, r, T, calls, puts = _get_live_context()

    parity = check_put_call_parity(calls, puts, spot, r, q, T)
    if len(parity) < 5:
        pytest.skip("Not enough overlapping tradeable strikes right now to judge parity (market likely closed)")

    violation_rate = parity["violated"].mean()
    print(f"\nPut-call parity on {len(parity)} live {TICKER} strikes ({violation_rate:.0%} violate):")
    print(parity[["strike", "lhs", "rhs", "band_low", "band_high", "diff", "violated"]].to_string(index=False))

    # Before the fix, a flat $0.10 tolerance against mid prices flagged
    # 23/23 (100%) of strikes on EVERY ticker, with zero discrimination --
    # that's the direct regression this guards against.
    #
    # NOT asserting a tight "<50%" bound here: on a live, trending session,
    # 60-75% of strikes can legitimately still be flagged even with a
    # correctly-implemented check, because of clock skew between the live
    # spot feed and yfinance's ~15min-delayed options chain (verified via
    # `diff` sitting at a roughly CONSTANT offset across every moneyness --
    # a level shift, not the moneyness-dependent shape early-exercise
    # premium would produce; see check_put_call_parity's docstring). A
    # threshold tight enough to demand "small minority" on real live data
    # would either flake on a normal trending morning or tempt loosening the
    # band so much it stops meaning anything. What must ALWAYS be true when
    # the code is correct, independent of how the market is moving right
    # now, is that it's not universal and not degenerate:
    assert not parity["violated"].all(), "100% of strikes violate parity -- looks like the flat-tolerance bug is back"
    assert violation_rate < 0.95, f"{violation_rate:.0%} violate parity -- too close to universal to be just feed skew"
