import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.models.black_scholes import OptionInputs, price
from backend.models.implied_vol import solve_implied_vol


def test_recovers_known_sigma_for_call():
    """Round-trip: price at a known sigma, solve IV from that price, recover sigma."""
    S, K, T, r, q, true_sigma = 100.0, 105.0, 0.5, 0.03, 0.01, 0.28
    market_price = price(OptionInputs(S, K, T, r, q, true_sigma, "call"))

    result = solve_implied_vol(market_price, S, K, T, r, q, "call")

    assert result["converged"]
    assert abs(result["iv"] - true_sigma) < 1e-4


def test_recovers_known_sigma_for_put():
    S, K, T, r, q, true_sigma = 100.0, 95.0, 1.0, 0.04, 0.0, 0.35
    market_price = price(OptionInputs(S, K, T, r, q, true_sigma, "put"))

    result = solve_implied_vol(market_price, S, K, T, r, q, "put")

    assert result["converged"]
    assert abs(result["iv"] - true_sigma) < 1e-4


def test_near_expiry_returns_documented_fallback():
    result = solve_implied_vol(1.0, S=100, K=100, T=1 / 730, r=0.03, q=0.0, option_type="call")
    assert not result["converged"]
    assert result["iv"] is None
    assert "near expiry" in result["message"].lower()


def test_price_below_intrinsic_returns_documented_fallback():
    """A call priced below its own discounted intrinsic value is impossible under BSM."""
    result = solve_implied_vol(1.0, S=200, K=100, T=0.5, r=0.03, q=0.0, option_type="call")
    assert not result["converged"]
    assert result["iv"] is None
    assert "intrinsic" in result["message"].lower()


def test_price_outside_reachable_range_returns_documented_fallback():
    """An absurdly high price has no sigma in [0.001, 5.0] that reproduces it."""
    result = solve_implied_vol(10_000.0, S=100, K=100, T=0.1, r=0.03, q=0.0, option_type="call")
    assert not result["converged"]
    assert result["iv"] is None
