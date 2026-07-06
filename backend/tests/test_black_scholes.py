import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.models.black_scholes import OptionInputs, price


def test_put_call_parity_holds_for_model_prices():
    """C - P == S*e^(-qT) - K*e^(-rT), by construction of the BSM formulas."""
    S, K, T, r, q, sigma = 100.0, 105.0, 0.5, 0.03, 0.01, 0.25

    call = price(OptionInputs(S, K, T, r, q, sigma, "call"))
    put = price(OptionInputs(S, K, T, r, q, sigma, "put"))

    lhs = call - put
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert math.isclose(lhs, rhs, abs_tol=1e-8)


def test_call_price_matches_known_reference_value():
    """
    Textbook reference case (Hull): S=42, K=40, r=0.10, sigma=0.20, T=0.5, q=0.
    Expected call price ~= 4.76.
    """
    inp = OptionInputs(S=42, K=40, T=0.5, r=0.10, q=0.0, sigma=0.20, option_type="call")
    assert math.isclose(price(inp), 4.76, abs_tol=0.01)


def test_deep_itm_call_approaches_intrinsic_minus_discounted_strike():
    inp = OptionInputs(S=1000, K=10, T=0.1, r=0.03, q=0.0, sigma=0.2, option_type="call")
    intrinsic_pv = 1000 - 10 * math.exp(-0.03 * 0.1)
    assert math.isclose(price(inp), intrinsic_pv, rel_tol=1e-4)


def test_invalid_option_type_raises():
    import pytest
    with pytest.raises(ValueError):
        OptionInputs(S=100, K=100, T=1, r=0.03, q=0.0, sigma=0.2, option_type="straddle")
