import math
import sys
import os

from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.models.black_scholes import OptionInputs, price, vega, theta, rho, _d1_d2


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


# --- Greek scaling regression tests -----------------------------------------
#
# vega, theta, and rho are all documented to return MARKET-CONVENTION units
# (per 1% of sigma, per calendar day, per 1% of r) rather than the raw
# calculus derivative's units (per full 1.0 change in sigma/r, per year).
# Each test below independently recomputes the RAW, unscaled textbook formula
# and checks the function divides by the documented factor. This is the
# class of test that would have caught rho shipping as the raw, unscaled
# dV/dr while its docstring/frontend tooltip already claimed "per 1
# percentage point" -- a label/value mismatch (rho was 100x too large)
# that slipped through because nothing pinned rho against its own formula.

def test_vega_is_raw_derivative_divided_by_100():
    inp = OptionInputs(S=100, K=105, T=0.5, r=0.03, q=0.01, sigma=0.25, option_type="call")
    d1, _ = _d1_d2(inp)
    raw_vega = inp.S * math.exp(-inp.q * inp.T) * norm.pdf(d1) * math.sqrt(inp.T)
    assert math.isclose(vega(inp), raw_vega / 100.0, rel_tol=1e-9)


def test_theta_is_raw_derivative_divided_by_365():
    inp = OptionInputs(S=100, K=105, T=0.5, r=0.03, q=0.01, sigma=0.25, option_type="call")
    d1, d2 = _d1_d2(inp)
    term1 = -(inp.S * math.exp(-inp.q * inp.T) * norm.pdf(d1) * inp.sigma) / (2 * math.sqrt(inp.T))
    term2 = -inp.r * inp.K * math.exp(-inp.r * inp.T) * norm.cdf(d2)
    term3 = inp.q * inp.S * math.exp(-inp.q * inp.T) * norm.cdf(d1)
    raw_annual_theta = term1 + term2 + term3
    assert math.isclose(theta(inp), raw_annual_theta / 365.0, rel_tol=1e-9)


def test_rho_is_raw_derivative_divided_by_100():
    """Regression test for the reported bug: rho shipped unscaled while its label already said /100."""
    call = OptionInputs(S=100, K=105, T=0.5, r=0.03, q=0.01, sigma=0.25, option_type="call")
    _, d2_call = _d1_d2(call)
    raw_rho_call = call.K * call.T * math.exp(-call.r * call.T) * norm.cdf(d2_call)
    assert math.isclose(rho(call), raw_rho_call / 100.0, rel_tol=1e-9)

    put = OptionInputs(S=100, K=105, T=0.5, r=0.03, q=0.01, sigma=0.25, option_type="put")
    _, d2_put = _d1_d2(put)
    raw_rho_put = -put.K * put.T * math.exp(-put.r * put.T) * norm.cdf(-d2_put)
    assert math.isclose(rho(put), raw_rho_put / 100.0, rel_tol=1e-9)


def test_rho_reproduces_reported_tsla_example():
    """
    The exact case that surfaced the bug: TSLA ATM call, K=417.50, T=0.0196,
    r=3.85%, sigma=42.60%. Before the fix this displayed 4.1118 (raw,
    unscaled); it should display ~0.0411 (raw / 100).
    """
    inp = OptionInputs(S=417.50, K=417.50, T=0.0196, r=0.0385, q=0.0, sigma=0.426, option_type="call")
    assert 0.03 < rho(inp) < 0.05
    assert rho(inp) < 1.0, "rho is still in the raw per-100%-rate-move scale, not per-1%"
