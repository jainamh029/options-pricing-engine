import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.models.black_scholes import OptionInputs, price
from backend.models.binomial_tree import binomial_price


def test_binomial_converges_to_bsm_for_european_call():
    inp = OptionInputs(S=100, K=105, T=0.5, r=0.03, q=0.01, sigma=0.25, option_type="call")
    bsm_price = price(inp)

    errors = [abs(binomial_price(inp, N=n, american=False) - bsm_price) for n in (50, 100, 200, 500, 1000)]

    # CRR trees oscillate between odd/even N, so error isn't strictly
    # monotonic step-to-step -- but across this wide a range it must trend
    # down, and the N=1000 error must be small.
    assert errors[-1] < errors[0]
    assert errors[-1] < 0.01


def test_binomial_converges_to_bsm_for_european_put():
    inp = OptionInputs(S=100, K=95, T=1.0, r=0.02, q=0.0, sigma=0.3, option_type="put")
    bsm_price = price(inp)
    tree_price = binomial_price(inp, N=1000, american=False)
    assert math.isclose(tree_price, bsm_price, abs_tol=0.01)


def test_american_put_is_worth_at_least_as_much_as_european():
    """Early exercise can only add value (or leave it unchanged), never subtract it."""
    inp = OptionInputs(S=100, K=110, T=1.0, r=0.05, q=0.0, sigma=0.2, option_type="put")
    european = binomial_price(inp, N=300, american=False)
    american = binomial_price(inp, N=300, american=True)
    assert american >= european - 1e-9


def test_american_call_equals_european_when_no_dividends():
    """With q=0, early-exercising a call is never optimal, so American == European exactly."""
    inp = OptionInputs(S=100, K=90, T=1.0, r=0.05, q=0.0, sigma=0.2, option_type="call")
    european = binomial_price(inp, N=300, american=False)
    american = binomial_price(inp, N=300, american=True)
    assert math.isclose(american, european, abs_tol=1e-6)


def test_american_call_exceeds_european_with_dividends():
    """With q > 0, early exercise of a call just before an ex-dividend date can be optimal."""
    inp = OptionInputs(S=100, K=90, T=1.0, r=0.02, q=0.08, sigma=0.2, option_type="call")
    european = binomial_price(inp, N=300, american=False)
    american = binomial_price(inp, N=300, american=True)
    assert american > european + 1e-6


def test_invalid_tree_probability_raises():
    """An extreme sigma/T combination that pushes p outside (0, 1) should fail loudly, not silently."""
    import pytest
    inp = OptionInputs(S=100, K=100, T=1.0, r=5.0, q=0.0, sigma=0.01, option_type="call")
    with pytest.raises(ValueError):
        binomial_price(inp, N=10, american=False)
