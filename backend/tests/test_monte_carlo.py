import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.models.black_scholes import OptionInputs, price
from backend.models.monte_carlo import price_european_mc, price_asian_mc, price_barrier_mc


def test_mc_converges_to_bsm_for_vanilla_call_within_confidence_interval():
    inp = OptionInputs(S=100, K=105, T=0.5, r=0.03, q=0.01, sigma=0.25, option_type="call")
    bsm = price(inp)
    result = price_european_mc(inp, n_paths=200_000, n_steps=1, antithetic=True, seed=7)
    # ~99.7% CI under CLT
    assert abs(result["price"] - bsm) < 3 * result["std_error"]


def test_antithetic_reduces_standard_error_vs_naive_sampling():
    inp = OptionInputs(S=100, K=100, T=1.0, r=0.02, q=0.0, sigma=0.3, option_type="call")
    n_paths = 20_000
    with_antithetic = price_european_mc(inp, n_paths=n_paths, n_steps=1, antithetic=True, seed=1)
    without_antithetic = price_european_mc(inp, n_paths=n_paths, n_steps=1, antithetic=False, seed=1)
    assert with_antithetic["std_error"] < without_antithetic["std_error"]


def test_asian_call_is_cheaper_than_vanilla_call():
    """Averaging reduces the effective volatility of the payoff, so an arithmetic Asian
    call is worth less than a vanilla call on the same underlying, all else equal."""
    inp = OptionInputs(S=100, K=100, T=1.0, r=0.02, q=0.0, sigma=0.3, option_type="call")
    n_steps = 50
    vanilla = price_european_mc(inp, n_paths=50_000, n_steps=n_steps, antithetic=True, seed=3)
    asian = price_asian_mc(inp, n_paths=50_000, n_steps=n_steps, antithetic=True, seed=3)
    assert asian["price"] < vanilla["price"]


def test_up_and_out_plus_up_and_in_equals_vanilla_on_shared_paths():
    """
    In/out parity: for any single path, either it breaches the barrier
    (up-and-in pays the vanilla payoff, up-and-out pays 0) or it doesn't
    (the reverse) -- so summed per-path, out + in == vanilla exactly. Using
    the same seed/n_steps/n_paths regenerates identical paths across all
    three calls, so this identity should hold to float precision, not just
    statistically.
    """
    inp = OptionInputs(S=100, K=100, T=1.0, r=0.03, q=0.0, sigma=0.2, option_type="call")
    barrier, n_steps, n_paths, seed = 130.0, 100, 5000, 123

    vanilla = price_european_mc(inp, n_paths=n_paths, n_steps=n_steps, antithetic=True, seed=seed)
    out = price_barrier_mc(inp, barrier, "up-and-out", n_paths=n_paths, n_steps=n_steps, antithetic=True, seed=seed)
    inn = price_barrier_mc(inp, barrier, "up-and-in", n_paths=n_paths, n_steps=n_steps, antithetic=True, seed=seed)

    assert math.isclose(vanilla["price"], out["price"] + inn["price"], abs_tol=1e-9)


def test_up_and_out_is_cheaper_than_vanilla():
    inp = OptionInputs(S=100, K=100, T=1.0, r=0.03, q=0.0, sigma=0.2, option_type="call")
    vanilla = price_european_mc(inp, n_paths=50_000, n_steps=100, antithetic=True, seed=9)
    knocked_out = price_barrier_mc(inp, 130.0, "up-and-out", n_paths=50_000, n_steps=100, antithetic=True, seed=9)
    assert knocked_out["price"] < vanilla["price"]


def test_invalid_barrier_type_raises():
    import pytest
    inp = OptionInputs(S=100, K=100, T=1.0, r=0.03, q=0.0, sigma=0.2, option_type="call")
    with pytest.raises(ValueError):
        price_barrier_mc(inp, 130.0, "sideways-and-confused", n_paths=1000, n_steps=10)
