"""
Implied volatility solver: given a real market price, solve for the sigma
that makes Black-Scholes-Merton reproduce it.

Brent's method (via scipy.optimize.brentq) is used rather than
Newton-Raphson. Brent's method is bracketed and guaranteed to converge
given a sign change across the bracket -- it never diverges the way
Newton-Raphson can on a bad initial guess, which matters here because the
inputs are noisy real market quotes, not clean synthetic prices.
"""

import math
from scipy.optimize import brentq

from .black_scholes import OptionInputs, price

SIGMA_LOWER = 0.001
SIGMA_UPPER = 5.0
MAX_ITER = 100
MIN_T_YEARS = 1 / 365  # near-expiry guard: below ~1 trading day, vega -> 0
                        # and small price noise maps to huge swings in IV


def solve_implied_vol(market_price: float, S: float, K: float, T: float,
                       r: float, q: float, option_type: str) -> dict:
    """
    Returns {"iv": float | None, "converged": bool, "message": str}.

    Never raises for expected failure modes (near-expiry, price outside
    the reachable range, price below intrinsic value) -- callers get a
    documented fallback message instead of a NaN or an exception.
    """
    if T < MIN_T_YEARS:
        return {"iv": None, "converged": False, "message": "IV undefined near expiry (T < 1 trading day)"}

    if market_price <= 0:
        return {"iv": None, "converged": False, "message": "No solution found -- non-positive market price"}

    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    discounted_intrinsic = intrinsic * math.exp(-r * T)
    if market_price < discounted_intrinsic - 1e-6:
        # A price below discounted intrinsic value is impossible under
        # BSM without arbitrage; in practice this means a stale or
        # crossed quote slipped past the data validators.
        return {
            "iv": None, "converged": False,
            "message": "No solution found -- price is below intrinsic value (likely a stale or crossed quote)",
        }

    def objective(sigma: float) -> float:
        inp = OptionInputs(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type=option_type)
        return price(inp) - market_price

    try:
        lo_val, hi_val = objective(SIGMA_LOWER), objective(SIGMA_UPPER)
        if lo_val * hi_val > 0:
            return {
                "iv": None, "converged": False,
                "message": f"No solution found -- price unreachable for sigma in [{SIGMA_LOWER}, {SIGMA_UPPER}] "
                           "(likely a stale or crossed quote)",
            }
        iv = brentq(objective, SIGMA_LOWER, SIGMA_UPPER, maxiter=MAX_ITER, xtol=1e-6)
        return {"iv": iv, "converged": True, "message": "ok"}
    except (ValueError, RuntimeError) as exc:
        return {"iv": None, "converged": False, "message": f"No solution found -- solver error: {exc}"}
