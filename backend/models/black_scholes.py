"""
Black-Scholes-Merton pricing with analytic Greeks.

Includes the continuous dividend yield term (q) -- the "Merton" extension.
All Greeks below are closed-form derivatives of the BSM formula, not
finite-difference approximations (finite differences are reserved for
models without a closed form, e.g. the binomial tree / Monte Carlo).
"""

import math
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class OptionInputs:
    S: float      # spot price
    K: float      # strike price
    T: float      # time to expiry, in years
    r: float      # risk-free rate (annualized, continuously compounded)
    q: float      # continuous dividend yield (annualized)
    sigma: float  # volatility (annualized)
    option_type: str = "call"  # "call" or "put"

    def __post_init__(self):
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {self.option_type!r}")
        if self.T <= 0:
            raise ValueError("T (time to expiry) must be > 0")
        if self.sigma <= 0:
            raise ValueError("sigma (volatility) must be > 0")
        if self.S <= 0 or self.K <= 0:
            raise ValueError("S and K must be > 0")


def _d1_d2(inp: OptionInputs) -> tuple[float, float]:
    S, K, T, r, q, sigma = inp.S, inp.K, inp.T, inp.r, inp.q, inp.sigma
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def price(inp: OptionInputs) -> float:
    """BSM price with continuous dividend yield."""
    S, K, T, r, q = inp.S, inp.K, inp.T, inp.r, inp.q
    d1, d2 = _d1_d2(inp)

    if inp.option_type == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def delta(inp: OptionInputs) -> float:
    d1, _ = _d1_d2(inp)
    if inp.option_type == "call":
        return math.exp(-inp.q * inp.T) * norm.cdf(d1)
    else:
        return math.exp(-inp.q * inp.T) * (norm.cdf(d1) - 1)


def gamma(inp: OptionInputs) -> float:
    """Same for calls and puts."""
    d1, _ = _d1_d2(inp)
    return (math.exp(-inp.q * inp.T) * norm.pdf(d1)) / (inp.S * inp.sigma * math.sqrt(inp.T))


def vega(inp: OptionInputs) -> float:
    """Same for calls and puts. Returned per 1.00 (100 percentage points) change in sigma."""
    d1, _ = _d1_d2(inp)
    return inp.S * math.exp(-inp.q * inp.T) * norm.pdf(d1) * math.sqrt(inp.T)


def theta(inp: OptionInputs) -> float:
    """Returned per year. Divide by 365 for per-calendar-day decay."""
    S, K, T, r, q, sigma = inp.S, inp.K, inp.T, inp.r, inp.q, inp.sigma
    d1, d2 = _d1_d2(inp)
    term1 = -(S * math.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))

    if inp.option_type == "call":
        term2 = -r * K * math.exp(-r * T) * norm.cdf(d2)
        term3 = q * S * math.exp(-q * T) * norm.cdf(d1)
        return term1 + term2 + term3
    else:
        term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)
        term3 = -q * S * math.exp(-q * T) * norm.cdf(-d1)
        return term1 + term2 + term3


def rho(inp: OptionInputs) -> float:
    """Returned per 1.00 (100 percentage points) change in r."""
    K, T, r = inp.K, inp.T, inp.r
    _, d2 = _d1_d2(inp)
    if inp.option_type == "call":
        return K * T * math.exp(-r * T) * norm.cdf(d2)
    else:
        return -K * T * math.exp(-r * T) * norm.cdf(-d2)


def greeks(inp: OptionInputs) -> dict:
    return {
        "price": price(inp),
        "delta": delta(inp),
        "gamma": gamma(inp),
        "vega": vega(inp),
        "theta": theta(inp),
        "rho": rho(inp),
    }
