"""
Monte Carlo pricer for path-dependent payoffs (Asian, barrier) that
closed-form (BSM) and lattice (binomial tree) methods can't handle
cleanly.

Uses antithetic variates -- each standard normal draw Z is paired with
-Z -- which cuts variance roughly in half for the same path count at
zero extra simulation cost. Every price returned here comes with its
standard error: a Monte Carlo estimate with no confidence interval is
not a serious estimate.

Barrier monitoring here is DISCRETE (checked at each simulated step),
not continuous. Discrete monitoring is known to systematically
underestimate true breach probability relative to continuous monitoring
(the path can cross a barrier between two discrete checkpoints and be
missed) -- this is a real, documented bias, not an oversight; using more
steps per unit time reduces it.
"""

import numpy as np

from .black_scholes import OptionInputs

VALID_BARRIER_TYPES = {"up-and-out", "down-and-out", "up-and-in", "down-and-in"}


def _simulate_paths(S0, r, q, sigma, T, n_steps, n_paths, antithetic, rng):
    """
    Returns (paths, n_pairs). When antithetic=True, paths is ordered as
    [Z-block; -Z-block] stacked along axis 0 -- row i and row (n_pairs + i)
    are a Z/-Z pair. n_paths is floored to the nearest even number in that
    case, to keep every path exactly paired (no leftover unpaired path).
    """
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    if antithetic:
        n_pairs = n_paths // 2
        if n_pairs < 1:
            raise ValueError("antithetic sampling requires n_paths >= 2")
        Z = rng.standard_normal((n_pairs, n_steps))
        Z = np.concatenate([Z, -Z], axis=0)
    else:
        n_pairs = None
        Z = rng.standard_normal((n_paths, n_steps))

    log_increments = drift + diffusion * Z
    log_paths = np.cumsum(log_increments, axis=1)
    n_used = Z.shape[0]
    paths = S0 * np.exp(np.concatenate([np.zeros((n_used, 1)), log_paths], axis=1))
    return paths, n_pairs


def _price_from_payoffs(payoffs: np.ndarray, r: float, T: float, n_pairs) -> dict:
    """
    n_pairs is not None for antithetic sampling: the variance reduction from
    pairing Z with -Z only shows up when the estimator is built from
    PER-PAIR averages, not from the naive std/sqrt(n) over all individual
    (correlated-by-construction) paths -- the latter silently discards the
    entire benefit of antithetic variates while still calling itself
    "antithetic."
    """
    discounted = np.exp(-r * T) * payoffs
    if n_pairs is not None:
        pair_avg = (discounted[:n_pairs] + discounted[n_pairs:2 * n_pairs]) / 2.0
        price = float(pair_avg.mean())
        std_error = float(pair_avg.std(ddof=1) / np.sqrt(n_pairs))
        n_paths_used = 2 * n_pairs
    else:
        price = float(discounted.mean())
        std_error = float(discounted.std(ddof=1) / np.sqrt(len(discounted)))
        n_paths_used = len(discounted)
    return {"price": price, "std_error": std_error, "n_paths": n_paths_used}


def price_european_mc(inp: OptionInputs, n_paths: int = 100_000, n_steps: int = 1,
                       antithetic: bool = True, seed=None) -> dict:
    """Plain European payoff via simulation -- serves as the sanity check against BSM."""
    rng = np.random.default_rng(seed)
    paths, n_pairs = _simulate_paths(inp.S, inp.r, inp.q, inp.sigma, inp.T, n_steps, n_paths, antithetic, rng)
    ST = paths[:, -1]
    payoffs = np.maximum(ST - inp.K, 0.0) if inp.option_type == "call" else np.maximum(inp.K - ST, 0.0)
    return _price_from_payoffs(payoffs, inp.r, inp.T, n_pairs)


def price_asian_mc(inp: OptionInputs, n_paths: int = 100_000, n_steps: int = 252,
                    antithetic: bool = True, seed=None) -> dict:
    """Arithmetic-average-price Asian option. No closed form exists for arithmetic averaging."""
    rng = np.random.default_rng(seed)
    paths, n_pairs = _simulate_paths(inp.S, inp.r, inp.q, inp.sigma, inp.T, n_steps, n_paths, antithetic, rng)
    avg_price = paths[:, 1:].mean(axis=1)  # excludes t=0, the standard convention
    payoffs = np.maximum(avg_price - inp.K, 0.0) if inp.option_type == "call" else np.maximum(inp.K - avg_price, 0.0)
    return _price_from_payoffs(payoffs, inp.r, inp.T, n_pairs)


def price_barrier_mc(inp: OptionInputs, barrier: float, barrier_type: str, n_paths: int = 100_000,
                      n_steps: int = 252, antithetic: bool = True, seed=None) -> dict:
    """
    barrier_type: one of 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'.
    Vanilla payoff at expiry, contingent on whether the barrier was breached
    along the (discretely monitored) path.
    """
    if barrier_type not in VALID_BARRIER_TYPES:
        raise ValueError(f"barrier_type must be one of {VALID_BARRIER_TYPES}, got {barrier_type!r}")

    rng = np.random.default_rng(seed)
    paths, n_pairs = _simulate_paths(inp.S, inp.r, inp.q, inp.sigma, inp.T, n_steps, n_paths, antithetic, rng)
    ST = paths[:, -1]
    vanilla_payoff = np.maximum(ST - inp.K, 0.0) if inp.option_type == "call" else np.maximum(inp.K - ST, 0.0)

    breached = (paths >= barrier).any(axis=1) if barrier_type.startswith("up") else (paths <= barrier).any(axis=1)

    if barrier_type.endswith("out"):
        payoffs = np.where(breached, 0.0, vanilla_payoff)
    else:  # "...-in"
        payoffs = np.where(breached, vanilla_payoff, 0.0)

    return _price_from_payoffs(payoffs, inp.r, inp.T, n_pairs)
