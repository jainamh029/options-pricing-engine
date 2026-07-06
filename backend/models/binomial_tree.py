"""
Cox-Ross-Rubinstein (CRR) binomial tree pricer.

Handles American-style early exercise, which the closed-form BSM formula
cannot: at each node, price = max(exercise value, discounted expected
continuation value). Passing american=False collapses this to a European
tree, which is the convergence benchmark against BSM in
test_bsm_vs_binomial_convergence.py -- as N -> infinity, the two must agree.

Greeks are computed via finite differences (bump-and-reprice), since no
closed form exists for a tree price (unlike BSM -- see black_scholes.py
for the analytic versions used there). Theta specifically is known to be
numerically noisy ("ringing") on a binomial tree: bumping T changes the
tree's time-step size (dt = T/N) and therefore its entire node structure,
not just its endpoint. This is a textbook limitation of bump-and-reprice
theta on a tree, not a bug -- more steps reduce it but never eliminate it.
"""

import numpy as np

from .black_scholes import OptionInputs


def binomial_price(inp: OptionInputs, N: int = 200, american: bool = True) -> float:
    S, K, T, r, q, sigma = inp.S, inp.K, inp.T, inp.r, inp.q, inp.sigma
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    if not (0.0 < p < 1.0):
        raise ValueError(
            f"CRR risk-neutral probability p={p:.4f} is outside (0, 1) -- "
            "inputs imply an arbitrage-inconsistent tree (try more steps, or check r/sigma/T)"
        )

    j = np.arange(N + 1)
    terminal_prices = S * u ** j * d ** (N - j)
    if inp.option_type == "call":
        values = np.maximum(terminal_prices - K, 0.0)
    else:
        values = np.maximum(K - terminal_prices, 0.0)

    for i in range(N - 1, -1, -1):
        values = disc * (p * values[1:i + 2] + (1 - p) * values[0:i + 1])
        if american:
            j = np.arange(i + 1)
            node_prices = S * u ** j * d ** (i - j)
            exercise = (
                np.maximum(node_prices - K, 0.0) if inp.option_type == "call"
                else np.maximum(K - node_prices, 0.0)
            )
            values = np.maximum(values, exercise)

    return float(values[0])


def _bumped(inp: OptionInputs, **overrides) -> OptionInputs:
    fields = dict(S=inp.S, K=inp.K, T=inp.T, r=inp.r, q=inp.q, sigma=inp.sigma, option_type=inp.option_type)
    fields.update(overrides)
    return OptionInputs(**fields)


def binomial_greeks(inp: OptionInputs, N: int = 200, american: bool = True) -> dict:
    """Bump-and-reprice Greeks off the binomial tree. See module docstring re: theta noise."""
    h_S = inp.S * 0.01
    h_sigma = 0.01
    h_r = 0.0001
    h_T = inp.T * 0.005

    def px(**overrides):
        return binomial_price(_bumped(inp, **overrides), N=N, american=american)

    base = px()
    delta = (px(S=inp.S + h_S) - px(S=inp.S - h_S)) / (2 * h_S)
    gamma = (px(S=inp.S + h_S) - 2 * base + px(S=inp.S - h_S)) / (h_S ** 2)
    vega = (px(sigma=inp.sigma + h_sigma) - px(sigma=inp.sigma - h_sigma)) / (2 * h_sigma)
    rho = (px(r=inp.r + h_r) - px(r=inp.r - h_r)) / (2 * h_r)
    theta = (px(T=inp.T - h_T) - px(T=inp.T + h_T)) / (2 * h_T)  # -dV/dT, matching black_scholes.theta's convention

    return {"price": base, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
