"""
Stage 4 demo: price path-dependent payoffs (Asian, barrier) via Monte
Carlo, using live-derived inputs (spot, r, q, and implied vol solved
from a real market quote). These payoffs have no listed market quote to
compare against directly -- unlike Stages 1-3, there's no exchange price
for "the arithmetic-average AAPL call" -- so this demo validates the
engine internally instead: a vanilla MC price against BSM, and a
variance-reduction comparison, both against the SAME live-derived inputs
used to price the exotics.

Run: python price_exotic_option.py [TICKER]
"""

import sys
import argparse

sys.path.insert(0, "backend")

from data import market_data
from data.risk_free_rate import get_risk_free_rate
from data.validators import annotate_chain
from models.black_scholes import OptionInputs, price as bsm_price
from models.implied_vol import solve_implied_vol
from models.monte_carlo import price_european_mc, price_asian_mc, price_barrier_mc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", nargs="?", default="AAPL")
    parser.add_argument("--n-paths", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ticker = args.ticker
    spot = market_data.get_spot_price(ticker)
    div_yield = market_data.get_dividend_yield(ticker)
    r = get_risk_free_rate()

    expiry = market_data.pick_default_expiry(ticker, min_days=30)
    T = market_data.years_to_expiry(expiry)

    calls_raw, _, _ = market_data.get_option_chain(ticker, expiry)
    calls = annotate_chain(calls_raw)
    atm_row = calls.iloc[(calls["strike"] - spot).abs().argsort().iloc[0]]
    strike = float(atm_row["strike"])

    iv_result = solve_implied_vol(atm_row["mid"], spot, strike, T, r, div_yield, "call")
    sigma = iv_result["iv"] if iv_result["converged"] else market_data.get_realized_volatility(ticker)
    sigma_source = "implied (ATM)" if iv_result["converged"] else "realized (IV solve failed)"

    print(f"Live inputs for {ticker}: spot={spot:.2f}  K={strike:.2f}  expiry={expiry} (T={T:.4f}y)")
    print(f"r={r:.4%}  q={div_yield:.4%}  sigma={sigma:.4%}  [{sigma_source}]\n")

    inp = OptionInputs(S=spot, K=strike, T=T, r=r, q=div_yield, sigma=sigma, option_type="call")

    # --- sanity check: MC vanilla vs BSM closed form ---
    bsm = bsm_price(inp)
    mc_vanilla = price_european_mc(inp, n_paths=args.n_paths, n_steps=1, antithetic=True, seed=args.seed)
    print("--- Sanity check: MC vanilla call vs. BSM closed form ---")
    print(f"BSM price:        {bsm:.4f}")
    print(f"MC price:         {mc_vanilla['price']:.4f}  +/- {1.96 * mc_vanilla['std_error']:.4f}  (95% CI, {mc_vanilla['n_paths']:,} paths)")
    within_ci = abs(mc_vanilla["price"] - bsm) < 1.96 * mc_vanilla["std_error"]
    print(f"BSM within 95% CI of MC estimate: {within_ci}\n")

    # --- variance reduction: antithetic vs naive, same path budget ---
    naive = price_european_mc(inp, n_paths=args.n_paths, n_steps=1, antithetic=False, seed=args.seed)
    reduction_pct = (1 - mc_vanilla["std_error"] / naive["std_error"]) * 100
    print("--- Antithetic variance reduction (same path budget, same seed) ---")
    print(f"Naive std error:      {naive['std_error']:.6f}")
    print(f"Antithetic std error: {mc_vanilla['std_error']:.6f}")
    print(f"Reduction: {reduction_pct:.1f}%\n")

    # --- Asian option: no closed form, no listed market quote to check against ---
    n_steps = max(int(T * 252), 5)  # ~ one simulated observation per trading day up to expiry
    asian = price_asian_mc(inp, n_paths=args.n_paths, n_steps=n_steps, antithetic=True, seed=args.seed)
    print(f"--- Arithmetic Asian call (avg over {n_steps} simulated observations) ---")
    print(f"Price: {asian['price']:.4f}  +/- {1.96 * asian['std_error']:.4f}  (95% CI)")
    print(f"Cheaper than vanilla by: {bsm - asian['price']:.4f} ({(bsm - asian['price']) / bsm * 100:.1f}%) "
          f"-- expected: averaging suppresses effective volatility\n")

    # --- Up-and-out barrier call, barrier set 15% above spot ---
    barrier = spot * 1.15
    knockout = price_barrier_mc(inp, barrier, "up-and-out", n_paths=args.n_paths, n_steps=n_steps,
                                 antithetic=True, seed=args.seed)
    print(f"--- Up-and-out call (barrier = {barrier:.2f}, +15% of spot) ---")
    print(f"Price: {knockout['price']:.4f}  +/- {1.96 * knockout['std_error']:.4f}  (95% CI)")
    print(f"Cheaper than vanilla by: {bsm - knockout['price']:.4f} ({(bsm - knockout['price']) / bsm * 100:.1f}%) "
          f"-- expected: knock-out risk strictly reduces value vs. vanilla")
    print("Note: barrier is monitored at each simulated step (discrete), not continuously -- "
          "discrete monitoring understates true breach probability, which biases this price slightly high "
          "relative to a continuously-monitored barrier. More steps reduce, but don't eliminate, this bias.")


if __name__ == "__main__":
    main()
