"""
Stage 3 demo: price a real, currently listed American-style option with
the CRR binomial tree, and show the early-exercise premium over the
European (BSM) price -- using the same live inputs for both.

Run: python price_american_option.py [TICKER] [--put]

Puts are the default here on purpose: with dividends this small, an
American call's early-exercise premium is close to zero (never optimal
to early-exercise a call absent a large enough dividend), while an ITM
American put's premium is real and visible even over a short expiry.
"""

import sys
import argparse

sys.path.insert(0, "backend")

from data import market_data
from data.risk_free_rate import get_risk_free_rate
from data.validators import annotate_chain
from models.black_scholes import OptionInputs, price as bsm_price
from models.binomial_tree import binomial_price, binomial_greeks
from models.implied_vol import solve_implied_vol


def pick_itm_strike(chain_df, spot: float, option_type: str) -> float:
    """An in-the-money strike, where early-exercise premium actually shows up."""
    if option_type == "put":
        candidates = chain_df[chain_df["strike"] > spot]
        target = spot * 1.05
    else:
        candidates = chain_df[chain_df["strike"] < spot]
        target = spot * 0.95
    if candidates.empty:
        candidates = chain_df
        target = spot
    return float(min(candidates["strike"], key=lambda k: abs(k - target)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", nargs="?", default="SPY")
    parser.add_argument("--put", action="store_true", default=True)
    parser.add_argument("--call", dest="put", action="store_false")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    ticker = args.ticker
    option_type = "put" if args.put else "call"

    spot = market_data.get_spot_price(ticker)
    div_yield = market_data.get_dividend_yield(ticker)
    r = get_risk_free_rate()

    expiry = market_data.pick_default_expiry(ticker, min_days=14)
    T = market_data.years_to_expiry(expiry)

    calls_raw, puts_raw, _ = market_data.get_option_chain(ticker, expiry)
    chain = annotate_chain(puts_raw if option_type == "put" else calls_raw)
    strike = pick_itm_strike(chain, spot, option_type)
    row = chain[chain["strike"] == strike].iloc[0]

    print(f"Pricing a live ITM {ticker} {option_type} (American exercise) with the CRR binomial tree\n")
    print(f"Spot: {spot:.2f}   Strike: {strike:.2f}   Expiry: {expiry} (T={T:.4f}y)   r={r:.4%}   q={div_yield:.4%}")

    market_mid = row["mid"]
    if market_mid is not None and market_mid > 0:
        iv_result = solve_implied_vol(market_mid, spot, strike, T, r, div_yield, option_type)
        sigma = iv_result["iv"] if iv_result["converged"] else market_data.get_realized_volatility(ticker)
        sigma_source = "implied (solved from market mid)" if iv_result["converged"] else "realized (IV solve failed: " + iv_result["message"] + ")"
    else:
        sigma = market_data.get_realized_volatility(ticker)
        sigma_source = "realized (no usable market mid)"

    print(f"Volatility: {sigma:.4%}  [{sigma_source}]\n")

    inp = OptionInputs(S=spot, K=strike, T=T, r=r, q=div_yield, sigma=sigma, option_type=option_type)

    european_bsm = bsm_price(inp)
    european_tree = binomial_price(inp, N=args.steps, american=False)
    american_tree = binomial_price(inp, N=args.steps, american=True)
    early_exercise_premium = american_tree - european_tree

    print("--- Pricing comparison (same live inputs) ---")
    print(f"European (BSM, closed-form):        {european_bsm:.4f}")
    print(f"European (binomial, N={args.steps}):        {european_tree:.4f}   (sanity check vs. BSM: diff = {european_tree - european_bsm:+.5f})")
    print(f"American (binomial, N={args.steps}):        {american_tree:.4f}")
    print(f"Early-exercise premium:              {early_exercise_premium:+.4f}  ({early_exercise_premium / european_tree * 100:+.2f}% of European price)")
    print()

    greeks = binomial_greeks(inp, N=args.steps, american=True)
    print("--- American option Greeks (finite-difference off the tree) ---")
    print(f"Delta:  {greeks['delta']:.4f}")
    print(f"Gamma:  {greeks['gamma']:.6f}")
    print(f"Vega:   {greeks['vega']:.4f}   (per 1 percentage point change in sigma)")
    print(f"Theta:  {greeks['theta']:.4f}  (per calendar day -- note: theta from a tree is known to be numerically noisier than BSM's closed form, see README)")
    print(f"Rho:    {greeks['rho']:.4f}    (per 1.00 change in r)")
    print()

    print("--- Real market quote ---")
    print(f"Bid/Ask: {row['bid']:.2f} / {row['ask']:.2f}   Mid: {market_mid if market_mid else 'n/a (crossed/zero)'}")
    if market_mid:
        diff = american_tree - market_mid
        print(f"American model - Mid diff: {diff:+.4f} ({diff / market_mid * 100:+.2f}%)")
        if sigma_source.startswith("implied") and early_exercise_premium > 0.01 and abs(diff - early_exercise_premium) < 0.05:
            print(
                "\nNote: this diff is suspiciously close to the early-exercise premium itself -- "
                "that's not a coincidence. Sigma was solved by matching the EUROPEAN BSM formula to "
                "a market price that already reflects American early-exercise value, so the resulting "
                "'implied vol' is biased downward to compensate. Feeding that same vol into the "
                "American tree then double-counts exercise value. A correct pipeline would solve IV "
                "against the American tree itself (bisection on binomial_price, not bsm price) -- "
                "not implemented in this stage."
            )


if __name__ == "__main__":
    main()
