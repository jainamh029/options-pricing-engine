"""
Stage 1 demo: price a real, current option end-to-end using only live data.

Run: python price_live_option.py [TICKER] [--put]

What's live vs. computed:
  - Spot price       -> yfinance, latest close
  - Dividend yield   -> yfinance .info
  - Risk-free rate   -> FRED DGS3MO
  - Volatility       -> trailing 1y realized vol from yfinance history
                        (a proxy for now -- Stage 2 replaces this with
                        market-implied vol backed out from a real option price)
  - Strike / expiry  -> nearest available listed expiry and an
                        approximately at-the-money strike, pulled live
                        from the actual chain (not fabricated)
"""

import sys

sys.path.insert(0, "backend")

from data import market_data
from data.risk_free_rate import get_risk_free_rate
from models.black_scholes import OptionInputs, greeks


def pick_atm_strike(chain_df, spot: float) -> float:
    strikes = chain_df["strike"].values
    return float(min(strikes, key=lambda k: abs(k - spot)))


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "AAPL"
    option_type = "put" if "--put" in sys.argv else "call"

    print(f"Pricing a live {ticker} {option_type} using real market data\n")

    spot = market_data.get_spot_price(ticker)
    div_yield = market_data.get_dividend_yield(ticker)
    r = get_risk_free_rate()
    sigma = market_data.get_realized_volatility(ticker)

    # Skip expiries inside the next 7 days for this demo -- near-zero-T options
    # have unstable Greeks (vega/theta blow up) and aren't a representative
    # example. The engine itself must still handle near-expiry gracefully
    # (see the IV solver's near-expiry guard).
    expiry = market_data.pick_default_expiry(ticker, min_days=7)
    T = market_data.years_to_expiry(expiry)

    calls, puts, fetched_at = market_data.get_option_chain(ticker, expiry)
    chain_df = calls if option_type == "call" else puts
    strike = pick_atm_strike(chain_df, spot)

    row = chain_df[chain_df["strike"] == strike].iloc[0]
    market_bid = float(row["bid"])
    market_ask = float(row["ask"])
    market_mid = (market_bid + market_ask) / 2 if market_bid > 0 and market_ask > 0 else None
    market_last = float(row["lastPrice"])

    print("--- Live inputs ---")
    print(f"Spot (S):            {spot:.2f}")
    print(f"Strike (K):          {strike:.2f}  (nearest ATM strike on live chain)")
    print(f"Expiry:              {expiry}  (T = {T:.4f} years)")
    print(f"Risk-free rate (r):  {r:.4%}  (FRED DGS3MO)")
    print(f"Dividend yield (q):  {div_yield:.4%}  (yfinance)")
    print(f"Volatility (sigma):  {sigma:.4%}  (trailing 1y realized vol -- proxy for now)")
    print()

    inp = OptionInputs(S=spot, K=strike, T=T, r=r, q=div_yield, sigma=sigma, option_type=option_type)
    result = greeks(inp)

    print("--- BSM model output ---")
    print(f"Price:  {result['price']:.4f}")
    print(f"Delta:  {result['delta']:.4f}")
    print(f"Gamma:  {result['gamma']:.6f}")
    print(f"Vega:   {result['vega']:.4f}   (per 1 percentage point change in sigma)")
    print(f"Theta:  {result['theta']:.4f}  (per calendar day)")
    print(f"Rho:    {result['rho']:.4f}    (per 1 percentage point change in r)")
    print()

    print("--- Real market quote (for comparison; IV solver comes in Stage 2) ---")
    print(f"Bid/Ask:    {market_bid:.2f} / {market_ask:.2f}")
    print(f"Last trade: {market_last:.2f}")
    if market_mid is not None:
        diff = result["price"] - market_mid
        pct = (diff / market_mid * 100) if market_mid else float("nan")
        print(f"Mid:        {market_mid:.2f}")
        print(f"Model - Mid diff: {diff:+.4f} ({pct:+.2f}%)")
        print("Note: this gap is EXPECTED -- realized vol != implied vol. "
              "Stage 2 solves for the vol the market is actually pricing in.")
    else:
        print("Mid: unavailable (bid or ask is zero -- illiquid/crossed quote)")


if __name__ == "__main__":
    main()
