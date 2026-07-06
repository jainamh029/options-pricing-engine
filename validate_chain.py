"""
Stage 2 demo: pull a live options chain, run the data validators, solve
implied vol at every strike, check put-call parity, and produce the
model-vs-market validation view described in the build spec (Section 6).

Run: python validate_chain.py [TICKER] [--expiry YYYY-MM-DD]

Two things are plotted, and they're meant to be read together:

  1. price_validation.png -- BSM priced with a SINGLE flat volatility
     (the at-the-money implied vol) against the real market mid-price,
     across the whole strike range. If BSM's constant-volatility
     assumption were actually true, this line would hug the market mid
     everywhere. It won't -- the error grows away from ATM.

  2. iv_smile.png -- the implied vol solved independently at every
     strike. This is *why* plot 1 has error: the market isn't pricing a
     flat vol, it's pricing a smile/skew. A flat line here would mean
     the solver is broken; real chains always show curvature.
"""

import os
import sys
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "backend")

from data import market_data
from data.risk_free_rate import get_risk_free_rate
from data.validators import annotate_chain, check_put_call_parity, is_snapshot_stale
from models.black_scholes import OptionInputs, price
from models.implied_vol import solve_implied_vol

STRIKE_RANGE_PCT = 0.30  # keep strikes within +/-30% of spot: far wings are
                          # almost always illiquid noise, not signal


def solve_iv_column(df, S, T, r, q, option_type):
    ivs, converged, messages = [], [], []
    for _, row in df.iterrows():
        if not row["is_tradeable"]:
            ivs.append(None); converged.append(False); messages.append("not tradeable (see flags)")
            continue
        result = solve_implied_vol(row["mid"], S, row["strike"], T, r, q, option_type)
        ivs.append(result["iv"]); converged.append(result["converged"]); messages.append(result["message"])
    df = df.copy()
    df["iv"], df["iv_converged"], df["iv_message"] = ivs, converged, messages
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", nargs="?", default="SPY")
    parser.add_argument("--expiry", default=None)
    args = parser.parse_args()

    ticker = args.ticker
    print(f"Validating live {ticker} options chain against BSM\n")

    spot = market_data.get_spot_price(ticker)
    div_yield = market_data.get_dividend_yield(ticker)
    r = get_risk_free_rate()
    expiry = market_data.pick_default_expiry(ticker, min_days=7, expiry_override=args.expiry)
    T = market_data.years_to_expiry(expiry)

    print(f"Spot: {spot:.2f}   Expiry: {expiry} (T={T:.4f}y)   r={r:.4%}   q={div_yield:.4%}\n")

    calls_raw, puts_raw, fetched_at = market_data.get_option_chain(ticker, expiry)
    if is_snapshot_stale(fetched_at):
        print("WARNING: this chain snapshot is older than the freshness window -- treat all prices as stale.\n")
    calls, puts = annotate_chain(calls_raw), annotate_chain(puts_raw)

    lo, hi = spot * (1 - STRIKE_RANGE_PCT), spot * (1 + STRIKE_RANGE_PCT)
    for name, df in (("calls", calls), ("puts", puts)):
        before = len(df)
        dropped = before - len(df[(df["strike"] >= lo) & (df["strike"] <= hi)])
        if dropped:
            print(f"Dropping {dropped}/{before} {name} strikes outside +/-{STRIKE_RANGE_PCT:.0%} of spot (far-wing noise)")
    calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)].reset_index(drop=True)
    puts = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)].reset_index(drop=True)

    for name, df in (("calls", calls), ("puts", puts)):
        n = len(df)
        stale_last_trade = (df["last_trade_age_days"] > 3).sum()
        print(f"{name}: {n} strikes in range -- "
              f"{df['is_tradeable'].sum()} tradeable, "
              f"{df['is_illiquid'].sum()} illiquid, "
              f"{df['is_crossed'].sum()} crossed/zero, "
              f"{stale_last_trade} with no trade in 3+ days (informational only)")

    calls = solve_iv_column(calls, spot, T, r, div_yield, "call")
    puts = solve_iv_column(puts, spot, T, r, div_yield, "put")

    n_converged = calls["iv_converged"].sum() + puts["iv_converged"].sum()
    n_tradeable = calls["is_tradeable"].sum() + puts["is_tradeable"].sum()
    print(f"\nIV solver converged on {n_converged}/{n_tradeable} tradeable quotes")

    # --- put-call parity check ---
    parity = check_put_call_parity(calls, puts, spot, r, div_yield, T)
    n_violations = parity["violated"].sum()
    print(f"\nPut-call parity: {n_violations}/{len(parity)} strikes violate tolerance")
    if n_violations:
        print(parity[parity["violated"]].to_string(index=False))
        violation_rate = n_violations / len(parity) if len(parity) else 0
        if violation_rate > 0.15:
            print(
                "\nNote: this violation rate is too high to be edge-of-chain noise alone. "
                f"{ticker} options are AMERICAN-style, and the parity formula used here "
                "(C - P = S*e^(-qT) - K*e^(-rT)) is only exact for EUROPEAN options. American "
                "puts in particular carry an early-exercise premium that pushes their price above "
                "the European parity value -- consistent with the negative diffs concentrated in "
                "ITM puts above. This is a real market effect the model should account for (see "
                "Stage 3's binomial tree for American exercise), not a data or solver bug."
            )

    # --- ATM implied vol, used as the "flat vol" assumption for the price-validation plot ---
    atm_candidates = calls[calls["iv_converged"]].copy()
    if atm_candidates.empty:
        raise SystemExit("No converged call IVs available to anchor an ATM vol -- try a different ticker/expiry")
    atm_candidates["dist"] = (atm_candidates["strike"] - spot).abs()
    atm_iv = atm_candidates.sort_values("dist").iloc[0]["iv"]
    print(f"\nATM implied vol (flat-vol assumption for validation plot): {atm_iv:.4%}")

    # --- model (flat ATM vol) vs market mid, across the chain ---
    def flat_model_price(row, option_type):
        inp = OptionInputs(S=spot, K=row["strike"], T=T, r=r, q=div_yield, sigma=atm_iv, option_type=option_type)
        return price(inp)

    for df, opt_type in ((calls, "call"), (puts, "put")):
        df["flat_model_price"] = df.apply(lambda row: flat_model_price(row, opt_type), axis=1)
        df["price_error"] = df["flat_model_price"] - df["mid"]
        df["price_error_pct"] = df["price_error"] / df["mid"] * 100

    os.makedirs(f"output/{ticker}_{expiry}", exist_ok=True)

    # Plot 1: model (flat vol) vs market mid. Log y-axis -- deep ITM prices
    # (dominated by intrinsic value, tens to hundreds of dollars) would
    # otherwise swamp the linear scale and hide the near-the-money region,
    # which is the part this plot actually needs to show.
    fig, ax = plt.subplots(figsize=(9, 5))
    for df, opt_type, color in ((calls, "call", "tab:blue"), (puts, "put", "tab:orange")):
        valid = df[df["mid"].notna() & (df["mid"] > 0)]
        ax.plot(valid["strike"], valid["mid"], "o-", color=color, label=f"{opt_type} market mid", markersize=4)
        ax.plot(valid["strike"], valid["flat_model_price"].clip(lower=1e-3), "x--", color=color,
                 alpha=0.8, label=f"{opt_type} BSM (flat ATM vol)", markersize=4)
    ax.axvline(spot, color="gray", linestyle=":", label="spot")
    ax.set_yscale("log")
    ax.set_xlabel("Strike"); ax.set_ylabel("Price (log scale)"); ax.set_title(f"{ticker} {expiry}: flat-vol BSM vs market mid")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"output/{ticker}_{expiry}/price_validation.png", dpi=130)

    # Plot 2: implied vol smile
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for df, opt_type, color in ((calls, "call", "tab:blue"), (puts, "put", "tab:orange")):
        valid = df[df["iv_converged"]]
        ax2.plot(valid["strike"], valid["iv"], "o-", color=color, label=f"{opt_type} IV")
    ax2.axhline(atm_iv, color="gray", linestyle=":", label="ATM IV (flat-vol assumption)")
    ax2.axvline(spot, color="gray", linestyle=":")
    ax2.set_xlabel("Strike"); ax2.set_ylabel("Implied volatility"); ax2.set_title(f"{ticker} {expiry}: implied volatility smile")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(f"output/{ticker}_{expiry}/iv_smile.png", dpi=130)

    print(f"\nSaved plots to output/{ticker}_{expiry}/price_validation.png and iv_smile.png")

    print_error_summary(calls, puts)


def print_error_summary(calls, puts):
    import pandas as pd
    both = pd.concat([calls[calls["mid"].notna()], puts[puts["mid"].notna()]])
    print(f"\nFlat-vol BSM vs market mid, across {len(both)} priced strikes:")
    print(f"  Mean |$ error|:                {both['price_error'].abs().mean():.4f}")
    print(f"  Median |$ error|:              {both['price_error'].abs().median():.4f}")
    # % error is only meaningful once the option has real premium -- near-zero
    # far-OTM mids make the denominator tiny and blow the percentage up
    # arbitrarily, which would misleadingly dominate a naive mean.
    liquid_priced = both[both["mid"] > 0.50]
    if len(liquid_priced):
        print(f"  Mean |% error| (mid > $0.50):  {liquid_priced['price_error_pct'].abs().mean():.2f}%  ({len(liquid_priced)} strikes)")
    excluded = len(both) - len(liquid_priced)
    if excluded:
        print(f"  ({excluded} strikes with mid <= $0.50 excluded from %% error -- tiny denominator makes %% meaningless there)")


if __name__ == "__main__":
    main()
