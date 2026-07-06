"""
FastAPI layer over the pricing engine. This is a thin wrapper: all the
actual pricing/validation logic lives in backend/models and backend/data;
this module resolves live inputs from a request, dispatches to the right
model, and shapes the response. No pricing math happens here.

Endpoints:
  GET /                health/index
  GET /expiries         listed option expiries for a ticker
  GET /price            price an option with bsm | binomial | monte_carlo
  GET /greeks           greeks with bsm | binomial (Monte Carlo greeks are
                        not implemented in this build -- see README)
  GET /chain             live chain, annotated + per-strike implied vol
  GET /iv-smile          strike -> implied vol, for plotting the smile
"""

import math
import sys
import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import market_data
from data.market_data import MarketDataError
from data.risk_free_rate import get_risk_free_rate, RiskFreeRateError
from data.validators import annotate_chain, check_put_call_parity, is_snapshot_stale
from models.black_scholes import OptionInputs, price as bsm_price, greeks as bsm_greeks
from models.binomial_tree import binomial_price, binomial_greeks
from models.monte_carlo import price_european_mc, price_asian_mc, price_barrier_mc
from models.implied_vol import solve_implied_vol

app = FastAPI(
    title="Real-Time Options Pricing Engine",
    description=(
        "Prices options from live market data (yfinance) and FRED. Options chain data "
        "is typically ~15min delayed on the free tier; the underlying spot price is close "
        "to live. See the README for what each model assumes and where it breaks."
    ),
    version="0.1.0",
)

# Permissive CORS for local frontend development. Tighten allow_origins to
# the deployed frontend's origin before shipping this to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _safe_int(x) -> Optional[int]:
    x = _safe_float(x)
    return None if x is None else int(x)


# --- response models -------------------------------------------------------

class PriceResponse(BaseModel):
    ticker: str
    option_type: str
    model: str
    expiry: str
    time_to_expiry_years: float
    spot: float
    strike: float
    risk_free_rate: float
    dividend_yield: float
    sigma: float
    sigma_source: str
    price: float
    std_error: Optional[float] = None
    market_bid: Optional[float] = None
    market_ask: Optional[float] = None
    market_mid: Optional[float] = None
    market_tradeable: Optional[bool] = None
    model_vs_market_diff: Optional[float] = None
    notes: list[str] = []


class GreeksResponse(BaseModel):
    ticker: str
    option_type: str
    model: str
    expiry: str
    time_to_expiry_years: float
    spot: float
    strike: float
    risk_free_rate: float
    dividend_yield: float
    sigma: float
    sigma_source: str
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    notes: list[str] = []


class ChainRow(BaseModel):
    strike: float
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    volume: Optional[int]
    open_interest: Optional[int]
    last_trade_age_days: Optional[float]
    is_tradeable: bool
    is_crossed: bool
    is_illiquid: bool
    iv: Optional[float]
    iv_converged: bool
    iv_message: str


class ChainResponse(BaseModel):
    ticker: str
    expiry: str
    time_to_expiry_years: float
    spot: float
    risk_free_rate: float
    dividend_yield: float
    snapshot_stale: bool
    calls: list[ChainRow]
    puts: list[ChainRow]
    put_call_parity_checked: int
    put_call_parity_violations: int


class IVSmilePoint(BaseModel):
    strike: float
    call_iv: Optional[float]
    put_iv: Optional[float]


class IVSmileResponse(BaseModel):
    ticker: str
    expiry: str
    spot: float
    atm_iv: Optional[float]
    points: list[IVSmilePoint]


# --- shared helpers ----------------------------------------------------------

def _solve_iv_column(df, S, T, r, q, option_type):
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


def _resolve_live_inputs(ticker: str, option_type: str, expiry_override: Optional[str],
                          strike_override: Optional[float], vol_source: str) -> dict:
    expiry = market_data.pick_default_expiry(ticker, min_days=7, expiry_override=expiry_override)
    T = market_data.years_to_expiry(expiry)
    spot = market_data.get_spot_price(ticker)
    q = market_data.get_dividend_yield(ticker)
    r = get_risk_free_rate()

    calls_raw, puts_raw, fetched_at = market_data.get_option_chain(ticker, expiry)
    side = annotate_chain(calls_raw if option_type == "call" else puts_raw)

    target = strike_override if strike_override is not None else spot
    strike = float(min(side["strike"], key=lambda k: abs(k - target)))
    row = side[side["strike"] == strike].iloc[0]

    market_mid = _safe_float(row["mid"])
    notes = []
    sigma, sigma_source = None, None

    if vol_source in ("implied", "auto"):
        if market_mid is not None and market_mid > 0:
            iv_result = solve_implied_vol(market_mid, spot, strike, T, r, q, option_type)
            if iv_result["converged"]:
                sigma, sigma_source = iv_result["iv"], "implied"
            elif vol_source == "implied":
                raise ValueError(f"Implied vol solve failed: {iv_result['message']}")
            else:
                notes.append(f"Implied vol solve failed ({iv_result['message']}) -- fell back to realized vol.")
        elif vol_source == "implied":
            raise ValueError("No usable market mid to solve implied vol from (illiquid or crossed quote)")
        else:
            notes.append("No usable market mid at this strike -- fell back to realized vol.")

    if sigma is None:
        sigma = market_data.get_realized_volatility(ticker)
        sigma_source = "realized"

    if is_snapshot_stale(fetched_at):
        notes.append("Chain snapshot is older than the freshness window -- treat market fields as stale.")

    return {
        "expiry": expiry, "T": T, "spot": spot, "q": q, "r": r, "strike": strike,
        "sigma": sigma, "sigma_source": sigma_source,
        "market_bid": _safe_float(row["bid"]), "market_ask": _safe_float(row["ask"]),
        "market_mid": market_mid, "market_tradeable": bool(row["is_tradeable"]),
        "notes": notes,
    }


def _chain_row(row) -> ChainRow:
    return ChainRow(
        strike=float(row["strike"]), bid=_safe_float(row["bid"]), ask=_safe_float(row["ask"]),
        mid=_safe_float(row["mid"]), volume=_safe_int(row.get("volume")),
        open_interest=_safe_int(row.get("openInterest")),
        last_trade_age_days=_safe_float(row.get("last_trade_age_days")),
        is_tradeable=bool(row["is_tradeable"]), is_crossed=bool(row["is_crossed"]),
        is_illiquid=bool(row["is_illiquid"]), iv=_safe_float(row.get("iv")),
        iv_converged=bool(row.get("iv_converged", False)), iv_message=str(row.get("iv_message", "")),
    )


def _as_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (MarketDataError, RiskFreeRateError)):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=f"Unexpected error: {exc}")


# --- endpoints ---------------------------------------------------------------

@app.get("/")
def index():
    return {
        "name": "Real-Time Options Pricing Engine",
        "docs": "/docs",
        "endpoints": ["/expiries", "/price", "/greeks", "/chain", "/iv-smile"],
    }


@app.get("/expiries")
def get_expiries(ticker: str):
    try:
        return {"ticker": ticker.upper(), "expiries": market_data.get_option_expiries(ticker)}
    except Exception as exc:
        raise _as_http_error(exc)


@app.get("/price", response_model=PriceResponse)
def get_price(
    ticker: str,
    option_type: Literal["call", "put"] = "call",
    model: Literal["bsm", "binomial", "monte_carlo"] = "bsm",
    expiry: Optional[str] = None,
    strike: Optional[float] = None,
    vol_source: Literal["auto", "implied", "realized"] = "auto",
    american: bool = True,
    steps: int = Query(200, ge=10, le=2000),
    mc_variant: Literal["vanilla", "asian", "barrier"] = "vanilla",
    barrier: Optional[float] = None,
    barrier_type: Optional[Literal["up-and-out", "down-and-out", "up-and-in", "down-and-in"]] = None,
    n_paths: int = Query(100_000, ge=1_000, le=1_000_000),
    seed: Optional[int] = None,
):
    try:
        ctx = _resolve_live_inputs(ticker, option_type, expiry, strike, vol_source)
        inp = OptionInputs(S=ctx["spot"], K=ctx["strike"], T=ctx["T"], r=ctx["r"], q=ctx["q"],
                            sigma=ctx["sigma"], option_type=option_type)

        std_error = None
        if model == "bsm":
            result_price = bsm_price(inp)
        elif model == "binomial":
            result_price = binomial_price(inp, N=steps, american=american)
        else:  # monte_carlo
            if mc_variant == "vanilla":
                mc = price_european_mc(inp, n_paths=n_paths, n_steps=1, antithetic=True, seed=seed)
            elif mc_variant == "asian":
                mc = price_asian_mc(inp, n_paths=n_paths, n_steps=steps, antithetic=True, seed=seed)
            else:
                if barrier is None or barrier_type is None:
                    raise ValueError("barrier and barrier_type are required when mc_variant='barrier'")
                mc = price_barrier_mc(inp, barrier, barrier_type, n_paths=n_paths, n_steps=steps,
                                       antithetic=True, seed=seed)
            result_price, std_error = mc["price"], mc["std_error"]

        # The live market quote fetched in _resolve_live_inputs is always for
        # the VANILLA contract at this strike/expiry -- comparing it against
        # an Asian or barrier price would be comparing two different
        # instruments. Only surface the market fields when we actually priced
        # the vanilla payoff.
        is_vanilla_payoff = model != "monte_carlo" or mc_variant == "vanilla"
        notes = list(ctx["notes"])
        if not is_vanilla_payoff:
            notes.append(
                f"market_mid/diff omitted: it's the vanilla contract's quote, not comparable to a "
                f"{mc_variant} payoff price (there is no listed market price for this exotic)."
            )

        return PriceResponse(
            ticker=ticker.upper(), option_type=option_type, model=model, expiry=ctx["expiry"],
            time_to_expiry_years=ctx["T"], spot=ctx["spot"], strike=ctx["strike"],
            risk_free_rate=ctx["r"], dividend_yield=ctx["q"], sigma=ctx["sigma"],
            sigma_source=ctx["sigma_source"], price=result_price, std_error=std_error,
            market_bid=ctx["market_bid"] if is_vanilla_payoff else None,
            market_ask=ctx["market_ask"] if is_vanilla_payoff else None,
            market_mid=ctx["market_mid"] if is_vanilla_payoff else None,
            market_tradeable=ctx["market_tradeable"] if is_vanilla_payoff else None,
            model_vs_market_diff=(result_price - ctx["market_mid"]) if (is_vanilla_payoff and ctx["market_mid"] is not None) else None,
            notes=notes,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _as_http_error(exc)


@app.get("/greeks", response_model=GreeksResponse)
def get_greeks(
    ticker: str,
    option_type: Literal["call", "put"] = "call",
    model: Literal["bsm", "binomial"] = "bsm",
    expiry: Optional[str] = None,
    strike: Optional[float] = None,
    vol_source: Literal["auto", "implied", "realized"] = "auto",
    american: bool = True,
    steps: int = Query(200, ge=10, le=2000),
):
    try:
        ctx = _resolve_live_inputs(ticker, option_type, expiry, strike, vol_source)
        inp = OptionInputs(S=ctx["spot"], K=ctx["strike"], T=ctx["T"], r=ctx["r"], q=ctx["q"],
                            sigma=ctx["sigma"], option_type=option_type)

        g = bsm_greeks(inp) if model == "bsm" else binomial_greeks(inp, N=steps, american=american)

        return GreeksResponse(
            ticker=ticker.upper(), option_type=option_type, model=model, expiry=ctx["expiry"],
            time_to_expiry_years=ctx["T"], spot=ctx["spot"], strike=ctx["strike"],
            risk_free_rate=ctx["r"], dividend_yield=ctx["q"], sigma=ctx["sigma"],
            sigma_source=ctx["sigma_source"], delta=g["delta"], gamma=g["gamma"], vega=g["vega"],
            theta=g["theta"], rho=g["rho"], notes=ctx["notes"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _as_http_error(exc)


@app.get("/chain", response_model=ChainResponse)
def get_chain(
    ticker: str,
    expiry: Optional[str] = None,
    range_pct: float = Query(0.30, ge=0.01, le=2.0, description="Keep strikes within +/- this fraction of spot"),
):
    try:
        expiry_resolved = market_data.pick_default_expiry(ticker, min_days=7, expiry_override=expiry)
        T = market_data.years_to_expiry(expiry_resolved)
        spot = market_data.get_spot_price(ticker)
        q = market_data.get_dividend_yield(ticker)
        r = get_risk_free_rate()

        calls_raw, puts_raw, fetched_at = market_data.get_option_chain(ticker, expiry_resolved)
        calls, puts = annotate_chain(calls_raw), annotate_chain(puts_raw)

        lo, hi = spot * (1 - range_pct), spot * (1 + range_pct)
        calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)].reset_index(drop=True)
        puts = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)].reset_index(drop=True)

        calls = _solve_iv_column(calls, spot, T, r, q, "call")
        puts = _solve_iv_column(puts, spot, T, r, q, "put")

        parity = check_put_call_parity(calls, puts, spot, r, q, T)

        return ChainResponse(
            ticker=ticker.upper(), expiry=expiry_resolved, time_to_expiry_years=T, spot=spot,
            risk_free_rate=r, dividend_yield=q, snapshot_stale=is_snapshot_stale(fetched_at),
            calls=[_chain_row(row) for _, row in calls.iterrows()],
            puts=[_chain_row(row) for _, row in puts.iterrows()],
            put_call_parity_checked=len(parity), put_call_parity_violations=int(parity["violated"].sum()),
        )
    except Exception as exc:
        raise _as_http_error(exc)


@app.get("/iv-smile", response_model=IVSmileResponse)
def get_iv_smile(ticker: str, expiry: Optional[str] = None, range_pct: float = Query(0.30, ge=0.01, le=2.0)):
    try:
        expiry_resolved = market_data.pick_default_expiry(ticker, min_days=7, expiry_override=expiry)
        T = market_data.years_to_expiry(expiry_resolved)
        spot = market_data.get_spot_price(ticker)
        q = market_data.get_dividend_yield(ticker)
        r = get_risk_free_rate()

        calls_raw, puts_raw, _ = market_data.get_option_chain(ticker, expiry_resolved)
        calls, puts = annotate_chain(calls_raw), annotate_chain(puts_raw)

        lo, hi = spot * (1 - range_pct), spot * (1 + range_pct)
        calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)].reset_index(drop=True)
        puts = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)].reset_index(drop=True)

        calls = _solve_iv_column(calls, spot, T, r, q, "call")
        puts = _solve_iv_column(puts, spot, T, r, q, "put")

        strikes = sorted(set(calls["strike"]) | set(puts["strike"]))
        call_iv_by_strike = dict(zip(calls["strike"], calls["iv"]))
        put_iv_by_strike = dict(zip(puts["strike"], puts["iv"]))

        points = [
            IVSmilePoint(strike=k, call_iv=_safe_float(call_iv_by_strike.get(k)),
                         put_iv=_safe_float(put_iv_by_strike.get(k)))
            for k in strikes
        ]

        converged_calls = calls[calls["iv_converged"]]
        atm_iv = None
        if not converged_calls.empty:
            nearest = converged_calls.iloc[(converged_calls["strike"] - spot).abs().argsort().iloc[0]]
            atm_iv = _safe_float(nearest["iv"])

        return IVSmileResponse(ticker=ticker.upper(), expiry=expiry_resolved, spot=spot, atm_iv=atm_iv, points=points)
    except Exception as exc:
        raise _as_http_error(exc)
