"""
API-layer tests using synthetic monkeypatched market data (no live network
calls, so these run in CI without hitting yfinance/FRED). The pricing math
itself is already covered by the model-level tests; these just check that
the API composes it correctly -- picks the right strike/expiry, routes to
the right model, and shapes the response.
"""

import sys
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.api import main as api_main

# ~60 days out, not decades: an expiry far in the future makes T huge, which
# makes the fixed synthetic option prices below theoretically unreachable
# under BSM (no sigma reproduces them), and the IV solver correctly refuses
# to converge on impossible data.
FAKE_EXPIRY = (datetime.now(timezone.utc) + timedelta(days=60)).strftime("%Y-%m-%d")


def _fake_chain():
    strikes = [95.0, 100.0, 105.0]
    now = pd.Timestamp.now(tz="UTC")
    calls = pd.DataFrame({
        "strike": strikes, "bid": [6.9, 3.9, 1.9], "ask": [7.1, 4.1, 2.1],
        "volume": [10, 10, 10], "openInterest": [10, 10, 10], "lastTradeDate": [now] * 3,
    })
    puts = pd.DataFrame({
        "strike": strikes, "bid": [1.9, 3.9, 6.9], "ask": [2.1, 4.1, 7.1],
        "volume": [10, 10, 10], "openInterest": [10, 10, 10], "lastTradeDate": [now] * 3,
    })
    return calls, puts


def _patch_market_data(monkeypatch):
    monkeypatch.setattr(api_main.market_data, "get_spot_price", lambda ticker: 100.0)
    monkeypatch.setattr(api_main.market_data, "get_dividend_yield", lambda ticker: 0.0)
    monkeypatch.setattr(api_main.market_data, "get_realized_volatility", lambda ticker: 0.25)
    monkeypatch.setattr(api_main.market_data, "get_option_expiries", lambda ticker: [FAKE_EXPIRY])
    monkeypatch.setattr(api_main.market_data, "get_option_chain", lambda ticker, expiry: (*_fake_chain(), 0.0))
    monkeypatch.setattr(api_main, "get_risk_free_rate", lambda: 0.03)
    monkeypatch.setattr(api_main, "is_snapshot_stale", lambda fetched_at, **kw: False)


client = TestClient(api_main.app)


def test_index():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "/price" in resp.json()["endpoints"]


def test_price_bsm(monkeypatch):
    _patch_market_data(monkeypatch)
    resp = client.get("/price", params={"ticker": "TEST", "option_type": "call", "vol_source": "realized"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["strike"] == 100.0
    assert body["sigma_source"] == "realized"
    assert body["price"] > 0


def test_price_binomial_american():
    resp = client.get("/price")  # no ticker -> should 422 (missing required query param)
    assert resp.status_code == 422


def test_price_binomial_american_with_data(monkeypatch):
    _patch_market_data(monkeypatch)
    resp = client.get("/price", params={
        "ticker": "TEST", "option_type": "put", "model": "binomial", "american": True, "vol_source": "realized",
    })
    assert resp.status_code == 200
    assert resp.json()["model"] == "binomial"


def test_price_monte_carlo_barrier_requires_params(monkeypatch):
    _patch_market_data(monkeypatch)
    resp = client.get("/price", params={
        "ticker": "TEST", "model": "monte_carlo", "mc_variant": "barrier", "vol_source": "realized",
    })
    assert resp.status_code == 400


def test_greeks_bsm(monkeypatch):
    _patch_market_data(monkeypatch)
    resp = client.get("/greeks", params={"ticker": "TEST", "vol_source": "realized"})
    assert resp.status_code == 200
    body = resp.json()
    assert 0 < body["delta"] < 1


def test_chain(monkeypatch):
    _patch_market_data(monkeypatch)
    resp = client.get("/chain", params={"ticker": "TEST", "range_pct": 0.5})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["calls"]) == 3
    assert len(body["puts"]) == 3


def test_iv_smile(monkeypatch):
    _patch_market_data(monkeypatch)
    resp = client.get("/iv-smile", params={"ticker": "TEST", "range_pct": 0.5})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["points"]) == 3
    assert body["atm_iv"] is not None


def test_invalid_model_returns_422():
    resp = client.get("/price", params={"ticker": "TEST", "model": "heston"})
    assert resp.status_code == 422
