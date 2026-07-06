import math
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.data.validators import check_put_call_parity, annotate_chain


def _annotated(strikes, bids, asks, volumes, open_interests):
    df = pd.DataFrame({
        "strike": strikes,
        "bid": bids,
        "ask": asks,
        "volume": volumes,
        "openInterest": open_interests,
        "lastTradeDate": [pd.Timestamp.now(tz="UTC")] * len(strikes),
    })
    return annotate_chain(df)


def test_parity_holds_within_tolerance_for_consistent_synthetic_prices():
    S, r, q, T = 100.0, 0.03, 0.0, 0.5
    strikes = [95.0, 100.0, 105.0]

    # Construct exactly-consistent call/put mids: C = P + S*e^(-qT) - K*e^(-rT)
    put_mids = [3.0, 5.0, 8.0]
    call_mids = [p + S * math.exp(-q * T) - k * math.exp(-r * T) for p, k in zip(put_mids, strikes)]

    calls = _annotated(strikes, [c - 0.05 for c in call_mids], [c + 0.05 for c in call_mids],
                        [10, 10, 10], [10, 10, 10])
    puts = _annotated(strikes, [p - 0.05 for p in put_mids], [p + 0.05 for p in put_mids],
                       [10, 10, 10], [10, 10, 10])

    result = check_put_call_parity(calls, puts, S, r, q, T, tolerance=0.10)

    assert len(result) == 3
    assert not result["violated"].any()


def test_parity_flags_a_genuine_violation():
    S, r, q, T = 100.0, 0.03, 0.0, 0.5
    strikes = [100.0]
    calls = _annotated(strikes, [9.95], [10.05], [10], [10])   # call mid ~10.00
    puts = _annotated(strikes, [0.95], [1.05], [10], [10])     # put mid ~1.00 -- badly mispriced vs parity

    result = check_put_call_parity(calls, puts, S, r, q, T, tolerance=0.10)

    assert len(result) == 1
    assert result.iloc[0]["violated"]


def test_non_tradeable_rows_are_excluded_from_the_check():
    S, r, q, T = 100.0, 0.03, 0.0, 0.5
    strikes = [100.0]
    calls = _annotated(strikes, [9.95], [10.05], [10], [10])
    # Crossed put quote (bid > ask) -> not tradeable -> should be dropped, not flagged as a violation
    puts = _annotated(strikes, [5.0], [1.0], [10], [10])

    result = check_put_call_parity(calls, puts, S, r, q, T, tolerance=0.10)

    assert len(result) == 0
