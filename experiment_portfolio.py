"""Experiment portfolio simulator and decision rules."""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def simulate_portfolio(n: int, seed: int, **kwargs) -> pd.DataFrame:
    """Simulate an experiment portfolio with true effects attached."""
    rng = np.random.default_rng(seed)

    is_positive = rng.random(n) < 0.15
    true = np.where(
        is_positive,
        rng.normal(0.08, 0.025, n),
        rng.normal(0.00, 0.01, n)
    )

    sample_n = rng.integers(500, 20001, n)
    se = np.sqrt(2 / sample_n)
    est = true + rng.normal(0, se)
    p = 2 * stats.norm.sf(np.abs(est / se))

    power = (
        stats.norm.sf(stats.norm.ppf(0.975) - 0.08 / se)
        + stats.norm.cdf(-stats.norm.ppf(0.975) - 0.08 / se)
    )

    return pd.DataFrame({
        "true": true,
        "n": sample_n,
        "se": se,
        "est": est,
        "p": p,
        "power": power,
        "is_positive": is_positive
    })


def apply_rule(log: pd.DataFrame, rule: str) -> pd.Series:
    """Return shipping decisions under the selected rule."""
    if rule == "threshold":
        return log["p"] < 0.05
    if rule == "power_floor":
        return (log["p"] < 0.05) & (log["power"] >= 0.80)
    raise ValueError("Unknown rule")


def evaluate(log: pd.DataFrame, rule: str) -> dict:
    """Return realised return, ship count, and false-ship rate."""
    shipped = apply_rule(log, rule)
    x = log[shipped]

    return {
        "realised_return": x["true"].sum(),
        "ship_count": len(x),
        "false_ship_rate": (~x["is_positive"]).mean()
    }


if __name__ == "__main__":
    log = simulate_portfolio(200, 42)

    a = evaluate(log, "threshold")
    b = evaluate(log, "power_floor")

    print("Threshold:", a)
    print("Power floor:", b)

    assert len(log) == 200
    assert apply_rule(log, "threshold").dtype == bool
    assert b["ship_count"] <= a["ship_count"]
