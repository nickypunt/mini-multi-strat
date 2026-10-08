"""
Step 3: the strategy "pods".

Each pod is a function that takes prices and returns TARGET WEIGHTS:
a table with one row per date and one column per ticker.
    +0.05 = 5% of the pod's capital long that stock
    -0.05 = 5% short
     0    = no position

Both pods here are long/short and dollar-neutral (50% long, 50% short),
like a market-neutral hedge fund pod: they try to profit from the
DIFFERENCE between stocks, not from the market going up.

Weights on date t only use prices up to date t. The backtest (step 5)
will trade them on t+1, so there is no look-ahead bias.

Run from the repo root to preview the current positions:
    python src/pods.py
"""

import pandas as pd

from data import load_prices

N_PER_SIDE = 10  # number of longs and number of shorts in each pod


def long_short_weights(signal: pd.DataFrame, n: int = N_PER_SIDE) -> pd.DataFrame:
    """
    Turn a signal (higher = more attractive) into long/short weights.
    Each day: long the top n stocks, short the bottom n, equal weight,
    50% of capital on each side.
    """
    rank_desc = signal.rank(axis=1, ascending=False)   # 1 = highest signal
    n_valid = signal.notna().sum(axis=1)

    is_long = rank_desc.le(n)
    is_short = rank_desc.gt(n_valid - n, axis=0)

    weights = is_long.astype(float) * (0.5 / n) - is_short.astype(float) * (0.5 / n)

    # Not enough stocks with data yet (early in the history) -> stay flat
    weights[n_valid < 2 * n] = 0.0
    return weights


def hold_between_rebalances(weights: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Only trade on the last trading day of each period ("M" = month,
    "W" = week) and hold those positions until the next rebalance.
    Cuts trading costs compared with changing positions every day.
    """
    periods = weights.index.to_period(freq)
    rebalance_days = weights.groupby(periods).tail(1).index
    held = weights.loc[rebalance_days].reindex(weights.index).ffill()
    return held.fillna(0.0)


def momentum_pod(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum: stocks that rose most over the past year tend to keep
    outperforming. Classic "12-1" signal: the return from 12 months ago
    to 1 month ago, skipping the latest month (which tends to reverse).
    Rebalanced monthly.
    """
    ret_12_1 = prices.shift(21) / prices.shift(252) - 1
    weights = long_short_weights(ret_12_1)
    return hold_between_rebalances(weights, "M")


def mean_reversion_pod(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Short-term mean reversion: the biggest losers of the past week tend
    to bounce, and the biggest winners tend to give some back.
    Signal = minus the 5-day return, so recent losers rank highest.
    Rebalanced weekly.
    """
    ret_5d = prices / prices.shift(5) - 1
    weights = long_short_weights(-ret_5d)
    return hold_between_rebalances(weights, "W")


def describe(name: str, weights: pd.DataFrame) -> None:
    """Print a quick summary and the latest positions for one pod."""
    active = weights[weights.abs().sum(axis=1) > 0]
    turnover = weights.diff().abs().sum(axis=1)
    latest = weights.iloc[-1]

    print(f"\n=== {name} ===")
    print(f"Active from {active.index[0].date()} to {active.index[-1].date()}")
    print(f"Net exposure (should be ~0):   {latest.sum():+.2f}")
    print(f"Gross exposure (should be 1):  {latest.abs().sum():.2f}")
    print(f"Average annual turnover:       {turnover.mean() * 252:.0f}x")
    print("Current longs: ", ", ".join(latest[latest > 0].index))
    print("Current shorts:", ", ".join(latest[latest < 0].index))


def main() -> None:
    prices = load_prices("stock")
    print(f"Loaded {prices.shape[1]} stocks, {prices.shape[0]:,} trading days")

    describe("Momentum pod (12-1, monthly)", momentum_pod(prices))
    describe("Mean-reversion pod (5-day, weekly)", mean_reversion_pod(prices))


if __name__ == "__main__":
    main()
