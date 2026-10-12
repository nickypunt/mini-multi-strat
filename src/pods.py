"""
Step 3: the strategy "pods".

Each pod is a function that takes prices and returns TARGET WEIGHTS:
a table with one row per date and one column per ticker.
    +0.05 = 5% of the pod's capital long that stock
    -0.05 = 5% short
     0    = no position

The three pods:
    momentum_pod        - past 12-month winners vs losers, monthly
    mean_reversion_pod  - past week's losers vs winners, weekly
    ml_pod              - a LightGBM model that learns which mix of
                          signals predicts next month's winners, monthly

All pods are long/short and dollar-neutral (50% long, 50% short),
like a market-neutral hedge fund pod: they try to profit from the
DIFFERENCE between stocks, not from the market going up.

Weights on date t only use prices up to date t. The backtest (step 5)
will trade them on t+1, so there is no look-ahead bias.

Run from the repo root to preview the current positions:
    python src/pods.py
"""

import lightgbm as lgb
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


# ---------------------------------------------------------------------------
# ML pod
# ---------------------------------------------------------------------------

ML_HORIZON = 21          # predict the return over the next ~1 month
ML_MIN_TRAIN_YEARS = 3   # don't trade until the model has 3 years to learn from


def ml_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Describe every stock on every day with a handful of numbers the model
    can learn from. All use prices up to that day only.

    Each feature is converted to a cross-sectional rank between -0.5 and
    +0.5 (e.g. +0.5 = highest of all stocks that day). Ranks keep the
    model focused on how stocks compare to EACH OTHER, which is what a
    long/short pod trades, and stop calm and wild years looking different.

    Returns a long table: one row per (date, ticker), one column per feature.
    """
    daily = prices.pct_change(fill_method=None)
    raw = {
        "ret_5d": prices / prices.shift(5) - 1,           # last week
        "ret_21d": prices / prices.shift(21) - 1,         # last month
        "ret_63d": prices / prices.shift(63) - 1,         # last quarter
        "mom_12_1": prices.shift(21) / prices.shift(252) - 1,
        "vol_21d": daily.rolling(21).std(),               # recent risk
        "vol_63d": daily.rolling(63).std(),
        "dist_52w_high": prices / prices.rolling(252).max() - 1,
    }
    ranked = {name: f.rank(axis=1, pct=True) - 0.5 for name, f in raw.items()}
    return pd.concat({name: f.stack() for name, f in ranked.items()}, axis=1)


def ml_target(prices: pd.DataFrame, horizon: int = ML_HORIZON) -> pd.Series:
    """
    What the model tries to predict: each stock's rank on its return over
    the NEXT `horizon` days, starting the day after the signal (because the
    backtest trades on t+1). Only used for training, never as an input.
    """
    fwd = prices.shift(-(horizon + 1)) / prices.shift(-1) - 1
    return (fwd.rank(axis=1, pct=True) - 0.5).stack().rename("target")


def ml_signal(prices: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Walk-forward predictions, so the model never sees the future:

    At the start of each calendar year, train a fresh model on every day
    whose answer was fully known by then, then use it to score stocks for
    that whole year. A day t's answer depends on prices up to t+22, so
    the last ~22 days before the retrain are dropped from training (a
    "purge" gap). Years before ML_MIN_TRAIN_YEARS of history are skipped.
    """
    features = ml_features(prices).dropna()
    target = ml_target(prices)
    data = features.join(target)

    dates = prices.index
    feat_dates = features.index.get_level_values(0)
    first_feature_day = feat_dates.min()

    predictions = []
    for year in sorted(set(dates.year)):
        year_days = dates[dates.year == year]
        start = year_days[0]
        cutoff_pos = dates.get_loc(start) - (ML_HORIZON + 1)
        if cutoff_pos < 0:
            continue
        cutoff = dates[cutoff_pos]
        if (cutoff - first_feature_day).days < ML_MIN_TRAIN_YEARS * 365:
            continue

        train = data[feat_dates <= cutoff].dropna(subset=["target"])
        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=15,
            min_child_samples=200,           # big leaves: daily stock data is noisy
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=0, verbose=-1,
        )
        model.fit(train[features.columns], train["target"])

        this_year = features[(feat_dates >= start) & (feat_dates <= year_days[-1])]
        predictions.append(pd.Series(model.predict(this_year), index=this_year.index))

        if verbose:
            top = pd.Series(model.feature_importances_, index=features.columns)
            top = top.sort_values(ascending=False).index[:3]
            print(f"  {year}: trained on {len(train):,} rows up to {cutoff.date()}, "
                  f"top features: {', '.join(top)}")

    if not predictions:
        return pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    signal = pd.concat(predictions).unstack()
    return signal.reindex(index=prices.index, columns=prices.columns)


def ml_pod(prices: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Machine-learning pod: instead of hand-picking one signal, let a LightGBM
    model learn how to combine several (momentum, reversal, volatility,
    distance from the 52-week high) to predict next month's relative winners.
    Long the 10 highest predictions, short the 10 lowest. Rebalanced monthly.

    Expect it to overlap partly with the other pods, since it sees the same
    ingredients. Whether it adds anything beyond them is what the
    backtest's correlation numbers will tell you.
    """
    weights = long_short_weights(ml_signal(prices, verbose=verbose))
    return hold_between_rebalances(weights, "M")


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

    print("\nTraining the ML pod (one model per year)...")
    describe("ML pod (LightGBM, monthly)", ml_pod(prices, verbose=True))


if __name__ == "__main__":
    main()
