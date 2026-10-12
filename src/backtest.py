"""
Step 5: backtest the pods.

Turns each pod's target weights into a daily P&L series, charges trading
costs, and reports how each pod would have performed.

Run from the repo root (after ingest.py has built the database):
    python src/backtest.py

Writes to reports/:
    pod_returns.csv       - daily net return of each pod (the allocator's input)
    backtest_summary.csv  - the performance table printed below
    equity_curves.png     - growth of $1 and drawdowns for each pod

How the simulation works
------------------------
1. Timing: weights decided with prices up to the close of day t are held
   during day t+1, so they earn the return from close t to close t+1.
   That means we never use a price we couldn't have seen yet.
   (It assumes we can trade at day t's close, a common simplification.
   Set LAG = 2 to be stricter and trade a full day later.)
2. P&L: each day, pod return = sum over stocks of weight x stock return.
3. Costs: every time a weight changes we pay COST_BPS on the amount traded.
   Turnover = sum of |today's weight - yesterday's weight| across stocks.
   Example: flipping a 5% long into a 5% short trades 10% of capital.
4. Simplification: between rebalances we treat weights as fixed, i.e. we
   quietly rebalance back to target every day at no cost. Real positions
   drift as prices move; for monthly/weekly pods the difference is small.

The return of a dollar-neutral pod is measured on its gross capital
(50% long + 50% short = 100%). We also assume zero interest on cash, so
the Sharpe ratio below is return / volatility with no risk-free rate.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_prices
from pods import mean_reversion_pod, momentum_pod

REPORTS = Path(__file__).resolve().parent.parent / "reports"

COST_BPS = 5          # trading cost per dollar traded, in basis points (0.05%)
LAG = 1               # days between deciding a weight and holding it
TRADING_DAYS = 252

# Every pod in the fund. Add new pods here and they flow through everything.
PODS = {
    "Momentum": momentum_pod,
    "Mean reversion": mean_reversion_pod,
}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_backtest(weights: pd.DataFrame, returns: pd.DataFrame,
                 cost_bps: float = COST_BPS, lag: int = LAG) -> pd.DataFrame:
    """
    Simulate one pod. Returns a table with one row per day:
        gross     - return before costs
        cost      - trading cost paid that day (as a return, >= 0)
        net       - gross - cost
        turnover  - fraction of capital traded that day
    """
    held = weights.shift(lag).reindex(returns.index).fillna(0.0)
    rets = returns[held.columns].fillna(0.0)   # missing price -> no P&L that day

    gross = (held * rets).sum(axis=1)
    turnover = held.diff().abs().sum(axis=1)
    turnover.iloc[0] = held.iloc[0].abs().sum()   # opening the first positions
    cost = turnover * cost_bps / 10_000

    out = pd.DataFrame({"gross": gross, "cost": cost,
                        "net": gross - cost, "turnover": turnover})

    # Drop the warm-up period before the pod ever held a position
    first_trade = held.abs().sum(axis=1).gt(0).idxmax()
    return out.loc[first_trade:]


# ---------------------------------------------------------------------------
# Performance statistics
# ---------------------------------------------------------------------------

def drawdown(daily: pd.Series) -> pd.Series:
    """How far below its previous peak the account is on each day (<= 0)."""
    wealth = (1 + daily).cumprod()
    return wealth / wealth.cummax() - 1


def summarize(result: pd.DataFrame, market: pd.Series) -> dict:
    """Headline numbers for one pod (or a combination of pods)."""
    net = result["net"]
    years = len(net) / TRADING_DAYS
    growth = (1 + net).prod()

    ann_vol = net.std() * np.sqrt(TRADING_DAYS)
    sharpe = net.mean() / net.std() * np.sqrt(TRADING_DAYS) if net.std() > 0 else np.nan

    # Beta to the market: should be near 0 for a market-neutral pod
    both = pd.concat([net, market], axis=1, join="inner").dropna()
    beta = both.iloc[:, 0].cov(both.iloc[:, 1]) / both.iloc[:, 1].var()

    return {
        "Start": net.index[0].date(),
        "Annual return": growth ** (1 / years) - 1,
        "Annual vol": ann_vol,
        "Sharpe": sharpe,
        "Max drawdown": drawdown(net).min(),
        "Hit rate (days up)": (net > 0).mean(),
        "Beta to SPY": beta,
        "Turnover / yr": result["turnover"].mean() * TRADING_DAYS,
        "Cost drag / yr": result["cost"].mean() * TRADING_DAYS,
        "Gross annual return": (1 + result["gross"]).prod() ** (1 / years) - 1,
    }


def format_table(table: pd.DataFrame) -> pd.DataFrame:
    pct = ["Annual return", "Annual vol", "Max drawdown", "Hit rate (days up)",
           "Cost drag / yr", "Gross annual return"]
    shown = table.copy().astype(object)
    for col in table.columns:
        if col in pct:
            shown[col] = table[col].map(lambda x: f"{x:6.1%}")
        elif col in ("Sharpe", "Beta to SPY"):
            shown[col] = table[col].map(lambda x: f"{x:5.2f}")
        elif col == "Turnover / yr":
            shown[col] = table[col].map(lambda x: f"{x:4.1f}x")
    return shown.T   # pods as columns reads better in a terminal


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # blue, orange, aqua (fixed order)


def plot(results: dict[str, pd.DataFrame], path: Path) -> None:
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    for (name, res), color in zip(results.items(), COLORS):
        wealth = (1 + res["net"]).cumprod()
        top.plot(wealth.index, wealth, color=color, linewidth=2, label=name)
        top.annotate(f" {name}", (wealth.index[-1], wealth.iloc[-1]),
                     color="#333333", fontsize=9, va="center")
        dd = drawdown(res["net"])
        bottom.plot(dd.index, dd * 100, color=color, linewidth=1.5)

    top.axhline(1, color="#999999", linewidth=0.8)
    top.set_ylabel("Growth of $1 (after costs)")
    top.set_title("Mini Multi-Strat: pod backtest", loc="left", fontsize=12)
    top.legend(frameon=False, loc="upper left")
    bottom.set_ylabel("Drawdown (%)")

    for ax in (top, bottom):
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#bbbbbb")
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.tick_params(colors="#555555")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> None:
    prices = load_prices("stock")
    returns = prices.pct_change(fill_method=None)
    market = load_prices("etf")["SPY"].pct_change(fill_method=None)
    print(f"Loaded {prices.shape[1]} stocks, {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Costs: {COST_BPS} bps per dollar traded, lag: {LAG} day(s)\n")

    results = {name: run_backtest(pod(prices), returns) for name, pod in PODS.items()}

    # Preview of the fund: split capital 50/50 across pods, every day.
    # Step 4's allocator will replace this with something smarter.
    combo = sum(results.values()) / len(results)
    results["50/50 combo"] = combo.dropna()

    table = pd.DataFrame({name: summarize(res, market) for name, res in results.items()}).T
    print(format_table(table).to_string())

    pod_net = pd.DataFrame({name: res["net"] for name, res in results.items()
                            if name in PODS})
    corr = pod_net.dropna().corr().iloc[0, 1] if len(PODS) == 2 else None
    if corr is not None:
        print(f"\nCorrelation of daily pod returns: {corr:+.2f}  "
              "(lower = more diversification from combining them)")

    REPORTS.mkdir(exist_ok=True)
    pod_net.to_csv(REPORTS / "pod_returns.csv")
    table.to_csv(REPORTS / "backtest_summary.csv")
    plot(results, REPORTS / "equity_curves.png")
    print(f"\nSaved pod_returns.csv, backtest_summary.csv and equity_curves.png to {REPORTS}")


if __name__ == "__main__":
    main()
