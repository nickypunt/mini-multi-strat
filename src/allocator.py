"""
Step 4: the allocator - run the pods together as one fund.

In a real multi-manager hedge fund, each pod (portfolio manager) runs its
own book, and a central risk team decides how much capital each pod gets
and cuts pods that lose too much. This script plays that central role.

Run from the repo root (after ingest.py has built the database):
    python src/allocator.py

Every month-end, using only information available that day:

1. ALLOCATION - risk parity across pods.
   Each pod gets capital in proportion to 1 / (its recent volatility),
   so a calm pod gets more dollars than a wild one and each contributes
   roughly the same risk. No pod may get more than MAX_POD_SHARE of the
   capital, so the fund never leans too hard on one strategy.

2. FUND VOLATILITY TARGET.
   The whole book is scaled up or down so the fund's expected volatility
   is TARGET_VOL a year (based on recent pod volatilities AND how they
   move together). Leverage is capped at MAX_LEVERAGE.

Every day:

3. POD DRAWDOWN LIMITS (the multi-manager "stop-out").
   Each pod's own track record (its return on the capital it's given) is
   watched. If it falls DD_CUT below its peak, its capital is halved, and
   it only gets full capital back once it recovers to within DD_RECOVER
   of its peak - otherwise a pod hovering at the limit would be cut and
   restored every other day, trading for nothing. If it falls DD_STOP
   below its peak, it is switched off for COOLDOWN_DAYS, then restarted
   with a fresh peak.

4. SINGLE-STOCK LIMIT.
   Pods can pile into the same stock. The fund's combined position in any
   one stock is capped at MAX_NAME_WEIGHT of capital (long or short).

Positions from all pods are added up into ONE fund book before trading,
so if one pod buys AAPL while another sells it, the fund only trades the
difference. The script reports how much trading cost that netting saves.

Writes to reports/:
    fund_returns.csv      - the fund's daily return
    fund_allocations.csv  - capital per pod each day (as a share of the fund)
    fund_summary.csv      - fund vs a naive equal-weight mix of the pods
    fund_chart.png        - growth, pod capital and drawdown over time
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import (COLORS, PODS, REPORTS, TRADING_DAYS, drawdown,
                      format_table, run_backtest, summarize)
from data import load_prices
from pods import hold_between_rebalances

# --- Allocation -------------------------------------------------------------
VOL_LOOKBACK = 63         # days of history used to measure each pod's risk
CORR_LOOKBACK = 126       # days used to measure how pods move together
MAX_POD_SHARE = 0.50      # no pod gets more than half the capital

# --- Fund-level risk --------------------------------------------------------
TARGET_VOL = 0.08         # aim for 8% annual fund volatility
MAX_LEVERAGE = 3.0        # never run more than 3x the fund's capital
MAX_NAME_WEIGHT = 0.10    # max 10% of capital in any one stock, long or short

# --- Pod drawdown limits ----------------------------------------------------
DD_CUT = -0.05            # pod down 5% from its peak -> capital halved
DD_RECOVER = -0.025       # ...and gets full capital back above -2.5%
DD_STOP = -0.10           # pod down 10% from its peak -> switched off
COOLDOWN_DAYS = 63        # ~3 months on the bench after a stop-out


# ---------------------------------------------------------------------------
# 1. Allocation
# ---------------------------------------------------------------------------

def cap_shares(shares: pd.Series, cap: float = MAX_POD_SHARE) -> pd.Series:
    """
    Scale shares to sum to 1, then cap each at `cap`, handing the excess
    to the pods still below the cap. If there aren't enough pods to absorb
    it (e.g. only one pod is live), the leftover capital stays in cash.
    """
    s = shares.dropna()
    s = s[s > 0]
    if s.empty:
        return shares * 0.0
    s = s / s.sum()
    for _ in range(len(s)):
        over = s > cap + 1e-12
        if not over.any():
            break
        excess = (s[over] - cap).sum()
        s[over] = cap
        room = s < cap - 1e-12
        if not room.any():
            break
        s[room] += excess * s[room] / s[room].sum()
    return s.reindex(shares.index).fillna(0.0)


def risk_parity_shares(pod_ret: pd.DataFrame) -> pd.DataFrame:
    """Share of capital per pod, set at each month-end from trailing volatility."""
    vol = pod_ret.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std()
    shares = (1 / vol).apply(cap_shares, axis=1)
    return hold_between_rebalances(shares, "M")


# ---------------------------------------------------------------------------
# 2. Fund volatility target
# ---------------------------------------------------------------------------

def vol_target_leverage(pod_ret: pd.DataFrame, shares: pd.DataFrame) -> pd.Series:
    """
    How much to scale the whole book, set at each month-end. Uses the
    covariance of pod returns, so pods that zig while others zag count as
    less risky together than on their own.
    """
    leverage = pd.Series(np.nan, index=pod_ret.index)
    month_ends = pod_ret.groupby(pod_ret.index.to_period("M")).tail(1).index
    for day in month_ends:
        w = shares.loc[day]
        if w.sum() == 0:
            continue
        window = pod_ret.loc[:day].tail(CORR_LOOKBACK).fillna(0.0)
        cov = window.cov().values * TRADING_DAYS
        fund_vol = float(np.sqrt(w.values @ cov @ w.values))
        leverage[day] = min(TARGET_VOL / fund_vol, MAX_LEVERAGE) if fund_vol > 0 else 0.0
    return leverage.ffill().fillna(0.0)


# ---------------------------------------------------------------------------
# 3. Pod drawdown limits
# ---------------------------------------------------------------------------

def drawdown_limits(pod_ret: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Daily capital multiplier per pod: 1 = full size, 0.5 = cut, 0 = stopped.
    The multiplier on day t uses the pod's returns up to and including day t;
    the backtest then trades it on t+1.

    The pod's own (paper) track keeps running while it's stopped, so we know
    where it stands when it comes back. Returns (multipliers, event log).
    """
    mult = pd.DataFrame(1.0, index=pod_ret.index, columns=pod_ret.columns)
    events = []
    for pod in pod_ret.columns:
        wealth, peak, bench_until, level = 1.0, 1.0, None, "full"
        for i, (day, r) in enumerate(pod_ret[pod].items()):
            if np.isnan(r):
                continue                       # pod not trading yet
            wealth *= 1 + r

            if bench_until is not None:        # serving a stop-out
                if i < bench_until:
                    mult.iat[i, mult.columns.get_loc(pod)] = 0.0
                    continue
                bench_until, peak, level = None, wealth, "full"
                events.append({"date": day.date(), "pod": pod, "event": "restarted"})

            peak = max(peak, wealth)
            dd = wealth / peak - 1
            if dd <= DD_STOP:
                bench_until, new = i + 1 + COOLDOWN_DAYS, 0.0
                events.append({"date": day.date(), "pod": pod,
                               "event": f"STOPPED OUT ({dd:.1%} from peak)"})
            elif level == "full" and dd <= DD_CUT:
                level, new = "cut", 0.5
                events.append({"date": day.date(), "pod": pod,
                               "event": f"capital halved ({dd:.1%} from peak)"})
            elif level == "cut" and dd > DD_RECOVER:
                level, new = "full", 1.0
                events.append({"date": day.date(), "pod": pod, "event": "back to full size"})
            else:
                new = 0.5 if level == "cut" else 1.0
            mult.iat[i, mult.columns.get_loc(pod)] = new
    return mult, events


# ---------------------------------------------------------------------------
# 4. Build the fund book
# ---------------------------------------------------------------------------

def cap_names(weights: pd.DataFrame, cap: float = MAX_NAME_WEIGHT) -> tuple[pd.DataFrame, float]:
    """Clip any single stock to +/- cap. Returns the clipped weights and the
    average share of gross exposure the cap trimmed off."""
    clipped = weights.clip(-cap, cap)
    gross = weights.abs().sum(axis=1)
    active = gross > 0
    trimmed = (1 - clipped.abs().sum(axis=1)[active] / gross[active]).mean() if active.any() else 0.0
    return clipped, trimmed


def build_fund(prices: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """Run every pod, allocate capital, apply the risk limits, and simulate the fund."""
    pod_weights = {name: pod(prices) for name, pod in PODS.items()}
    pod_results = {name: run_backtest(w, returns) for name, w in pod_weights.items()}
    pod_ret = pd.DataFrame({n: r["net"] for n, r in pod_results.items()}).reindex(prices.index)

    shares = risk_parity_shares(pod_ret)
    leverage = vol_target_leverage(pod_ret, shares)
    mult, events = drawdown_limits(pod_ret)
    capital = shares.mul(leverage, axis=0) * mult        # share of fund capital per pod

    # One combined book: add up every pod's positions, scaled by its capital
    book = sum(pod_weights[n].mul(capital[n], axis=0) for n in PODS)
    book, name_cap_trimmed = cap_names(book)
    fund = run_backtest(book, returns)

    # What trading would have cost if each pod traded on its own (no netting)
    separate_cost = sum(
        run_backtest(pod_weights[n].mul(capital[n], axis=0), returns)["cost"]
        .reindex(fund.index).fillna(0.0)
        for n in PODS
    )

    return {"pod_results": pod_results, "pod_ret": pod_ret, "capital": capital,
            "leverage": leverage, "events": events, "book": book, "fund": fund,
            "separate_cost": separate_cost, "name_cap_trimmed": name_cap_trimmed}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def plot(fund: pd.DataFrame, naive: pd.DataFrame, capital: pd.DataFrame, path: Path) -> None:
    fig, (top, mid, bottom) = plt.subplots(
        3, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 1]})

    for res, name, color in [(fund, "Fund (allocator + risk limits)", "#1f1f1f"),
                             (naive, "Equal-weight pods, no limits", "#888888")]:
        wealth = (1 + res["net"]).cumprod()
        top.plot(wealth.index, wealth, color=color, linewidth=2, label=name)
    top.axhline(1, color="#999999", linewidth=0.8)
    top.set_ylabel("Growth of $1 (after costs)")
    top.set_title("Mini Multi-Strat: the fund", loc="left", fontsize=12)
    top.legend(frameon=False, loc="upper left")

    shown = capital.loc[fund.index[0]:]
    for name, color in zip(shown.columns, COLORS):   # same pod colors as backtest.py
        mid.plot(shown.index, shown[name] * 100, color=color, linewidth=1.5, label=name)
    mid.set_ylabel("Pod capital (% of fund,\nincl. leverage)")
    mid.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
               ncol=len(shown.columns), borderaxespad=0.2)

    bottom.plot(fund.index, drawdown(fund["net"]) * 100, color="#1f1f1f", linewidth=1.5)
    bottom.set_ylabel("Fund drawdown (%)")

    for ax in (top, mid, bottom):
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#bbbbbb")
        ax.tick_params(colors="#555555")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    prices = load_prices("stock")
    returns = prices.pct_change(fill_method=None)
    market = load_prices("etf")["SPY"].pct_change(fill_method=None)
    print(f"Loaded {prices.shape[1]} stocks, {prices.index[0].date()} to {prices.index[-1].date()}")
    print("Running pods and allocator (the ML pod trains for ~10 seconds)...\n")

    out = build_fund(prices, returns)
    fund, capital, pod_ret = out["fund"], out["capital"], out["pod_ret"]

    # Fair comparison: only count days when every pod was live
    all_live = pod_ret.dropna().index[0]
    naive = sum(out["pod_results"].values()) / len(PODS)
    naive = naive.dropna().loc[all_live:]
    fund_live = fund.loc[all_live:]

    table = pd.DataFrame({
        "Fund, full history": summarize(fund, market),
        "Fund, all pods live": summarize(fund_live, market),
        "Equal-weight, no limits": summarize(naive, market),
    }).T
    print(format_table(table).to_string())
    print("\nThe fund targets", f"{TARGET_VOL:.0%} volatility while the equal-weight mix runs",
          "unlevered, so compare them on Sharpe and drawdown, not raw return.")

    held = capital.loc[fund.index[0]:]
    print("\n=== Average capital per pod (% of fund equity, including leverage) ===")
    print(held.mean().map(lambda x: f"{x:.0%}").to_string())
    print(f"Average fund leverage: {out['leverage'].loc[fund.index[0]:].mean():.2f}x")

    gross = out["book"].abs().sum(axis=1).loc[fund.index[0]:]
    net = out["book"].sum(axis=1).loc[fund.index[0]:]
    saved = (out["separate_cost"].sum() - fund["cost"].sum()) / len(fund) * TRADING_DAYS
    print("\n=== Fund-level risk ===")
    print(f"Average gross exposure:          {gross.mean():.2f}x capital")
    print(f"Average net exposure:            {net.mean():+.3f} (0 = market neutral)")
    print(f"Exposure trimmed by the {MAX_NAME_WEIGHT:.0%} single-stock cap: "
          f"{out['name_cap_trimmed']:.1%} of gross")
    print(f"Cost saved by netting pods' trades: {saved:.2%} a year")

    print("\n=== Pod risk events ===")
    if out["events"]:
        print(pd.DataFrame(out["events"]).to_string(index=False))
    else:
        print("None: no pod hit its drawdown limits.")

    REPORTS.mkdir(exist_ok=True)
    fund.to_csv(REPORTS / "fund_returns.csv")
    capital.loc[fund.index[0]:].to_csv(REPORTS / "fund_allocations.csv")
    table.to_csv(REPORTS / "fund_summary.csv")
    plot(fund, naive, capital, REPORTS / "fund_chart.png")
    print(f"\nSaved fund_returns.csv, fund_allocations.csv, fund_summary.csv "
          f"and fund_chart.png to {REPORTS}")


if __name__ == "__main__":
    main()
