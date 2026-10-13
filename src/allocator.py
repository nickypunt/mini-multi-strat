"""
Step 4: the allocator - run the pods together as one fund.

In a real multi-manager hedge fund, each pod (portfolio manager) runs its
own book, and a central risk team decides how much capital each pod gets
and cuts pods that lose too much. This script plays that central role.

Run from the repo root (after ingest.py has built the database):
    python src/allocator.py

The core idea: RISK BUDGETS
---------------------------
The pods naturally swing by different amounts (momentum ~12% a year, ML
~9%). If every pod faced the same loss limit, that limit would mean
"normal bad stretch" for a wild pod and "something is broken" for a calm
one. So, like a real multi-manager fund, every pod is first resized to
the same risk budget - POD_RISK_BUDGET of annual volatility - and every
rule below is then set in units of that budget, identical for all pods.

Every month-end, using only information available that day:

1. RISK BUDGET SIZING.
   Each pod is scaled by POD_RISK_BUDGET / (its recent volatility), capped
   at MAX_POD_SCALE, so all pods run at roughly the same volatility.

2. ALLOCATION.
   Every live pod gets an equal share of the fund's risk (no more than
   MAX_POD_SHARE each, so the fund never leans too hard on one strategy).

3. FUND VOLATILITY TARGET.
   The whole fund is scaled so its expected volatility is TARGET_VOL a
   year, using how the pods move together (pods that zig while others zag
   lower fund risk). Total gross exposure is capped at MAX_LEVERAGE.

Every day:

4. POD DRAWDOWN LIMITS (the multi-manager "stop-out").
   Each pod's track record at its risk budget is watched. Limits are set in
   multiples of the budget, so they mean the same thing for every pod:
     - down DD_CUT from its peak      -> capital halved
     - recovers to within DD_RECOVER  -> full capital back
     - down DD_STOP from its peak     -> switched off for COOLDOWN_DAYS,
                                         then restarted with a fresh peak
   These multiples were chosen once, up front, and NOT tuned on the
   results (tuning them to the backtest would just fit the past).

5. SINGLE-STOCK LIMIT.
   The fund's combined position in any one stock is capped at
   MAX_NAME_WEIGHT of capital (long or short).

Positions from all pods are added up into ONE fund book before trading,
so if one pod buys AAPL while another sells it, the fund only trades the
difference. The script reports how much trading cost that netting saves.

To show whether the drawdown limits actually help, the fund is also run
with them switched off, and both are compared with a simple equal-weight
mix of the pods.

Writes to reports/:
    fund_returns.csv      - the fund's daily return
    fund_allocations.csv  - capital per pod each day (multiples of fund equity)
    fund_risk_events.csv  - every cut, stop-out and restart
    fund_summary.csv      - fund with limits vs without vs equal-weight
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

# --- Risk budgets -----------------------------------------------------------
POD_RISK_BUDGET = 0.10    # every pod is sized to ~10% annual volatility
POD_VOL_LOOKBACK = 126    # ~6 months of history to measure a pod's volatility
MAX_POD_SCALE = 3.0       # never lever a single pod's book more than 3x to hit it

# --- Allocation -------------------------------------------------------------
CORR_LOOKBACK = 126       # days used to measure how pods move together
MAX_POD_SHARE = 0.50      # no pod gets more than half the fund's risk

# --- Fund-level risk --------------------------------------------------------
TARGET_VOL = 0.08         # aim for 8% annual fund volatility
MAX_LEVERAGE = 3.0        # total gross exposure across all pods, x fund capital
MAX_NAME_WEIGHT = 0.10    # max 10% of capital in any one stock, long or short

# --- Pod drawdown limits, in multiples of the risk budget --------------------
DD_CUT = -0.75 * POD_RISK_BUDGET      # -7.5%: capital halved
DD_RECOVER = -0.25 * POD_RISK_BUDGET  # -2.5%: full capital back
DD_STOP = -1.5 * POD_RISK_BUDGET      # -15%:  switched off
COOLDOWN_DAYS = 63                    # ~3 months on the bench after a stop-out


# ---------------------------------------------------------------------------
# 1. Risk budget sizing
# ---------------------------------------------------------------------------

def risk_budget_scale(pod_ret: pd.DataFrame) -> pd.DataFrame:
    """
    How much to scale each pod's book so it runs at POD_RISK_BUDGET
    volatility. Set at each month-end from the pod's trailing volatility.
    NaN until a pod has POD_VOL_LOOKBACK days of history.
    """
    vol = pod_ret.rolling(POD_VOL_LOOKBACK, min_periods=POD_VOL_LOOKBACK).std() * np.sqrt(TRADING_DAYS)
    scale = (POD_RISK_BUDGET / vol).clip(upper=MAX_POD_SCALE)
    month_end = scale.groupby(scale.index.to_period("M")).tail(1).index
    return scale.loc[month_end].reindex(scale.index).ffill().where(scale.notna())


def budget_returns(pod_ret: pd.DataFrame, scale: pd.DataFrame) -> pd.DataFrame:
    """
    Each pod's daily return at its risk budget. The scale decided at day
    t-1's close sizes the positions that earn day t's return.
    """
    return pod_ret * scale.shift(1)


# ---------------------------------------------------------------------------
# 2. Allocation
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


def equal_risk_shares(budget_ret: pd.DataFrame) -> pd.DataFrame:
    """Equal share of the fund's risk for every live pod, set monthly."""
    live = budget_ret.notna().astype(float).where(budget_ret.notna())
    return hold_between_rebalances(live.apply(cap_shares, axis=1), "M")


# ---------------------------------------------------------------------------
# 3. Fund volatility target
# ---------------------------------------------------------------------------

def vol_target_leverage(budget_ret: pd.DataFrame, shares: pd.DataFrame) -> pd.Series:
    """
    How much to scale the whole fund, set at each month-end, from the
    covariance of the pods' budget returns.
    """
    leverage = pd.Series(np.nan, index=budget_ret.index)
    month_ends = budget_ret.groupby(budget_ret.index.to_period("M")).tail(1).index
    for day in month_ends:
        w = shares.loc[day]
        if w.sum() == 0:
            continue
        window = budget_ret.loc[:day].tail(CORR_LOOKBACK).fillna(0.0)
        cov = window.cov().values * TRADING_DAYS
        fund_vol = float(np.sqrt(w.values @ cov @ w.values))
        leverage[day] = TARGET_VOL / fund_vol if fund_vol > 0 else 0.0
    return leverage.ffill().fillna(0.0)


# ---------------------------------------------------------------------------
# 4. Pod drawdown limits
# ---------------------------------------------------------------------------

def drawdown_limits(budget_ret: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Daily capital multiplier per pod: 1 = full size, 0.5 = cut, 0 = stopped.
    The multiplier on day t uses the pod's returns up to and including day t;
    the backtest then trades it on t+1.

    The pod's own (paper) track keeps running while it's cut or stopped, so
    we know where it stands when it comes back. Returns (multipliers, events).
    """
    mult = pd.DataFrame(1.0, index=budget_ret.index, columns=budget_ret.columns)
    events = []
    for pod in budget_ret.columns:
        col = mult.columns.get_loc(pod)
        wealth, peak, bench_until, level = 1.0, 1.0, None, "full"
        for i, (day, r) in enumerate(budget_ret[pod].items()):
            if np.isnan(r):
                continue                       # pod not trading yet
            wealth *= 1 + r

            if bench_until is not None:        # serving a stop-out
                if i < bench_until:
                    mult.iat[i, col] = 0.0
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
            mult.iat[i, col] = new
    return mult, events


# ---------------------------------------------------------------------------
# 5. Build the fund book
# ---------------------------------------------------------------------------

def cap_names(weights: pd.DataFrame, cap: float = MAX_NAME_WEIGHT) -> tuple[pd.DataFrame, float]:
    """Clip any single stock to +/- cap. Returns the clipped weights and the
    average share of gross exposure the cap trimmed off."""
    clipped = weights.clip(-cap, cap)
    gross = weights.abs().sum(axis=1)
    active = gross > 0
    trimmed = (1 - clipped.abs().sum(axis=1)[active] / gross[active]).mean() if active.any() else 0.0
    return clipped, trimmed


def run_pods(prices: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """Run every pod once: its target weights and its standalone backtest."""
    weights = {name: pod(prices) for name, pod in PODS.items()}
    results = {name: run_backtest(w, returns) for name, w in weights.items()}
    ret = pd.DataFrame({n: r["net"] for n, r in results.items()}).reindex(prices.index)
    return {"weights": weights, "results": results, "ret": ret}


def build_fund(pods: dict, returns: pd.DataFrame, use_limits: bool = True) -> dict:
    """Size, allocate and limit the pods, then simulate the combined fund."""
    scale = risk_budget_scale(pods["ret"])
    budget_ret = budget_returns(pods["ret"], scale)

    shares = equal_risk_shares(budget_ret)
    leverage = vol_target_leverage(budget_ret, shares)
    if use_limits:
        mult, events = drawdown_limits(budget_ret)
    else:
        mult, events = pd.DataFrame(1.0, index=budget_ret.index, columns=budget_ret.columns), []

    # Capital per pod, in units of that pod's own 1x book (50% long / 50% short)
    capital = (shares.mul(leverage, axis=0) * mult * scale).fillna(0.0)
    total = capital.sum(axis=1)
    capital = capital.mul((MAX_LEVERAGE / total).clip(upper=1.0).fillna(1.0), axis=0)

    # One combined book: add up every pod's positions, scaled by its capital
    book = sum(pods["weights"][n].mul(capital[n], axis=0) for n in PODS)
    book, name_cap_trimmed = cap_names(book)
    fund = run_backtest(book, returns)

    # What trading would have cost if each pod traded on its own (no netting)
    separate_cost = sum(
        run_backtest(pods["weights"][n].mul(capital[n], axis=0), returns)["cost"]
        .reindex(fund.index).fillna(0.0)
        for n in PODS
    )

    return {"capital": capital, "scale": scale, "events": events, "book": book,
            "fund": fund, "separate_cost": separate_cost,
            "name_cap_trimmed": name_cap_trimmed}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def style(ax) -> None:
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bbbbbb")
    ax.tick_params(colors="#555555")


def plot(lines: dict[str, tuple[pd.DataFrame, str, str]], fund: pd.DataFrame,
         capital: pd.DataFrame, path: Path) -> None:
    fig, (top, mid, bottom) = plt.subplots(
        3, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 1]})

    for name, (res, color, ls) in lines.items():
        wealth = (1 + res["net"]).cumprod()
        top.plot(wealth.index, wealth, color=color, linewidth=2, linestyle=ls, label=name)
    top.axhline(1, color="#999999", linewidth=0.8)
    top.set_ylabel("Growth of $1 (after costs)")
    top.set_title("Mini Multi-Strat: the fund", loc="left", fontsize=12)
    top.legend(frameon=False, loc="upper left")

    shown = capital.loc[fund.index[0]:]
    for name, color in zip(shown.columns, COLORS):   # same pod colors as backtest.py
        mid.plot(shown.index, shown[name], color=color, linewidth=1.5, label=name)
    mid.set_ylabel("Pod capital\n(x fund equity)")
    mid.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
               ncol=len(shown.columns), borderaxespad=0.2)

    bottom.plot(fund.index, drawdown(fund["net"]) * 100, color="#1f1f1f", linewidth=1.5)
    bottom.set_ylabel("Fund drawdown (%)")

    for ax in (top, mid, bottom):
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    prices = load_prices("stock")
    returns = prices.pct_change(fill_method=None)
    market = load_prices("etf")["SPY"].pct_change(fill_method=None)
    print(f"Loaded {prices.shape[1]} stocks, {prices.index[0].date()} to {prices.index[-1].date()}")
    print("Running pods and allocator (the ML pod trains for ~10 seconds)...\n")

    pods = run_pods(prices, returns)
    out = build_fund(pods, returns, use_limits=True)
    no_limits = build_fund(pods, returns, use_limits=False)
    fund, capital = out["fund"], out["capital"]

    # Fair comparison: only count days when every pod was live and sized
    all_live = out["scale"].dropna().index[0]
    naive = (sum(pods["results"].values()) / len(PODS)).dropna().loc[all_live:]

    table = pd.DataFrame({
        "Fund with limits": summarize(fund.loc[all_live:], market),
        "Fund, no limits": summarize(no_limits["fund"].loc[all_live:], market),
        "Equal-weight pods": summarize(naive, market),
    }).T
    print(f"=== From {all_live.date()}, when every pod is live ===")
    print(format_table(table).to_string())
    print(f"\nBoth funds target {TARGET_VOL:.0%} volatility; the equal-weight mix runs unlevered,")
    print("so compare on Sharpe and drawdown, not raw return.")
    print("'Fund with limits' vs 'Fund, no limits' shows what the drawdown limits add.")

    full = summarize(fund, market)
    print(f"\nFund with limits over its full history (from {full['Start']}): "
          f"{full['Annual return']:.1%} a year, Sharpe {full['Sharpe']:.2f}, "
          f"max drawdown {full['Max drawdown']:.1%}")

    held = capital.loc[fund.index[0]:]
    print("\n=== Risk budgets ===")
    print(f"Each pod sized to {POD_RISK_BUDGET:.0%} volatility. Limits: halve at {DD_CUT:.1%}, "
          f"restore above {DD_RECOVER:.1%}, stop out at {DD_STOP:.1%}")
    print("Average scale needed per pod:  " +
          ", ".join(f"{n} {v:.2f}x" for n, v in out["scale"].mean().items()))
    print("Average capital per pod:       " +
          ", ".join(f"{n} {v:.2f}x" for n, v in held.mean().items()) + "  (x fund equity)")

    gross = out["book"].abs().sum(axis=1).loc[fund.index[0]:]
    net = out["book"].sum(axis=1).loc[fund.index[0]:]
    saved = (out["separate_cost"].sum() - fund["cost"].sum()) / len(fund) * TRADING_DAYS
    print("\n=== Fund-level risk ===")
    print(f"Average gross exposure:          {gross.mean():.2f}x capital")
    print(f"Average net exposure:            {net.mean():+.3f} (0 = market neutral)")
    print(f"Exposure trimmed by the {MAX_NAME_WEIGHT:.0%} single-stock cap: "
          f"{out['name_cap_trimmed']:.1%} of gross")
    print(f"Cost saved by netting pods' trades: {saved:.2%} a year")

    events = pd.DataFrame(out["events"], columns=["date", "pod", "event"])
    print("\n=== Pod risk events ===")
    if events.empty:
        print("None: no pod hit its drawdown limits.")
    else:
        kind = events["event"].str.extract(r"^(STOPPED OUT|capital halved)")[0]
        counts = pd.crosstab(events["pod"], kind.rename("count")).reindex(index=list(PODS), fill_value=0)
        counts.index.name, counts.columns.name = None, None
        print(counts.to_string())
        stops = events[kind == "STOPPED OUT"]
        if not stops.empty:
            print("\nStop-outs:")
            print(stops.to_string(index=False))

    REPORTS.mkdir(exist_ok=True)
    fund.to_csv(REPORTS / "fund_returns.csv")
    held.to_csv(REPORTS / "fund_allocations.csv")
    events.to_csv(REPORTS / "fund_risk_events.csv", index=False)
    table.to_csv(REPORTS / "fund_summary.csv")
    lines = {
        "Fund with limits": (fund, "#1f1f1f", "-"),
        "Fund, no limits": (no_limits["fund"], "#1f1f1f", ":"),
        "Equal-weight pods": (naive, "#888888", "-"),
    }
    plot(lines, fund, capital, REPORTS / "fund_chart.png")
    print(f"\nSaved fund_returns.csv, fund_allocations.csv, fund_risk_events.csv, "
          f"fund_summary.csv and fund_chart.png to {REPORTS}")


if __name__ == "__main__":
    main()
