"""
Shared helper: load prices from the DuckDB database built by ingest.py.

Every other script (pods, allocator, backtest) imports from here, so the
data is always read the same way.
"""

from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "market.duckdb"


def load_prices(asset_type: str = "stock") -> pd.DataFrame:
    """
    Return adjusted close prices as a wide table:
    one row per date, one column per ticker.

    asset_type: "stock" (what the pods trade) or "etf" (benchmarks).
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    long = con.execute(
        """
        SELECT p.date, p.ticker, p.close
        FROM prices p
        JOIN tickers t USING (ticker)
        WHERE t.asset_type = ?
        ORDER BY p.date, p.ticker
        """,
        [asset_type],
    ).df()
    con.close()

    long["date"] = pd.to_datetime(long["date"])
    wide = long.pivot(index="date", columns="ticker", values="close")
    return wide.sort_index()
