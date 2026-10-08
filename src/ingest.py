"""
Step 2: download daily prices and store them in DuckDB.

Run from the repo root (with your .venv activated):
    python src/ingest.py

Creates data/market.duckdb with:
    tickers  - one row per ticker (asset type, sector)
    prices   - daily adjusted OHLCV in long format (date, ticker, ...)
    returns  - a SQL view of daily returns computed with a window function
"""

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import yfinance as yf

from universe import ETFS, START_DATE, STOCKS

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "market.duckdb"


def download_prices(tickers: list[str], start: str) -> pd.DataFrame:
    """Download adjusted daily prices and return one long table."""
    print(f"Downloading {len(tickers)} tickers from {start}...")
    raw = yf.download(
        tickers,
        start=start,
        end=date.today().isoformat(),
        auto_adjust=True,      # prices adjusted for splits and dividends
        group_by="ticker",     # raw[ticker] -> Open/High/Low/Close/Volume
        progress=True,
        threads=True,
    )

    frames = []
    missing = []
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            missing.append(t)
            continue
        df = raw[t].dropna(subset=["Close"]).copy()
        if df.empty:
            missing.append(t)
            continue
        df["ticker"] = t
        frames.append(df)

    if missing:
        print(f"WARNING: no data for {missing}")

    prices = pd.concat(frames).reset_index()
    prices = prices.rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    return prices[["date", "ticker", "open", "high", "low", "close", "volume"]]


def build_ticker_table() -> pd.DataFrame:
    rows = [{"ticker": t, "asset_type": "stock", "sector": s} for t, s in STOCKS.items()]
    rows += [{"ticker": t, "asset_type": "etf", "sector": d} for t, d in ETFS.items()]
    return pd.DataFrame(rows)


def write_to_duckdb(prices: pd.DataFrame, tickers: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    # DuckDB can query pandas DataFrames in scope by their variable name.
    con.execute("CREATE OR REPLACE TABLE tickers AS SELECT * FROM tickers")
    con.execute("CREATE OR REPLACE TABLE prices AS SELECT * FROM prices ORDER BY ticker, date")

    # Daily returns via a window function: today's close / yesterday's close - 1
    con.execute("""
        CREATE OR REPLACE VIEW returns AS
        SELECT
            date,
            ticker,
            close / LAG(close) OVER (PARTITION BY ticker ORDER BY date) - 1 AS ret
        FROM prices
    """)
    return con


def quality_checks(con: duckdb.DuckDBPyConnection) -> None:
    print("\n=== Coverage by ticker (shortest histories first) ===")
    print(con.sql("""
        SELECT ticker, COUNT(*) AS days, MIN(date) AS first_date, MAX(date) AS last_date
        FROM prices
        GROUP BY ticker
        ORDER BY days
        LIMIT 10
    """).df().to_string(index=False))

    print("\n=== Bad values (should all be 0) ===")
    print(con.sql("""
        SELECT
            SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END)  AS null_close,
            SUM(CASE WHEN close <= 0 THEN 1 ELSE 0 END)     AS nonpositive_close,
            SUM(CASE WHEN high < low THEN 1 ELSE 0 END)     AS high_below_low
        FROM prices
    """).df().to_string(index=False))

    print("\n=== Largest one-day moves (sanity check: real events, not data errors?) ===")
    print(con.sql("""
        SELECT date, ticker, ROUND(ret * 100, 1) AS pct_move
        FROM returns
        WHERE ret IS NOT NULL
        ORDER BY ABS(ret) DESC
        LIMIT 10
    """).df().to_string(index=False))


def main() -> None:
    tickers = build_ticker_table()
    prices = download_prices(tickers["ticker"].tolist(), START_DATE)
    con = write_to_duckdb(prices, tickers)

    n_rows = con.sql("SELECT COUNT(*) FROM prices").fetchone()[0]
    print(f"\nSaved {n_rows:,} rows to {DB_PATH}")

    quality_checks(con)
    con.close()


if __name__ == "__main__":
    main()
