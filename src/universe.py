"""
The investable universe for Mini Multi-Strat.

~52 large, liquid US stocks across every major sector, plus 8 ETFs used as
benchmarks / macro references (not traded by the stock pods).

Note on survivorship bias: these are companies that are large *today*.
Picking them with hindsight flatters backtest results, so call this out
in the README's limitations section.
"""

STOCKS = {
    # Technology & communication
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Communication", "AMZN": "Consumer Discretionary",
    "META": "Communication", "ORCL": "Technology", "CSCO": "Technology",
    "INTC": "Technology", "ADBE": "Technology", "CRM": "Technology",
    "AMD": "Technology",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "WFC": "Financials", "C": "Financials",
    "BLK": "Financials", "AXP": "Financials",
    # Health care
    "JNJ": "Health Care", "UNH": "Health Care", "PFE": "Health Care",
    "MRK": "Health Care", "ABBV": "Health Care", "LLY": "Health Care",
    "TMO": "Health Care",
    # Consumer
    "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "WMT": "Consumer Staples",
    "COST": "Consumer Staples", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "HD": "Consumer Discretionary",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    # Industrials
    "CAT": "Industrials", "BA": "Industrials", "GE": "Industrials",
    "HON": "Industrials", "UPS": "Industrials", "DE": "Industrials",
    # Utilities, real estate, materials, telecom, media
    "NEE": "Utilities", "DUK": "Utilities", "AMT": "Real Estate",
    "LIN": "Materials", "VZ": "Communication", "T": "Communication",
    "DIS": "Communication",
}

ETFS = {
    "SPY": "US large cap",
    "QQQ": "US tech / Nasdaq 100",
    "IWM": "US small cap",
    "EFA": "Developed international",
    "EEM": "Emerging markets",
    "TLT": "Long-term Treasuries",
    "IEF": "Intermediate Treasuries",
    "GLD": "Gold",
}

START_DATE = "2015-01-01"
