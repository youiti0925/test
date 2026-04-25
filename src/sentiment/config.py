"""Symbol → search-query mapping and curated influencer lists.

Edit this file to tune what each collector looks for. Kept in code
rather than YAML/JSON because the lists rarely change and source-control
catches edits.
"""
from __future__ import annotations

# yfinance symbol → keywords / cashtags / subreddits to search.
SYMBOL_PROFILES: dict[str, dict] = {
    "USDJPY=X": {
        "stocktwits_symbol": "USDJPY",
        "tradingview_symbol": "USDJPY",
        "reddit_subs": ["forex", "Forex", "Daytrading"],
        "reddit_keywords": ["USDJPY", "USD/JPY", "dollar yen", "ドル円"],
        "twitter_keywords": ["$USDJPY", "USDJPY", "dollar yen"],
        "rss_keywords": ["USD/JPY", "yen", "Bank of Japan", "BOJ"],
    },
    "EURUSD=X": {
        "stocktwits_symbol": "EURUSD",
        "tradingview_symbol": "EURUSD",
        "reddit_subs": ["forex", "Forex"],
        "reddit_keywords": ["EURUSD", "EUR/USD", "euro dollar"],
        "twitter_keywords": ["$EURUSD", "EURUSD", "euro dollar"],
        "rss_keywords": ["EUR/USD", "euro", "ECB"],
    },
    "GBPUSD=X": {
        "stocktwits_symbol": "GBPUSD",
        "tradingview_symbol": "GBPUSD",
        "reddit_subs": ["forex", "Forex"],
        "reddit_keywords": ["GBPUSD", "GBP/USD", "cable"],
        "twitter_keywords": ["$GBPUSD", "GBPUSD", "cable"],
        "rss_keywords": ["GBP/USD", "pound sterling", "Bank of England", "BoE"],
    },
    "BTC-USD": {
        "stocktwits_symbol": "BTC.X",
        "tradingview_symbol": "BTCUSD",
        "reddit_subs": ["Bitcoin", "CryptoCurrency", "BitcoinMarkets"],
        "reddit_keywords": ["BTC", "Bitcoin"],
        "twitter_keywords": ["$BTC", "Bitcoin"],
        "rss_keywords": ["Bitcoin", "BTC"],
    },
    "ETH-USD": {
        "stocktwits_symbol": "ETH.X",
        "tradingview_symbol": "ETHUSD",
        "reddit_subs": ["ethereum", "ethfinance", "CryptoCurrency"],
        "reddit_keywords": ["ETH", "Ethereum"],
        "twitter_keywords": ["$ETH", "Ethereum"],
        "rss_keywords": ["Ethereum", "ETH"],
    },
}


# Curated list of "famous" Twitter / X accounts to scrape per category.
# Public profile pages only. We intentionally keep this list small so
# the per-symbol fetch stays cheap and respectful.
TWITTER_INFLUENCERS: dict[str, list[str]] = {
    "fx": [
        "ForexLive",
        "fxmacroguy",
        "Newsquawk",
        "RANsquawk",
        "WSJecon",
        "FxAnalystic",
        "GlobalProTrader",
    ],
    "macro": [
        "zerohedge",
        "LizAnnSonders",
        "biancoresearch",
        "Schuldensuehner",
        "RaoulGMI",
    ],
    "boj_jpy": [
        "tomokoamayaJP",   # JP macro
        "MlivePulse",
    ],
    "crypto": [
        "VitalikButerin",
        "cz_binance",
        "saylor",
        "elonmusk",         # noisy but moves crypto
        "APompliano",
        "WClementeIII",
    ],
}


# Map each symbol to which influencer categories are relevant.
SYMBOL_INFLUENCER_GROUPS: dict[str, list[str]] = {
    "USDJPY=X": ["fx", "macro", "boj_jpy"],
    "EURUSD=X": ["fx", "macro"],
    "GBPUSD=X": ["fx", "macro"],
    "BTC-USD":  ["crypto", "macro"],
    "ETH-USD":  ["crypto"],
}


# RSS feeds to scan globally (filtered per-symbol by keyword match).
RSS_FEEDS: list[str] = [
    "https://www.cnbc.com/id/15839135/device/rss/rss.html",       # Forex
    "https://www.cnbc.com/id/19854910/device/rss/rss.html",       # World economy
    "https://www.investing.com/rss/news_25.rss",                  # Forex news
    "https://cointelegraph.com/rss",                              # Crypto
    "https://decrypt.co/feed",                                    # Crypto
]


def profile_for(symbol: str) -> dict | None:
    return SYMBOL_PROFILES.get(symbol)


def influencers_for(symbol: str) -> list[str]:
    groups = SYMBOL_INFLUENCER_GROUPS.get(symbol, [])
    accts: list[str] = []
    for g in groups:
        accts.extend(TWITTER_INFLUENCERS.get(g, []))
    # Preserve order, drop dupes
    seen: set[str] = set()
    return [a for a in accts if not (a in seen or seen.add(a))]
