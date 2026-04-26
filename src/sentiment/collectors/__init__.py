"""Per-source collectors. Each returns a list of base.Post."""
from .reddit import RedditCollector
from .stocktwits import StocktwitsCollector
from .tradingview import TradingViewCollector
from .twitter import TwitterCollector
from .rss import RSSCollector

__all__ = [
    "RedditCollector",
    "StocktwitsCollector",
    "TradingViewCollector",
    "TwitterCollector",
    "RSSCollector",
]
