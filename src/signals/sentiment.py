"""
Sentiment Scoring Layer.

Parses RSS feeds from 12+ crypto news sources and scores sentiment
using keyword analysis with source credibility weighting.

Score range: -100 to +100
Positive = bullish sentiment dominance
Negative = bearish sentiment dominance
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import feedparser

logger = logging.getLogger(__name__)


# Bullish keywords with intensity weights (1-3)
BULLISH_KEYWORDS = {
    # Strong bullish (weight 3)
    "rally": 3, "surge": 3, "soar": 3, "breakout": 3, "all-time high": 3,
    "ath": 3, "moon": 3, "parabolic": 3, "explosive": 3, "skyrocket": 3,
    # Moderate bullish (weight 2)
    "bullish": 2, "pump": 2, "gain": 2, "rise": 2, "jump": 2,
    "recovery": 2, "bounce": 2, "uptick": 2, "outperform": 2, "adoption": 2,
    "institutional": 2, "accumulate": 2, "accumulation": 2, "buy": 2,
    "upgrade": 2, "approval": 2, "etf": 2, "milestone": 2,
    # Mild bullish (weight 1)
    "positive": 1, "growth": 1, "optimistic": 1, "confidence": 1,
    "support": 1, "uptrend": 1, "green": 1, "higher": 1, "strong": 1,
    "partnership": 1, "launch": 1, "integrate": 1, "expand": 1,
    "invest": 1, "fund": 1, "backing": 1, "innovation": 1,
}

# Bearish keywords with intensity weights (1-3)
BEARISH_KEYWORDS = {
    # Strong bearish (weight 3)
    "crash": 3, "collapse": 3, "plunge": 3, "dump": 3, "capitulation": 3,
    "liquidation": 3, "bankrupt": 3, "fraud": 3, "scam": 3, "hack": 3,
    "exploit": 3, "rug pull": 3, "ponzi": 3,
    # Moderate bearish (weight 2)
    "bearish": 2, "sell-off": 2, "selloff": 2, "decline": 2, "drop": 2,
    "fall": 2, "correction": 2, "regulation": 2, "ban": 2, "crackdown": 2,
    "lawsuit": 2, "sec": 2, "enforcement": 2, "fine": 2, "fear": 2,
    "uncertainty": 2, "risk": 2, "warning": 2,
    # Mild bearish (weight 1)
    "negative": 1, "bearish": 1, "weak": 1, "lower": 1, "red": 1,
    "concern": 1, "caution": 1, "volatile": 1, "downtrend": 1,
    "resistance": 1, "overbought": 1, "bubble": 1, "delay": 1,
    "postpone": 1, "reject": 1, "struggle": 1,
}

# Symbol-specific keywords for targeted sentiment
SYMBOL_KEYWORDS = {
    "BTCUSDT": ["bitcoin", "btc", "satoshi"],
    "ETHUSDT": ["ethereum", "eth", "vitalik", "merge", "layer 2"],
    "BNBUSDT": ["bnb", "binance", "cz"],
    "SOLUSDT": ["solana", "sol"],
    "AVAXUSDT": ["avalanche", "avax"],
    "LINKUSDT": ["chainlink", "link", "oracle"],
    "DOTUSDT": ["polkadot", "dot", "parachain"],
    "ADAUSDT": ["cardano", "ada"],
    "MATICUSDT": ["polygon", "matic"],
    "DOGEUSDT": ["doge", "dogecoin", "elon"],
    "1000PEPEUSDT": ["pepe"],
    "1000SHIBUSDT": ["shib", "shiba"],
    "1000FLOKIUSDT": ["floki"],
    "1000BONKUSDT": ["bonk"],
}


@dataclass
class ArticleSentiment:
    """Sentiment analysis result for a single article."""
    title: str
    source: str
    published: Optional[datetime]
    bullish_score: float
    bearish_score: float
    net_score: float
    matched_keywords: List[str]
    relevance: float  # 0-1, how relevant to specific symbol


@dataclass
class SentimentCache:
    """Cached sentiment state."""
    articles: List[ArticleSentiment] = field(default_factory=list)
    last_update: Optional[datetime] = None
    feed_status: Dict[str, str] = field(default_factory=dict)


class SentimentAnalyzer:
    """
    RSS-based crypto sentiment analyzer.

    Methodology:
    1. Fetch articles from 12+ RSS feeds
    2. Score each article's title and summary for bullish/bearish keywords
    3. Weight by source credibility and article recency
    4. For symbol-specific scores, filter by symbol keywords
    5. Aggregate into composite sentiment score
    """

    def __init__(self, feeds: Optional[List[Dict]] = None, max_age_hours: int = 24):
        self.feeds = feeds or self._default_feeds()
        self.max_age_hours = max_age_hours
        self._cache = SentimentCache()

    @staticmethod
    def _default_feeds() -> List[Dict]:
        return [
            {"url": "https://www.coindesk.com/arc/outboundfeeds/rss/", "weight": 1.0, "name": "CoinDesk"},
            {"url": "https://cointelegraph.com/rss", "weight": 1.0, "name": "CoinTelegraph"},
            {"url": "https://www.theblock.co/rss.xml", "weight": 0.9, "name": "TheBlock"},
            {"url": "https://decrypt.co/feed", "weight": 0.8, "name": "Decrypt"},
            {"url": "https://bitcoinmagazine.com/.rss/full/", "weight": 0.7, "name": "BitcoinMagazine"},
            {"url": "https://cryptoslate.com/feed/", "weight": 0.7, "name": "CryptoSlate"},
            {"url": "https://beincrypto.com/feed/", "weight": 0.6, "name": "BeInCrypto"},
            {"url": "https://cryptopotato.com/feed/", "weight": 0.6, "name": "CryptoPotato"},
            {"url": "https://u.today/rss", "weight": 0.5, "name": "UToday"},
            {"url": "https://ambcrypto.com/feed/", "weight": 0.5, "name": "AMBCrypto"},
            {"url": "https://www.newsbtc.com/feed/", "weight": 0.5, "name": "NewsBTC"},
            {"url": "https://bitcoinist.com/feed/", "weight": 0.5, "name": "Bitcoinist"},
        ]

    def update_feeds(self):
        """
        Fetch and parse all RSS feeds, updating the sentiment cache.

        This is a synchronous operation (feedparser does its own HTTP).
        Call this from the scheduler every 5 minutes.
        """
        logger.info("Updating sentiment feeds (%d sources)...", len(self.feeds))
        all_articles = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.max_age_hours)

        for feed_config in self.feeds:
            feed_name = feed_config["name"]
            feed_url = feed_config["url"]
            source_weight = feed_config["weight"]

            try:
                feed = feedparser.parse(feed_url)

                if feed.bozo and not feed.entries:
                    self._cache.feed_status[feed_name] = f"error: {feed.bozo_exception}"
                    logger.warning("Feed %s parse error: %s", feed_name, feed.bozo_exception)
                    continue

                count = 0
                for entry in feed.entries[:20]:  # Max 20 per feed
                    title = entry.get("title", "")
                    summary = entry.get("summary", entry.get("description", ""))
                    text = f"{title} {summary}".lower()

                    # Parse publication date
                    published = self._parse_date(entry)
                    if published and published < cutoff:
                        continue

                    # Score the article
                    bull_score, bear_score, matched = self._score_text(text)

                    # Apply source credibility weight
                    bull_score *= source_weight
                    bear_score *= source_weight

                    # Apply recency decay
                    if published:
                        age_hours = (datetime.now(timezone.utc) - published).total_seconds() / 3600
                        recency_factor = max(0.3, 1.0 - (age_hours / self.max_age_hours) * 0.7)
                        bull_score *= recency_factor
                        bear_score *= recency_factor

                    net = bull_score - bear_score

                    article = ArticleSentiment(
                        title=title[:200],
                        source=feed_name,
                        published=published,
                        bullish_score=bull_score,
                        bearish_score=bear_score,
                        net_score=net,
                        matched_keywords=matched,
                        relevance=1.0,  # Will be adjusted per-symbol
                    )
                    all_articles.append(article)
                    count += 1

                self._cache.feed_status[feed_name] = f"ok ({count} articles)"
                logger.debug("Feed %s: %d articles processed", feed_name, count)

            except Exception as e:
                self._cache.feed_status[feed_name] = f"error: {str(e)[:100]}"
                logger.error("Failed to process feed %s: %s", feed_name, e)

        self._cache.articles = all_articles
        self._cache.last_update = datetime.now(timezone.utc)
        logger.info("Sentiment update complete: %d articles from %d feeds", len(all_articles), len(self.feeds))

    def score(self, symbol: Optional[str] = None) -> Dict:
        """
        Generate sentiment score, optionally filtered for a specific symbol.

        Args:
            symbol: If provided, weight articles by relevance to this symbol.
                    If None, return overall market sentiment.

        Returns:
            Dict with composite score and breakdown.
        """
        articles = self._cache.articles

        if not articles:
            return {
                "score": 0.0,
                "article_count": 0,
                "last_update": None,
                "note": "no_articles_cached - call update_feeds() first",
            }

        # Filter and weight by symbol relevance
        if symbol:
            keywords = SYMBOL_KEYWORDS.get(symbol, [])
            scored_articles = []
            for article in articles:
                relevance = self._compute_relevance(article, keywords)
                if relevance > 0:
                    scored_articles.append((article, relevance))

            if not scored_articles:
                # Fall back to general sentiment
                scored_articles = [(a, 0.5) for a in articles]
        else:
            scored_articles = [(a, 1.0) for a in articles]

        # Aggregate scores
        total_bull = 0.0
        total_bear = 0.0
        total_weight = 0.0

        for article, relevance in scored_articles:
            weight = relevance
            total_bull += article.bullish_score * weight
            total_bear += article.bearish_score * weight
            total_weight += weight

        if total_weight == 0:
            return {
                "score": 0.0,
                "article_count": len(scored_articles),
                "last_update": self._cache.last_update.isoformat() if self._cache.last_update else None,
            }

        avg_bull = total_bull / total_weight
        avg_bear = total_bear / total_weight

        # Normalize to -100 to +100 range
        raw_score = avg_bull - avg_bear
        # Empirical scaling: typical raw scores are in range -5 to +5
        normalized_score = np.clip(raw_score * 15, -100, 100)

        # Sentiment distribution
        positive_count = sum(1 for a, _ in scored_articles if a.net_score > 0)
        negative_count = sum(1 for a, _ in scored_articles if a.net_score < 0)
        neutral_count = sum(1 for a, _ in scored_articles if a.net_score == 0)

        # Top bullish and bearish articles
        sorted_by_score = sorted(scored_articles, key=lambda x: x[0].net_score, reverse=True)
        top_bullish = [
            {"title": a.title, "source": a.source, "score": float(a.net_score)}
            for a, _ in sorted_by_score[:3] if a.net_score > 0
        ]
        top_bearish = [
            {"title": a.title, "source": a.source, "score": float(a.net_score)}
            for a, _ in sorted_by_score[-3:] if a.net_score < 0
        ]

        return {
            "score": float(normalized_score),
            "raw_bullish": float(avg_bull),
            "raw_bearish": float(avg_bear),
            "article_count": len(scored_articles),
            "distribution": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count,
            },
            "top_bullish": top_bullish,
            "top_bearish": top_bearish,
            "last_update": self._cache.last_update.isoformat() if self._cache.last_update else None,
            "feed_status": self._cache.feed_status,
            "symbol_filter": symbol,
        }

    @staticmethod
    def _score_text(text: str) -> Tuple[float, float, List[str]]:
        """
        Score text for bullish and bearish sentiment.

        Returns (bullish_score, bearish_score, matched_keywords).
        """
        text_lower = text.lower()
        bull_score = 0.0
        bear_score = 0.0
        matched = []

        for keyword, weight in BULLISH_KEYWORDS.items():
            # Use word boundary matching to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            count = len(re.findall(pattern, text_lower))
            if count > 0:
                bull_score += weight * count
                matched.append(f"+{keyword}({count})")

        for keyword, weight in BEARISH_KEYWORDS.items():
            pattern = r'\b' + re.escape(keyword) + r'\b'
            count = len(re.findall(pattern, text_lower))
            if count > 0:
                bear_score += weight * count
                matched.append(f"-{keyword}({count})")

        # Negation detection: "not bullish" should flip the score
        negation_patterns = [
            r"not\s+(?:very\s+)?(?:bullish|positive|optimistic)",
            r"no\s+(?:rally|breakout|recovery)",
            r"(?:unlikely|fail)\s+to\s+(?:rally|rise|break)",
        ]
        for pattern in negation_patterns:
            if re.search(pattern, text_lower):
                # Swap scores partially
                bull_score *= 0.5
                bear_score += 1.0
                matched.append("negation_detected")

        return bull_score, bear_score, matched

    @staticmethod
    def _compute_relevance(article: ArticleSentiment, keywords: List[str]) -> float:
        """
        Compute how relevant an article is to a specific symbol.

        Returns 0.0 (not relevant) to 1.0 (highly relevant).
        """
        if not keywords:
            return 0.3  # Base relevance for untracked symbols

        text = article.title.lower()
        relevance = 0.0

        for kw in keywords:
            if kw.lower() in text:
                relevance += 0.5

        return min(relevance, 1.0)

    @staticmethod
    def _parse_date(entry) -> Optional[datetime]:
        """Parse publication date from feed entry."""
        for date_field in ["published_parsed", "updated_parsed"]:
            parsed = entry.get(date_field)
            if parsed:
                try:
                    from time import mktime
                    dt = datetime.fromtimestamp(mktime(parsed), tz=timezone.utc)
                    return dt
                except (ValueError, OverflowError):
                    continue

        # Try string parsing
        for date_field in ["published", "updated"]:
            date_str = entry.get(date_field, "")
            if date_str:
                for fmt in [
                    "%a, %d %b %Y %H:%M:%S %z",
                    "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%dT%H:%M:%SZ",
                ]:
                    try:
                        return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue

        return None

    def get_status(self) -> Dict:
        """Return sentiment system status."""
        return {
            "feeds_configured": len(self.feeds),
            "cached_articles": len(self._cache.articles),
            "last_update": self._cache.last_update.isoformat() if self._cache.last_update else None,
            "feed_status": self._cache.feed_status,
        }


# Need numpy for clip function used above
import numpy as np
