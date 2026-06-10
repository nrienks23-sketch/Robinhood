# Install dependencies:
# pip install robin_stocks yfinance pandas_ta pandas numpy pytz finvizfinance requests

import os
import re
import json
import time
import logging
import math
import urllib.request
from datetime import datetime, timezone, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

try:
    from finvizfinance.screener.overview import Overview as FinvizOverview
    FINVIZ_AVAILABLE = True
except ImportError:
    FINVIZ_AVAILABLE = False

import robin_stocks.robinhood as r

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Set DRY_RUN = True to scan and log signals without placing any real orders.
# Great for testing. Set to False when you're ready to trade live.
DRY_RUN = True

POSITIONS_FILE = Path(__file__).parent / "positions.json"
LOG_FILE = Path(__file__).parent / "trader.log"
POSITION_SIZE_USD = 50.0
SCAN_INTERVAL_SECONDS = 300  # 5 minutes
PREMARKET_START = dtime(8, 0, 0)   # start pre-market scan at 8:00 AM ET
MARKET_OPEN = dtime(9, 30, 0)
MARKET_CLOSE = dtime(16, 0, 0)
EASTERN = pytz.timezone("America/New_York")

# Volume / market-cap filters
MIN_MARKET_CAP = 150_000_000   # $150M
MAX_MARKET_CAP = 20_000_000_000  # $20B
VOLUME_MULTIPLIER = 2.0         # projected full-day volume must be >= 2x 20-day avg

# Position limits
MAX_POSITIONS = 3               # never hold more than 3 positions at once

# RSI thresholds
RSI_LOW = 50
RSI_HIGH = 70

# Tiered exit levels: (profit_pct, fraction_of_original_shares_to_sell)
# 10% at +2%, 20% at +3%, 10% at +4%, 20% at +5% = 60% sold via tiers
# Remaining 40% is held for the trailing stop to capture further upside
TIERS = [
    (0.02, 0.10),
    (0.03, 0.20),
    (0.04, 0.10),
    (0.05, 0.20),
]

# Trailing stop: activates once HWM hits +7.5%, trails 2.5% below HWM
# Sells ALL remaining shares when price falls to the floor
TRAILING_STOP_ACTIVATION_PCT = 0.075   # activate once HWM hits +7.5%
TRAILING_STOP_DROP_ALLOWED = 0.025     # trail 2.5% below HWM
TRAILING_STOP_FLOOR_PCT = 0.05         # absolute floor: never let it drop below +5% from entry

# Hard stop loss (before any tiers trigger): sell everything at -3%
HARD_STOP_LOSS_PCT = -0.03

# Breakeven stop: once the first tier triggers, tighten stop to -1%
# so a winner can never fully reverse into a loss
BREAKEVEN_STOP_PCT = -0.01

# ---------------------------------------------------------------------------
# Ticker universe (~200 small/mid-cap names)
# ---------------------------------------------------------------------------

TICKER_UNIVERSE = [
    # Meme / high-vol retail favorites
    "AMC", "GME", "SPCE", "MARA", "RIOT", "PLUG", "FCEL",
    "WKHS", "CLOV", "SNDL", "TLRY", "ACB",
    "CGC", "CRON", "PLTR", "BB", "NOK",
    "KOSS", "UWMC", "CLNE", "LCID",
    "RIVN", "HYLN", "BLNK", "CHPT",
    # Biotech / pharma small-cap
    "OCGN", "NVAX", "ATOS",
    "TNXP", "NKTR", "VXRT", "ADMA", "PRTS", "ORMP",
    "GILD", "CRSP", "EDIT", "NTLA", "BEAM",
    "PMVP", "IMVT", "ARDX", "LGVN", "RCUS", "CDNA", "PACB", "FATE",
    # Tech small/mid
    "BBAI", "PAYO", "BRZE", "DDOG", "MNDY", "GTLB",
    "LSPD", "TOST", "RELY", "ALKT", "HIMS", "OPEN", "OPFI",
    "AFRM", "UPST", "SOFI", "HOOD", "COUR", "DUOL", "DOCN", "FSLY",
    "ESTC", "SAIL", "WEAV", "NRDS", "KVYO", "RDDT", "IBTA",
    # Energy / clean energy
    "SPWR", "ARRY", "ENPH", "RUN", "FTCI",
    "FLNC", "STEM", "BLDP", "ITM", "CWEN",
    # Mining / metals
    "MVIS", "MP", "USAS", "EXK", "AG", "HL", "CDE", "PAAS",
    "MUX", "FSM",
    # Consumer / retail small-cap
    "LOVE", "PRPL", "REAL", "RVLV", "CPNG", "GREE", "EVGO",
    # Financial small-cap
    "GPRO", "LMND", "ROOT", "KPLT", "LPRO", "TREE",
    "CUBI", "PRAA", "ECPG", "ENVA", "ATLC", "MFIN",
    # Misc momentum names
    "CELH", "BYND", "SKIN", "ACMR",
    "XPEV", "NIO", "LI", "GFAI",
    "BKKT", "COIN", "MSTR", "BTBT", "HUT", "CLSK", "CIFR",
    "WULF", "IREN", "CORZ", "ARBK", "SOS", "EBON", "CAN",
]
# deduplicate while preserving order
seen = set()
TICKER_UNIVERSE_DEDUP = []
for t in TICKER_UNIVERSE:
    if t not in seen:
        seen.add(t)
        TICKER_UNIVERSE_DEDUP.append(t)
TICKER_UNIVERSE = TICKER_UNIVERSE_DEDUP

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Positions persistence
# ---------------------------------------------------------------------------

def load_positions() -> dict:
    if POSITIONS_FILE.exists():
        try:
            with open(POSITIONS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load positions.json: %s", exc)
    return {}


def sync_positions_from_robinhood(positions: dict) -> None:
    """Pull live Robinhood holdings and add any not already tracked."""
    try:
        holdings = r.account.build_holdings()
        if not holdings:
            return
        added = 0
        for ticker, data in holdings.items():
            if ticker in positions:
                continue
            quantity = float(data.get("quantity", 0))
            avg_buy_price = float(data.get("average_buy_price", 0))
            equity = float(data.get("equity", 0))
            if quantity <= 0 or avg_buy_price <= 0:
                continue
            positions[ticker] = {
                "entry_price": avg_buy_price,
                "dollars_invested": equity,
                "dollars_remaining": equity,
                "entry_time": datetime.now(EASTERN).isoformat(),
                "high_water_mark": avg_buy_price,
                "tiers_triggered": [],
                "trailing_stop_active": False,
                "trailing_stop_floor_pct": None,
                "news_flag": "imported from Robinhood",
            }
            added += 1
            logger.info("Imported existing position: %s — %s shares @ $%.2f", ticker, quantity, avg_buy_price)
        if added:
            save_positions(positions)
            logger.info("Synced %d position(s) from Robinhood into tracker.", added)
        else:
            logger.info("No new positions to sync from Robinhood.")
    except Exception as exc:
        logger.error("Failed to sync positions from Robinhood: %s", exc)


def save_positions(positions: dict) -> None:
    try:
        with open(POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=2)
    except OSError as exc:
        logger.error("Failed to save positions.json: %s", exc)

# ---------------------------------------------------------------------------
# Market hours helpers
# ---------------------------------------------------------------------------

def get_eastern_now() -> datetime:
    return datetime.now(EASTERN)


def is_market_open() -> bool:
    now = get_eastern_now()
    if now.weekday() >= 5:
        return False
    current_time = now.time()
    return MARKET_OPEN <= current_time < MARKET_CLOSE


def is_premarket() -> bool:
    now = get_eastern_now()
    if now.weekday() >= 5:
        return False
    return PREMARKET_START <= now.time() < MARKET_OPEN


def is_market_closing_soon() -> bool:
    """Return True if we are at or past 4:00 PM ET (end of day close-out)."""
    now = get_eastern_now()
    return now.time() >= MARKET_CLOSE


def is_afterhours() -> bool:
    """True between 4:00 PM and 8:00 PM ET on weekdays."""
    now = get_eastern_now()
    if now.weekday() >= 5:
        return False
    return MARKET_CLOSE <= now.time() < dtime(20, 0, 0)

# ---------------------------------------------------------------------------
# Robinhood helpers
# ---------------------------------------------------------------------------

def rh_login() -> bool:
    username = os.environ.get("ROBINHOOD_USERNAME", "nrienks23@gmail.com")
    password = os.environ.get("ROBINHOOD_PASSWORD", "LanD1989$$")
    if not username or not password:
        logger.error("ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD must be set.")
        return False
    try:
        r.login(username, password)
        logger.info("Logged into Robinhood successfully.")
        return True
    except Exception as exc:
        logger.error("Robinhood login failed: %s", exc)
        return False


def get_current_price(ticker: str) -> float | None:
    try:
        prices = r.stocks.get_latest_price(ticker)
        if prices and prices[0] is not None:
            return float(prices[0])
    except Exception as exc:
        logger.warning("Failed to get price for %s via Robinhood: %s", ticker, exc)
    return None


def place_buy_dollars(ticker: str, amount_usd: float) -> bool:
    if amount_usd < 1.0:
        logger.warning("Buy amount $%.2f too small for %s — skipping.", amount_usd, ticker)
        return False
    if DRY_RUN:
        logger.info("[DRY RUN] Would BUY $%.2f of %s", amount_usd, ticker)
        return True
    try:
        result = r.orders.order_buy_fractional_by_price(ticker, amount_usd)
        logger.info("BUY $%.2f of %s — order id: %s", amount_usd, ticker, result.get("id"))
        return True
    except Exception as exc:
        logger.error("BUY order failed for %s: %s", ticker, exc)
        return False


def place_sell_dollars(ticker: str, amount_usd: float) -> bool:
    if amount_usd < 1.0:
        logger.warning("Sell amount $%.2f too small for %s — skipping.", amount_usd, ticker)
        return False
    if DRY_RUN:
        logger.info("[DRY RUN] Would SELL $%.2f of %s", amount_usd, ticker)
        return True
    try:
        result = r.orders.order_sell_fractional_by_price(ticker, amount_usd)
        logger.info("SELL $%.2f of %s — order id: %s", amount_usd, ticker, result.get("id"))
        return True
    except Exception as exc:
        logger.error("SELL order failed for %s: %s", ticker, exc)
        return False

# ---------------------------------------------------------------------------
# Technical analysis helpers
# ---------------------------------------------------------------------------

def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame | None:
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df is None or df.empty or len(df) < 50:
                if attempt < 2:
                    time.sleep(1.5)
                    continue
                return None
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            df.dropna(inplace=True)
            return df
        except Exception as exc:
            if attempt < 2:
                time.sleep(1.5)
                continue
            logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
            return None
    return None


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def is_bullish_engulfing(df: pd.DataFrame, idx: int) -> bool:
    """True if candle at idx engulfs the previous bearish candle."""
    if idx < 1:
        return False
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]
    prev_bearish = prev["close"] < prev["open"]
    curr_bullish = curr["close"] > curr["open"]
    if not (prev_bearish and curr_bullish):
        return False
    return curr["close"] > prev["open"] and curr["open"] < prev["close"]


def is_hammer(df: pd.DataFrame, idx: int) -> bool:
    """True if candle at idx is a hammer (small body, long lower wick)."""
    candle = df.iloc[idx]
    body = abs(candle["close"] - candle["open"])
    total_range = candle["high"] - candle["low"]
    if total_range == 0:
        return False
    lower_wick = min(candle["close"], candle["open"]) - candle["low"]
    upper_wick = candle["high"] - max(candle["close"], candle["open"])
    return (lower_wick >= 2 * body) and (upper_wick <= body) and (body / total_range < 0.35)


def is_morning_star(df: pd.DataFrame, idx: int) -> bool:
    """True if candles at idx-2, idx-1, idx form a morning star pattern."""
    if idx < 2:
        return False
    c1 = df.iloc[idx - 2]
    c2 = df.iloc[idx - 1]
    c3 = df.iloc[idx]
    c1_bearish = c1["close"] < c1["open"]
    c3_bullish = c3["close"] > c3["open"]
    if not (c1_bearish and c3_bullish):
        return False
    c2_small_body = abs(c2["close"] - c2["open"]) < abs(c1["close"] - c1["open"]) * 0.3
    c3_recovers = c3["close"] > (c1["open"] + c1["close"]) / 2
    return c2_small_body and c3_recovers


def has_bullish_pattern(df: pd.DataFrame) -> bool:
    """Check last 2 candles of any timeframe dataframe for bullish patterns."""
    last_idx = len(df) - 1
    for idx in [last_idx, last_idx - 1]:
        if idx < 0:
            continue
        if is_bullish_engulfing(df, idx):
            return True
        if is_hammer(df, idx):
            return True
        if is_morning_star(df, idx):
            return True
    return False


def fetch_intraday_ohlcv(ticker: str, interval: str = "5m", period: str = "1d") -> pd.DataFrame | None:
    """Pull intraday candles for pattern detection at market open."""
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True, prepost=False)
        if df is None or df.empty or len(df) < 3:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.dropna(inplace=True)
        return df
    except Exception as exc:
        logger.debug("Intraday fetch failed for %s: %s", ticker, exc)
        return None


def has_bullish_pattern_intraday(ticker: str) -> tuple[bool, str]:
    """
    Check intraday (5-min) candles first, fall back to daily.
    Returns (pattern_found, description).
    """
    # Try 5-min candles first — most relevant for day trading
    df_5m = fetch_intraday_ohlcv(ticker, interval="5m", period="1d")
    if df_5m is not None and len(df_5m) >= 3:
        if has_bullish_pattern(df_5m):
            return True, "bullish pattern on 5m candles"

    # Try 15-min candles as second option
    df_15m = fetch_intraday_ohlcv(ticker, interval="15m", period="5d")
    if df_15m is not None and len(df_15m) >= 3:
        if has_bullish_pattern(df_15m):
            return True, "bullish pattern on 15m candles"

    # Fall back to daily candles
    df_1d = fetch_ohlcv(ticker, period="60d", interval="1d")
    if df_1d is not None and len(df_1d) >= 3:
        if has_bullish_pattern(df_1d):
            return True, "bullish pattern on daily candles"

    return False, "no bullish pattern on 5m, 15m, or daily candles"


def check_news_catalyst(ticker: str) -> tuple[bool, str]:
    """
    Scan multiple sources for recent news and social buzz on a ticker.
    Returns (is_safe_to_trade, summary_string).

    Blocks entry if:
    - Negative keywords found in recent headlines (lawsuit, fraud, SEC, bankruptcy, etc.)
    - No news at all AND no social buzz (dead stock, not worth trading)

    Boosts confidence if:
    - Bullish keywords in headlines (earnings beat, partnership, FDA approval, upgrade, etc.)
    - Reddit mentions today
    - StockTwits bullish sentiment majority
    - High Google News article count in last 24h
    """
    now_utc = datetime.now(timezone.utc)
    summary_parts = []
    negative_hit = False
    positive_score = 0

    NEGATIVE_KEYWORDS = [
        'lawsuit', 'fraud', 'sec investigation', 'bankruptcy', 'bankrupt',
        'delisted', 'delisting', 'class action', 'restatement', 'restated',
        'going concern', 'default', 'insolvency', 'ponzi', 'investigated',
        'criminal', 'indicted', 'misleading', 'recall', 'fda rejection',
        'rejected', 'halt', 'suspended trading',
    ]
    POSITIVE_KEYWORDS = [
        'earnings beat', 'raised guidance', 'upgrade', 'buy rating', 'strong buy',
        'partnership', 'contract', 'fda approval', 'approved', 'acquisition',
        'merger', 'buyout', 'beats estimates', 'record revenue', 'record sales',
        'short squeeze', 'unusual options', 'insider buying', 'breakout',
    ]

    # --- 1. Yahoo Finance news (last 24 hours) ---
    try:
        yf_news = yf.Ticker(ticker).news or []
        recent = []
        for article in yf_news:
            pub = article.get('content', {}).get('pubDate') or article.get('providerPublishTime')
            # handle both timestamp int and ISO string
            if isinstance(pub, int):
                age_hours = (now_utc.timestamp() - pub) / 3600
            elif isinstance(pub, str):
                try:
                    pub_dt = datetime.fromisoformat(pub.replace('Z', '+00:00'))
                    age_hours = (now_utc - pub_dt).total_seconds() / 3600
                except Exception:
                    age_hours = 999
            else:
                age_hours = 999
            if age_hours <= 24:
                title = (article.get('content', {}).get('title') or
                         article.get('title') or '').lower()
                recent.append(title)

        if recent:
            summary_parts.append(f"Yahoo:{len(recent)} articles")
            for title in recent:
                for kw in NEGATIVE_KEYWORDS:
                    if kw in title:
                        negative_hit = True
                        summary_parts.append(f"⚠️ negative keyword '{kw}'")
                        break
                for kw in POSITIVE_KEYWORDS:
                    if kw in title:
                        positive_score += 1
                        break
    except Exception as exc:
        logger.debug("Yahoo news fetch failed for %s: %s", ticker, exc)

    # --- 2. StockTwits sentiment ---
    if REQUESTS_AVAILABLE:
        try:
            resp = _requests.get(
                f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
                timeout=5,
            )
            if resp.status_code == 200:
                messages = resp.json().get('messages', [])
                bullish = sum(1 for m in messages
                              if m.get('entities', {}).get('sentiment', {}) and
                              m['entities']['sentiment'].get('basic') == 'Bullish')
                bearish = sum(1 for m in messages
                              if m.get('entities', {}).get('sentiment', {}) and
                              m['entities']['sentiment'].get('basic') == 'Bearish')
                total_sentiment = bullish + bearish
                if total_sentiment > 0:
                    bull_pct = bullish / total_sentiment * 100
                    summary_parts.append(f"StockTwits:{len(messages)}msgs {bull_pct:.0f}%bull")
                    if bull_pct >= 60:
                        positive_score += 1
                    elif bull_pct <= 30:
                        negative_hit = True
                        summary_parts.append("⚠️ StockTwits mostly bearish")
        except Exception as exc:
            logger.debug("StockTwits fetch failed for %s: %s", ticker, exc)

    # --- 3. Reddit mentions (WSB, stocks, pennystocks) ---
    if REQUESTS_AVAILABLE:
        reddit_count = 0
        headers = {'User-Agent': 'trader-scanner/1.0'}
        for sub in ['wallstreetbets', 'stocks', 'pennystocks', 'RobinHoodPennyStocks', 'investing']:
            try:
                url = f"https://www.reddit.com/r/{sub}/search.json?q={ticker}&sort=new&limit=10&t=day"
                resp = _requests.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    posts = resp.json().get('data', {}).get('children', [])
                    reddit_count += len(posts)
                    for post in posts:
                        title = post.get('data', {}).get('title', '').lower()
                        for kw in NEGATIVE_KEYWORDS:
                            if kw in title:
                                negative_hit = True
                                summary_parts.append(f"⚠️ Reddit negative: '{kw}'")
                                break
            except Exception:
                continue
        if reddit_count > 0:
            summary_parts.append(f"Reddit:{reddit_count} posts today")
            positive_score += min(2, reddit_count // 3)  # up to +2 for heavy Reddit activity

    # --- 4. Google News RSS ---
    try:
        gurl = f"https://news.google.com/rss/search?q={ticker}+stock+%22{ticker}%22&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(gurl, headers={'User-Agent': 'Mozilla/5.0'})
        content = urllib.request.urlopen(req, timeout=5).read().decode('utf-8')
        # Parse pub dates to filter last 24h
        items = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
        recent_google = 0
        for item in items:
            pub_match = re.search(r'<pubDate>(.*?)</pubDate>', item)
            title_match = re.search(r'<title>(.*?)</title>', item)
            if pub_match:
                try:
                    from email.utils import parsedate_to_datetime
                    pub_dt = parsedate_to_datetime(pub_match.group(1))
                    age_hours = (now_utc - pub_dt.astimezone(timezone.utc)).total_seconds() / 3600
                    if age_hours <= 24:
                        recent_google += 1
                        if title_match:
                            title = title_match.group(1).lower()
                            title = re.sub(r'<[^>]+>', '', title)
                            for kw in NEGATIVE_KEYWORDS:
                                if kw in title:
                                    negative_hit = True
                                    summary_parts.append(f"⚠️ Google News negative: '{kw}'")
                                    break
                            for kw in POSITIVE_KEYWORDS:
                                if kw in title:
                                    positive_score += 1
                                    break
                except Exception:
                    pass
        if recent_google > 0:
            summary_parts.append(f"GoogleNews:{recent_google} articles(24h)")
            if recent_google >= 3:
                positive_score += 1
    except Exception as exc:
        logger.debug("Google News fetch failed for %s: %s", ticker, exc)

    # --- Summary (never blocks — just flags for awareness) ---
    summary = " | ".join(summary_parts) if summary_parts else "no news found"
    flag = "⚠️ NEGATIVE NEWS FLAG" if negative_hit else "✅ news clear"
    return True, f"{flag} (score={positive_score}) — {summary}"


def check_entry_signals(ticker: str) -> tuple[int, float | None, str]:
    """
    Scores all 6 signals and returns (score, current_price, detail_string).
    Score 6 = all signals pass (strongest)
    Score 5 = one signal missing
    Score 4 = two signals missing
    Score < 4 = skip entirely
    Never hard-blocks on a single signal — everything is scored.
    """
    _, news_summary = check_news_catalyst(ticker)
    logger.info("News flag for %s: %s", ticker, news_summary)

    df = fetch_ohlcv(ticker, period="1y", interval="1d")
    if df is None or len(df) < 50:
        return 0, None, "insufficient data"

    close = df["close"]
    volume = df["volume"]
    current_price = float(close.iloc[-1])

    # Market cap gate — hard filter, not scored (just skip out-of-range stocks)
    try:
        info = yf.Ticker(ticker).fast_info
        market_cap = getattr(info, "market_cap", None) or getattr(info, "marketCap", None)
        if market_cap is not None and not (MIN_MARKET_CAP <= market_cap <= MAX_MARKET_CAP):
            return 0, current_price, f"market cap out of range"
    except Exception:
        pass

    signals = {}

    # Hard filter: skip stocks that already ran today (>8% up = already moved)
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else current_price
    today_change_pct = (current_price - prev_close) / prev_close if prev_close > 0 else 0
    if today_change_pct > 0.08:
        return 0, current_price, f"already up {today_change_pct*100:.1f}% today — skip"

    # 1. Volume spike
    now_et = datetime.now(pytz.timezone("America/New_York"))
    market_open_dt = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    minutes_elapsed = max(1, (now_et - market_open_dt).seconds // 60)
    avg_vol_20 = float(volume.iloc[-21:-1].mean())
    current_vol = float(volume.iloc[-1])
    projected_vol = current_vol * (390 / minutes_elapsed)
    signals["volume"] = (avg_vol_20 > 0 and projected_vol >= VOLUME_MULTIPLIER * avg_vol_20,
                         f"vol {projected_vol/avg_vol_20:.1f}x" if avg_vol_20 > 0 else "vol N/A")

    # 2. Price above SMA50
    sma50 = compute_sma(close, 50)
    sma50_val = float(sma50.iloc[-1])
    signals["sma50"] = (current_price > sma50_val, f"price {'>' if current_price > sma50_val else '<'} SMA50")

    # 3. Golden cross (SMA50 > SMA200)
    if len(df) >= 200:
        sma200_val = float(compute_sma(close, 200).iloc[-1])
        signals["golden_cross"] = (sma50_val > sma200_val,
                                   "golden cross ✅" if sma50_val > sma200_val else "no golden cross")
    else:
        signals["golden_cross"] = (False, "SMA200 N/A")

    # 4. RSI
    rsi_val = float(compute_rsi(close).iloc[-1])
    signals["rsi"] = (RSI_LOW <= rsi_val <= RSI_HIGH, f"RSI {rsi_val:.1f}")

    # 5. MACD
    macd_line, signal_line = compute_macd(close)
    macd_bull = float(macd_line.iloc[-1]) > float(signal_line.iloc[-1])
    signals["macd"] = (macd_bull, "MACD ✅" if macd_bull else "MACD ❌")

    # 6. Candlestick pattern
    pattern_found, pattern_desc = has_bullish_pattern_intraday(ticker)
    signals["candle"] = (pattern_found, pattern_desc)

    score = sum(1 for v, _ in signals.values() if v)

    # Quiet accumulation bonus label (volume building, price not broken out yet)
    change_label = f"today {today_change_pct*100:+.1f}%"
    if signals["volume"][0] and abs(today_change_pct) < 0.03:
        change_label += " 🔍 QUIET"

    detail = " | ".join(f"{'✅' if v else '❌'} {desc}" for _, (v, desc) in signals.items())
    full_detail = f"score={score}/6 | ${current_price:.2f} | {change_label} | {detail}"

    return score, current_price, full_detail

# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def get_dynamic_universe() -> list[str]:
    """
    Pull a live universe from Finviz each morning:
    - Market cap $150M–$20B (small + mid cap)
    - Average volume over 500K (liquid enough to trade)
    - Pulls both flat/slightly-up stocks (pre-breakout) AND up stocks
    Falls back to the hardcoded TICKER_UNIVERSE if Finviz is unavailable.
    """
    if not FINVIZ_AVAILABLE:
        logger.warning("finvizfinance not installed — using hardcoded universe.")
        return TICKER_UNIVERSE

    tickers = []
    # Two passes per cap tier: "Up" (any up) catches broad momentum,
    # no Change filter catches flat/consolidating stocks building quietly
    for cap_filter in ['Small ($300mln to $2bln)', 'Mid ($2bln to $10bln)']:
        for change_filter in ['Up', None]:
            try:
                screener = FinvizOverview()
                f = {'Market Cap.': cap_filter, 'Average Volume': 'Over 500K'}
                if change_filter:
                    f['Change'] = change_filter
                screener.set_filter(filters_dict=f)
                df = screener.screener_view(verbose=0)
                if df is not None and not df.empty:
                    tickers.extend(df['Ticker'].tolist())
            except Exception as exc:
                logger.warning("Finviz screener failed for %s/%s: %s", cap_filter, change_filter, exc)

    # Also grab micro cap ($150M–$300M) separately
    try:
        screener = FinvizOverview()
        screener.set_filter(filters_dict={
            'Market Cap.': 'Micro ($50mln to $300mln)',
            'Average Volume': 'Over 500K',
        })
        df = screener.screener_view(verbose=0)
        if df is not None and not df.empty:
            tickers.extend(df['Ticker'].tolist())
    except Exception as exc:
        logger.warning("Finviz micro-cap screener failed: %s", exc)

    # Deduplicate
    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        logger.warning("Finviz returned no results — falling back to hardcoded universe.")
        return TICKER_UNIVERSE

    logger.info("Dynamic universe: %d tickers from Finviz screener.", len(tickers))
    return tickers


def fast_volume_filter(tickers: list[str]) -> list[str]:
    """Quick pre-filter: only keep tickers with a recent volume spike."""
    survivors = []
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            raw = yf.download(
                batch,
                period="25d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )
        except Exception as exc:
            logger.warning("Batch volume download failed: %s", exc)
            time.sleep(1)
            continue

        for ticker in batch:
            try:
                if len(batch) == 1:
                    vol_series = raw["Volume"]
                else:
                    vol_series = raw[ticker]["Volume"]
                if vol_series is None or len(vol_series) < 2:
                    continue
                vol_series = vol_series.dropna()
                avg_vol = float(vol_series.iloc[-21:-1].mean()) if len(vol_series) >= 21 else float(vol_series[:-1].mean())
                latest_vol = float(vol_series.iloc[-1])
                if avg_vol > 0 and latest_vol >= VOLUME_MULTIPLIER * avg_vol:
                    survivors.append(ticker)
            except Exception:
                continue

        time.sleep(0.5)  # rate limit yfinance

    return survivors


def premarket_scan() -> None:
    """
    Run during 8:00–9:30 AM ET. Pulls pre-market price + technicals for the
    full universe and saves a ranked watchlist to premarket_watchlist.json.
    At market open the main loop reads this file and fires entries immediately.
    """
    logger.info("PRE-MARKET SCAN starting...")
    watchlist = []
    universe = get_dynamic_universe()

    for i, ticker in enumerate(universe):
        try:
            df = fetch_ohlcv(ticker, period="1y", interval="1d")
            if df is None or len(df) < 50:
                continue

            close = df["close"]
            volume = df["volume"]

            # Pre-market price via 1m data (last available pre-market bar)
            pm_df = yf.download(ticker, period="1d", interval="1m", progress=False,
                                auto_adjust=True, prepost=True)
            if pm_df is not None and not pm_df.empty:
                if isinstance(pm_df.columns, pd.MultiIndex):
                    pm_df.columns = pm_df.columns.get_level_values(0)
                pm_df.columns = [c.lower() for c in pm_df.columns]
                premarket_price = float(pm_df["close"].iloc[-1])
                premarket_vol = float(pm_df["volume"].sum())
            else:
                premarket_price = float(close.iloc[-1])
                premarket_vol = 0

            prev_close = float(close.iloc[-1])
            premarket_chg_pct = (premarket_price - prev_close) / prev_close * 100

            sma50 = float(close.rolling(50).mean().iloc[-1])
            sma200 = float(close.rolling(200).mean().iloc[-1]) if len(df) >= 200 else None
            rsi = float(compute_rsi(close).iloc[-1])
            macd_line, signal_line = compute_macd(close)
            macd_ok = float(macd_line.iloc[-1]) > float(signal_line.iloc[-1])

            avg_vol_20 = float(volume.iloc[-21:-1].mean())

            above_sma50 = premarket_price > sma50
            golden_cross = sma200 is not None and sma50 > sma200
            rsi_ok = 50 <= rsi <= 75
            vol_building = premarket_vol > avg_vol_20 * 0.1  # 10% of avg by pre-market is strong

            score = sum([above_sma50, golden_cross, rsi_ok, macd_ok, vol_building])
            # Must be moving up pre-market to make the list
            if premarket_chg_pct > 0.5 and score >= 3:
                watchlist.append({
                    "ticker": ticker,
                    "premarket_price": round(premarket_price, 2),
                    "premarket_chg_pct": round(premarket_chg_pct, 2),
                    "premarket_vol": int(premarket_vol),
                    "rsi": round(rsi, 1),
                    "sma50": round(sma50, 2),
                    "sma200": round(sma200, 2) if sma200 else None,
                    "macd_ok": macd_ok,
                    "score": score,
                })
                logger.info(
                    "PRE-MARKET WATCHLIST: %s +%.1f%% @ $%.2f | score %d/5 | RSI %.1f",
                    ticker, premarket_chg_pct, premarket_price, score, rsi,
                )
        except Exception as exc:
            logger.warning("Pre-market scan error on %s: %s", ticker, exc)

        if i % 10 == 0:
            time.sleep(0.5)

    watchlist.sort(key=lambda x: (x["score"], x["premarket_chg_pct"]), reverse=True)

    watchlist_file = Path(__file__).parent / "premarket_watchlist.json"
    with open(watchlist_file, "w") as f:
        json.dump(watchlist, f, indent=2)

    logger.info("PRE-MARKET SCAN complete. %d tickers on watchlist.", len(watchlist))
    print("\n" + "=" * 60)
    print(f"  PRE-MARKET WATCHLIST ({len(watchlist)} stocks)")
    print("=" * 60)
    for s in watchlist[:10]:
        print(f"  {s['ticker']:<7} +{s['premarket_chg_pct']:.1f}%  "
              f"@ ${s['premarket_price']:.2f}  RSI={s['rsi']:.0f}  score={s['score']}/5")
    print("=" * 60 + "\n")


def load_premarket_watchlist() -> list[str]:
    """Load tickers from the pre-market watchlist file, highest score first."""
    watchlist_file = Path(__file__).parent / "premarket_watchlist.json"
    if not watchlist_file.exists():
        return []
    try:
        with open(watchlist_file, "r") as f:
            data = json.load(f)
        return [item["ticker"] for item in data]
    except Exception:
        return []


def afterhours_scan(positions: dict) -> None:
    """Runs 4:00–8:00 PM ET — monitors open positions with after-hours prices."""
    now_str = get_eastern_now().strftime("%Y-%m-%d %H:%M:%S ET")
    print("\n" + "=" * 60)
    print(f"  AFTER-HOURS UPDATE  |  {now_str}")
    print("=" * 60)

    if positions:
        print(f"\n  Open positions (after-hours prices):")
        for ticker, pos in positions.items():
            try:
                ah_price = get_current_price(ticker)
                if not ah_price:
                    continue
                entry = pos["entry_price"]
                pct = (ah_price - entry) / entry * 100
                dollars_rem = pos.get("dollars_remaining", 0)
                tiers_done = [t for t, _ in pos.get("tiers_triggered", [])]
                next_tier = next(((t, frac) for t, frac in TIERS if t not in tiers_done), None)
                if pct <= HARD_STOP_LOSS_PCT * 100:
                    action = "🛑 CONSIDER SELLING — at hard stop"
                elif next_tier and pct >= next_tier[0] * 100:
                    action = f"✂️  TRIM {int(next_tier[1]*100)}% at open — +{next_tier[0]*100:.0f}% tier hit"
                else:
                    action = f"⏳ hold — P&L {pct:+.1f}%"
                print(f"    {ticker:6s}  entry=${entry:.2f}  now=${ah_price:.2f}  "
                      f"P&L={pct:+.1f}%  remaining=${dollars_rem:.2f}  → {action}")
            except Exception as exc:
                logger.warning("After-hours check failed for %s: %s", ticker, exc)
    else:
        print("\n  No open positions to monitor.")

    print("=" * 60 + "\n")



def scan_for_entries(existing_positions: dict) -> dict:
    """
    Scan universe and return results bucketed by signal score.
    Returns dict: {
        6: [(ticker, price, detail), ...],   # 6/6 — strongest
        5: [(ticker, price, detail), ...],   # 5/6
        4: [(ticker, price, detail), ...],   # 4/6
    }
    Bot auto-enters 6/6 and 5/6. Shows 4/6 as watchlist only.
    """
    watchlist = load_premarket_watchlist()
    dynamic = get_dynamic_universe()
    universe = watchlist + [t for t in dynamic if t not in watchlist]
    logger.info("Scanning %d tickers (watchlist: %d prioritized)...", len(universe), len(watchlist))
    candidates = fast_volume_filter(universe)
    logger.info("Volume filter: %d tickers pass volume spike test", len(candidates))

    results = {6: [], 5: [], 4: []}

    for ticker in candidates:
        if ticker in existing_positions:
            continue
        try:
            score, price, detail = check_entry_signals(ticker)
            if score >= 4 and price is not None:
                results[min(score, 6)].append((ticker, price, detail))
                logger.info("SCORE %d/6: %s @ $%.2f — %s", score, ticker, price, detail)
            else:
                logger.debug("SKIP %s (score %d) — %s", ticker, score, detail)
        except Exception as exc:
            logger.warning("Error analysing %s: %s", ticker, exc)
        time.sleep(0.3)

    # Sort each bucket by score desc within bucket (all same score, so just keep order)
    return results

# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------

def enter_position(ticker: str, price: float, positions: dict) -> None:
    _, news_summary = check_news_catalyst(ticker)

    success = place_buy_dollars(ticker, POSITION_SIZE_USD)
    if not success:
        return

    positions[ticker] = {
        "entry_price": price,
        "dollars_invested": POSITION_SIZE_USD,
        "dollars_remaining": POSITION_SIZE_USD,
        "entry_time": datetime.now(EASTERN).isoformat(),
        "high_water_mark": price,
        "tiers_triggered": [],
        "trailing_stop_active": False,
        "trailing_stop_floor_pct": None,
        "news_flag": news_summary,
    }
    logger.info(
        "ENTERED %s: $%.2f @ $%.2f per share | %s",
        ticker, POSITION_SIZE_USD, price, news_summary,
    )
    save_positions(positions)


def update_position_exit(ticker: str, pos: dict, current_price: float, positions: dict) -> None:
    """Evaluate and execute exit logic for a single position."""
    entry = pos["entry_price"]
    dollars_remaining = pos["dollars_remaining"]
    dollars_invested = pos["dollars_invested"]
    hwm = pos["high_water_mark"]
    tiers_triggered = pos["tiers_triggered"]

    if dollars_remaining <= 0.50:
        del positions[ticker]
        save_positions(positions)
        return

    pct_gain = (current_price - entry) / entry

    # Update high water mark
    if current_price > hwm:
        pos["high_water_mark"] = current_price
        hwm = current_price

    hwm_pct = (hwm - entry) / entry

    # Activate trailing stop once HWM reaches activation threshold
    if hwm_pct >= TRAILING_STOP_ACTIVATION_PCT and not pos["trailing_stop_active"]:
        pos["trailing_stop_active"] = True
        pos["trailing_stop_floor_pct"] = TRAILING_STOP_FLOOR_PCT
        logger.info("TRAILING STOP ACTIVATED for %s (HWM +%.1f%%)", ticker, hwm_pct * 100)

    # Dynamic trailing stop: floor is max(fixed floor, hwm - drop_allowed)
    if pos["trailing_stop_active"]:
        dynamic_floor_pct = hwm_pct - TRAILING_STOP_DROP_ALLOWED
        floor_pct = max(pos.get("trailing_stop_floor_pct") or TRAILING_STOP_FLOOR_PCT, dynamic_floor_pct)
        pos["trailing_stop_floor_pct"] = floor_pct
        if pct_gain <= floor_pct:
            sell_usd = round(dollars_remaining * (1 + pct_gain), 2)
            logger.info(
                "TRAILING STOP HIT for %s: price $%.2f (%.2f%% <= floor %.2f%%) — selling $%.2f",
                ticker, current_price, pct_gain * 100, floor_pct * 100, sell_usd,
            )
            place_sell_dollars(ticker, sell_usd)
            del positions[ticker]
            save_positions(positions)
            return

    # Stop loss:
    # - Before any tier: hard stop at -3%
    # - After first tier: tighten to -1% breakeven stop
    stop_pct = HARD_STOP_LOSS_PCT if not tiers_triggered else BREAKEVEN_STOP_PCT
    if pct_gain <= stop_pct:
        sell_usd = round(dollars_remaining * (1 + pct_gain), 2)
        stop_label = "HARD STOP LOSS" if not tiers_triggered else "BREAKEVEN STOP"
        logger.info(
            "%s for %s: price $%.2f (%.2f%%) — selling $%.2f",
            stop_label, ticker, current_price, pct_gain * 100, sell_usd,
        )
        place_sell_dollars(ticker, sell_usd)
        del positions[ticker]
        save_positions(positions)
        return

    # Tiered profit taking — sell % of original dollars invested
    for tier_pct, fraction in TIERS:
        tier_label = f"{tier_pct:.3f}"
        if tier_label in tiers_triggered:
            continue
        if pct_gain >= tier_pct:
            sell_usd = round(dollars_invested * fraction * (1 + pct_gain), 2)
            sell_usd = min(sell_usd, dollars_remaining)
            logger.info(
                "TIER +%.1f%% hit for %s: selling $%.2f (%.0f%% of original $%.2f)",
                tier_pct * 100, ticker, sell_usd, fraction * 100, dollars_invested,
            )
            success = place_sell_dollars(ticker, sell_usd)
            if success:
                pos["dollars_remaining"] -= round(dollars_invested * fraction, 2)
                pos["tiers_triggered"].append(tier_label)
                dollars_remaining = pos["dollars_remaining"]
                if dollars_remaining <= 0.50:
                    del positions[ticker]
                    save_positions(positions)
                    return

    save_positions(positions)


def manage_positions(positions: dict) -> None:
    """Check all open positions and apply exit logic."""
    if not positions:
        return

    for ticker in list(positions.keys()):
        pos = positions.get(ticker)
        if pos is None:
            continue

        current_price = get_current_price(ticker)
        if current_price is None:
            logger.warning("Could not fetch price for open position %s — skipping.", ticker)
            continue

        update_position_exit(ticker, pos, current_price, positions)
        time.sleep(0.2)


def close_all_positions(positions: dict) -> None:
    """Market-close sweep: sell everything remaining."""
    logger.info("MARKET CLOSE — closing all %d open positions.", len(positions))
    for ticker in list(positions.keys()):
        pos = positions.get(ticker)
        if pos and pos.get("dollars_remaining", 0) > 0.50:
            current_price = get_current_price(ticker)
            entry = pos.get("entry_price", 1)
            pct_gain = ((current_price - entry) / entry) if current_price else 0
            sell_usd = round(pos["dollars_remaining"] * (1 + pct_gain), 2)
            place_sell_dollars(ticker, sell_usd)
        del positions[ticker]
    save_positions(positions)

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(positions: dict, scan_entries: list) -> None:
    now_str = get_eastern_now().strftime("%Y-%m-%d %H:%M:%S ET")
    print("\n" + "=" * 60)
    print(f"  TRADER SUMMARY  |  {now_str}")
    print("=" * 60)

    if positions:
        print(f"  Open positions ({len(positions)}):")
        for ticker, pos in positions.items():
            price = get_current_price(ticker)
            if not price:
                print(f"    {ticker:6s}  ⚠️  Could not fetch price")
                continue
            entry = pos["entry_price"]
            pct = (price - entry) / entry * 100
            dollars_rem = pos.get("dollars_remaining", 0)
            dollars_inv = pos.get("dollars_invested", 0)
            tiers_done = pos.get("tiers_triggered", [])
            trailing_active = pos.get("trailing_stop_active", False)
            trailing_floor = pos.get("trailing_stop_floor_pct")
            news_flag = pos.get("news_flag", "")
            news_display = f"  [{news_flag}]" if "⚠️" in news_flag else ""

            # Figure out what action to show
            hard_stop = entry * (1 + HARD_STOP_LOSS_PCT)
            breakeven_stop = entry * (1 + BREAKEVEN_STOP_PCT)
            if pct <= HARD_STOP_LOSS_PCT * 100:
                action = "🛑 SELL NOW — hard stop hit"
            elif tiers_done and pct <= BREAKEVEN_STOP_PCT * 100:
                action = "🛑 SELL — breakeven stop hit"
            elif trailing_active and trailing_floor and pct <= trailing_floor * 100:
                action = "🛑 SELL — trailing stop hit"
            elif trailing_active:
                action = f"⚡ TRAILING STOP active (floor {trailing_floor*100:.1f}%)"
            else:
                # Check next tier
                next_tier = next(((t, frac) for t, frac in TIERS if t not in [x[0] for x in tiers_done]), None)
                if next_tier:
                    t_pct, t_frac = next_tier
                    if pct >= t_pct * 100:
                        action = f"✂️  TRIM {int(t_frac*100)}% — +{t_pct*100:.0f}% tier reached"
                    else:
                        action = f"⏳ hold — next trim at +{t_pct*100:.0f}% (currently {pct:+.1f}%)"
                else:
                    action = f"⏳ hold {pct:+.1f}%"

            print(
                f"    {ticker:6s}  entry=${entry:.2f}  now=${price:.2f}  "
                f"P&L={pct:+.1f}%  remaining=${dollars_rem:.2f}  "
                f"→ {action}{news_display}"
            )
    else:
        print("  No open positions.")

    total_signals = sum(len(v) for v in scan_entries.values()) if isinstance(scan_entries, dict) else 0
    if total_signals:
        for score in [6, 5, 4]:
            bucket = scan_entries.get(score, [])
            if not bucket:
                continue
            label = {6: "🔥 6/6", 5: "✅ 5/6", 4: "👀 4/6 — WATCH ONLY"}[score]
            print(f"\n  {label} ({len(bucket)} stocks):")
            for ticker, price, detail in bucket:
                print(f"    {ticker:6s} @ ${price:.2f}  |  {detail}")
    else:
        print("\n  No signals found this scan.")

    print("=" * 60 + "\n")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    if DRY_RUN:
        logger.info("=" * 60)
        logger.info("  DRY RUN MODE — no real orders will be placed")
        logger.info("=" * 60)
    logger.info("Starting day trading bot.")

    if not rh_login():
        logger.error("Cannot proceed without Robinhood login. Exiting.")
        return

    positions = load_positions()
    logger.info("Loaded %d existing positions from file.", len(positions))
    sync_positions_from_robinhood(positions)
    logger.info("Tracking %d position(s) total after Robinhood sync.", len(positions))

    closed_today = False
    premarket_scanned = False
    last_premarket_scan = None

    while True:
        try:
            now_et = get_eastern_now()

            # After-hours window: 4:00–8:00 PM ET
            if is_afterhours():
                if not closed_today:
                    close_all_positions(positions)
                    closed_today = True
                    logger.info("Market closed — running after-hours mode.")
                try:
                    afterhours_scan(positions)
                except Exception as exc:
                    logger.error("After-hours scan crashed: %s", exc, exc_info=True)
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue

            # End-of-day (after 8 PM ET) — just sleep
            if is_market_closing_soon() and not is_afterhours():
                if not closed_today:
                    close_all_positions(positions)
                    closed_today = True
                    logger.info("All positions closed for end of day. Waiting for tomorrow.")
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue

            # Reset flags each morning before pre-market
            if now_et.time() < PREMARKET_START:
                closed_today = False
                premarket_scanned = False
                last_premarket_scan = None
                logger.info("Waiting for pre-market window (8:00 AM ET). Sleeping 60s.")
                time.sleep(60)
                continue

            # Pre-market window: 8:00–9:30 AM ET — scan every 5 minutes
            if is_premarket():
                now_minute = now_et.hour * 60 + now_et.minute
                if last_premarket_scan is None or (now_minute - last_premarket_scan) >= 5:
                    premarket_scan()
                    last_premarket_scan = now_minute
                else:
                    logger.info("Pre-market: next scan in %dm. Sleeping 30s.",
                                5 - (now_minute - last_premarket_scan))
                time.sleep(30)
                continue

            if not is_market_open():
                logger.info("Market closed. Sleeping %ds.", SCAN_INTERVAL_SECONDS)
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue

            # Manage existing positions first
            manage_positions(positions)

            # Scan for new entries — returns {6: [...], 5: [...], 4: [...]}
            new_entries = scan_for_entries(positions)

            print_summary(positions, new_entries)

            # Ask for buy confirmation if there are actionable signals
            buyable = [
                (score, ticker, price, detail)
                for score in [6, 5]
                for ticker, price, detail in new_entries.get(score, [])
                if ticker not in positions
            ]
            if buyable and len(positions) < MAX_POSITIONS:
                print("  Enter ticker(s) to BUY (comma-separated), or press Enter to skip:")
                print("  Available slots: %d / %d" % (MAX_POSITIONS - len(positions), MAX_POSITIONS))
                try:
                    raw = input("  > ").strip().upper()
                except EOFError:
                    raw = ""
                if raw:
                    chosen = [t.strip() for t in raw.split(",") if t.strip()]
                    buyable_map = {ticker: (price, detail) for _, ticker, price, detail in buyable}
                    for ticker in chosen:
                        if len(positions) >= MAX_POSITIONS:
                            print("  Max positions reached — stopping buys.")
                            break
                        if ticker in buyable_map:
                            price, detail = buyable_map[ticker]
                            enter_position(ticker, price, positions)
                        else:
                            print(f"  {ticker} not in signal list — skipping.")

            # Manual trade tracker — ask if they bought anything on their own
            print("\n  Did you manually buy anything? Enter TICKER PRICE (e.g. NVDA 124.50)")
            print("  or press Enter to skip:")
            try:
                manual = input("  > ").strip().upper()
            except EOFError:
                manual = ""
            if manual:
                parts = manual.split()
                if len(parts) == 2:
                    try:
                        mticker = parts[0]
                        mprice = float(parts[1])
                        if mticker not in positions:
                            positions[mticker] = {
                                "entry_price": mprice,
                                "dollars_invested": POSITION_SIZE_USD,
                                "dollars_remaining": POSITION_SIZE_USD,
                                "entry_time": get_eastern_now().isoformat(),
                                "high_water_mark": mprice,
                                "tiers_triggered": [],
                                "trailing_stop_active": False,
                                "trailing_stop_floor_pct": None,
                                "news_flag": "manual entry",
                            }
                            save_positions(positions)
                            print(f"  ✅ Tracking {mticker} @ ${mprice:.2f} — I'll tell you when to trim/sell.")
                        else:
                            print(f"  Already tracking {mticker}.")
                    except ValueError:
                        print("  Couldn't parse that. Format is: TICKER PRICE (e.g. NVDA 124.50)")
                else:
                    print("  Format is: TICKER PRICE (e.g. NVDA 124.50)")

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received — shutting down.")
            break
        except Exception as exc:
            logger.error("Unhandled exception in main loop: %s", exc, exc_info=True)

        time.sleep(SCAN_INTERVAL_SECONDS)

    logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
