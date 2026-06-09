# Install dependencies:
# pip install robin_stocks yfinance pandas_ta pandas numpy pytz

import os
import json
import time
import logging
import math
from datetime import datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

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
SCAN_INTERVAL_SECONDS = 30
PREMARKET_START = dtime(8, 0, 0)   # start pre-market scan at 8:00 AM ET
MARKET_OPEN = dtime(9, 30, 0)
MARKET_CLOSE = dtime(16, 0, 0)
EASTERN = pytz.timezone("America/New_York")

# Volume / market-cap filters
MIN_MARKET_CAP = 150_000_000   # $150M
MAX_MARKET_CAP = 20_000_000_000  # $20B
VOLUME_MULTIPLIER = 2.0         # must be >= 2x 20-day avg

# RSI thresholds
RSI_LOW = 50
RSI_HIGH = 70

# Tiered exit levels: (profit_pct, fraction_of_original_shares_to_sell)
TIERS = [
    (0.03, 0.10),
    (0.045, 0.20),
    (0.06, 0.50),
    (0.10, 0.20),
]

# Trailing stop parameters
TRAILING_STOP_ACTIVATION_PCT = 0.075   # activate once HWM hits +7.5%
TRAILING_STOP_DROP_ALLOWED = 0.025     # allow 2.5% drop from HWM floor
TRAILING_STOP_FLOOR_PCT = 0.05         # hard floor once activated: +5% from entry

# Hard stop loss
HARD_STOP_LOSS_PCT = -0.03

# ---------------------------------------------------------------------------
# Ticker universe (~200 small/mid-cap names)
# ---------------------------------------------------------------------------

TICKER_UNIVERSE = [
    # Meme / high-vol retail favorites
    "AMC", "GME", "BBBY", "SPCE", "MARA", "RIOT", "PLUG", "FCEL",
    "NKLA", "WKHS", "GOEV", "CLOV", "WISH", "SNDL", "TLRY", "ACB",
    "APHA", "HEXO", "CGC", "CRON", "PLTR", "BB", "NOK", "EXPR",
    "NAKD", "KOSS", "UWMC", "CLNE", "RIDE", "XL", "FSR", "LCID",
    "RIVN", "ARVL", "HYLN", "SOLO", "AYRO", "PTRA", "BLNK", "CHPT",
    # Biotech / pharma small-cap
    "SAVA", "OCGN", "NVAX", "SRNE", "ATOS", "VERB", "MRIN", "OBSV",
    "ZYNE", "TNXP", "NKTR", "AGTC", "VXRT", "ADMA", "PRTS", "ORMP",
    "EYEG", "GILD", "SGEN", "CRSP", "EDIT", "NTLA", "BEAM", "VERV",
    "PMVP", "IMVT", "ARDX", "LGVN", "RCUS", "CDNA", "PACB", "FATE",
    # Tech small/mid
    "BBAI", "PAYO", "BRZE", "DDOG", "CFLT", "MNDY", "GTLB", "SMAR",
    "LSPD", "TOST", "OLO", "RELY", "ALKT", "HIMS", "OPEN", "OPFI",
    "AFRM", "UPST", "SOFI", "HOOD", "COUR", "DUOL", "DOCN", "FSLY",
    "ESTC", "SAIL", "WEAV", "NRDS", "KVYO", "RDDT", "IBTA", "KYNDRYL",
    # Energy / clean energy
    "SUNW", "MAXN", "SPWR", "NOVA", "ARRY", "ENPH", "RUN", "FTCI",
    "FLNC", "STEM", "BLDP", "ITM", "CWEN", "NOVA", "AZRE", "REGI",
    # Mining / metals
    "MVIS", "MP", "NOVN", "USAS", "EXK", "AG", "HL", "CDE", "PAAS",
    "SILV", "MUX", "FSM", "BCML", "MFAC",
    # Consumer / retail small-cap
    "LOVE", "PRPL", "LAZY", "BIGC", "REAL", "POSH", "RVLV", "CPNG",
    "WISH", "OTRK", "XELA", "TLGA", "GREE", "FFIE", "MULN", "EVGO",
    # Financial small-cap
    "GPRO", "LMND", "ROOT", "MILE", "KPLT", "LPRO", "TREE", "UWMC",
    "GHIV", "CUBI", "PRAA", "ECPG", "ENVA", "ATLC", "MFIN", "NICK",
    # Misc momentum names
    "TTOO", "NUO", "CELH", "BYND", "OATLY", "PNTM", "SKIN", "ACMR",
    "XPEV", "NIO", "LI", "NKLA", "SOLO", "ZEV", "IDEX", "GFAI",
    "BKKT", "COIN", "MSTR", "BTBT", "BITF", "HUT", "CLSK", "CIFR",
    "BTCY", "WULF", "IREN", "CORZ", "ARBK", "DMGI", "SOS", "BTCM",
    "NCTY", "EBON", "CAN", "BRPHF", "FRMO",
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

# ---------------------------------------------------------------------------
# Robinhood helpers
# ---------------------------------------------------------------------------

def rh_login() -> bool:
    username = os.environ.get("ROBINHOOD_USERNAME")
    password = os.environ.get("ROBINHOOD_PASSWORD")
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


def place_buy(ticker: str, shares: int) -> bool:
    if shares < 1:
        logger.warning("Attempted to buy 0 shares of %s — skipping.", ticker)
        return False
    if DRY_RUN:
        logger.info("[DRY RUN] Would BUY %d shares of %s", shares, ticker)
        return True
    try:
        result = r.orders.order_buy_market(ticker, shares)
        logger.info("BUY %d shares of %s — order id: %s", shares, ticker, result.get("id"))
        return True
    except Exception as exc:
        logger.error("BUY order failed for %s: %s", ticker, exc)
        return False


def place_sell(ticker: str, shares: int) -> bool:
    if shares < 1:
        logger.warning("Attempted to sell 0 shares of %s — skipping.", ticker)
        return False
    if DRY_RUN:
        logger.info("[DRY RUN] Would SELL %d shares of %s", shares, ticker)
        return True
    try:
        result = r.orders.order_sell_market(ticker, shares)
        logger.info("SELL %d shares of %s — order id: %s", shares, ticker, result.get("id"))
        return True
    except Exception as exc:
        logger.error("SELL order failed for %s: %s", ticker, exc)
        return False

# ---------------------------------------------------------------------------
# Technical analysis helpers
# ---------------------------------------------------------------------------

def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 50:
            return None
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.dropna(inplace=True)
        return df
    except Exception as exc:
        logger.warning("yfinance fetch failed for %s: %s", ticker, exc)
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


def check_entry_signals(ticker: str) -> tuple[bool, float | None, str]:
    """
    Returns (should_enter, current_price, reason).
    reason is a human-readable string explaining the outcome.
    """
    df = fetch_ohlcv(ticker, period="1y", interval="1d")
    if df is None or len(df) < 200:
        return False, None, "insufficient data"

    close = df["close"]
    volume = df["volume"]
    current_price = float(close.iloc[-1])

    # --- Market cap check (approximate using shares_outstanding from yfinance) ---
    try:
        info = yf.Ticker(ticker).fast_info
        market_cap = getattr(info, "market_cap", None)
        if market_cap is None:
            market_cap = getattr(info, "marketCap", None)
        if market_cap is not None:
            if not (MIN_MARKET_CAP <= market_cap <= MAX_MARKET_CAP):
                return False, current_price, f"market cap {market_cap:.0f} out of range"
    except Exception:
        pass  # skip market cap filter if unavailable

    # --- Volume spike ---
    avg_vol_20 = float(volume.iloc[-21:-1].mean())
    current_vol = float(volume.iloc[-1])
    if avg_vol_20 == 0 or current_vol < VOLUME_MULTIPLIER * avg_vol_20:
        return False, current_price, (
            f"volume {current_vol:.0f} < {VOLUME_MULTIPLIER}x avg {avg_vol_20:.0f}"
        )

    # --- SMA checks ---
    sma50 = compute_sma(close, 50)
    sma200 = compute_sma(close, 200)
    if current_price <= float(sma50.iloc[-1]):
        return False, current_price, "price below SMA50"
    if float(sma50.iloc[-1]) <= float(sma200.iloc[-1]):
        return False, current_price, "SMA50 below SMA200 (no golden cross)"

    # --- RSI ---
    rsi = compute_rsi(close)
    rsi_val = float(rsi.iloc[-1])
    if not (RSI_LOW <= rsi_val <= RSI_HIGH):
        return False, current_price, f"RSI {rsi_val:.1f} outside [{RSI_LOW}, {RSI_HIGH}]"

    # --- MACD ---
    macd_line, signal_line = compute_macd(close)
    if float(macd_line.iloc[-1]) <= float(signal_line.iloc[-1]):
        return False, current_price, "MACD below signal"

    # --- Candlestick pattern ---
    if not has_bullish_pattern(df):
        return False, current_price, "no bullish candlestick pattern"

    return True, current_price, (
        f"all signals passed | price={current_price:.2f} RSI={rsi_val:.1f} "
        f"vol={current_vol:.0f} ({current_vol/avg_vol_20:.1f}x avg)"
    )

# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

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

    for i, ticker in enumerate(TICKER_UNIVERSE):
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


def scan_for_entries(existing_positions: dict) -> list[tuple[str, float]]:
    """Return list of (ticker, price) that pass all entry signals."""
    # Prioritize pre-market watchlist tickers so they get checked first at open
    watchlist = load_premarket_watchlist()
    universe = watchlist + [t for t in TICKER_UNIVERSE if t not in watchlist]
    logger.info("Scanning %d tickers (watchlist: %d prioritized)...", len(universe), len(watchlist))
    candidates = fast_volume_filter(universe)
    logger.info("Volume filter: %d tickers pass volume spike test", len(candidates))

    entries = []
    for ticker in candidates:
        if ticker in existing_positions:
            continue  # already holding
        try:
            should_enter, price, reason = check_entry_signals(ticker)
            if should_enter:
                logger.info("SIGNAL: %s @ $%.2f — %s", ticker, price, reason)
                entries.append((ticker, price))
            else:
                logger.debug("SKIP %s — %s", ticker, reason)
        except Exception as exc:
            logger.warning("Error analysing %s: %s", ticker, exc)
        time.sleep(0.3)  # rate limit

    return entries

# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------

def enter_position(ticker: str, price: float, positions: dict) -> None:
    shares = math.floor(POSITION_SIZE_USD / price)
    if shares < 1:
        logger.warning("Cannot buy %s at $%.2f — price too high for $%d budget.", ticker, price, POSITION_SIZE_USD)
        return

    success = place_buy(ticker, shares)
    if not success:
        return

    positions[ticker] = {
        "entry_price": price,
        "shares": shares,
        "original_shares": shares,
        "entry_time": datetime.now(EASTERN).isoformat(),
        "high_water_mark": price,
        "tiers_triggered": [],
        "trailing_stop_active": False,
        "trailing_stop_floor_pct": None,
    }
    logger.info(
        "ENTERED %s: %d shares @ $%.2f (cost ~$%.2f)",
        ticker, shares, price, shares * price,
    )
    save_positions(positions)


def update_position_exit(ticker: str, pos: dict, current_price: float, positions: dict) -> None:
    """Evaluate and execute exit logic for a single position."""
    entry = pos["entry_price"]
    shares = pos["shares"]
    original_shares = pos["original_shares"]
    hwm = pos["high_water_mark"]
    tiers_triggered = pos["tiers_triggered"]

    if shares <= 0:
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
        logger.info(
            "TRAILING STOP ACTIVATED for %s (HWM +%.1f%%)",
            ticker, hwm_pct * 100,
        )

    # Dynamic trailing stop: floor is max(fixed floor, hwm - drop_allowed)
    if pos["trailing_stop_active"]:
        dynamic_floor_pct = hwm_pct - TRAILING_STOP_DROP_ALLOWED
        floor_pct = max(pos.get("trailing_stop_floor_pct") or TRAILING_STOP_FLOOR_PCT, dynamic_floor_pct)
        pos["trailing_stop_floor_pct"] = floor_pct
        if pct_gain <= floor_pct:
            logger.info(
                "TRAILING STOP HIT for %s: price $%.2f (pct_gain=%.2f%% <= floor=%.2f%%)",
                ticker, current_price, pct_gain * 100, floor_pct * 100,
            )
            place_sell(ticker, shares)
            del positions[ticker]
            save_positions(positions)
            return

    # Hard stop loss (only if no tiers have been triggered)
    if not tiers_triggered and pct_gain <= HARD_STOP_LOSS_PCT:
        logger.info(
            "STOP LOSS for %s: price $%.2f (%.2f%%)",
            ticker, current_price, pct_gain * 100,
        )
        place_sell(ticker, shares)
        del positions[ticker]
        save_positions(positions)
        return

    # Tiered profit taking
    for tier_pct, fraction in TIERS:
        tier_label = f"{tier_pct:.3f}"
        if tier_label in tiers_triggered:
            continue
        if pct_gain >= tier_pct:
            sell_shares = max(1, math.floor(original_shares * fraction))
            sell_shares = min(sell_shares, shares)  # can't sell more than held
            logger.info(
                "TIER %.1f%% hit for %s: selling %d shares (%.0f%% of original)",
                tier_pct * 100, ticker, sell_shares, fraction * 100,
            )
            success = place_sell(ticker, sell_shares)
            if success:
                pos["shares"] -= sell_shares
                pos["tiers_triggered"].append(tier_label)
                shares = pos["shares"]
                if shares <= 0:
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
        if pos and pos.get("shares", 0) > 0:
            place_sell(ticker, pos["shares"])
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
            if price:
                pnl = (price - pos["entry_price"]) * pos["shares"]
                pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
                print(
                    f"    {ticker:6s}  entry=${pos['entry_price']:.2f}  "
                    f"curr=${price:.2f}  shares={pos['shares']}  "
                    f"P&L=${pnl:+.2f} ({pct:+.1f}%)"
                )
    else:
        print("  No open positions.")

    if scan_entries:
        print(f"\n  New signals this scan ({len(scan_entries)}):")
        for ticker, price in scan_entries:
            print(f"    {ticker:6s} @ ${price:.2f}")
    else:
        print("\n  No new entry signals this scan.")

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

    closed_today = False
    premarket_scanned = False
    last_premarket_scan = None

    while True:
        try:
            now_et = get_eastern_now()

            # End-of-day close-out
            if is_market_closing_soon():
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

            # Scan for new entries
            new_entries = scan_for_entries(positions)

            for ticker, price in new_entries:
                if ticker not in positions:
                    enter_position(ticker, price, positions)

            print_summary(positions, new_entries)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received — shutting down.")
            break
        except Exception as exc:
            logger.error("Unhandled exception in main loop: %s", exc, exc_info=True)

        time.sleep(SCAN_INTERVAL_SECONDS)

    logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
