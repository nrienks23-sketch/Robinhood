# Robinhood day-trading bot

`trader.py` scans a small/mid-cap universe pre-market, scores entry signals
(6-point system), asks for buy confirmation during market hours, and manages
exits automatically (tiered profit-taking, trailing stop, hard stop, 4 PM ET
close-out). State lives in `positions.json`; every buy/sell is appended to
`trades.json`.

## Setup

```
pip install robin_stocks yfinance pandas_ta pandas numpy pytz finvizfinance requests
```

Copy `local_config.example.json` to `local_config.json` and fill it in:

```json
{
  "robinhood_username": "you@example.com",
  "robinhood_password": "your-password",
  "discord_webhook_pnl": "https://discord.com/api/webhooks/...",
  "discord_webhook_bop": "https://discord.com/api/webhooks/..."
}
```

`local_config.json` is gitignored — **never commit passwords or webhook URLs
to this repo; it is public.** Environment variables (`ROBINHOOD_USERNAME`,
`ROBINHOOD_PASSWORD`, `DISCORD_WEBHOOK_PNL`, `DISCORD_WEBHOOK_BOP`) work too
and take priority if set.

## Run

```
python trader.py                  # the bot (DRY_RUN flag is at the top of trader.py)
python trader.py --test-discord   # post a test message to both Discord channels
```

## Discord notifications

- **#pnl** — one compact line per trading day (posted after the 4 PM close-out,
  or on shutdown if the bot is stopped mid-day):
  `2026-06-10 | realized $+1.84 | sells 4 (3W/1L) | buys 2 | open 0`
- **#bop** — an end-of-day summary: realized P&L, every entry and exit with
  its reason (tier, trailing stop, hard stop, market close), and anything
  still open.

Notifications only fire when `DRY_RUN = False` — dry runs stay silent so the
channels only ever show real-money activity. Posting is fire-and-forget: a
broken webhook is logged and ignored, it can never interrupt trading.
