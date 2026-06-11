"""Discord webhook notifications for trader.py (one-way, fire-and-forget).

Every public function swallows all errors and returns a bool — a dead or
missing webhook must never affect the trading loop.

Webhook URLs are read from environment variables first, then from
local_config.json (which is gitignored — never commit real URLs or
passwords to this public repo; copy local_config.example.json to
local_config.json and fill it in locally).

Channels:
  pnl — one compact line per trading day (and on shutdown).
  bop — a fuller end-of-day summary of what the bot did.

When the bot runs with DRY_RUN = True nothing is posted, so the channels
only ever contain real-money activity. Use `python trader.py --test-discord`
to verify the webhooks are wired up.
"""

import json
import logging
import os
import urllib.request
from pathlib import Path

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

LOCAL_CONFIG_FILE = Path(__file__).parent / "local_config.json"
TIMEOUT_SECONDS = 5
MAX_CONTENT = 1900  # Discord rejects content over 2000 chars

_dry_run = False


def configure(dry_run: bool) -> None:
    """Called once by trader.py at startup so dry runs stay silent."""
    global _dry_run
    _dry_run = dry_run


def load_local_config() -> dict:
    """Read local_config.json (gitignored). Returns {} if absent or broken."""
    if LOCAL_CONFIG_FILE.exists():
        try:
            with open(LOCAL_CONFIG_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read local_config.json: %s", exc)
    return {}


def _webhook_url(channel: str) -> str:
    """Env var wins (DISCORD_WEBHOOK_PNL / DISCORD_WEBHOOK_BOP), then local_config.json."""
    env_value = os.environ.get(f"DISCORD_WEBHOOK_{channel.upper()}", "")
    if env_value:
        return env_value
    return load_local_config().get(f"discord_webhook_{channel.lower()}", "")


def _post(url: str, content: str) -> bool:
    payload = {"content": content[:MAX_CONTENT]}
    try:
        if REQUESTS_AVAILABLE:
            resp = _requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
            resp.raise_for_status()
        else:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS)
        return True
    except Exception as exc:
        logger.warning("Discord notify failed: %s", exc)
        return False


def _send(channel: str, content: str) -> bool:
    if _dry_run:
        logger.info("[DRY RUN] Discord post to #%s suppressed: %s",
                    channel, content.splitlines()[0])
        return False
    url = _webhook_url(channel)
    if not url:
        logger.debug("No webhook configured for #%s — skipping post.", channel)
        return False
    return _post(url, content)


def post_pnl(line: str) -> bool:
    """One spreadsheet-style row per trading day to #pnl. Inline code so
    columns line up across days."""
    return _send("pnl", f"`{line}`")


def post_bop(message: str) -> bool:
    """End-of-day summary to #bop."""
    return _send("bop", message)


def send_test() -> None:
    """Post a test message to both channels regardless of DRY_RUN.
    Run via: python trader.py --test-discord"""
    for channel in ("pnl", "bop"):
        url = _webhook_url(channel)
        if not url:
            print(f"  #{channel}: no webhook URL configured "
                  f"(set discord_webhook_{channel} in local_config.json)")
            continue
        ok = _post(url, f"\U0001f9ea Test message from trader.py — #{channel} webhook is working.")
        print(f"  #{channel}: {'OK — check Discord' if ok else 'FAILED — see log for the error'}")
