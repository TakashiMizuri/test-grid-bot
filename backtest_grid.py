"""Spot geometric grid backtest on Binance BTCUSDT 1m candles.

Idea (no lookahead):
  At the test start, set grid bounds from the prior 30 days of high/low,
  padded 1% and widened if needed so the entry price sits inside.
  Geometric step is targeted near 0.50%. Fee is zero in this run.

This is a research backtest, not a trading bot.
"""

from __future__ import annotations

import bisect
import csv
import io
import json
import math
import urllib.request
import zipfile
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "btcusdt_1m.csv"
SUMMARY_PATH = ROOT / "results" / "summary.json"

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
CAPITAL = 10_000.0
FEE = 0.0  # zero commission scenario
SMA_DAYS = 20
DAY_DROP = 0.03  # block buys after a 3% drop versus 24h ago
TARGET_STEP = 0.005  # 0.50% geometric spacing
MAX_HOLD = 6  # open grid slots at once
MIN_HALF_SPAN = 0.015  # band at least wide enough for six 0.50% steps
LOOKBACK_DAYS = 30
# Calendar 2026 through the last published day. The first 30 days only set
# the grid; the scored window is the rest of the year.
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 11, 23, 0, 0, tzinfo=timezone.utc)
HISTORY_DAYS = (END - START).days
TEST_DAYS = HISTORY_DAYS - LOOKBACK_DAYS
RANGE_PAD = 0.01

VISION = "https://data.binance.vision/data/spot"


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def archive_urls(start: datetime, end: datetime) -> list[str]:
    """Monthly zips for complete months, daily zips for the partial tail."""
    urls: list[str] = []
    # Walk each calendar month overlapping [start, end).
    cursor = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while cursor < end:
        if cursor.month == 12:
            nxt = datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            nxt = datetime(cursor.year, cursor.month + 1, 1, tzinfo=timezone.utc)
        # A finished month is one zip. The in-progress month (November) is daily files,
        # because the monthly archive is not published yet.
        if nxt <= end:
            stamp = cursor.strftime("%Y-%m")
            urls.append(
                f"{VISION}/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{stamp}.zip"
            )
        else:
            day = max(cursor, datetime(start.year, start.month, start.day, tzinfo=timezone.utc))
            # Daily files are named by the UTC date. `end` is exclusive.
            last_day = (end - timedelta(seconds=1)).date()
            while day.date() <= last_day:
                stamp = day.strftime("%Y-%m-%d")
                urls.append(
                    f"{VISION}/daily/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{stamp}.zip"
                )
                day += timedelta(days=1)
        cursor = nxt
    return urls


def parse_zip(payload: bytes) -> list[tuple[int, float, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float, float]] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        text = zf.read(name).decode("utf-8")
    for line in text.splitlines():
        if not line or line.startswith("open_time"):
            continue
        parts = line.split(",")
        open_raw = int(parts[0])
        # Vision switched spot klines to microseconds; API-era dumps are milliseconds.
        open_ms = open_raw // 1000 if open_raw > 10**14 else open_raw
        rows.append(
            (
                open_ms,
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
        )
    return rows


def read_csv() -> list[tuple[int, float, float, float, float]]:
    candles = []
    with DATA_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append(
                (
                    int(row["open_time"]),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                )
            )
    return candles


def load_or_download() -> list[tuple[int, float, float, float, float]]:
    start = START
    if DATA_PATH.exists():
        candles = read_csv()
        if candles and candles[0][0] <= ms(start):
            print(f"reading {DATA_PATH} ({len(candles)} candles)")
            return candles
        print(f"csv starts after {start:%Y-%m-%d}, rebuilding the year file")

    urls = archive_urls(start, END)
    print(f"downloading {len(urls)} archive files, {start:%Y-%m-%d} -> {END:%Y-%m-%d} UTC")
    archive_dir = ROOT / "data" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    merged: dict[int, tuple[int, float, float, float, float, float]] = {}
    start_ms = ms(start)
    end_ms = ms(END)
    for url in urls:
        name = url.rsplit("/", 1)[-1]
        dest = archive_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            payload = dest.read_bytes()
            print(f"  cached {name}")
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "test-grid-bot/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = resp.read()
            dest.write_bytes(payload)
            print(f"  saved {name} ({len(payload) // 1024} KB)")
        for row in parse_zip(payload):
            if start_ms <= row[0] < end_ms:
                merged[row[0]] = row

    ordered = [merged[k] for k in sorted(merged)]
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["open_time", "open", "high", "low", "close", "volume"])
        writer.writerows(ordered)
    print(f"saved {len(ordered)} candles -> {DATA_PATH}")
    return [(r[0], r[1], r[2], r[3], r[4]) for r in ordered]


def build_levels(lower: float, upper: float) -> list[float]:
    n = round(math.log(upper / lower) / math.log(1 + TARGET_STEP))
    n = max(8, min(80, n))
    ratio = (upper / lower) ** (1 / n)
    levels = [lower * ratio**i for i in range(n + 1)]
    levels[0] = lower
    levels[-1] = upper
    return levels


def band_levels(
    price: float,
    high: float,
    low: float,
    target_step: float = TARGET_STEP,
    min_half_span: float = MIN_HALF_SPAN,
    max_intervals: int = 80,
) -> list[float]:
    half = max((high - low) / price / 2, min_half_span)
    lower = price * (1 - half)
    upper = price * (1 + half)
    n = round(math.log(upper / lower) / math.log(1 + target_step))
    n = max(MAX_HOLD, min(max_intervals, n))
    ratio = (upper / lower) ** (1 / n)
    levels = [lower * ratio**i for i in range(n + 1)]
    levels[0] = lower
    levels[-1] = upper
    return levels


def simulate_floating(
    candles: list[tuple[int, float, float, float, float]],
    test_ms: int,
    entry_price: float,
) -> dict:
    """Moving grid: at most 6 slots, band recenters when a daily close leaves it,
    and the book is sold when price is below the prior 20-day average.
    """
    daily_close: dict[str, float] = {}
    for open_ms, _o, _h, _l, close in candles:
        day = datetime.fromtimestamp(open_ms / 1000, timezone.utc).strftime("%Y-%m-%d")
        daily_close[day] = close
    ordered_days = sorted(daily_close)
    sma_by_day: dict[str, float] = {}
    for i in range(SMA_DAYS, len(ordered_days)):
        window = [daily_close[d] for d in ordered_days[i - SMA_DAYS : i]]
        sma_by_day[ordered_days[i]] = sum(window) / SMA_DAYS

    warmup = [c for c in candles if c[0] < test_ms]
    test = [c for c in candles if c[0] >= test_ms]
    if not warmup or not test:
        raise ValueError("floating sim needs warmup and test candles")

    cash = CAPITAL
    btc = 0.0
    avg_cost = 0.0
    realized = 0.0
    buy_fills = 0
    sell_fills = 0
    round_trips = 0
    flattens = 0
    recenters = 0
    skipped_cap = 0
    rt_pnl: list[float] = []

    def equity_at(price: float) -> float:
        return cash + btc * price

    def buy(qty: float, price: float) -> float:
        nonlocal cash, btc, avg_cost, buy_fills
        if qty <= 1e-12 or price <= 0:
            return 0.0
        affordable = cash / (price * (1 + FEE))
        qty = min(qty, affordable)
        if qty <= 1e-12:
            return 0.0
        cost = qty * price
        new_btc = btc + qty
        avg_cost = (avg_cost * btc + price * qty) / new_btc
        cash -= cost * (1 + FEE)
        btc = new_btc
        buy_fills += 1
        return qty

    def sell(qty: float, price: float) -> float:
        nonlocal cash, btc, avg_cost, realized, sell_fills
        qty = min(qty, btc)
        if qty <= 1e-12 or price <= 0:
            return 0.0
        pnl = (price - avg_cost) * qty - qty * price * FEE
        realized += pnl
        rt_pnl.append(pnl)
        cash += qty * price * (1 - FEE)
        btc -= qty
        sell_fills += 1
        if btc <= 1e-10:
            btc = 0.0
            avg_cost = 0.0
        return qty

    levels = band_levels(entry_price, max(c[2] for c in warmup), min(c[3] for c in warmup))
    q = 0.0
    long: list[bool] = []

    def install(price: float, allow_seed: bool) -> None:
        nonlocal q, long
        n = len(levels) - 1
        equity = equity_at(price)
        buy_sum = sum(levels[i] for i in range(n))
        q = equity / ((1 + FEE) * buy_sum) if buy_sum > 0 else 0.0
        above = [i for i in range(n) if levels[i] > price]
        want = above[:MAX_HOLD] if allow_seed else []
        target = q * len(want)
        if btc > target:
            sell(btc - target, price)
        elif target > btc:
            buy(target - btc, price)
        long = [i in want for i in range(n)]

    def rebuild(price: float, high: float, low: float, allow_seed: bool) -> None:
        nonlocal levels
        levels = band_levels(price, high, low)
        install(price, allow_seed)

    first_day = datetime.fromtimestamp(test[0][0] / 1000, timezone.utc).strftime("%Y-%m-%d")
    sma0 = sma_by_day[first_day]
    levels = band_levels(entry_price, max(c[2] for c in warmup), min(c[3] for c in warmup))
    install(entry_price, entry_price >= sma0)

    def flatten(price: float) -> None:
        nonlocal flattens, long
        if btc > 0:
            sell(btc, price)
            flattens += 1
        long = [False] * (len(levels) - 1)

    def grid_move(a: float, b: float, buys_ok: bool) -> None:
        nonlocal skipped_cap, round_trips
        n = len(levels) - 1
        if b < a:
            hi = bisect.bisect_left(levels, a) - 1
            lo = bisect.bisect_left(levels, b)
            for i in range(min(hi, n - 1), lo - 1, -1):
                if i < 0 or long[i] or not buys_ok:
                    continue
                if sum(long) >= MAX_HOLD:
                    skipped_cap += 1
                    continue
                if buy(q, levels[i]) > 0:
                    long[i] = True
        elif b > a:
            lo = bisect.bisect_right(levels, a)
            hi = bisect.bisect_right(levels, b) - 1
            for lvl_i in range(lo, hi + 1):
                slot = lvl_i - 1
                if 0 <= slot < n and long[slot]:
                    sold = sell(q, levels[lvl_i])
                    long[slot] = False
                    if sold > 0:
                        round_trips += 1

    def walk(a: float, b: float, sma: float, buys_ok: bool) -> bool:
        if a >= sma and b < sma:
            grid_move(a, sma, buys_ok)
            flatten(sma)
            if b < sma:
                grid_move(sma, b, False)
            return False
        grid_move(a, b, buys_ok and min(a, b) >= sma)
        return buys_ok

    peak = equity_at(entry_price)
    max_dd = 0.0
    max_frac = (btc * entry_price / peak) if peak else 0.0
    daily: list[dict] = []
    last = entry_price
    current_day = None
    buys_ok = entry_price >= sma0
    window: list[tuple[int, float, float]] = [
        (c[0], c[2], c[3]) for c in warmup if c[0] >= test_ms - 30 * 86_400_000
    ]

    def finish_day(day: str, close: float) -> None:
        nonlocal buys_ok, recenters, peak, max_dd, max_frac
        sma = sma_by_day[day]
        if close < sma and btc > 0:
            flatten(close)
            buys_ok = False
        elif close >= sma:
            buys_ok = True
        if close < levels[0] or close > levels[-1]:
            hh = max(item[1] for item in window)
            ll = min(item[2] for item in window)
            rebuild(close, hh, ll, close >= sma)
            recenters += 1
        eq = equity_at(close)
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak if peak else 0.0
        if dd < max_dd:
            max_dd = dd
        frac = (btc * close / eq) if eq > 0 else 0.0
        if frac > max_frac:
            max_frac = frac
        daily.append(
            {
                "day": day,
                "eq": eq,
                "btc_pct": frac * 100,
                "px": close,
                "lower": levels[0],
                "upper": levels[-1],
            }
        )

    for open_ms, o, h, l, c in test:
        day = datetime.fromtimestamp(open_ms / 1000, timezone.utc).strftime("%Y-%m-%d")
        sma = sma_by_day[day]
        if current_day is None:
            current_day = day
        elif day != current_day:
            finish_day(current_day, last)
            current_day = day
        if o != last:
            buys_ok = walk(last, o, sma, buys_ok)
        path = (l, h, c) if c >= o else (h, l, c)
        prev = o
        for px in path:
            if px != prev:
                buys_ok = walk(prev, px, sma, buys_ok)
            prev = px
        last = c
        eq = equity_at(c)
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak if peak else 0.0
        if dd < max_dd:
            max_dd = dd
        frac = (btc * c / eq) if eq > 0 else 0.0
        if frac > max_frac:
            max_frac = frac
        window.append((open_ms, h, l))
        cutoff = open_ms - 30 * 86_400_000
        while window and window[0][0] < cutoff:
            window.pop(0)

    if current_day is not None:
        finish_day(current_day, last)

    final_eq = equity_at(last)
    # Daily-return Sharpe, same definition as the fixed-grid sim.
    rets = []
    for a, b in zip(daily, daily[1:]):
        if a["eq"] > 0:
            rets.append(b["eq"] / a["eq"] - 1)
    if len(rets) > 2:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        sharpe = (mean / math.sqrt(var)) * math.sqrt(365) if var > 0 else 0.0
    else:
        sharpe = 0.0
    months: list[dict] = []
    by_month: dict[str, list[dict]] = {}
    for point in daily:
        by_month.setdefault(point["day"][:7], []).append(point)
    prev_eq = equity_at(entry_price)
    # Entry equity is marked at the seed price; monthly returns use day closes.
    prev_eq = CAPITAL if not daily else None
    first = True
    running = CAPITAL
    for key in sorted(by_month):
        last_p = by_month[key][-1]
        base = running
        months.append(
            {
                "month": key,
                "grid_pct": (last_p["eq"] / base - 1) * 100 if base else 0.0,
                "end_eq": last_p["eq"],
            }
        )
        running = last_p["eq"]
        first = False
    _ = (prev_eq, first)

    return {
        "final_eq": final_eq,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "realized": realized,
        "unrealized": final_eq - CAPITAL - realized,
        "round_trips": round_trips,
        "buy_fills": buy_fills,
        "sell_fills": sell_fills,
        "capped_buys": skipped_cap,
        "paused_buys": flattens,
        "btc": btc,
        "cash": cash,
        "end_btc_frac": (btc * last / final_eq) if final_eq else 0.0,
        "max_btc_frac": max_frac,
        "avg_rt": (sum(rt_pnl) / len(rt_pnl)) if rt_pnl else 0.0,
        "months": months,
        "daily": daily,
        "recenters": recenters,
        "flattens": flattens,
        "lower": levels[0],
        "upper": levels[-1],
    }


def _sma_table(
    candles: list[tuple[int, float, float, float, float]],
) -> dict[str, float]:
    daily_close: dict[str, float] = {}
    for open_ms, _o, _h, _l, close in candles:
        day = datetime.fromtimestamp(open_ms / 1000, timezone.utc).strftime("%Y-%m-%d")
        daily_close[day] = close
    ordered = sorted(daily_close)
    sma: dict[str, float] = {}
    for i in range(SMA_DAYS, len(ordered)):
        window = [daily_close[d] for d in ordered[i - SMA_DAYS : i]]
        sma[ordered[i]] = sum(window) / SMA_DAYS
    return sma


def _months_from_daily(daily: list[dict]) -> list[dict]:
    by_month: dict[str, list[dict]] = {}
    for point in daily:
        by_month.setdefault(point["day"][:7], []).append(point)
    months: list[dict] = []
    running = CAPITAL
    for key in sorted(by_month):
        end_eq = by_month[key][-1]["eq"]
        months.append(
            {
                "month": key,
                "grid_pct": (end_eq / running - 1) * 100 if running else 0.0,
                "end_eq": end_eq,
            }
        )
        running = end_eq
    return months


def _sharpe(daily: list[dict]) -> float:
    rets = []
    for a, b in zip(daily, daily[1:]):
        if a["eq"] > 0:
            rets.append(b["eq"] / a["eq"] - 1)
    if len(rets) < 3:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return (mean / math.sqrt(var)) * math.sqrt(365) if var > 0 else 0.0


def simulate_slots(
    candles: list[tuple[int, float, float, float, float]],
    test_ms: int,
    entry_price: float,
    stop_mode: str = "step",
    stop_frac: float = 1.0,
    side: int = 1,
    leverage: float = 1.0,
    adverse_frac: float | None = None,
    block_rising_sma: int = 0,
    notional_leverage: float | None = None,
    reentry_step_block: bool = False,
    breakeven_stop: bool = False,
    slippage_bps: float = 0.0,
    maker_take: bool = False,
    target_step: float | None = None,
    breakeven_arm_frac: float = 0.5,
    breakeven_buffer_bps: float = 0.0,
    min_range: float = 0.0,
) -> dict:
    """Six working slots. No opening seed.
    notional_leverage is the slot size as a fraction of equity: 1.0 is 100% of the
    deposit, 2.0 is 200%. When it is unset, qty stays on the old leverage / 6 sizing.
    side 1 is long: take-profit is the next level up, stop is stop_frac of the way
    toward the next level down. side -1 with adverse_frac unset flips those same two
    prices. adverse_frac sets the short stop to that fraction of the step up and
    leaves the take a full step down. reentry_step_block refuses a new short until
    price has risen one full step above the latest stop. breakeven_stop moves the
    stop to the entry once price has covered half the way to the take.
    breakeven_arm_frac is that fraction of the way. breakeven_buffer_bps parks the
    moved stop that many basis points beyond the entry, on the adverse side.
    min_range blocks a new entry when the prior 30-day high-low range, as a
    fraction of price, is below that value. Open positions still exit.
    slippage_bps worsens the fill by that many basis points on entry and again
    on exit. The order still triggers at the signal price. maker_take leaves the
    take fill at the signal price: that exit is a resting limit. Stops, including
    the breakeven scratch, still pay slippage.
    target_step overrides the 0.50% grid. The minimum corridor scales with it,
    so a wider step is not squeezed back to 0.50% on a quiet month.
    block_rising_sma skips a new entry when today's
    SMA20 is above the SMA20 that many calendar days earlier. Open positions still
    exit only at their own stop or take. stop_mode "band" still sells leftovers only
    when the daily close breaks the lower band.
    """
    sma_by_day = _sma_table(candles)
    warmup = [c for c in candles if c[0] < test_ms]
    test = [c for c in candles if c[0] >= test_ms]
    if not warmup or not test:
        raise ValueError("slot sim needs warmup and test candles")

    cash = CAPITAL
    btc = 0.0
    avg_cost = 0.0
    realized = 0.0
    buy_fills = 0
    sell_fills = 0
    takes = 0
    stops = 0
    skipped_cap = 0
    skipped_trend = 0
    skipped_rise = 0
    skipped_reentry = 0
    skipped_narrow = 0
    reentry_block_px = 0.0
    breakeven_moves = 0
    trade_bps: list[float] = []
    recenters = 0
    win_pnl = 0.0
    loss_pnl = 0.0
    first_stop_pnl = 0.0
    repeat_stop_pnl = 0.0
    first_stops = 0
    repeat_stops = 0
    fresh_take_pnl = 0.0
    reentry_take_pnl = 0.0
    reentry_takes = 0
    ambiguous_stops = 0
    ambiguous_stop_pnl = 0.0
    stopped_levels: dict[float, int] = {}
    bar_high = entry_price
    bar_low = entry_price

    def equity_at(price: float) -> float:
        return cash + btc * price

    def buy(qty: float, price: float) -> float:
        nonlocal cash, btc, avg_cost, buy_fills
        if qty <= 1e-12 or price <= 0:
            return 0.0
        qty = min(qty, cash / (price * (1 + FEE)))
        if qty <= 1e-12:
            return 0.0
        new_btc = btc + qty
        avg_cost = (avg_cost * btc + price * qty) / new_btc
        cash -= qty * price * (1 + FEE)
        btc = new_btc
        buy_fills += 1
        return qty

    def sell(qty: float, price: float) -> None:
        nonlocal cash, btc, avg_cost, realized, sell_fills
        qty = min(qty, btc)
        if qty <= 1e-12 or price <= 0:
            return
        realized += (price - avg_cost) * qty - qty * price * FEE
        cash += qty * price * (1 - FEE)
        btc -= qty
        sell_fills += 1
        if btc <= 1e-10:
            btc = 0.0
            avg_cost = 0.0

    slip = slippage_bps / 10_000

    def open_fill(price: float) -> float:
        return price * (1 - slip) if side == -1 else price * (1 + slip)

    def close_fill(price: float, kind: str) -> float:
        if maker_take and kind == "take":
            return price
        return price * (1 + slip) if side == -1 else price * (1 - slip)

    step = TARGET_STEP if target_step is None else target_step
    half_floor = MIN_HALF_SPAN if target_step is None else MIN_HALF_SPAN * (step / TARGET_STEP)

    def levels_at(price: float, high: float, low: float) -> list[float]:
        cap = 80 if target_step is None else 400
        return band_levels(price, high, low, step, half_floor, cap)

    levels = levels_at(entry_price, max(c[2] for c in warmup), min(c[3] for c in warmup))
    # entry, stop, target, qty — prices are fixed at the fill
    positions: list[list[float]] = []

    def close_at(index: int, price: float, kind: str) -> None:
        nonlocal takes, stops, win_pnl, loss_pnl, first_stop_pnl, repeat_stop_pnl, first_stops, repeat_stops, fresh_take_pnl, reentry_take_pnl, reentry_takes, ambiguous_stops, ambiguous_stop_pnl, cash, btc, realized, sell_fills, reentry_block_px
        entry_px, _stop, target, qty = positions.pop(index)[:4]
        entry_fill = open_fill(entry_px)
        exit_fill = close_fill(price, kind)
        if entry_fill > 0:
            trade_bps.append((exit_fill - entry_fill) / entry_fill * side * 10_000)
        if reentry_step_block and kind != "take" and _stop > entry_px:
            reentry_block_px = _stop + 2.0 * (_stop - entry_px)
        pnl = (exit_fill - entry_fill) * qty * side
        if kind == "take":
            takes += 1
            win_pnl += pnl
            key = round(entry_px, 2)
            if stopped_levels.get(key, 0) >= 1:
                reentry_take_pnl += pnl
                reentry_takes += 1
            else:
                fresh_take_pnl += pnl
            stopped_levels.pop(key, None)
        else:
            stops += 1
            loss_pnl += pnl
            key = round(entry_px, 2)
            if stopped_levels.get(key, 0) >= 1:
                repeat_stop_pnl += pnl
                repeat_stops += 1
            else:
                first_stop_pnl += pnl
                first_stops += 1
            stopped_levels[key] = stopped_levels.get(key, 0) + 1
            if side == 1 and bar_high >= target and bar_low <= _stop:
                ambiguous_stops += 1
                ambiguous_stop_pnl += pnl
        qty = min(qty, btc) if side == 1 else min(qty, -btc)
        if side == 1:
            cash += qty * exit_fill * (1 - FEE)
            btc -= qty
        else:
            cash -= qty * exit_fill * (1 + FEE)
            btc += qty
        realized += pnl
        sell_fills += 1
        if abs(btc) <= 1e-10:
            btc = 0.0

    def try_open(price: float) -> None:
        nonlocal skipped_cap, cash, btc, buy_fills
        if len(positions) >= MAX_HOLD:
            skipped_cap += 1
            return
        if any(abs(p[0] - price) <= price * 1e-8 for p in positions):
            return
        i = min(range(len(levels)), key=lambda k: abs(levels[k] - price))
        if i + 1 < len(levels):
            target = levels[i + 1]
        else:
            target = price * (1 + step)
        if i > 0:
            full_stop = levels[i - 1]
        else:
            full_stop = price / (1 + step)
        if side == 1 or adverse_frac is None:
            down = price - (price - full_stop) * stop_frac
            if target <= price or down >= price:
                return
            # Long: stop below, take above. Short uses the same two prices the other way.
            adverse, favorable = (down, target) if side == 1 else (target, down)
        else:
            adverse = price + (target - price) * adverse_frac
            favorable = full_stop
            if favorable >= price or adverse <= price:
                return
        equity = equity_at(price)
        if equity <= 0:
            return
        if notional_leverage is None:
            qty = equity * leverage / MAX_HOLD / price
        else:
            qty = equity * notional_leverage / price
        opened = open_fill(price)
        if side == 1 and leverage <= 1:
            qty = buy(qty, opened)
        elif side == 1:
            cash -= qty * opened * (1 + FEE)
            btc += qty
            buy_fills += 1
        else:
            cash += qty * opened * (1 - FEE)
            btc -= qty
            buy_fills += 1
        if qty > 0:
            slot = [price, adverse, favorable, qty]
            if breakeven_stop:
                slot.append(price + breakeven_arm_frac * (favorable - price))
            positions.append(slot)

    def events(cur: float, dest: float, sma: float) -> list[tuple[float, str]]:
        down = dest < cur
        found: list[tuple[float, str]] = []

        def consider(price: float, kind: str) -> None:
            if down:
                if dest <= price < cur:
                    found.append((price, kind))
            elif cur < price <= dest:
                found.append((price, kind))

        for level in levels:
            consider(level, "level")
        for pos in positions:
            if stop_mode == "step":
                consider(pos[1], "stop")
            consider(pos[2], "target")
            if breakeven_stop and len(pos) > 4 and pos[4] > 0:
                consider(pos[4], "arm")
        consider(sma, "sma")
        return found

    def walk(a: float, b: float, sma: float, buys_ok: bool, rise_block: bool, narrow_block: bool) -> bool:
        nonlocal skipped_trend, skipped_rise, skipped_reentry, skipped_narrow, reentry_block_px, breakeven_moves
        if reentry_block_px > 0 and max(a, b) >= reentry_block_px:
            reentry_block_px = 0.0
        cur = a
        for _ in range(500):
            if cur == b:
                break
            found = events(cur, b, sma)
            if not found:
                break
            if b < cur:
                price = max(price for price, _kind in found)
            else:
                price = min(price for price, _kind in found)
            kinds = {kind for px, kind in found if abs(px - price) <= max(1e-6, abs(price) * 1e-10)}
            # Exits first, so a stop and a new level at the same print don't double-count a slot.
            tol = max(1e-6, abs(price) * 1e-10)
            for idx in range(len(positions) - 1, -1, -1):
                entry_px, stop, target = positions[idx][0], positions[idx][1], positions[idx][2]
                if "stop" in kinds and abs(stop - price) <= tol:
                    close_at(idx, stop, "stop")
                elif "target" in kinds and abs(target - price) <= tol:
                    close_at(idx, target, "take")
            if breakeven_stop and "arm" in kinds:
                for pos in positions:
                    if len(pos) > 4 and pos[4] > 0 and abs(pos[4] - price) <= tol:
                        buffer = breakeven_buffer_bps / 10_000
                        pos[1] = pos[0] * (1 + buffer) if side == -1 else pos[0] * (1 - buffer)
                        pos[4] = 0.0
                        breakeven_moves += 1
            going_down = b < cur
            if "level" in kinds and going_down:
                if price + 1e-8 < sma or not buys_ok:
                    skipped_trend += 1
                elif rise_block:
                    skipped_rise += 1
                elif narrow_block:
                    skipped_narrow += 1
                elif reentry_block_px > 0 and price < reentry_block_px:
                    skipped_reentry += 1
                else:
                    try_open(price)
            if going_down and price <= sma:
                buys_ok = False
            elif (not going_down) and price >= sma:
                buys_ok = True
            cur = price
        else:
            raise RuntimeError("slot segment did not finish")
        return buys_ok

    first_day = datetime.fromtimestamp(test[0][0] / 1000, timezone.utc).strftime("%Y-%m-%d")
    sma0 = sma_by_day[first_day]
    buys_ok = entry_price >= sma0
    peak = CAPITAL
    max_dd = 0.0
    max_frac = 0.0
    daily: list[dict] = []
    last = entry_price
    current_day = None
    window: list[tuple[int, float, float]] = [
        (c[0], c[2], c[3]) for c in warmup if c[0] >= test_ms - 30 * 86_400_000
    ]
    hi_q: deque[tuple[int, float]] = deque()
    lo_q: deque[tuple[int, float]] = deque()

    def push_span(ms: int, high: float, low: float) -> None:
        cutoff_ms = ms - 30 * 86_400_000
        while hi_q and hi_q[0][0] < cutoff_ms:
            hi_q.popleft()
        while lo_q and lo_q[0][0] < cutoff_ms:
            lo_q.popleft()
        while hi_q and hi_q[-1][1] <= high:
            hi_q.pop()
        hi_q.append((ms, high))
        while lo_q and lo_q[-1][1] >= low:
            lo_q.pop()
        lo_q.append((ms, low))

    if min_range > 0:
        for ms, high, low in window:
            push_span(ms, high, low)

    def mark_extrema(price: float) -> None:
        nonlocal peak, max_dd, max_frac
        eq = equity_at(price)
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak if peak else 0.0
        if dd < max_dd:
            max_dd = dd
        frac = (btc * price / eq) if eq > 0 else 0.0
        if abs(frac) > max_frac:
            max_frac = abs(frac)

    def finish_day(day: str, close: float) -> None:
        nonlocal buys_ok, recenters, levels
        sma = sma_by_day[day]
        buys_ok = close >= sma
        if stop_mode == "band" and positions and close < levels[0]:
            while positions:
                close_at(len(positions) - 1, close, "stop")
            buys_ok = False
        if close < levels[0] or close > levels[-1]:
            hh = max(item[1] for item in window)
            ll = min(item[2] for item in window)
            levels = levels_at(close, hh, ll)
            recenters += 1
        mark_extrema(close)
        eq = equity_at(close)
        daily.append(
            {
                "day": day,
                "eq": eq,
                "btc_pct": (btc * close / eq * 100) if eq > 0 else 0.0,
                "px": close,
                "lower": levels[0],
                "upper": levels[-1],
            }
        )

    for open_ms, o, h, l, c in test:
        bar_high = h
        bar_low = l
        day = datetime.fromtimestamp(open_ms / 1000, timezone.utc).strftime("%Y-%m-%d")
        sma = sma_by_day[day]
        if current_day is None:
            current_day = day
        elif day != current_day:
            finish_day(current_day, last)
            current_day = day
        rise_block = False
        if block_rising_sma > 0:
            prev_day = (
                datetime.fromtimestamp(open_ms / 1000, timezone.utc) - timedelta(days=block_rising_sma)
            ).strftime("%Y-%m-%d")
            prev_sma = sma_by_day.get(prev_day)
            rise_block = prev_sma is not None and sma > prev_sma
        narrow_block = False
        if min_range > 0 and hi_q and lo_q and last > 0:
            narrow_block = (hi_q[0][1] - lo_q[0][1]) / last < min_range
        if o != last:
            buys_ok = walk(last, o, sma, buys_ok, rise_block, narrow_block)
        path = (l, h, c) if c >= o else (h, l, c)
        prev = o
        for px in path:
            if px != prev:
                buys_ok = walk(prev, px, sma, buys_ok, rise_block, narrow_block)
            prev = px
        last = c
        mark_extrema(c)
        window.append((open_ms, h, l))
        if min_range > 0:
            push_span(open_ms, h, l)
        cutoff = open_ms - 30 * 86_400_000
        while window and window[0][0] < cutoff:
            window.pop(0)

    if current_day is not None:
        finish_day(current_day, last)

    final_eq = equity_at(last)
    return {
        "final_eq": final_eq,
        "max_dd": max_dd,
        "sharpe": _sharpe(daily),
        "realized": realized,
        "unrealized": final_eq - CAPITAL - realized,
        "round_trips": takes,
        "buy_fills": buy_fills,
        "sell_fills": sell_fills,
        "capped_buys": skipped_cap,
        "paused_buys": skipped_trend,
        "paused_rise": skipped_rise,
        "paused_reentry": skipped_reentry,
        "paused_narrow": skipped_narrow,
        "breakeven_moves": breakeven_moves,
        "trade_bps": trade_bps,
        "btc": btc,
        "cash": cash,
        "end_btc_frac": (btc * last / final_eq) if final_eq else 0.0,
        "max_btc_frac": max_frac,
        "avg_rt": (win_pnl / takes) if takes else 0.0,
        "months": _months_from_daily(daily),
        "daily": daily,
        "recenters": recenters,
        "flattens": stops,
        "takes": takes,
        "stops": stops,
        "win_pnl": win_pnl,
        "loss_pnl": loss_pnl,
        "first_stops": first_stops,
        "repeat_stops": repeat_stops,
        "first_stop_pnl": first_stop_pnl,
        "repeat_stop_pnl": repeat_stop_pnl,
        "reentry_takes": reentry_takes,
        "reentry_take_pnl": reentry_take_pnl,
        "fresh_take_pnl": fresh_take_pnl,
        "ambiguous_stops": ambiguous_stops,
        "ambiguous_stop_pnl": ambiguous_stop_pnl,
        "open_slots": len(positions),
        "lower": levels[0],
        "upper": levels[-1],
    }


def simulate(
    candles: list[tuple[int, float, float, float, float]],
    levels: list[float],
    entry_price: float,
    max_btc_frac: float | None = None,
    allow_buy=None,
) -> dict:
    n = len(levels) - 1
    long = [entry_price < levels[i] for i in range(n)]
    if max_btc_frac is not None:
        # Opening inventory counts toward the same ceiling as later buys.
        # Drop the farthest seeded slots until the share at the entry price fits.
        while any(long):
            seeded = sum(long)
            buy_sum_now = sum(levels[i] for i in range(n) if not long[i])
            gross = seeded * entry_price + buy_sum_now
            if gross <= 0 or (seeded * entry_price) / gross <= max_btc_frac:
                break
            for i in range(n - 1, -1, -1):
                if long[i]:
                    long[i] = False
                    break
    long_count = sum(long)
    buy_sum = sum(levels[i] for i in range(n) if not long[i])
    q = CAPITAL / ((1 + FEE) * (long_count * entry_price + buy_sum))
    seed_notional = q * entry_price * long_count
    seed_fee = seed_notional * FEE
    cash = CAPITAL - seed_notional - seed_fee
    btc = q * long_count
    fees = seed_fee
    realized = -seed_fee  # seed fee is a realized cost; inventory MTM is unrealized
    buy_fills = 0
    sell_fills = 0
    round_trips = 0
    rt_pnl: list[float] = []
    # Seeded slots were bought at the market, not at their lower grid line.
    entry_cost = [entry_price if long[i] else levels[i] for i in range(n)]
    skipped_buys = 0
    capped_buys = 0
    paused_buys = 0
    max_frac = 0.0

    def equity_at(price: float) -> float:
        return cash + btc * price

    peak = equity_at(entry_price)
    max_dd = 0.0
    max_dd_at = None
    min_equity = peak
    max_equity = peak

    # Daily samples taken at the last close of each UTC date.
    daily: list[dict] = []
    outside_min = 0
    below_min = 0
    above_min = 0

    def try_buy(i: int, ts: int) -> None:
        nonlocal cash, btc, fees, buy_fills, skipped_buys, capped_buys, paused_buys
        if long[i]:
            return
        if allow_buy is not None and not allow_buy(levels[i], ts):
            paused_buys += 1
            return
        notional = q * levels[i]
        fee = notional * FEE
        if max_btc_frac is not None:
            new_btc = btc + q
            new_eq = cash - notional - fee + new_btc * levels[i]
            if new_eq <= 0 or (new_btc * levels[i]) / new_eq > max_btc_frac:
                capped_buys += 1
                return
        if cash + 1e-4 < notional + fee:
            skipped_buys += 1
            return
        cash -= notional + fee
        btc += q
        fees += fee
        entry_cost[i] = levels[i]
        realized_delta = -fee
        # buy fee hits realized; the position itself is still unrealized
        nonlocal_realized(realized_delta)
        long[i] = True
        buy_fills += 1

    def nonlocal_realized(delta: float) -> None:
        nonlocal realized
        realized += delta

    def try_sell(i: int) -> None:
        nonlocal cash, btc, fees, sell_fills, round_trips
        if not long[i]:
            return
        if btc + 1e-9 < q:
            return
        notional = q * levels[i + 1]
        fee = notional * FEE
        buy_fee = q * entry_cost[i] * FEE
        pnl = (levels[i + 1] - entry_cost[i]) * q - buy_fee - fee
        cash += notional - fee
        btc -= q
        fees += fee
        # buy fee was already in `realized` at fill time; add sell proceeds vs cost
        nonlocal_realized((levels[i + 1] - entry_cost[i]) * q - fee)
        long[i] = False
        sell_fills += 1
        round_trips += 1
        rt_pnl.append(pnl)

    def move(a: float, b: float, ts: int) -> None:
        if b < a:
            # Cross buy levels downward: higher levels first.
            hi = bisect.bisect_left(levels, a) - 1  # last level < a
            lo = bisect.bisect_left(levels, b)  # first level >= b
            for i in range(min(hi, n - 1), lo - 1, -1):
                if 0 <= i < n:
                    try_buy(i, ts)
        elif b > a:
            # Cross sell levels upward: lower levels first. Sell of slot i is levels[i+1].
            lo = bisect.bisect_right(levels, a)  # first level > a
            hi = bisect.bisect_right(levels, b) - 1  # last level <= b
            for lvl_i in range(lo, hi + 1):
                slot = lvl_i - 1
                if 0 <= slot < n:
                    try_sell(slot)

    last = entry_price
    test_start_equity = equity_at(entry_price)

    # Buy & hold benchmarks, same entry fee, marked to market (no exit fee).
    bh_btc = (CAPITAL * (1 - FEE)) / entry_price
    half_btc = ((CAPITAL / 2) * (1 - FEE)) / entry_price
    half_cash = CAPITAL / 2

    for open_ms, o, h, l, c in candles:
        if o != last:
            move(last, o, open_ms)
        if c >= o:
            path = (l, h, c)
        else:
            path = (h, l, c)
        prev = o
        for px in path:
            if px != prev:
                move(prev, px, open_ms)
            prev = px
        last = c

        eq = equity_at(c)
        frac = (btc * c / eq) if eq > 0 else 0.0
        if frac > max_frac:
            max_frac = frac
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak
        if dd < max_dd:
            max_dd = dd
            max_dd_at = open_ms
        min_equity = min(min_equity, eq)
        max_equity = max(max_equity, eq)

        if c < levels[0]:
            outside_min += 1
            below_min += 1
        elif c > levels[-1]:
            outside_min += 1
            above_min += 1

        day = datetime.fromtimestamp(open_ms / 1000, timezone.utc).strftime("%Y-%m-%d")
        point = {
            "day": day,
            "t": open_ms,
            "eq": eq,
            "bh": bh_btc * c,
            "half": half_cash + half_btc * c,
            "px": c,
            "btc": btc,
            "btc_pct": frac * 100,
        }
        if daily and daily[-1]["day"] == day:
            daily[-1] = point
        else:
            daily.append(point)

    final_eq = equity_at(last)
    final_bh = bh_btc * last
    final_half = half_cash + half_btc * last
    unrealized = final_eq - CAPITAL - realized

    # Daily returns for Sharpe (equity).
    rets = []
    for a, b in zip(daily, daily[1:]):
        if a["eq"] > 0:
            rets.append(b["eq"] / a["eq"] - 1)
    if len(rets) > 2:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        sharpe = (mean / math.sqrt(var)) * math.sqrt(365) if var > 0 else 0.0
    else:
        sharpe = 0.0

    # Weekly returns (ISO week), last point of each week vs previous.
    weeks: list[dict] = []
    bucket: list[dict] = []

    def flush_week(points: list[dict], prev_eq: float, prev_bh: float) -> None:
        if not points:
            return
        last_p = points[-1]
        weeks.append(
            {
                "label": points[0]["day"][5:],  # MM-DD of week start in the sample
                "grid": (last_p["eq"] / prev_eq - 1) * 100,
                "bh": (last_p["bh"] / prev_bh - 1) * 100,
            }
        )

    prev_eq = test_start_equity
    prev_bh = bh_btc * entry_price
    current_key = None
    for point in daily:
        dt = datetime.fromtimestamp(point["t"] / 1000, timezone.utc)
        key = dt.strftime("%G-W%V")
        if current_key is None:
            current_key = key
        if key != current_key:
            flush_week(bucket, prev_eq, prev_bh)
            prev_eq = bucket[-1]["eq"]
            prev_bh = bucket[-1]["bh"]
            bucket = []
            current_key = key
        bucket.append(point)
    flush_week(bucket, prev_eq, prev_bh)

    # Monthly
    months: list[dict] = []
    by_month: dict[str, list[dict]] = {}
    for point in daily:
        by_month.setdefault(point["day"][:7], []).append(point)
    prev_eq = test_start_equity
    prev_bh = bh_btc * entry_price
    for key in sorted(by_month):
        last_p = by_month[key][-1]
        months.append(
            {
                "month": key,
                "grid_pct": (last_p["eq"] / prev_eq - 1) * 100,
                "bh_pct": (last_p["bh"] / prev_bh - 1) * 100,
                "end_eq": last_p["eq"],
            }
        )
        prev_eq = last_p["eq"]
        prev_bh = last_p["bh"]

    wins = [p for p in rt_pnl if p > 0]
    losses = [p for p in rt_pnl if p <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)

    # Round-trip pnl histogram in percent of notional (approx vs buy price).
    # Use USDT pnl buckets.
    if rt_pnl:
        avg_rt = sum(rt_pnl) / len(rt_pnl)
    else:
        avg_rt = 0.0

    return {
        "q": q,
        "long_seeded": long_count,
        "cash": cash,
        "btc": btc,
        "fees": fees,
        "realized": realized,
        "unrealized": unrealized,
        "final_eq": final_eq,
        "final_bh": final_bh,
        "final_half": final_half,
        "start_eq": test_start_equity,
        "max_dd": max_dd,
        "max_dd_at": max_dd_at,
        "min_equity": min_equity,
        "max_equity": max_equity,
        "buy_fills": buy_fills,
        "sell_fills": sell_fills,
        "round_trips": round_trips,
        "avg_rt": avg_rt,
        "win_rate": (len(wins) / len(rt_pnl)) if rt_pnl else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
        "sharpe": sharpe,
        "outside_min": outside_min,
        "below_min": below_min,
        "above_min": above_min,
        "bars": len(candles),
        "skipped_buys": skipped_buys,
        "capped_buys": capped_buys,
        "paused_buys": paused_buys,
        "max_btc_frac": max_frac,
        "end_btc_frac": (btc * last / final_eq) if final_eq > 0 else 0.0,
        "daily": daily,
        "weeks": weeks,
        "months": months,
        "rt_pnl_sum": sum(rt_pnl),
    }


def self_check() -> None:
    """One round trip: buy the upper slot on the way down, sell it on the way up."""
    levels = [100.0, 110.0, 121.0]
    # entry 115 -> slot 0 empty, slot 1 empty (115 >= 110). No seed.
    # Fall 115 -> 105 buys slot 1 at 110. Rise 105 -> 125 sells slot 1 at 121.
    candles = [
        (0, 115.0, 115.0, 105.0, 105.0),
        (60_000, 105.0, 125.0, 105.0, 125.0),
    ]
    # simulate() uses CAPITAL. Just check fills via a tiny local copy of move logic
    # by running simulate and inspecting round trips. Entry fee model uses CAPITAL,
    # so pnl sign is what we assert, not the dollar amount vs a toy account.
    stats = simulate(candles, levels, 115.0)
    assert stats["buy_fills"] == 1, stats["buy_fills"]
    assert stats["sell_fills"] == 1, stats["sell_fills"]
    assert stats["round_trips"] == 1
    # Gross edge (121-110)*q minus both fees must be positive at 0.1% vs 10% step.
    assert stats["rt_pnl_sum"] > 0, stats["rt_pnl_sum"]
    blocked = simulate(candles, levels, 115.0, max_btc_frac=0.0)
    assert blocked["buy_fills"] == 0, blocked["buy_fills"]
    assert blocked["capped_buys"] >= 1, blocked["capped_buys"]
    print(f"self-check ok  rt_pnl={stats['rt_pnl_sum']:.4f} USDT")


def main() -> None:
    self_check()
    candles = load_or_download()
    if len(candles) < 10_000:
        raise SystemExit(f"too few candles: {len(candles)}")

    test_start = END - timedelta(days=TEST_DAYS)
    test_ms = ms(test_start)
    warmup_ms = ms(test_start - timedelta(days=LOOKBACK_DAYS))
    warmup = [c for c in candles if warmup_ms <= c[0] < test_ms]
    test = [c for c in candles if c[0] >= test_ms]
    if not warmup or not test:
        raise SystemExit("warmup/test split is empty")

    lo = min(c[3] for c in warmup)
    hi = max(c[2] for c in warmup)
    entry = test[0][1]
    lower = min(lo * (1 - RANGE_PAD), entry * (1 - RANGE_PAD))
    upper = max(hi * (1 + RANGE_PAD), entry * (1 + RANGE_PAD))
    levels = build_levels(lower, upper)
    step = levels[1] / levels[0] - 1

    print(
        f"warmup {datetime.fromtimestamp(warmup[0][0]/1000, timezone.utc):%Y-%m-%d}"
        f" -> {datetime.fromtimestamp(warmup[-1][0]/1000, timezone.utc):%Y-%m-%d}"
        f"  high {hi:.2f}  low {lo:.2f}"
    )
    print(
        f"test   {datetime.fromtimestamp(test[0][0]/1000, timezone.utc):%Y-%m-%d %H:%M}"
        f" -> {datetime.fromtimestamp(test[-1][0]/1000, timezone.utc):%Y-%m-%d %H:%M}"
        f"  entry {entry:.2f}"
    )
    print(
        f"grid   {levels[0]:.2f} .. {levels[-1]:.2f}   "
        f"lines {len(levels)}  step {step*100:.3f}%  intervals {len(levels)-1}"
    )

    def pack_variant(run: dict, cap: float | None) -> dict:
        equity_peak = CAPITAL
        dd_daily: list[float] = []
        daily_eq: list[float] = []
        daily_btc: list[float] = []
        for point in run["daily"]:
            equity_peak = max(equity_peak, point["eq"])
            dd_daily.append(round((point["eq"] / equity_peak - 1) * 100, 2))
            daily_eq.append(round(point["eq"], 2))
            daily_btc.append(round(point["btc_pct"], 2))
        return {
            "cap": cap,
            "final_equity": round(run["final_eq"], 2),
            "return_pct": round(run["final_eq"] / CAPITAL * 100 - 100, 2),
            "max_dd_pct": round(run["max_dd"] * 100, 2),
            "sharpe": round(run["sharpe"], 2),
            "realized": round(run["realized"], 2),
            "unrealized": round(run["unrealized"], 2),
            "round_trips": run["round_trips"],
            "buy_fills": run["buy_fills"],
            "sell_fills": run["sell_fills"],
            "capped_buys": run["capped_buys"],
            "paused_buys": run["paused_buys"],
            "end_btc": round(run["btc"], 6),
            "end_cash": round(run["cash"], 2),
            "end_btc_pct": round(run["end_btc_frac"] * 100, 2),
            "max_btc_pct": round(run["max_btc_frac"] * 100, 2),
            "avg_rt": round(run["avg_rt"], 4),
            "months": [
                {
                    "month": m["month"],
                    "grid_pct": round(m["grid_pct"], 2),
                    "end_eq": round(m["end_eq"], 2),
                }
                for m in run["months"]
            ],
            "daily_eq": daily_eq,
            "daily_btc_pct": daily_btc,
            "dd_daily_pct": dd_daily,
            "oct31_eq": next(
                (round(p["eq"], 2) for p in run["daily"] if p["day"] == "2026-10-31"),
                None,
            ),
            "oct31_btc_pct": next(
                (round(p["btc_pct"], 2) for p in run["daily"] if p["day"] == "2026-10-31"),
                None,
            ),
        }

    variants: list[dict] = []
    stats = None
    for cap in (None, 0.25, 0.40, 0.50):
        run = simulate(test, levels, entry, max_btc_frac=cap)
        if cap is None:
            stats = run
        packed = pack_variant(run, cap)
        variants.append(packed)
        name = "no cap" if cap is None else f"cap {cap:.0%}"
        print(
            f"{name:<8} equity {packed['final_equity']:.2f}  "
            f"return {packed['return_pct']}%  maxDD {packed['max_dd_pct']}%  "
            f"trips {packed['round_trips']}  realized {packed['realized']:.2f}  "
            f"unreal {packed['unrealized']:.2f}  end BTC {packed['end_btc_pct']}%  "
            f"blocked {packed['capped_buys']}"
        )
    assert stats is not None

    daily_close: dict[str, float] = {}
    for open_ms, _o, _h, _l, close in candles:
        day = datetime.fromtimestamp(open_ms / 1000, timezone.utc).strftime("%Y-%m-%d")
        daily_close[day] = close
    ordered_days = sorted(daily_close)
    sma_by_day: dict[str, float] = {}
    for i in range(SMA_DAYS, len(ordered_days)):
        window = [daily_close[d] for d in ordered_days[i - SMA_DAYS : i]]
        sma_by_day[ordered_days[i]] = sum(window) / SMA_DAYS
    close_by_ms = {row[0]: row[4] for row in candles}

    def make_gate(mode: str):
        def allow(level: float, open_ms: int) -> bool:
            day = datetime.fromtimestamp(open_ms / 1000, timezone.utc).strftime("%Y-%m-%d")
            sma = sma_by_day[day]
            ref = close_by_ms.get(open_ms - 1440 * 60_000)
            below_sma = level < sma
            dropped = ref is not None and ref > 0 and level / ref - 1 < -DAY_DROP
            if mode == "sma":
                blocked = below_sma
            elif mode == "drop":
                blocked = dropped
            elif mode == "and":
                blocked = below_sma and dropped
            elif mode == "or":
                blocked = below_sma or dropped
            else:
                raise ValueError(mode)
            return not blocked

        return allow

    half_span = (hi - lo) / entry / 2
    wide_lower = entry * (1 - half_span)
    wide_upper = entry * (1 + half_span)
    wide_levels = build_levels(wide_lower, wide_upper)
    wide_step = wide_levels[1] / wide_levels[0] - 1
    print(
        f"wide   {wide_levels[0]:.2f} .. {wide_levels[-1]:.2f}   "
        f"intervals {len(wide_levels) - 1}  step {wide_step * 100:.3f}%  "
        f"half-span {half_span * 100:.2f}%"
    )

    theory_runs: list[tuple[str, list[float], float | None, object]] = [
        ("Пауза в тренде", levels, None, make_gate("or")),
        ("Пауза + потолок 25%", levels, 0.25, make_gate("or")),
        ("Широкий коридор", wide_levels, None, None),
        ("Коридор + потолок 25%", wide_levels, 0.25, None),
    ]
    theories: list[dict] = []
    base = variants[0].copy()
    base["name"] = "База"
    base["lower"] = round(levels[0], 2)
    base["upper"] = round(levels[-1], 2)
    theories.append(base)
    for label, grid, cap, gate in theory_runs:
        run = simulate(test, grid, entry, max_btc_frac=cap, allow_buy=gate)
        packed = pack_variant(run, cap)
        packed["name"] = label
        packed["lower"] = round(grid[0], 2)
        packed["upper"] = round(grid[-1], 2)
        packed["grids"] = len(grid) - 1
        theories.append(packed)
        print(
            f"{label:<24} equity {packed['final_equity']:.2f}  "
            f"return {packed['return_pct']}%  maxDD {packed['max_dd_pct']}%  "
            f"trips {packed['round_trips']}  realized {packed['realized']:.2f}  "
            f"unreal {packed['unrealized']:.2f}  end BTC {packed['end_btc_pct']}%  "
            f"paused {packed['paused_buys']}  "
            f"31 Oct BTC {packed['oct31_btc_pct']}%"
        )

    followups: list[dict] = []
    followup_runs: list[tuple[str, list[float], object]] = [
        ("Только SMA20", levels, make_gate("sma")),
        ("Только −3% за сутки", levels, make_gate("drop")),
        ("Оба условия сразу", levels, make_gate("and")),
        ("Коридор + пауза", wide_levels, make_gate("or")),
    ]
    for label, grid, gate in followup_runs:
        run = simulate(test, grid, entry, allow_buy=gate)
        packed = pack_variant(run, None)
        packed["name"] = label
        packed["lower"] = round(grid[0], 2)
        packed["upper"] = round(grid[-1], 2)
        packed["grids"] = len(grid) - 1
        followups.append(packed)
        print(
            f"{label:<24} equity {packed['final_equity']:.2f}  "
            f"return {packed['return_pct']}%  maxDD {packed['max_dd_pct']}%  "
            f"trips {packed['round_trips']}  realized {packed['realized']:.2f}  "
            f"unreal {packed['unrealized']:.2f}  end BTC {packed['end_btc_pct']}%  "
            f"paused {packed['paused_buys']}  "
            f"31 Oct BTC {packed['oct31_btc_pct']}%"
        )

    floating_run = simulate_floating(candles, test_ms, entry)
    floating = pack_variant(floating_run, None)
    floating["name"] = "Плавающая"
    floating["recenters"] = floating_run["recenters"]
    floating["flattens"] = floating_run["flattens"]
    floating["lower"] = round(floating_run["lower"], 2)
    floating["upper"] = round(floating_run["upper"], 2)
    floating["daily_lower"] = [round(p["lower"], 2) for p in floating_run["daily"]]
    floating["daily_upper"] = [round(p["upper"], 2) for p in floating_run["daily"]]
    print(
        f"{'Плавающая':<24} equity {floating['final_equity']:.2f}  "
        f"return {floating['return_pct']}%  maxDD {floating['max_dd_pct']}%  "
        f"trips {floating['round_trips']}  realized {floating['realized']:.2f}  "
        f"unreal {floating['unrealized']:.2f}  end BTC {floating['end_btc_pct']}%  "
        f"recenters {floating['recenters']}  flattens {floating['flattens']}"
    )

    def pack_slots(run: dict, name: str) -> dict:
        packed = pack_variant(run, None)
        packed["name"] = name
        packed["recenters"] = run["recenters"]
        packed["stops"] = run["stops"]
        packed["takes"] = run["takes"]
        packed["win_pnl"] = round(run["win_pnl"], 2)
        packed["loss_pnl"] = round(run["loss_pnl"], 2)
        return packed

    slots_step = pack_slots(simulate_slots(candles, test_ms, entry, stop_mode="step"), "Стоп на шаг")
    slots_band = pack_slots(simulate_slots(candles, test_ms, entry, stop_mode="band"), "Выход из коридора")
    for packed in (slots_step, slots_band):
        print(
            f"{packed['name']:<24} equity {packed['final_equity']:.2f}  "
            f"return {packed['return_pct']}%  maxDD {packed['max_dd_pct']}%  "
            f"takes {packed['takes']}  stops {packed['stops']}  "
            f"end BTC {packed['end_btc_pct']}%  max BTC {packed['max_btc_pct']}%"
        )

    total_min = stats["bars"]
    ret = stats["final_eq"] / CAPITAL - 1
    bh_ret = stats["final_bh"] / CAPITAL - 1
    half_ret = stats["final_half"] / CAPITAL - 1

    # Downsample daily to keep the canvas small: already daily.
    def rnd(x: float, n: int = 2) -> float:
        return round(x, n)

    summary = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "capital": CAPITAL,
        "fee": FEE,
        "target_step": TARGET_STEP,
        "lookback_days": LOOKBACK_DAYS,
        "test_days": TEST_DAYS,
        "warmup_start": datetime.fromtimestamp(warmup[0][0] / 1000, timezone.utc).strftime(
            "%Y-%m-%d"
        ),
        "warmup_end": datetime.fromtimestamp(warmup[-1][0] / 1000, timezone.utc).strftime(
            "%Y-%m-%d"
        ),
        "test_start": datetime.fromtimestamp(test[0][0] / 1000, timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        ),
        "test_end": datetime.fromtimestamp(test[-1][0] / 1000, timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        ),
        "warmup_high": rnd(hi),
        "warmup_low": rnd(lo),
        "entry": rnd(entry),
        "last": rnd(test[-1][4]),
        "lower": rnd(levels[0]),
        "upper": rnd(levels[-1]),
        "grids": len(levels) - 1,
        "step_pct": rnd(step * 100, 3),
        "qty_btc": rnd(stats["q"], 6),
        "seeded_slots": stats["long_seeded"],
        "final_equity": rnd(stats["final_eq"]),
        "return_pct": rnd(ret * 100, 2),
        "bh_equity": rnd(stats["final_bh"]),
        "bh_return_pct": rnd(bh_ret * 100, 2),
        "half_equity": rnd(stats["final_half"]),
        "half_return_pct": rnd(half_ret * 100, 2),
        "excess_vs_bh_pct": rnd((ret - bh_ret) * 100, 2),
        "excess_vs_half_pct": rnd((ret - half_ret) * 100, 2),
        "max_dd_pct": rnd(stats["max_dd"] * 100, 2),
        "max_dd_at": (
            datetime.fromtimestamp(stats["max_dd_at"] / 1000, timezone.utc).strftime(
                "%Y-%m-%d"
            )
            if stats["max_dd_at"]
            else None
        ),
        "sharpe": rnd(stats["sharpe"], 2),
        "fees": rnd(stats["fees"]),
        "realized": rnd(stats["realized"]),
        "unrealized": rnd(stats["unrealized"]),
        "end_cash": rnd(stats["cash"]),
        "end_btc": rnd(stats["btc"], 6),
        "buy_fills": stats["buy_fills"],
        "sell_fills": stats["sell_fills"],
        "round_trips": stats["round_trips"],
        "avg_rt": rnd(stats["avg_rt"], 4),
        "win_rate_pct": rnd(stats["win_rate"] * 100, 1),
        "profit_factor": (
            rnd(stats["profit_factor"], 2) if stats["profit_factor"] is not None else None
        ),
        "rt_pnl": rnd(stats["rt_pnl_sum"]),
        "outside_pct": rnd(100 * stats["outside_min"] / total_min, 2),
        "below_pct": rnd(100 * stats["below_min"] / total_min, 2),
        "above_pct": rnd(100 * stats["above_min"] / total_min, 2),
        "bars": total_min,
        "min_px": rnd(min(c[3] for c in test)),
        "max_px": rnd(max(c[2] for c in test)),
        "months": [
            {
                "month": m["month"],
                "grid_pct": rnd(m["grid_pct"], 2),
                "bh_pct": rnd(m["bh_pct"], 2),
                "end_eq": rnd(m["end_eq"]),
            }
            for m in stats["months"]
        ],
        "weeks": [
            {"label": w["label"], "grid": rnd(w["grid"], 2), "bh": rnd(w["bh"], 2)}
            for w in stats["weeks"]
        ],
        "daily": [
            {
                "day": d["day"][5:],  # MM-DD
                "date": d["day"],
                "eq": rnd(d["eq"]),
                "bh": rnd(d["bh"]),
                "half": rnd(d["half"]),
                "px": rnd(d["px"]),
                "btc": rnd(d["btc"], 4),
            }
            for d in stats["daily"]
        ],
    }

    # Drawdown series from daily equity (display). Peak-to-trough on the daily curve.
    peak = CAPITAL
    dd_series = []
    for d in summary["daily"]:
        peak = max(peak, d["eq"])
        dd_series.append(rnd((d["eq"] / peak - 1) * 100, 2))
    summary["dd_daily_pct"] = dd_series
    summary["variants"] = variants
    summary["theories"] = theories
    summary["followups"] = followups
    summary["floating"] = floating
    summary["slots_step"] = slots_step
    summary["slots_band"] = slots_band
    summary["sma_days"] = SMA_DAYS
    summary["day_drop_pct"] = DAY_DROP * 100
    summary["wide_half_span_pct"] = round(half_span * 100, 2)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {SUMMARY_PATH}")
    print(
        f"equity {summary['final_equity']:.2f}  return {summary['return_pct']}%  "
        f"B&H {summary['bh_return_pct']}%  maxDD {summary['max_dd_pct']}%  "
        f"trips {summary['round_trips']}  fees {summary['fees']:.2f}  "
        f"skipped_buys {stats['skipped_buys']}"
    )


if __name__ == "__main__":
    main()
