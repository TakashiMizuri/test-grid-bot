"""Download calendar 2025 and run the same short/long slot rules. Does not touch the 2026 CSV."""

import csv
import json
import urllib.request
from datetime import datetime, timedelta, timezone

import backtest_grid as g

START = datetime(2025, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 1, 1, tzinfo=timezone.utc)
PATH = g.ROOT / "data" / "btcusdt_1m_2025.csv"
OUT = g.ROOT / "results" / "summary_2025.json"


def load() -> list[tuple[int, float, float, float, float]]:
    if PATH.exists() and PATH.stat().st_size > 0:
        candles = []
        with PATH.open(newline="") as f:
            for row in csv.DictReader(f):
                candles.append(
                    (int(row["open_time"]), float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]))
                )
        if candles and candles[0][0] <= g.ms(START) and candles[-1][0] >= g.ms(END) - 60_000:
            print(f"reading {PATH} ({len(candles)} candles)")
            return candles
    urls = g.archive_urls(START, END)
    print(f"downloading {len(urls)} monthly files")
    archive = g.ROOT / "data" / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    merged: dict[int, tuple] = {}
    start_ms, end_ms = g.ms(START), g.ms(END)
    for url in urls:
        name = url.rsplit("/", 1)[-1]
        dest = archive / name
        if dest.exists() and dest.stat().st_size > 0:
            payload = dest.read_bytes()
            print(f"  cached {name}")
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "test-grid-bot/1.0"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = resp.read()
            dest.write_bytes(payload)
            print(f"  saved {name} ({len(payload) // 1024} KB)")
        for row in g.parse_zip(payload):
            if start_ms <= row[0] < end_ms:
                merged[row[0]] = row
    ordered = [merged[k] for k in sorted(merged)]
    with PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["open_time", "open", "high", "low", "close", "volume"])
        writer.writerows(ordered)
    print(f"saved {len(ordered)} candles -> {PATH}")
    return [(r[0], r[1], r[2], r[3], r[4]) for r in ordered]


def main() -> None:
    candles = load()
    expected = (END - START).days * 1440
    gaps = sum(1 for a, b in zip(candles, candles[1:]) if b[0] - a[0] != 60_000)
    print(
        f"bars {len(candles)} expected {expected} gaps {gaps} "
        f"first {datetime.fromtimestamp(candles[0][0]/1000, timezone.utc)} "
        f"last {datetime.fromtimestamp(candles[-1][0]/1000, timezone.utc)}"
    )
    test_start = END - timedelta(days=(END - START).days - g.LOOKBACK_DAYS)
    test_ms = g.ms(test_start)
    entry = next(c[1] for c in candles if c[0] >= test_ms)
    test = [c for c in candles if c[0] >= test_ms]
    print(
        f"warmup through {test_start:%Y-%m-%d} entry {entry:.2f} "
        f"last {test[-1][4]:.2f} min {min(c[3] for c in test):.2f} max {max(c[2] for c in test):.2f} "
        f"days {len(test)/1440:.0f}"
    )
    packed = []
    for side in (1, -1):
        for frac in (1.0, 0.5, 0.25):
            run = g.simulate_slots(candles, test_ms, entry, stop_mode="step", stop_frac=frac, side=side)
            item = {
                "side": side,
                "stop_frac": frac,
                "final_equity": round(run["final_eq"], 2),
                "return_pct": round(run["final_eq"] / g.CAPITAL * 100 - 100, 2),
                "max_dd_pct": round(run["max_dd"] * 100, 2),
                "sharpe": round(run["sharpe"], 2),
                "takes": run["takes"],
                "stops": run["stops"],
                "win_pnl": round(run["win_pnl"], 2),
                "loss_pnl": round(run["loss_pnl"], 2),
                "end_btc_pct": round(run["end_btc_frac"] * 100, 2),
                "max_btc_pct": round(run["max_btc_frac"] * 100, 2),
                "months": [round(m["grid_pct"], 2) for m in run["months"]],
                "daily_eq": [round(p["eq"], 2) for p in run["daily"]],
                "days": [p["day"] for p in run["daily"]],
            }
            packed.append(item)
            print(
                f"side {side:+d} frac {frac:<4} eq {item['final_equity']:.2f} ret {item['return_pct']} "
                f"dd {item['max_dd_pct']} sharpe {item['sharpe']} "
                f"takes {item['takes']} {item['win_pnl']} stops {item['stops']} {item['loss_pnl']} "
                f"maxbtc {item['max_btc_pct']}"
            )
            print("  months", item["months"])
    # Weekly closes for the canvas, same sampling as 2026: from the second day, every 7th.
    days = packed[0]["days"]
    idx = list(range(1, len(days), 7))
    print("WEEKS", [days[i][5:] for i in idx])
    for item in packed:
        print("EQ", item["side"], item["stop_frac"], [item["daily_eq"][i] for i in idx])
    # Drop the long day lists from the saved summary to keep it smaller, but keep daily_eq.
    OUT.write_text(json.dumps({"test": "2025-01-31..2025-12-31", "entry": entry, "runs": packed}, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
