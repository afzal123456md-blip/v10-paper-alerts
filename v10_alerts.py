"""v10 paper-trading alerts -> Telegram. Runs every 15 minutes on GitHub Actions (free).
Each run: update the candle store (Binance public archive once, then the latest CoinDCX candles),
replay the exact v10 rules on 15 streams, and send a Telegram message for every NEW paper entry or exit.
Secrets (set in GitHub > Settings > Secrets and variables > Actions): TELEGRAM_TOKEN, TELEGRAM_CHAT_ID."""
import io, json, os, sys, time, zipfile, urllib.request, urllib.error, urllib.parse
import numpy as np
import pandas as pd

# ================================ SETTINGS ================================
PAPER_START = "2026-10-27"          # IST date your paper test started
START_CAPITAL_INR = 34733.63
USDT_INR = 95.175
RISK_PER_TRADE = 0.0015             # 0.15% per trade
MAX_TOTAL_RISK = 0.015              # 1.5% max open risk (paper account)
COST_ROUND_TRIP = 0.0014
FUNDING_PER_8H = 0.0001
DAILY_SUMMARY_IST_HOUR = 9          # morning summary around 09:00 IST
COINS_1H = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC", "DOT", "TRX"]
COINS_15M = ["BTC", "ETH", "SOL"]
PRESETS = {
    "1H":  dict(tf="1h",  coins=COINS_1H,  breakout=10, ema=100, adx_min=0,  stop_atr=6.0, target_r=4.0, max_bars=480),
    "15M": dict(tf="15m", coins=COINS_15M, breakout=80, ema=100, adx_min=20, stop_atr=8.0, target_r=4.0, max_bars=768),
}
KEEP_DAYS = {"15m": 45, "1h": 120, "4h": 240}      # history downloaded when the store is first built
MAX_KEEP_DAYS = 730                                # the store keeps growing (stable replay), trimmed after 2 years
DATA_DIR = os.environ.get("V10_DATA_DIR", "store")
# ==========================================================================

MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
IST = pd.Timedelta(hours=5, minutes=30)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
NOW_MS = int(os.environ.get("V10_NOW_MS", str(int(time.time() * 1000))))   # overridable for testing


def http_get(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


# ------------------------------ data sources ------------------------------
def fetch_coindcx_latest(coin, interval):
    url = f"https://public.coindcx.com/market_data/candles/?pair=B-{coin}_USDT&interval={interval}&limit=1000"
    for attempt in range(3):
        try:
            batch = json.loads(http_get(url, 30)); break
        except Exception:
            if attempt == 2: raise
            time.sleep(3)
    df = pd.DataFrame(batch).rename(columns={"time": "open_time_ms"})[["open_time_ms", "open", "high", "low", "close"]].astype(float)
    df["open_time_ms"] = df.open_time_ms.astype("int64")
    return df


def fetch_archive(coin, interval, days):
    sym = f"{coin}USDT"; base = "https://data.binance.vision/data/futures/um"
    now = pd.Timestamp(NOW_MS, unit="ms").normalize(); start = now - pd.Timedelta(days=days)
    frames = []
    def read_zip(url):
        try: raw = http_get(url)
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            raise
        with zipfile.ZipFile(io.BytesIO(raw)) as z: f = z.read(z.namelist()[0]).decode()
        df = pd.read_csv(io.StringIO(f), header=0 if f.startswith("open_time") else None).iloc[:, :5]
        df.columns = ["open_time_ms", "open", "high", "low", "close"]; return df
    for m in pd.date_range(start.replace(day=1), now.replace(day=1), freq="MS"):
        df = read_zip(f"{base}/monthly/klines/{sym}/{interval}/{sym}-{interval}-{m:%Y-%m}.zip") if m < now.replace(day=1) else None
        if df is None:
            for d in pd.date_range(max(m, start), min(m + pd.offsets.MonthEnd(0), now - pd.Timedelta(days=1)), freq="D"):
                x = read_zip(f"{base}/daily/klines/{sym}/{interval}/{sym}-{interval}-{d:%Y-%m-%d}.zip")
                if x is not None: frames.append(x)
        else:
            frames.append(df)
    if not frames: raise RuntimeError(f"no archive data for {sym} {interval}")
    df = pd.concat(frames).astype(float); df["open_time_ms"] = df.open_time_ms.astype("int64")
    return df


def update_store(coin, tf, log):
    """Keep a rolling candle file per series. Bootstrap from the archive if missing or too old."""
    path = os.path.join(DATA_DIR, f"{coin}_{tf}.csv")
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["open_time_ms", "open", "high", "low", "close"])
    coverage_ms = 1000 * MS[tf] * 0.9
    try:
        new = fetch_coindcx_latest(coin, tf)
    except Exception as e:
        log.append(f"{coin} {tf}: CoinDCX failed ({e})"); new = None
    need_boot = old.empty or (new is not None and new.open_time_ms.min() > old.open_time_ms.max() + MS[tf]) \
                or (new is None and (NOW_MS - (old.open_time_ms.max() if len(old) else 0)) > coverage_ms)
    parts = [old]
    if need_boot:
        try: parts.append(fetch_archive(coin, tf, KEEP_DAYS[tf]))
        except Exception as e: log.append(f"{coin} {tf}: archive failed ({e})")
    if new is not None: parts.append(new)
    df = pd.concat(parts).dropna()
    df["open_time_ms"] = df.open_time_ms.astype("int64")
    df = df.drop_duplicates("open_time_ms", keep="last").sort_values("open_time_ms")
    df = df[df.open_time_ms + MS[tf] <= NOW_MS]                                  # closed candles only
    df = df[df.open_time_ms >= NOW_MS - MAX_KEEP_DAYS * 86_400_000]
    os.makedirs(DATA_DIR, exist_ok=True); df.to_csv(path, index=False)
    return df.reset_index(drop=True)


# ------------------------------ v10 rules (same as the notebooks) ------------------------------
def rma(x, n): return pd.Series(x).ewm(alpha=1.0 / n, adjust=False).mean().values
def ema(x, n): return pd.Series(x).ewm(span=n, adjust=False).mean().values
def atr(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]; return rma(np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc))), n)
def adx(h, l, c, n=14):
    up = h - np.r_[h[0], h[:-1]]; dn = np.r_[l[0], l[:-1]] - l
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(h, l, c, n); pdi = 100 * rma(pdm, n) / a; mdi = 100 * rma(mdm, n) / a
    return rma(100 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-9), n)
def trend_direction(chart, htf, tf, ema_len, adx_min):
    c4 = htf.close.values; e4 = ema(c4, ema_len); a4 = adx(htf.high.values, htf.low.values, c4)
    idx = np.searchsorted(htf.open_time_ms.values + MS["4h"], chart.open_time_ms.values + MS[tf], side="right") - 1
    ok = idx >= 0; ii = np.where(ok, idx, 0)
    return np.where(ok & (c4[ii] > e4[ii]) & (a4[ii] >= adx_min), 1, np.where(ok & (c4[ii] < e4[ii]) & (a4[ii] >= adx_min), -1, 0))

def run_stream(coin, chart, htf, p):
    o, h, l, c, t = (chart[k].values for k in ("open", "high", "low", "close", "open_time_ms"))
    n = len(c); a = atr(h, l, c); d = trend_direction(chart, htf, p["tf"], p["ema"], p["adx_min"])
    hi = pd.Series(h).rolling(p["breakout"]).max().values; lo = pd.Series(l).rolling(p["breakout"]).min().values
    trades = []; i = 300; pending = None
    while i < n - 1:
        s = d[i]
        if s == 0 or np.isnan(hi[i]): i += 1; continue
        lvl = hi[i] if s == 1 else lo[i]; risk = p["stop_atr"] * a[i]; sl = lvl - s * risk; e_ = i + 1
        triggered = h[e_] >= lvl if s == 1 else l[e_] <= lvl
        if not triggered or not (0.0015 <= risk / lvl <= 0.15): i += 1; continue
        entry = max(lvl, o[e_]) if s == 1 else min(lvl, o[e_])
        if s * (entry - sl) <= 0: i += 1; continue
        tp = lvl + s * p["target_r"] * risk; ex = None; why = "OPEN"
        for j in range(e_, min(n - 1, e_ + p["max_bars"]) + 1):
            if j > e_ and s * (o[j] - sl) <= 0: ex, why = o[j], "stop"; break
            if (s == 1 and l[j] <= sl) or (s == -1 and h[j] >= sl): ex, why = sl, "stop"; break
            if (s == 1 and h[j] >= tp) or (s == -1 and l[j] <= tp): ex, why = tp, "target"; break
        if ex is None: ex = c[j]; why = "time-stop" if j == e_ + p["max_bars"] else "OPEN"
        funding = (t[j] // 28_800_000 - t[e_] // 28_800_000) * FUNDING_PER_8H * entry
        R = (s * (ex - entry) - COST_ROUND_TRIP * entry - funding) / risk
        trades.append(dict(key=f"{coin}|{p['tf']}|{int(t[e_])}", coin=coin, chart=p["tf"].upper(), side="LONG" if s == 1 else "SHORT",
                           entry_ms=int(t[e_]), exit_ms=int(t[j]) + MS[p["tf"]], entry=float(entry), stop=float(sl), target=float(tp),
                           exit=float(ex), status=why, R=float(R)))
        if why == "OPEN": break
        i = j
    if not trades or trades[-1]["status"] != "OPEN":
        k = n - 1; s = d[k]
        if s != 0 and not np.isnan(hi[k]):
            lvl = hi[k] if s == 1 else lo[k]; risk = p["stop_atr"] * a[k]
            pending = dict(coin=coin, chart=p["tf"].upper(), order="BUY STOP" if s == 1 else "SELL STOP", trigger=float(lvl),
                           stop=float(lvl - s * risk), target=float(lvl + s * p["target_r"] * risk))
    return trades, pending


def paper_account(trades, start_ms):
    """Fake-money balance since PAPER_START with 0.15% risk per trade and the 1.5% cap."""
    T = sorted([x for x in trades if x["entry_ms"] >= start_ms], key=lambda x: x["entry_ms"])
    realised, book, rows = [], [], []
    for x in T:
        eq_now = START_CAPITAL_INR + sum(p for ex, p in realised if ex <= x["entry_ms"])
        book = [b for b in book if b["status"] == "OPEN" or b["exit_ms"] > x["entry_ms"]]
        if sum(b["risk_inr"] for b in book) + eq_now * RISK_PER_TRADE > eq_now * MAX_TOTAL_RISK + 1e-9:
            rows.append(dict(x, taken=False, risk_inr=0.0, pnl_inr=0.0)); continue
        y = dict(x, taken=True, risk_inr=eq_now * RISK_PER_TRADE); y["pnl_inr"] = y["R"] * y["risk_inr"]
        if y["status"] != "OPEN": realised.append((y["exit_ms"], y["pnl_inr"]))
        book.append(y); rows.append(y)
    return START_CAPITAL_INR + sum(p for _, p in realised), rows


# ------------------------------ Telegram ------------------------------
def send(text):
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat or os.environ.get("V10_DRY_RUN"):
        print("---- TELEGRAM (dry run) ----\n" + text); return
    data = urllib.parse.urlencode({"chat_id": chat, "text": text, "disable_web_page_preview": "true"}).encode()
    for attempt in range(3):
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=data), timeout=30); return
        except Exception as e:
            if attempt == 2: print("Telegram send failed:", e)
            time.sleep(2)

def fp(x): return f"{x:,.2f}" if abs(x) >= 100 else f"{x:.5g}"
def ist(ms): return (pd.Timestamp(ms, unit="ms") + IST).strftime("%d %b %H:%M")


# ------------------------------ main ------------------------------
def main():
    log = []; data = {}
    need = sorted({(c, "4h") for c in set(COINS_1H) | set(COINS_15M)} | {(c, "1h") for c in COINS_1H} | {(c, "15m") for c in COINS_15M})
    for coin, tf in need:
        try: data[(coin, tf)] = update_store(coin, tf, log)
        except Exception as e: log.append(f"{coin} {tf}: {e}")
    trades, pend = [], []
    for p in PRESETS.values():
        for coin in p["coins"]:
            ch, h4 = data.get((coin, p["tf"])), data.get((coin, "4h"))
            if ch is None or h4 is None or len(ch) < 400: continue
            tr, pe = run_stream(coin, ch, h4, p); trades += tr
            if pe: pend.append(pe)
    state_path = os.path.join(DATA_DIR, "state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else None
    start_ms = int((pd.Timestamp(PAPER_START) - IST).timestamp() * 1000)
    bal, rows = paper_account(trades, start_ms)
    by_key = {r["key"]: r for r in rows}
    latest = max((int(d.open_time_ms.iloc[-1]) + MS[tf] for (c, tf), d in data.items() if len(d)), default=NOW_MS)

    if state is None:   # first run: set the baseline, no historical spam
        opn = [x for x in trades if x["status"] == "OPEN"]
        msg = [f"v10 paper alerts are ON. Checking every 15 min.", f"Paper balance: Rs {bal:,.0f} (start Rs {START_CAPITAL_INR:,.0f}, from {PAPER_START})",
               f"Open paper positions now: {len(opn)}"]
        msg += [f"  {x['coin']} {x['chart']} {x['side']} @ {fp(x['entry'])} | SL {fp(x['stop'])} | TP {fp(x['target'])}" for x in opn]
        send("\n".join(msg))
        state = dict(seen_entries=[x["key"] for x in trades], seen_exits=[x["key"] for x in trades if x["status"] != "OPEN"],
                     last_summary_day="")
    else:
        seen_e, seen_x = set(state["seen_entries"]), set(state["seen_exits"])
        guard = state.get("latest_ms", 0) - 2 * MS["1h"]      # only events newer than the previous run are alerted
        for x in sorted(trades, key=lambda z: z["entry_ms"]):
            if x["key"] not in seen_e and x["entry_ms"] < guard: seen_e.add(x["key"])
            if x["status"] != "OPEN" and x["key"] not in seen_x and x["exit_ms"] < guard: seen_x.add(x["key"])
            r = by_key.get(x["key"]); note = "" if r is None else ("" if r["taken"] else "  (skipped in paper account: 1.5% risk cap)")
            if x["key"] not in seen_e:
                size = "" if r is None or not r["taken"] else f"\nPaper risk Rs {r['risk_inr']:,.0f} | size {r['risk_inr'] / USDT_INR / abs(x['entry'] - x['stop']):.4g} {x['coin']}"
                send(f"v10 PAPER ENTRY: {x['coin']} {x['chart']} {x['side']}\nEntry {fp(x['entry'])} ({x['chart']} candle from {ist(x['entry_ms'])} IST)\n"
                     f"Stop {fp(x['stop'])} | Target {fp(x['target'])}{size}{note}")
                seen_e.add(x["key"])
            if x["status"] != "OPEN" and x["key"] not in seen_x:
                pnl = "" if r is None or not r["taken"] else f" | Rs {r['pnl_inr']:+,.0f}"
                send(f"v10 PAPER EXIT: {x['coin']} {x['chart']} {x['side']} ({x['status']})\nExit {fp(x['exit'])} ({ist(x['exit_ms'])} IST) | "
                     f"{x['R']:+.2f}R{pnl}\nPaper balance: Rs {bal:,.0f} ({bal / START_CAPITAL_INR - 1:+.2%}){note}")
                seen_x.add(x["key"])
        state["seen_entries"], state["seen_exits"] = sorted(seen_e), sorted(seen_x)

    now_ist = pd.Timestamp(NOW_MS, unit="ms") + IST
    if now_ist.hour == DAILY_SUMMARY_IST_HOUR and state.get("last_summary_day") != now_ist.strftime("%Y-%m-%d"):
        closed = [r for r in rows if r["taken"] and r["status"] != "OPEN"]; opn = [r for r in rows if r["taken"] and r["status"] == "OPEN"]
        w = sum(r["pnl_inr"] > 0 for r in closed)
        msg = [f"v10 DAILY PAPER SUMMARY ({now_ist:%d %b})", f"Balance Rs {bal:,.0f} ({bal / START_CAPITAL_INR - 1:+.2%}) since {PAPER_START}",
               f"Closed trades {len(closed)} (wins {w}, losses {len(closed) - w}) | open {len(opn)} (unrealised Rs {sum(r['pnl_inr'] for r in opn):+,.0f})"]
        if pend:
            msg.append("Resting orders now:")
            msg += [f"  {p_['coin']} {p_['chart']} {p_['order']} {fp(p_['trigger'])} | SL {fp(p_['stop'])} | TP {fp(p_['target'])}" for p_ in pend]
        send("\n".join(msg)); state["last_summary_day"] = now_ist.strftime("%Y-%m-%d")

    if log:
        print("Data notes:\n  " + "\n  ".join(log))
        if len(log) >= 10: send("v10 alerts: data download problems this run:\n" + "\n".join(log[:10]))
    state["latest_ms"] = int(latest); state["last_run_ist"] = now_ist.strftime("%Y-%m-%d %H:%M"); state["latest_candle_ist"] = ist(latest)
    json.dump(state, open(state_path, "w"))
    print(f"Run OK at {now_ist:%d %b %H:%M} IST | {len(trades)} trades in history | paper balance Rs {bal:,.0f}")


if __name__ == "__main__":
    if "--test" in sys.argv: send("v10 alerts test message: Telegram is connected."); sys.exit(0)
    main()
