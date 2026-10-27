"""v11 paper-trading alerts -> Telegram. Runs every 15 minutes on GitHub Actions (free).
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
MAX_OPEN = 6                        # at most 6 paper trades open at once
TOTAL_RISK = 0.01                   # 1% of the account at risk in total -> 1% / 6 = 0.167% per trade
RISK_PER_TRADE = TOTAL_RISK / MAX_OPEN
COST_ROUND_TRIP = 0.0014
FUNDING_PER_8H = 0.0001
DAILY_SUMMARY_IST_HOUR = 9          # morning summary around 09:00 IST
# ---- manual-trading scan (published hourly to scan/latest.json, Telegram summary once a day) ----
ALIVE_HOURS = 1                     # "bot alive" Telegram message every hour (change to 6 for fewer messages)
SCAN_HOUR_IST = 10                  # Telegram "manual setup scan" at the first run after 10:00 IST
SCAN_RISK_INR = 150                 # risk per manual trade (v2.0 cap); 2 trades = Rs 300 = 0.86% of capital
SCAN_LEVERAGE = 5
EXTRA_1H = ["NEAR", "APT", "ARB", "OP", "SUI", "INJ", "FIL", "ATOM", "BCH", "ETC"]   # price/ATR for funding-squeeze checks
COINS_1H = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC", "DOT", "TRX"]
COINS_15M = ["BTC", "ETH", "SOL"]
PRESETS = {
    # target = next support/resistance (lowest low / highest high of the last 50 candles); skip if < 1R away; cap 4R
    "1H":  dict(tf="1h",  coins=COINS_1H,  breakout=10, ema=100, adx_min=0,  stop_atr=6.0, target_r=4.0, max_bars=480, struct=50, min_r=1.0),
    "15M": dict(tf="15m", coins=COINS_15M, breakout=80, ema=100, adx_min=20, stop_atr=8.0, target_r=4.0, max_bars=768, struct=50, min_r=1.0),
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
    HH = pd.Series(h).rolling(p["struct"]).max().values; LL = pd.Series(l).rolling(p["struct"]).min().values
    def struct_target(k, s, lvl, risk):
        """Next support (short) / resistance (long) beyond the breakout; None = too close (skip the trade)."""
        st = LL[k] if s == -1 else HH[k]
        dist = s * (st - lvl) / risk if not np.isnan(st) else np.nan
        if np.isnan(dist) or dist <= 0.05: return lvl + s * p["target_r"] * risk      # open field: cap at 4R
        if dist < p["min_r"]: return None
        return lvl + s * min(dist, p["target_r"]) * risk
    trades = []; i = 300; pending = None
    while i < n - 1:
        s = d[i]
        if s == 0 or np.isnan(hi[i]): i += 1; continue
        lvl = hi[i] if s == 1 else lo[i]; risk = p["stop_atr"] * a[i]; sl = lvl - s * risk; e_ = i + 1
        triggered = h[e_] >= lvl if s == 1 else l[e_] <= lvl
        if not triggered or not (0.0015 <= risk / lvl <= 0.15): i += 1; continue
        entry = max(lvl, o[e_]) if s == 1 else min(lvl, o[e_])
        if s * (entry - sl) <= 0: i += 1; continue
        tp = struct_target(i, s, lvl, risk)
        if tp is None: i += 1; continue
        ex = None; why = "OPEN"
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
            lvl = hi[k] if s == 1 else lo[k]; risk = p["stop_atr"] * a[k]; tpk = struct_target(k, s, lvl, risk)
            if tpk is not None and 0.0015 <= risk / lvl <= 0.15:
                pending = dict(coin=coin, chart=p["tf"].upper(), order="BUY STOP" if s == 1 else "SELL STOP", trigger=float(lvl),
                               stop=float(lvl - s * risk), target=float(tpk))
    return trades, pending


def paper_account(trades, start_ms):
    """Fake-money balance since PAPER_START: 1% total risk split over at most MAX_OPEN trades (0.167% each)."""
    T = sorted([x for x in trades if x["entry_ms"] >= start_ms], key=lambda x: x["entry_ms"])
    realised, book, rows = [], [], []
    for x in T:
        eq_now = START_CAPITAL_INR + sum(p for ex, p in realised if ex <= x["entry_ms"])
        book = [b for b in book if b["status"] == "OPEN" or b["exit_ms"] > x["entry_ms"]]
        if len(book) >= MAX_OPEN:
            rows.append(dict(x, taken=False, risk_inr=0.0, pnl_inr=0.0)); continue
        # size so that a stop-out INCLUDING fees costs about 1R (the stop distance plus the round-trip fee = the risk budget)
        frac = abs(x["entry"] - x["stop"]) / (abs(x["entry"] - x["stop"]) + COST_ROUND_TRIP * x["entry"])
        y = dict(x, taken=True, risk_inr=eq_now * RISK_PER_TRADE, frac=frac); y["pnl_inr"] = y["R"] * y["risk_inr"] * frac
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


# ------------------------------ manual-trading scan ------------------------------
def scan_publish(data, trades, pend, rows, state, now_ist):
    """Rank the resting v11 orders for manual execution and publish them (scan/latest.json, hourly)."""
    open_now = {(x["coin"], x["chart"]) for x in trades if x["status"] == "OPEN"}
    coins = {}
    for (c, tf), d in data.items():
        if tf != "1h" or len(d) < 60: continue
        h, l, cl = d.high.values, d.low.values, d.close.values; a = atr(h, l, cl)
        info = dict(price=float(cl[-1]), atr_1h=float(a[-1]), atr_1h_pct=float(a[-1] / cl[-1] * 100),
                    chg_24h_pct=float((cl[-1] / cl[-25] - 1) * 100) if len(cl) > 25 else None)
        h4 = data.get((c, "4h"))
        if h4 is not None and len(h4) > 120:
            c4 = h4.close.values; e4 = ema(c4, 100); a4 = adx(h4.high.values, h4.low.values, c4)
            info.update(trend_4h="UP" if c4[-1] > e4[-1] else "DOWN", ema100_4h=float(e4[-1]), adx_4h=float(a4[-1]))
        coins[c] = info
    cands = []
    for p_ in pend:
        c = p_["coin"]; side = 1 if p_["order"] == "BUY STOP" else -1
        tfk = p_["chart"].lower(); d = data.get((c, tfk))
        if d is None or c not in coins: continue
        price = float(d.close.values[-1]); trig, sl, tp = p_["trigger"], p_["stop"], p_["target"]
        dist = side * (trig - price)                       # > 0: price still has to travel to the trigger
        if dist <= 0: continue                             # already through the level: do not chase
        risk_px = abs(trig - sl); rr = abs(tp - trig) / risk_px
        a1 = coins[c]["atr_1h"]; dist_atr = dist / a1
        qty = SCAN_RISK_INR / USDT_INR / risk_px; notional = qty * trig
        close_s = 40 * (1 - min(dist_atr, 4) / 4); rew_s = 30 * min(rr, 4) / 4; adx_s = 30 * min(coins[c].get("adx_4h", 0), 40) / 40
        cands.append(dict(coin=c, chart=p_["chart"], side="LONG" if side == 1 else "SHORT", order=p_["order"],
                          price=price, trigger=trig, stop=sl, target=tp, rr=round(rr, 2),
                          dist_pct=round(dist / price * 100, 3), dist_atr_1h=round(dist_atr, 2), stop_pct=round(risk_px / trig * 100, 2),
                          qty=qty, notional_usdt=round(notional, 2), margin_usdt_5x=round(notional / SCAN_LEVERAGE, 2),
                          risk_inr=SCAN_RISK_INR, reward_inr=round(SCAN_RISK_INR * rr, 0),
                          score=round(close_s + rew_s + adx_s), in_paper_trade=(c, p_["chart"]) in open_now))
    cands.sort(key=lambda z: -z["score"])
    hour_key = now_ist.strftime("%Y-%m-%d %H")
    out = dict(generated_ist=now_ist.strftime("%Y-%m-%d %H:%M"), hour_key=hour_key, usdt_inr=USDT_INR,
               rules="v11: stop order at trigger (1H: 10-candle high/low, 15M: 80-candle), 4H EMA100 trend, stop 6xATR(1H) / 8xATR(15M), "
                     "target = next 50-candle structure (1-4R), time-stop 20 days (1H) / 8 days (15M). Funding squeeze: buy when Binance "
                     "funding <= -0.05%, stop 4xATR(1H), exit after 72h.",
               candidates=cands, coins=coins,
               paper_open=[dict(coin=x["coin"], chart=x["chart"], side=x["side"], entry=x["entry"], stop=x["stop"], target=x["target"])
                           for x in trades if x["status"] == "OPEN"])
    path = os.path.join("scan", "latest.json"); old_key = None
    if os.path.exists(path):
        try: old_key = json.load(open(path)).get("hour_key")
        except Exception: pass
    if old_key != hour_key:                                # publish at most once per hour (keeps the repo history small)
        os.makedirs("scan", exist_ok=True); json.dump(out, open(path, "w"), indent=1)
    # Telegram: once a day after SCAN_HOUR_IST
    day = now_ist.strftime("%Y-%m-%d")
    if now_ist.hour >= SCAN_HOUR_IST and state.get("last_scan_day") != day:
        top, used = [], set()
        for z in cands:
            if z["coin"] in used or z["dist_atr_1h"] > 3: continue
            top.append(z); used.add(z["coin"])
            if len(top) == 2: break
        msg = [f"MANUAL SETUP SCAN {now_ist:%d %b %H:%M} IST (tested v11 rules, Rs {SCAN_RISK_INR} risk each, isolated {SCAN_LEVERAGE}x)"]
        if not top: msg.append("No setup within 3 ATR of its trigger today. Stay flat.")
        for k, z in enumerate(top, 1):
            msg.append(f"{k}) {z['coin']} {z['chart']} {z['side']} | {('STOP-LIMIT BUY' if z['side'] == 'LONG' else 'STOP-LIMIT SELL')} trigger {fp(z['trigger'])} (now {fp(z['price'])}, {z['dist_pct']}% away)\n"
                       f"   SL {fp(z['stop'])} ({z['stop_pct']}%) | TP {fp(z['target'])} ({z['rr']}R, ~Rs {z['reward_inr']:.0f}) | qty {z['qty']:.4g} | margin ~{z['margin_usdt_5x']} USDT | score {z['score']}")
        msg.append("Claude adds the news check in the morning briefing. Cancel unfilled orders at the next scan.")
        send("\n".join(msg)); state["last_scan_day"] = day


# ------------------------------ main ------------------------------
def main():
    log = []; data = {}
    need = sorted({(c, "4h") for c in set(COINS_1H) | set(COINS_15M)} | {(c, "1h") for c in COINS_1H + EXTRA_1H} | {(c, "15m") for c in COINS_15M})
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

    if state is None or state.get("version") != "v11":   # first run (or upgrade from v10): set the baseline, no historical spam
        opn = [r for r in rows if r["taken"] and r["status"] == "OPEN"]
        msg = [f"v11 paper alerts are ON: max {MAX_OPEN} trades open, {TOTAL_RISK:.0%} total risk (Rs {START_CAPITAL_INR * RISK_PER_TRADE:,.0f} per trade), "
               f"targets at the next support/resistance (1R to 4R). Checking every 15 min.",
               f"Paper balance: Rs {bal:,.0f} (start Rs {START_CAPITAL_INR:,.0f}, from {PAPER_START})", f"Open paper positions now: {len(opn)}/{MAX_OPEN}"]
        msg += [f"  {x['coin']} {x['chart']} {x['side']} @ {fp(x['entry'])} | SL {fp(x['stop'])} | TP {fp(x['target'])}" for x in opn]
        send("\n".join(msg))
        state = dict(state or {}, version="v11", seen_entries=[x["key"] for x in trades],
                     seen_exits=[x["key"] for x in trades if x["status"] != "OPEN"], last_summary_day=(state or {}).get("last_summary_day", ""))
    else:
        seen_e, seen_x = set(state["seen_entries"]), set(state["seen_exits"])
        guard = state.get("latest_ms", 0) - 2 * MS["1h"]      # only events newer than the previous run are alerted
        for x in sorted(trades, key=lambda z: z["entry_ms"]):
            if x["key"] not in seen_e and x["entry_ms"] < guard: seen_e.add(x["key"])
            if x["status"] != "OPEN" and x["key"] not in seen_x and x["exit_ms"] < guard: seen_x.add(x["key"])
            r = by_key.get(x["key"]); note = ""
            if r is not None and not r["taken"]:      # skipped because 6 trades were already open: no Telegram noise
                if x["key"] not in seen_e: print(f"skipped (already {MAX_OPEN} open): {x['key']}")
                seen_e.add(x["key"]); seen_x.add(x["key"]); continue
            if x["key"] not in seen_e:
                size = "" if r is None or not r["taken"] else f"\nPaper risk Rs {r['risk_inr']:,.0f} | size {r['risk_inr'] * r['frac'] / USDT_INR / abs(x['entry'] - x['stop']):.4g} {x['coin']}"
                n_open = sum(1 for z in rows if z["taken"] and z["status"] == "OPEN")
                tr_ = abs(x['target'] - x['entry']) / abs(x['entry'] - x['stop'])
                send(f"v11 PAPER ENTRY: {x['coin']} {x['chart']} {x['side']}\nEntry {fp(x['entry'])} ({x['chart']} candle from {ist(x['entry_ms'])} IST)\n"
                     f"Stop {fp(x['stop'])} | Target {fp(x['target'])} (about {tr_:.1f}R, next structure){size}\nOpen paper trades: {n_open}/{MAX_OPEN}")
                seen_e.add(x["key"])
            if x["status"] != "OPEN" and x["key"] not in seen_x:
                pnl = "" if r is None or not r["taken"] else f" | Rs {r['pnl_inr']:+,.0f}"
                send(f"v11 PAPER EXIT: {x['coin']} {x['chart']} {x['side']} ({x['status']})\nExit {fp(x['exit'])} ({ist(x['exit_ms'])} IST) | "
                     f"{x['R']:+.2f}R{pnl}\nPaper balance: Rs {bal:,.0f} ({bal / START_CAPITAL_INR - 1:+.2%}){note}")
                seen_x.add(x["key"])
        state["seen_entries"], state["seen_exits"] = sorted(seen_e), sorted(seen_x)

    now_ist = pd.Timestamp(NOW_MS, unit="ms") + IST
    if now_ist.hour == DAILY_SUMMARY_IST_HOUR and state.get("last_summary_day") != now_ist.strftime("%Y-%m-%d"):
        closed = [r for r in rows if r["taken"] and r["status"] != "OPEN"]; opn = [r for r in rows if r["taken"] and r["status"] == "OPEN"]
        w = sum(r["pnl_inr"] > 0 for r in closed)
        msg = [f"v11 DAILY PAPER SUMMARY ({now_ist:%d %b})", f"Balance Rs {bal:,.0f} ({bal / START_CAPITAL_INR - 1:+.2%}) since {PAPER_START}",
               f"Closed trades {len(closed)} (wins {w}, losses {len(closed) - w}) | open {len(opn)} (unrealised Rs {sum(r['pnl_inr'] for r in opn):+,.0f})"]
        if pend:
            msg.append("Resting orders now:")
            msg += [f"  {p_['coin']} {p_['chart']} {p_['order']} {fp(p_['trigger'])} | SL {fp(p_['stop'])} | TP {fp(p_['target'])}" for p_ in pend]
        send("\n".join(msg)); state["last_summary_day"] = now_ist.strftime("%Y-%m-%d")

    # ---- "bot alive" heartbeat every ALIVE_HOURS ----
    state["runs_since_alive"] = state.get("runs_since_alive", 0) + 1
    if NOW_MS - state.get("last_alive_ms", 0) >= ALIVE_HOURS * 3_600_000 - 300_000:
        in_trade = sum(1 for x in trades if x["status"] == "OPEN")
        paper_open = sum(1 for r in rows if r["taken"] and r["status"] == "OPEN")
        send(f"\u2705 Bot alive {now_ist:%d %b %H:%M} IST | {state['runs_since_alive']} runs OK since last check\n"
             f"Latest candle: {ist(latest)} IST | data problems: {len(log)}\n"
             f"Streams in a trade: {in_trade}/15 | resting orders: {len(pend)} | paper positions: {paper_open}/{MAX_OPEN}\n"
             f"Paper balance: Rs {bal:,.0f} ({bal / START_CAPITAL_INR - 1:+.2%})")
        state["last_alive_ms"] = NOW_MS; state["runs_since_alive"] = 0

    try: scan_publish(data, trades, pend, rows, state, now_ist)
    except Exception as e: print("scan failed:", e)

    if log:
        print("Data notes:\n  " + "\n  ".join(log))
        if len(log) >= 10: send("v11 alerts: data download problems this run:\n" + "\n".join(log[:10]))
    state["latest_ms"] = int(latest); state["last_run_ist"] = now_ist.strftime("%Y-%m-%d %H:%M"); state["latest_candle_ist"] = ist(latest)
    json.dump(state, open(state_path, "w"))
    print(f"Run OK at {now_ist:%d %b %H:%M} IST | {len(trades)} trades in history | paper balance Rs {bal:,.0f}")


if __name__ == "__main__":
    if "--test" in sys.argv: send("v11 alerts test message: Telegram is connected."); sys.exit(0)
    main()
