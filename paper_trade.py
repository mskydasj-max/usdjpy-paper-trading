"""USD/JPY retrospective H1 paper journal. No broker access or live orders."""
import glob, json, math, os, subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "state.json"
DASHBOARD_PATH = ROOT / "dashboard.json"
DATA_DIR = ROOT / ".paper_data"
INITIAL_JPY = 100000.0
UNITS = 1000
ROUND_TRIP_COST_JPY_PER_USD = 0.02
MAX_RISK_JPY = 500.0
DAILY_LIMIT_JPY = 1000.0
TOTAL_LIMIT_JPY = 5000.0
REWARD_MULTIPLE = 1.5

def atomic_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)

def load_bars():
    DATA_DIR.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=45)).strftime("%Y-%m-%d")
    end = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    cmd = ["npx", "--yes", "dukascopy-node", "-i", "usdjpy", "-from", start,
           "-to", end, "-t", "h1", "-f", "csv", "-dir", str(DATA_DIR),
           "-fn", "usdjpy_h1", "--cache", "--retries", "3"]
    print("Downloading Dukascopy H1:", start, "to", end)
    subprocess.run(cmd, check=True, timeout=420)
    paths = glob.glob(str(DATA_DIR / "*.csv"))
    if not paths:
        raise RuntimeError("No Dukascopy CSV downloaded")
    raw = pd.read_csv(max(paths, key=os.path.getmtime))
    raw.columns = [str(c).lower() for c in raw.columns]
    required = {"timestamp", "open", "high", "low", "close"}
    if not required.issubset(raw.columns):
        raise RuntimeError(f"Unexpected columns: {raw.columns.tolist()}")
    raw["time"] = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
    d = raw.set_index("time")[["open", "high", "low", "close"]].astype(float).sort_index()
    d = d[~d.index.duplicated(keep="last")].dropna()
    d = d[d.index < pd.Timestamp(now).floor("h")]
    if len(d) < 300:
        raise RuntimeError("Not enough complete H1 bars")
    for n in (50, 200):
        d[f"ma{n}"] = d.close.rolling(n).mean()
    pc = d.close.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(),
                    (d.low - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    return d

def run():
    d = load_bars()
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        print("Resuming journal from", state["last_bar"])
    else:
        state = {"cash": INITIAL_JPY, "position": None,
                 "last_bar": d.index[-1].isoformat(), "halted": False,
                 "day": None, "day_start": INITIAL_JPY,
                 "blocked_day": False, "pending_exit": False, "trades": []}
        print("Initialized new journal without retroactive trades")

    last_bar = pd.Timestamp(state["last_bar"])
    if last_bar < d.index[0]:
        raise RuntimeError("Saved bar older than data window; journal unchanged")
    new_bars = d.index[d.index > last_bar]
    for t in new_bars:
        i = d.index.get_loc(t)
        if i < 2:
            continue
        b, p, q = d.iloc[i], d.iloc[i-1], d.iloc[i-2]
        o, h, l, c = (float(b[x]) for x in ("open", "high", "low", "close"))
        day = t.strftime("%Y-%m-%d")
        if day != state["day"]:
            state["day"] = day
            state["day_start"] = state["cash"]
            state["blocked_day"] = False

        pos = state["position"]
        if pos is not None:
            side, entry, sl, tp = pos["side"], pos["entry"], pos["sl"], pos["tp"]
            exit_price = None
            reason = None
            if state["pending_exit"]:
                exit_price, reason = o, "risk_exit_next_open"
            elif side == 1:
                if o <= sl: exit_price, reason = o, "gap_stop"
                elif o >= tp: exit_price, reason = o, "gap_target"
                elif l <= sl: exit_price, reason = sl, "stop"
                elif h >= tp: exit_price, reason = tp, "target"
            else:
                if o >= sl: exit_price, reason = o, "gap_stop"
                elif o <= tp: exit_price, reason = o, "gap_target"
                elif h >= sl: exit_price, reason = sl, "stop"
                elif l <= tp: exit_price, reason = tp, "target"
            if exit_price is not None:
                pnl = (exit_price-entry)*side*UNITS - ROUND_TRIP_COST_JPY_PER_USD*UNITS
                state["cash"] += pnl
                state["trades"].append({"opened": pos["opened"], "closed": t.isoformat(),
                    "side": "BUY" if side == 1 else "SELL", "entry": entry,
                    "exit": exit_price, "pnl_jpy": round(pnl, 2), "reason": reason})
                state["position"] = None
                state["pending_exit"] = False

        if state["position"] is None and not state["halted"] and not state["blocked_day"]:
            side = int(q.ma50 <= q.ma200 and p.ma50 > p.ma200) - int(q.ma50 >= q.ma200 and p.ma50 < p.ma200)
            dist = 1.5 * float(p.atr)
            if side and math.isfinite(dist) and dist > 0 and dist*UNITS + ROUND_TRIP_COST_JPY_PER_USD*UNITS <= MAX_RISK_JPY:
                state["position"] = {"side": side, "entry": o,
                    "sl": o-side*dist, "tp": o+side*dist*REWARD_MULTIPLE,
                    "opened": t.isoformat()}
                pos = state["position"]
                exit_price = None
                reason = None
                if side == 1:
                    if l <= pos["sl"]: exit_price, reason = pos["sl"], "stop_same_bar"
                    elif h >= pos["tp"]: exit_price, reason = pos["tp"], "target_same_bar"
                else:
                    if h >= pos["sl"]: exit_price, reason = pos["sl"], "stop_same_bar"
                    elif l <= pos["tp"]: exit_price, reason = pos["tp"], "target_same_bar"
                if exit_price is not None:
                    pnl = (exit_price-o)*side*UNITS - ROUND_TRIP_COST_JPY_PER_USD*UNITS
                    state["cash"] += pnl
                    state["trades"].append({"opened": t.isoformat(), "closed": t.isoformat(),
                        "side": "BUY" if side == 1 else "SELL", "entry": o,
                        "exit": exit_price, "pnl_jpy": round(pnl, 2), "reason": reason})
                    state["position"] = None

        pos = state["position"]
        floating = ((c-pos["entry"])*pos["side"]*UNITS -
                    ROUND_TRIP_COST_JPY_PER_USD*UNITS) if pos else 0
        equity = state["cash"] + floating
        if equity <= INITIAL_JPY-TOTAL_LIMIT_JPY:
            state["halted"] = True
            if pos: state["pending_exit"] = True
        if equity <= state["day_start"]-DAILY_LIMIT_JPY:
            state["blocked_day"] = True
            if pos: state["pending_exit"] = True
        state["last_bar"] = t.isoformat()

    latest = d.iloc[-1]
    public = {"updated_at": datetime.now(timezone.utc).isoformat(),
        "last_processed_bar": state["last_bar"],
        "cash_jpy": round(state["cash"], 2),
        "realized_pnl_jpy": round(state["cash"]-INITIAL_JPY, 2),
        "position": state["position"], "trades": state["trades"][-100:],
        "trade_count": len(state["trades"]), "halted": state["halted"],
        "latest_price": round(float(latest.close), 4),
        "ma50": round(float(latest.ma50), 4),
        "ma200": round(float(latest.ma200), 4),
        "note": "Retrospective Dukascopy bid-only H1 paper simulation; 2-sen round-trip illustrative cost. No live orders, slippage or swap."}
    atomic_json(STATE_PATH, state)
    atomic_json(DASHBOARD_PATH, public)
    print("Processed", len(new_bars), "bars; cash", round(state["cash"], 2),
          "trades", len(state["trades"]), "position", state["position"])
    print("Simulation only. No live orders.")

if __name__ == "__main__":
    run()
