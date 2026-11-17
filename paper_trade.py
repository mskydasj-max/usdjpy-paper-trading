"""USD/JPY paper trading only. No broker connections or real orders."""
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "state.json"
PUBLIC_FILE = ROOT / "dashboard.json"
INITIAL_CASH = 100000.0
UNITS = 1000
COST_JPY = 40.0  # 4 sen round-trip per USD x 1000
MAX_RISK_JPY = 500.0
MAX_TOTAL_LOSS = 5000.0


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_bars():
    df = yf.download("JPY=X", period="60d", interval="1h", auto_adjust=False, progress=False, threads=False)
    if df.empty:
        raise RuntimeError("No USD/JPY data. State was not changed.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # Only bars whose entire 1-hour window has ended.
    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index + pd.Timedelta(hours=1) <= now]
    if len(df) < 220:
        raise RuntimeError("Not enough completed hourly candles.")
    df["ma50"] = df.close.rolling(50).mean()
    df["ma200"] = df.close.rolling(200).mean()
    prev_close = df.close.shift()
    tr = pd.concat([(df.high - df.low), (df.high - prev_close).abs(), (df.low - prev_close).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    return df


def save_json(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def run():
    df = load_bars()
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    else:
        # Start fresh; no retroactive hypothetical trades.
        state = {
            "cash": INITIAL_CASH, "position": None, "trades": [],
            "last_bar": df.index[-1].isoformat(), "halted": False,
            "created_at": utc_now(), "version": 1
        }
    last_bar = pd.Timestamp(state["last_bar"])
    new = df[df.index > last_bar]
    if len(new) and last_bar < df.index[0]:
        raise RuntimeError("Historical data gap exceeds Yahoo 60-day window. State not changed.")

    for ts, bar in new.iterrows():
        i = df.index.get_loc(ts)
        if i < 201:
            continue
        prev = df.iloc[i - 1]
        prev2 = df.iloc[i - 2]
        # Existing position: stop/target evaluated on current H1 bar.
        pos = state["position"]
        if pos:
            side = pos["side"]
            stop_hit = (bar.low <= pos["stop"]) if side == 1 else (bar.high >= pos["stop"])
            target_hit = (bar.high >= pos["target"]) if side == 1 else (bar.low <= pos["target"])
            # Conservative tie-break: stop wins if both reached in same candle.
            if stop_hit or target_hit:
                price = pos["stop"] if stop_hit else pos["target"]
                # Opening gaps: stop fills at worse opening price, never at guaranteed stop.
                if stop_hit:
                    price = min(price, float(bar.open)) if side == 1 else max(price, float(bar.open))
                pnl = (price - pos["entry"]) * side * UNITS - COST_JPY
                state["cash"] = round(state["cash"] + pnl, 2)
                state["trades"].append({
                    "opened": pos["opened"], "closed": ts.isoformat(),
                    "side": "BUY" if side == 1 else "SELL",
                    "entry": pos["entry"], "exit": round(float(price), 4),
                    "pnl_jpy": round(pnl, 2),
                    "reason": "STOP" if stop_hit else "TARGET"
                })
                state["position"] = None
        if state["cash"] <= INITIAL_CASH - MAX_TOTAL_LOSS:
            state["halted"] = True
        # Enter at the opening of current bar, based only on PREVIOUS closed bar.
        # Never enter on a candle that just closed a trade.
        closed_this_bar = bool(state["trades"] and state["trades"][-1]["closed"] == ts.isoformat())
        if not state["position"] and not state["halted"] and not closed_this_bar:
            crossed_up = prev2.ma50 <= prev2.ma200 and prev.ma50 > prev.ma200
            crossed_down = prev2.ma50 >= prev2.ma200 and prev.ma50 < prev.ma200
            if (crossed_up or crossed_down) and math.isfinite(prev.atr14):
                side = 1 if crossed_up else -1
                distance = 1.5 * float(prev.atr14)
                if distance > 0 and distance * UNITS + COST_JPY <= MAX_RISK_JPY:
                    entry = float(bar.open)
                    state["position"] = {
                        "side": side, "entry": round(entry, 4),
                        "stop": round(entry - side * distance, 4),
                        "target": round(entry + side * distance * 1.5, 4),
                        "opened": ts.isoformat()
                    }
        state["last_bar"] = ts.isoformat()

    state["updated_at"] = utc_now()
    save_json(STATE_FILE, state)
    latest = df.iloc[-1]
    public = {
        "updated_at": state["updated_at"], "last_processed_bar": state["last_bar"],
        "cash_jpy": state["cash"], "realized_pnl_jpy": round(state["cash"] - INITIAL_CASH, 2),
        "position": state["position"], "trades": state["trades"][-100:],
        "trade_count": len(state["trades"]), "halted": state["halted"],
        "latest_price": round(float(latest.close), 4),
        "ma50": round(float(latest.ma50), 4), "ma200": round(float(latest.ma200), 4),
        "note": "Simulation only. Yahoo H1 quotes; fixed 4-sen round-trip cost, no broker fills, spread dynamics, slippage or swap. Not investment advice."
    }
    save_json(PUBLIC_FILE, public)
    print("Processed", len(new), "bars; cash", state["cash"], "trades", len(state["trades"]), "position", state["position"])


if __name__ == "__main__":
    run()
