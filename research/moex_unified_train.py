#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

SEED = 20260904
OUT = Path("artifacts/moex_unified")
OUT.mkdir(parents=True, exist_ok=True)
TODAY = os.environ.get("MOEX_TILL", "2026-09-04")

FEATURES = [
    "direction",
    "side_ret1", "side_ret3", "side_ret6", "side_ret12",
    "atr_pct", "vol_ratio", "efficiency", "noise", "impulse",
    "rsi_side", "rsi8_side", "rsi21_side",
    "macd_side", "ewo_side", "zema20_side", "zema55_side",
    "yellow_strength", "extreme_strength", "exhaust_score",
    "reject_score", "div_score", "activity_rank", "wpr_side",
    "wick_side", "close_side", "qdiv_score",
    "ell_valid", "ell_fibq", "ell_w5ratio", "ell_age_norm",
    "ell_trunc", "ell_diag", "ell_w5of5",
]


def fetch_moex_candles(secid: str, date_from: str, date_till: str) -> pd.DataFrame:
    """Fetch H1 candles from official MOEX ISS. Tries board and board-agnostic paths."""
    endpoints = [
        f"https://iss.moex.com/iss/engines/stock/markets/index/boards/SNDX/securities/{secid}/candles.json",
        f"https://iss.moex.com/iss/engines/stock/markets/index/securities/{secid}/candles.json",
    ]
    headers = {"User-Agent": "MOEX-Unified-Reversal-Research/1.0"}
    all_frames: List[pd.DataFrame] = []
    y0 = pd.Timestamp(date_from).year
    y1 = pd.Timestamp(date_till).year
    for year in range(y0, y1 + 1):
        a = max(pd.Timestamp(date_from), pd.Timestamp(f"{year}-01-01"))
        b = min(pd.Timestamp(date_till), pd.Timestamp(f"{year}-12-31"))
        got_year = False
        for endpoint in endpoints:
            start = 0
            rows = []
            cols = None
            while True:
                params = {
                    "interval": 60,
                    "from": a.strftime("%Y-%m-%d"),
                    "till": b.strftime("%Y-%m-%d"),
                    "start": start,
                    "iss.meta": "off",
                    "iss.only": "candles",
                }
                r = requests.get(endpoint, params=params, headers=headers, timeout=30)
                if r.status_code == 429:
                    time.sleep(1.0)
                    continue
                r.raise_for_status()
                js = r.json()
                block = js.get("candles", {})
                data = block.get("data", [])
                cols = block.get("columns", cols)
                if not data:
                    break
                rows.extend(data)
                start += len(data)
                if len(data) < 500:
                    break
                time.sleep(0.03)
            if rows and cols:
                dfy = pd.DataFrame(rows, columns=cols)
                dfy["secid_source"] = secid
                all_frames.append(dfy)
                got_year = True
                break
        print(f"{secid} {year}: {'OK' if got_year else 'no H1'}")
    if not all_frames:
        return pd.DataFrame()
    df = pd.concat(all_frames, ignore_index=True)
    return df


def load_history() -> pd.DataFrame:
    legacy = fetch_moex_candles("MICEXINDEXCF", "1999-01-01", min(TODAY, "2017-12-31"))
    modern = fetch_moex_candles("IMOEX", "2018-01-01", TODAY)
    parts = [x for x in (legacy, modern) if not x.empty]
    if not parts:
        # Last attempt: IMOEX code across the whole requested range.
        fallback = fetch_moex_candles("IMOEX", "1999-01-01", TODAY)
        parts = [fallback] if not fallback.empty else []
    if not parts:
        raise RuntimeError("Official MOEX ISS returned no IMOEX/MICEXINDEXCF H1 candles")
    df = pd.concat(parts, ignore_index=True)
    time_col = "begin" if "begin" in df.columns else "end"
    df["time"] = pd.to_datetime(df[time_col], errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "value"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    df = df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    df["volume"] = df["volume"].fillna(0.0)
    df["value"] = df["value"].fillna(0.0)
    df.to_csv(OUT / "imoex_h1_official.csv", index=False)
    return df


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df.close.shift(1)
    tr = pd.concat([(df.high - df.low), (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def roll_norm(s: pd.Series, n: int) -> pd.Series:
    lo = s.rolling(n, min_periods=max(10, n // 4)).min()
    hi = s.rolling(n, min_periods=max(10, n // 4)).max()
    return ((s - lo) / (hi - lo).replace(0, np.nan)).clip(0, 1).fillna(0.5)


def zscore(s: pd.Series, n: int) -> pd.Series:
    m = s.rolling(n, min_periods=max(10, n // 4)).mean()
    sd = s.rolling(n, min_periods=max(10, n // 4)).std(ddof=0)
    return ((s - m) / sd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def wpr(df: pd.DataFrame, n: int) -> pd.Series:
    hh = df.high.rolling(n, min_periods=n).max()
    ll = df.low.rolling(n, min_periods=n).min()
    return (-100 * (hh - df.close) / (hh - ll).replace(0, np.nan)).fillna(-50.0)


def closeness(x: float, target: float, width: float) -> float:
    return max(0.0, 1.0 - abs(x - target) / width) if np.isfinite(x) else 0.0


def fib_quality(w2: float, w3: float, w4: float, w5: float) -> float:
    q2 = max(closeness(w2, 0.5, 0.35), closeness(w2, 0.618, 0.35))
    q3 = max(closeness(w3, 1.618, 1.15), closeness(w3, 2.618, 1.15))
    q4 = max(closeness(w4, 0.382, 0.30), closeness(w4, 0.5, 0.30))
    q5 = max(closeness(w5, 0.618, 0.55), closeness(w5, 1.0, 0.55), closeness(w5, 1.618, 0.55))
    return float(np.clip((q2 + q3 + q4 + q5) / 4.0, 0, 1))


def add_elliott_features(df: pd.DataFrame) -> pd.DataFrame:
    """Causal pivot-confirmed 0-4 anchors with live Wave-5 terminal features."""
    n = len(df)
    pivot_len = 5
    high = df.high.to_numpy(float)
    low = df.low.to_numpy(float)
    a = df.atr14.to_numpy(float)
    out = {
        k: np.zeros(n, dtype=float)
        for k in [
            "ell_sell_valid", "ell_buy_valid", "ell_sell_fibq", "ell_buy_fibq",
            "ell_sell_w5ratio", "ell_buy_w5ratio", "ell_sell_age", "ell_buy_age",
            "ell_sell_trunc", "ell_buy_trunc", "ell_sell_diag", "ell_buy_diag",
            "ell_sell_w5of5", "ell_buy_w5of5", "ell_sell_extreme", "ell_buy_extreme",
        ]
    }
    pivots: List[Tuple[int, int, float]] = []  # (index,type(+1 high/-1 low),price)
    bull_anchor = None
    bear_anchor = None

    def add_pivot(idx: int, typ: int, price: float):
        nonlocal bull_anchor, bear_anchor, pivots
        if pivots and pivots[-1][1] == typ:
            old = pivots[-1]
            better = (typ == 1 and price >= old[2]) or (typ == -1 and price <= old[2])
            if better:
                pivots[-1] = (idx, typ, price)
        else:
            pivots.append((idx, typ, price))
        if len(pivots) > 40:
            pivots = pivots[-40:]
        if len(pivots) >= 5:
            p5 = pivots[-5:]
            types = [p[1] for p in p5]
            if types == [-1, 1, -1, 1, -1]:
                bull_anchor = tuple(p5)
            elif types == [1, -1, 1, -1, 1]:
                bear_anchor = tuple(p5)

    for i in range(n):
        j = i - pivot_len
        if j >= pivot_len:
            sl = slice(j - pivot_len, i + 1)
            is_hi = high[j] >= np.nanmax(high[sl])
            is_lo = low[j] <= np.nanmin(low[sl])
            if is_hi and is_lo:
                if pivots:
                    prev = pivots[-1][2]
                    if abs(high[j] - prev) >= abs(low[j] - prev):
                        is_lo = False
                    else:
                        is_hi = False
                else:
                    is_lo = False
            if is_hi:
                add_pivot(j, 1, high[j])
            if is_lo:
                add_pivot(j, -1, low[j])

        atr_i = a[i] if np.isfinite(a[i]) and a[i] > 0 else max(high[i] - low[i], 1e-9)

        if bull_anchor is not None:
            p0, p1, p2, p3, p4 = bull_anchor
            i0, i1, i2, i3, i4 = [x[0] for x in bull_anchor]
            if i - i4 > 80 or low[i] < p4[2] - 1.2 * atr_i:
                bull_anchor = None
            else:
                w1 = p1[2] - p0[2]
                w3abs = p3[2] - p2[2]
                if w1 > 0 and w3abs > 0:
                    p5 = np.nanmax(high[i4:i + 1])
                    prev5 = np.nanmax(high[i4:i]) if i > i4 else -np.inf
                    w2 = (p1[2] - p2[2]) / w1
                    w3 = w3abs / w1
                    w4 = (p3[2] - p4[2]) / w3abs
                    w5 = (p5 - p4[2]) / w1
                    age = i - i4
                    base = 0.236 <= w2 <= 0.886 and w3 >= 0.80 and 0.146 <= w4 <= 0.786 and p3[2] > p1[2]
                    trunc = base and w3 >= 1.45 and abs(p5 - p3[2]) <= 0.35 * atr_i and age >= 2
                    overlap = p4[2] <= p1[2] + 0.20 * atr_i
                    diag = base and overlap and 0.20 <= w5 <= 1.35 and age >= 3
                    valid = base and age >= 2 and 0.15 <= w5 <= 3.0 and (p5 >= p3[2] or trunc or diag)
                    inner_count = sum(1 for q in pivots if q[0] > i4)
                    out["ell_sell_valid"][i] = float(valid)
                    out["ell_sell_fibq"][i] = fib_quality(w2, w3, w4, w5) if base else 0.0
                    out["ell_sell_w5ratio"][i] = np.clip(w5, 0, 3)
                    out["ell_sell_age"][i] = age
                    out["ell_sell_trunc"][i] = float(trunc)
                    out["ell_sell_diag"][i] = float(diag)
                    out["ell_sell_w5of5"][i] = float(valid and inner_count >= 3 and age >= 5)
                    out["ell_sell_extreme"][i] = float(high[i] >= p5 - 1e-12 and p5 >= prev5 - 1e-12)

        if bear_anchor is not None:
            p0, p1, p2, p3, p4 = bear_anchor
            i0, i1, i2, i3, i4 = [x[0] for x in bear_anchor]
            if i - i4 > 80 or high[i] > p4[2] + 1.2 * atr_i:
                bear_anchor = None
            else:
                w1 = p0[2] - p1[2]
                w3abs = p2[2] - p3[2]
                if w1 > 0 and w3abs > 0:
                    p5 = np.nanmin(low[i4:i + 1])
                    prev5 = np.nanmin(low[i4:i]) if i > i4 else np.inf
                    w2 = (p2[2] - p1[2]) / w1
                    w3 = w3abs / w1
                    w4 = (p4[2] - p3[2]) / w3abs
                    w5 = (p4[2] - p5) / w1
                    age = i - i4
                    base = 0.236 <= w2 <= 0.886 and w3 >= 0.80 and 0.146 <= w4 <= 0.786 and p3[2] < p1[2]
                    trunc = base and w3 >= 1.45 and abs(p5 - p3[2]) <= 0.35 * atr_i and age >= 2
                    overlap = p4[2] >= p1[2] - 0.20 * atr_i
                    diag = base and overlap and 0.20 <= w5 <= 1.35 and age >= 3
                    valid = base and age >= 2 and 0.15 <= w5 <= 3.0 and (p5 <= p3[2] or trunc or diag)
                    inner_count = sum(1 for q in pivots if q[0] > i4)
                    out["ell_buy_valid"][i] = float(valid)
                    out["ell_buy_fibq"][i] = fib_quality(w2, w3, w4, w5) if base else 0.0
                    out["ell_buy_w5ratio"][i] = np.clip(w5, 0, 3)
                    out["ell_buy_age"][i] = age
                    out["ell_buy_trunc"][i] = float(trunc)
                    out["ell_buy_diag"][i] = float(diag)
                    out["ell_buy_w5of5"][i] = float(valid and inner_count >= 3 and age >= 5)
                    out["ell_buy_extreme"][i] = float(low[i] <= p5 + 1e-12 and p5 <= prev5 + 1e-12)

    for k, v in out.items():
        df[k] = v
    return df


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df.close
    df["ret1"] = np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan)
    for h in [3, 6, 12]:
        df[f"ret{h}"] = np.log(c / c.shift(h)).replace([np.inf, -np.inf], np.nan)

    df["atr14"] = atr(df, 14)
    df["atr104"] = atr(df, 104)
    safe_atr = df.atr14.replace(0, np.nan)
    df["atr_pct"] = (df.atr14 / c).clip(0, 0.2)
    df["vol_ratio"] = (df.atr14 / df.atr104.replace(0, np.nan)).clip(0.3, 4.0)

    travel = c.diff().abs().rolling(26, min_periods=26).sum()
    displacement = (c - c.shift(26)).abs()
    df["efficiency"] = (displacement / travel.replace(0, np.nan)).clip(0, 1).fillna(0.0)
    flip = (c.diff() * c.diff().shift(1) < 0).astype(float)
    turn_rate = flip.rolling(26, min_periods=26).mean().fillna(0.5)
    df["noise"] = (0.62 * (1 - df.efficiency) + 0.38 * turn_rate).clip(0, 1)
    df["impulse"] = (((df.vol_ratio - 0.82) / 1.10).clip(0, 1) * ((df.efficiency - 0.08) / 0.74).clip(0, 1)).fillna(0)

    df["rsi8"] = rsi(c, 8)
    df["rsi13"] = rsi(c, 13)
    df["rsi21"] = rsi(c, 21)
    e12, e26 = ema(c, 12), ema(c, 26)
    macd = e12 - e26
    macd_sig = ema(macd, 9)
    df["macd_hist"] = macd - macd_sig
    df["macd_atr"] = (df.macd_hist / safe_atr).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["ewo_atr"] = ((ema(c, 5) - ema(c, 35)) / safe_atr).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["zema20"] = ((c - ema(c, 20)) / safe_atr).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["zema55"] = ((c - ema(c, 55)) / safe_atr).replace([np.inf, -np.inf], np.nan).fillna(0)

    windows = [6, 10, 16, 24, 36]
    weights = [0.10, 0.15, 0.20, 0.25, 0.30]
    low_strength = pd.Series(0.0, index=df.index)
    high_strength = pd.Series(0.0, index=df.index)
    for w, wt in zip(windows, weights):
        low_strength += wt * (df.low <= df.low.rolling(w, min_periods=w).min() + 1e-12).astype(float)
        high_strength += wt * (df.high >= df.high.rolling(w, min_periods=w).max() - 1e-12).astype(float)
    df["low_strength"] = low_strength
    df["high_strength"] = high_strength

    # 3D Yellow Cluster family, implemented with causal rolling normalization.
    tr = pd.concat([(df.high - df.low), (df.high - c.shift()).abs(), (df.low - c.shift()).abs()], axis=1).max(axis=1)
    proxy = (tr / (c.abs() * 1e-6).clip(lower=1e-9)) * (1 + (c - df.open).abs() / tr.replace(0, np.nan))
    raw_activity = df.volume.where(df.volume > 0, proxy).clip(lower=0)
    df["activity"] = np.log1p(raw_activity).replace([np.inf, -np.inf], np.nan).fillna(0)
    typical = (df.high + df.low + c) / 3
    p_rank = roll_norm(typical, 160)
    a_rank = roll_norm(df.activity, 160)
    p_vel = p_rank.diff().fillna(0)
    a_vel = a_rank.diff().fillna(0)
    k_raw = np.sqrt((p_rank.rolling(20).std(ddof=0) * a_rank.rolling(20).std(ddof=0)).clip(lower=0)) * p_vel.abs() * a_vel.abs()
    k_rank = roll_norm(k_raw, 160)
    p_ret = np.log(typical / typical.shift()).replace([np.inf, -np.inf], np.nan).fillna(0)
    a_ret = df.activity.diff().fillna(0)
    p_acc = p_ret.diff().fillna(0)
    a_acc = a_ret.diff().fillna(0)
    rv = p_ret.rolling(20).std(ddof=0).fillna(0)
    tensor = (zscore(p_acc, 64).abs() + zscore(a_acc, 64).abs() + zscore(rv.diff().fillna(0), 64).abs()) / 3
    tensor_rank = roll_norm(tensor, 160)
    intensity = (p_ret.rolling(12).std(ddof=0) * a_ret.rolling(12).std(ddof=0)).abs()
    intensity_rank = roll_norm(intensity, 160)
    df["activity_rank"] = a_rank
    df["yellow_strength"] = (0.44 * k_rank + 0.26 * tensor_rank + 0.15 * intensity_rank + 0.15 * a_rank).clip(0, 1)

    df["wpr9"] = wpr(df, 9)
    df["wpr34"] = wpr(df, 34)
    rsi_rank = roll_norm(df.rsi13, 160)
    bull_deep = (rsi_rank <= 0.15) & (df.wpr9 <= -80) & (df.wpr34 <= -75)
    bear_deep = (rsi_rank >= 0.85) & (df.wpr9 >= -20) & (df.wpr34 >= -25)
    bull_recent = bull_deep.rolling(7, min_periods=1).max().astype(bool)
    bear_recent = bear_deep.rolling(7, min_periods=1).max().astype(bool)
    bull_release = bull_recent & (df.wpr9 > df.wpr9.shift()) & (df.rsi13 >= df.rsi13.shift())
    bear_release = bear_recent & (df.wpr9 < df.wpr9.shift()) & (df.rsi13 <= df.rsi13.shift())
    df["bull_exhaust"] = ((1 - rsi_rank) + 0.35 * bull_release.astype(float)).clip(0, 1)
    df["bear_exhaust"] = (rsi_rank + 0.35 * bear_release.astype(float)).clip(0, 1)

    eq = ema(c, 34)
    stretch = (c - eq) / safe_atr
    stretch_rank = roll_norm(stretch, 160)
    span = (df.high - df.low).replace(0, np.nan)
    df["lower_wick"] = ((np.minimum(df.open, c) - df.low) / span).clip(0, 1).fillna(0)
    df["upper_wick"] = ((df.high - np.maximum(df.open, c)) / span).clip(0, 1).fillna(0)
    df["close_pos"] = ((c - df.low) / span).clip(0, 1).fillna(0.5)
    df["bull_reject"] = ((1 - stretch_rank) * 0.55 + df.lower_wick * 0.30 + df.close_pos * 0.15).clip(0, 1)
    df["bear_reject"] = (stretch_rank * 0.55 + df.upper_wick * 0.30 + (1 - df.close_pos) * 0.15).clip(0, 1)

    signed_flow = df.activity * np.sign(c - df.open)
    flow_osc = ema(signed_flow, 5) - ema(signed_flow, 13)
    mf_mult = (((c - df.low) - (df.high - c)) / span).replace([np.inf, -np.inf], np.nan).fillna(0)
    vol_for_cmf = df.volume.where(df.volume > 0, np.exp(df.activity) - 1)
    cmf = (mf_mult * vol_for_cmf).rolling(20).sum() / vol_for_cmf.rolling(20).sum().replace(0, np.nan)
    cmf = cmf.fillna(0)
    bull_pts = (
        ((df.low < df.low.shift(8)) & (df.rsi13 > df.rsi13.shift(8))).astype(float)
        + ((df.low < df.low.shift(13)) & (df.macd_hist > df.macd_hist.shift(13))).astype(float)
        + 0.8 * ((df.low < df.low.shift(21)) & (flow_osc > flow_osc.shift(21))).astype(float)
        + 0.7 * ((df.low < df.low.shift(13)) & (cmf > cmf.shift(13))).astype(float)
    )
    bear_pts = (
        ((df.high > df.high.shift(8)) & (df.rsi13 < df.rsi13.shift(8))).astype(float)
        + ((df.high > df.high.shift(13)) & (df.macd_hist < df.macd_hist.shift(13))).astype(float)
        + 0.8 * ((df.high > df.high.shift(21)) & (flow_osc < flow_osc.shift(21))).astype(float)
        + 0.7 * ((df.high > df.high.shift(13)) & (cmf < cmf.shift(13))).astype(float)
    )
    q_price = roll_norm(c, 64)
    q_rsi = roll_norm(df.rsi13, 64)
    df["qdiv_bull"] = (q_rsi - q_price).clip(0, 1)
    df["qdiv_bear"] = (q_price - q_rsi).clip(0, 1)
    df["bull_div"] = ((bull_pts + 0.8 * df.qdiv_bull) / 4.3).clip(0, 1)
    df["bear_div"] = ((bear_pts + 0.8 * df.qdiv_bear) / 4.3).clip(0, 1)

    df = add_elliott_features(df)
    return df


def side_frame(df: pd.DataFrame, direction: int) -> pd.DataFrame:
    long = direction == 1
    z = pd.DataFrame(index=df.index)
    z["time"] = df.time
    z["bar_index"] = np.arange(len(df))
    z["direction"] = float(direction)
    for h in [1, 3, 6, 12]:
        z[f"side_ret{h}"] = direction * df[f"ret{h}"]
    for c in ["atr_pct", "vol_ratio", "efficiency", "noise", "impulse", "yellow_strength", "activity_rank"]:
        z[c] = df[c]
    z["rsi_side"] = ((50 - df.rsi13) / 50 if long else (df.rsi13 - 50) / 50).clip(-1, 1)
    z["rsi8_side"] = ((50 - df.rsi8) / 50 if long else (df.rsi8 - 50) / 50).clip(-1, 1)
    z["rsi21_side"] = ((50 - df.rsi21) / 50 if long else (df.rsi21 - 50) / 50).clip(-1, 1)
    z["macd_side"] = direction * df.macd_atr
    z["ewo_side"] = direction * df.ewo_atr
    z["zema20_side"] = direction * df.zema20
    z["zema55_side"] = direction * df.zema55
    z["extreme_strength"] = df.low_strength if long else df.high_strength
    z["exhaust_score"] = df.bull_exhaust if long else df.bear_exhaust
    z["reject_score"] = df.bull_reject if long else df.bear_reject
    z["div_score"] = df.bull_div if long else df.bear_div
    z["qdiv_score"] = df.qdiv_bull if long else df.qdiv_bear
    z["wpr_side"] = (-df.wpr9 / 100 if long else 1 + df.wpr9 / 100).clip(0, 1)
    z["wick_side"] = df.lower_wick if long else df.upper_wick
    z["close_side"] = df.close_pos if long else 1 - df.close_pos
    prefix = "ell_buy" if long else "ell_sell"
    z["ell_valid"] = df[f"{prefix}_valid"]
    z["ell_fibq"] = df[f"{prefix}_fibq"]
    z["ell_w5ratio"] = df[f"{prefix}_w5ratio"]
    z["ell_age_norm"] = (df[f"{prefix}_age"] / 40.0).clip(0, 2)
    z["ell_trunc"] = df[f"{prefix}_trunc"]
    z["ell_diag"] = df[f"{prefix}_diag"]
    z["ell_w5of5"] = df[f"{prefix}_w5of5"]
    z["ell_extreme"] = df[f"{prefix}_extreme"]
    z["close"] = df.close
    z["high"] = df.high
    z["low"] = df.low
    z["atr14"] = df.atr14

    structural = z.extreme_strength >= 0.25
    confluence = (
        (z.yellow_strength >= 0.42)
        | (z.exhaust_score >= 0.62)
        | (z.reject_score >= 0.62)
        | (z.div_score >= 0.25)
        | (z.qdiv_score >= 0.25)
    )
    z["candidate"] = (structural & confluence) | ((z.ell_valid > 0.5) & (z.ell_extreme > 0.5))
    return z


def build_events(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    long = side_frame(df, 1)
    short = side_frame(df, -1)
    events = pd.concat([long[long.candidate], short[short.candidate]], ignore_index=True)
    events = events.sort_values(["time", "direction"]).reset_index(drop=True)

    # Future paths are indexed back to the original H1 bar for leakage-free labeling only.
    future_close = df.close.shift(-horizon)
    future_hi = pd.concat([df.high.shift(-k) for k in range(1, horizon + 1)], axis=1).max(axis=1)
    future_lo = pd.concat([df.low.shift(-k) for k in range(1, horizon + 1)], axis=1).min(axis=1)
    idx = events.bar_index.astype(int).to_numpy()
    fc = future_close.to_numpy()[idx]
    fh = future_hi.to_numpy()[idx]
    fl = future_lo.to_numpy()[idx]
    entry = events.close.to_numpy(float)
    at = events.atr14.to_numpy(float)
    d = events.direction.to_numpy(float)
    signed = d * (fc / entry - 1.0)
    mfe = np.where(d > 0, (fh - entry) / at, (entry - fl) / at)
    mae = np.where(d > 0, (entry - fl) / at, (fh - entry) / at)
    events["signed_ret"] = signed
    events["signed_ret_bps"] = signed * 10000.0
    events["mfe_atr"] = mfe
    events["mae_atr"] = mae
    events["target"] = ((signed > 0) & (mfe >= 0.35)).astype(int)
    events = events.replace([np.inf, -np.inf], np.nan)
    events = events.dropna(subset=FEATURES + ["target", "signed_ret_bps", "mfe_atr"]).reset_index(drop=True)
    return events


def model_factory(iterations: int = 140) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=4,
        learning_rate=0.035,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        l2_leaf_reg=5.0,
        random_strength=0.35,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )


def safe_auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def pf_from_bps(r: np.ndarray) -> float:
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    return float(pos / neg) if neg > 0 else float("inf")


def threshold_metrics(frame: pd.DataFrame, threshold: float) -> Dict[str, float]:
    s = frame[frame.prob >= threshold]
    if len(s) == 0:
        return {"threshold": threshold, "n": 0, "precision": np.nan, "mean_bps": np.nan, "pf": np.nan, "coverage": 0.0}
    return {
        "threshold": float(threshold),
        "n": int(len(s)),
        "precision": float(s.target.mean()),
        "mean_bps": float(s.signed_ret_bps.mean()),
        "pf": pf_from_bps(s.signed_ret_bps.to_numpy()),
        "coverage": float(len(s) / len(frame)),
    }


def choose_threshold(oof: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    rows = []
    min_n = max(30, int(len(oof) * 0.01))
    for t in np.arange(0.50, 0.801, 0.01):
        m = threshold_metrics(oof, float(round(t, 2)))
        if m["n"] >= min_n:
            m["objective"] = (m["precision"] - 0.5) * 100.0 + m["mean_bps"] / 10.0 + min(m["n"], 300) / 300.0
        else:
            m["objective"] = -1e9
        rows.append(m)
    tab = pd.DataFrame(rows)
    best = tab.sort_values(["objective", "precision", "mean_bps"], ascending=False).iloc[0]
    return float(best.threshold), tab


def cross_validate(events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    events = events.sort_values("time").reset_index(drop=True)
    unique_times = np.array(sorted(events.time.unique()))
    cuts = np.quantile(np.arange(len(unique_times)), [0.45, 0.58, 0.71, 0.84, 1.0]).astype(int)
    frames = []
    fold_rows = []
    for fold in range(4):
        v0i = min(cuts[fold], len(unique_times) - 1)
        v1i = min(cuts[fold + 1], len(unique_times))
        v0 = unique_times[v0i]
        v1 = unique_times[v1i - 1]
        train = events[events.time < v0].copy()
        valid = events[(events.time >= v0) & (events.time <= v1)].copy()
        if len(train) < 300 or len(valid) < 50:
            continue
        # Purge the last 12 H1 bars before validation boundary.
        purge_time = pd.Timestamp(v0) - pd.Timedelta(hours=12)
        train = train[train.time < purge_time]
        model = model_factory(120)
        model.fit(train[FEATURES], train.target)
        valid["prob"] = model.predict_proba(valid[FEATURES])[:, 1]
        valid["fold"] = fold
        frames.append(valid)
        fold_rows.append({
            "fold": fold,
            "train_from": train.time.min(), "train_to": train.time.max(),
            "valid_from": valid.time.min(), "valid_to": valid.time.max(),
            "n_train": len(train), "n_valid": len(valid),
            "auc": safe_auc(valid.target, valid.prob),
        })
    if not frames:
        raise RuntimeError("Not enough candidate events for expanding walk-forward CV")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(fold_rows)


@dataclass
class StudyResult:
    horizon: int
    threshold: float
    cv_auc: float
    cv_precision: float
    cv_mean_bps: float
    cv_pf: float
    cv_n: int
    hold_auc: float
    hold_precision: float
    hold_mean_bps: float
    hold_pf: float
    hold_n: int
    events: pd.DataFrame
    oof: pd.DataFrame
    folds: pd.DataFrame
    threshold_table: pd.DataFrame


def run_horizon(events: pd.DataFrame, horizon: int) -> StudyResult:
    events = events.sort_values("time").reset_index(drop=True)
    # Final 15% of chronological events is frozen from threshold/model selection.
    split_time = events.time.sort_values().iloc[int(len(events) * 0.85)]
    dev = events[events.time < split_time].copy()
    hold = events[events.time >= split_time].copy()
    oof, folds = cross_validate(dev)
    threshold, thtab = choose_threshold(oof)
    cvm = threshold_metrics(oof, threshold)
    cv_auc = safe_auc(oof.target, oof.prob)

    train = dev[dev.time < split_time - pd.Timedelta(hours=max(12, horizon))].copy()
    hold_model = model_factory(140)
    hold_model.fit(train[FEATURES], train.target)
    hold["prob"] = hold_model.predict_proba(hold[FEATURES])[:, 1]
    hm = threshold_metrics(hold, threshold)
    hold_auc = safe_auc(hold.target, hold.prob)

    return StudyResult(
        horizon=horizon, threshold=threshold,
        cv_auc=cv_auc, cv_precision=cvm["precision"], cv_mean_bps=cvm["mean_bps"], cv_pf=cvm["pf"], cv_n=cvm["n"],
        hold_auc=hold_auc, hold_precision=hm["precision"], hold_mean_bps=hm["mean_bps"], hold_pf=hm["pf"], hold_n=hm["n"],
        events=events, oof=oof, folds=folds, threshold_table=thtab,
    )


def export_model(events: pd.DataFrame) -> Tuple[CatBoostClassifier, dict]:
    model = model_factory(140)
    model.fit(events[FEATURES], events.target)
    model.save_model(str(OUT / "imoex_unified_catboost.cbm"))
    model.save_model(str(OUT / "imoex_unified_catboost.json"), format="json")
    with open(OUT / "imoex_unified_catboost.json", "r", encoding="utf-8") as f:
        js = json.load(f)
    # Verify JSON symmetric-tree evaluator exactly matches CatBoost raw prediction.
    sample = events[FEATURES].tail(min(500, len(events))).to_numpy(float)
    trees = js["oblivious_trees"]
    sb = js.get("scale_and_bias", [1.0, [0.0]])
    scale = float(sb[0])
    bias_obj = sb[1]
    bias = float(bias_obj[0] if isinstance(bias_obj, list) else bias_obj)
    manual = []
    for row in sample:
        total = 0.0
        for tr in trees:
            idx = 0
            for d, sp in enumerate(tr["splits"]):
                fi = int(sp["float_feature_index"])
                if row[fi] > float(sp["border"]):
                    idx |= (1 << d)
            total += float(tr["leaf_values"][idx])
        manual.append(scale * total + bias)
    raw = np.asarray(model.predict(sample, prediction_type="RawFormulaVal"), dtype=float).reshape(-1)
    err = float(np.max(np.abs(np.asarray(manual) - raw)))
    if err > 1e-6:
        raise RuntimeError(f"CatBoost JSON evaluator mismatch: {err}")
    with open(OUT / "export_validation.json", "w", encoding="utf-8") as f:
        json.dump({"trees": len(trees), "features": len(FEATURES), "max_abs_raw_error": err, "scale": scale, "bias": bias}, f, indent=2)
    return model, js


def pine_constants(js: dict) -> str:
    trees = js["oblivious_trees"]
    lines = []
    lines.append("var cbFeat = array.new_int()")
    lines.append("var cbBorder = array.new_float()")
    lines.append("var cbLeaf = array.new_float()")
    lines.append("f_cbAddTree(_f0,_b0,_f1,_b1,_f2,_b2,_f3,_b3,_l0,_l1,_l2,_l3,_l4,_l5,_l6,_l7,_l8,_l9,_l10,_l11,_l12,_l13,_l14,_l15) =>")
    lines += [
        "    array.push(cbFeat, int(_f0)), array.push(cbBorder, float(_b0))",
        "    array.push(cbFeat, int(_f1)), array.push(cbBorder, float(_b1))",
        "    array.push(cbFeat, int(_f2)), array.push(cbBorder, float(_b2))",
        "    array.push(cbFeat, int(_f3)), array.push(cbBorder, float(_b3))",
    ]
    for i in range(16):
        lines.append(f"    array.push(cbLeaf, float(_l{i}))")
    lines.append("    0")
    part_names = []
    for p0 in range(0, len(trees), 10):
        name = f"f_cbInit{p0//10+1:02d}"
        part_names.append(name)
        lines.append(f"{name}() =>")
        for tr in trees[p0:p0+10]:
            sp = tr["splits"]
            if len(sp) != 4:
                raise RuntimeError("Pine exporter expects CatBoost symmetric depth=4 trees")
            sv = []
            for s in sp:
                sv += [str(int(s["float_feature_index"])), f"{float(s['border']):.10g}"]
            lv = [f"{float(x):.12g}" for x in tr["leaf_values"]]
            lines.append("    f_cbAddTree(" + ",".join(sv + lv) + ")")
        lines.append("    0")
    lines.append("if barstate.isfirst")
    for name in part_names:
        lines.append(f"    {name}()")
    return "\n".join(lines)


def generate_pine(js: dict, threshold: float, horizon: int, hist_from: str, hist_to: str) -> str:
    sb = js.get("scale_and_bias", [1.0, [0.0]])
    scale = float(sb[0])
    bias_obj = sb[1]
    bias = float(bias_obj[0] if isinstance(bias_obj, list) else bias_obj)
    const = pine_constants(js)
    feat_comment = ", ".join(FEATURES)
    return f'''//@version=6
// IMOEX Unified Reversal CatBoost V1
// Official MOEX H1 training history actually available: {hist_from} -> {hist_to}
// Selected forward horizon: H{horizon}. Frozen threshold: {threshold:.2f}
// Model families: causal Elliott + 3D Yellow Cluster + adaptive extrema + dual W%R/RSI exhaustion
// + percentile stretch/wick rejection + RSI/MACD/flow/QDIV divergence + regime/volatility context.
// CatBoost is frozen; no future bars are used in live features. Signals fire only on confirmed bars.
indicator("IMOEX Unified Reversal CatBoost V1", shorttitle="IMOEX CB REV V1", overlay=true, max_bars_back=700, max_labels_count=300)

float cbThreshold = input.float({threshold:.2f}, "CatBoost probability threshold", minval=0.40, maxval=0.90, step=0.01)
float probGap = input.float(0.03, "BUY/SELL probability separation", minval=0.0, maxval=0.20, step=0.01)
int cooldown = input.int(3, "Signal cooldown, bars", minval=0, maxval=50)
bool showLabels = input.bool(true, "Show probability labels")
bool strictH1 = input.bool(true, "Require 1-hour chart")

f_clamp(float x, float lo, float hi) => math.max(lo, math.min(hi, x))
f_div(float n, float d) => math.abs(d) < 1e-12 ? 0.0 : n / d
f_norm(float x, simple int len) =>
    float lo = ta.lowest(x, len)
    float hi = ta.highest(x, len)
    na(lo) or na(hi) or hi <= lo ? 0.5 : f_clamp((x-lo)/(hi-lo), 0.0, 1.0)
f_z(float x, simple int len) =>
    float m = ta.sma(x, len)
    float s = ta.stdev(x, len)
    na(m) or na(s) or s <= 1e-12 ? 0.0 : (x-m)/s
f_wpr(simple int len) =>
    float hh = ta.highest(high,len)
    float ll = ta.lowest(low,len)
    -100.0*f_div(hh-close, math.max(hh-ll, syminfo.mintick))
f_close(float x,float t,float w) => math.max(0.0,1.0-math.abs(x-t)/w)
f_fibq(float w2,float w3,float w4,float w5) =>
    float q2=math.max(f_close(w2,0.5,0.35),f_close(w2,0.618,0.35))
    float q3=math.max(f_close(w3,1.618,1.15),f_close(w3,2.618,1.15))
    float q4=math.max(f_close(w4,0.382,0.30),f_close(w4,0.5,0.30))
    float q5=math.max(f_close(w5,0.618,0.55),math.max(f_close(w5,1.0,0.55),f_close(w5,1.618,0.55)))
    f_clamp((q2+q3+q4+q5)/4.0,0.0,1.0)

float atr14=ta.atr(14)
float atr104=ta.atr(104)
float atrSafe=math.max(atr14,syminfo.mintick)
float ret1=math.log(close/close[1])
float ret3=math.log(close/close[3])
float ret6=math.log(close/close[6])
float ret12=math.log(close/close[12])
float atrPct=f_clamp(atr14/close,0.0,0.2)
float volRatio=f_clamp(f_div(atr14,atr104),0.3,4.0)
float travel=math.sum(math.abs(close-close[1]),26)
float efficiency=f_clamp(f_div(math.abs(close-close[26]),travel),0.0,1.0)
float flip=(close-close[1])*(close[1]-close[2])<0?1.0:0.0
float turnRate=ta.sma(flip,26)
float noise=f_clamp(0.62*(1.0-efficiency)+0.38*nz(turnRate,0.5),0.0,1.0)
float impulse=f_clamp((volRatio-0.82)/1.10,0.0,1.0)*f_clamp((efficiency-0.08)/0.74,0.0,1.0)
float rsi8=ta.rsi(close,8)
float rsi13=ta.rsi(close,13)
float rsi21=ta.rsi(close,21)
[ml,ms,mh]=ta.macd(close,12,26,9)
float macdAtr=mh/atrSafe
float ewoAtr=(ta.ema(close,5)-ta.ema(close,35))/atrSafe
float zema20=(close-ta.ema(close,20))/atrSafe
float zema55=(close-ta.ema(close,55))/atrSafe

float lowStrength=(low<=ta.lowest(low,6)?0.10:0)+(low<=ta.lowest(low,10)?0.15:0)+(low<=ta.lowest(low,16)?0.20:0)+(low<=ta.lowest(low,24)?0.25:0)+(low<=ta.lowest(low,36)?0.30:0)
float highStrength=(high>=ta.highest(high,6)?0.10:0)+(high>=ta.highest(high,10)?0.15:0)+(high>=ta.highest(high,16)?0.20:0)+(high>=ta.highest(high,24)?0.25:0)+(high>=ta.highest(high,36)?0.30:0)

float trNow=math.max(ta.tr(true),syminfo.mintick)
float activityProxy=(trNow/math.max(math.abs(close)*1e-6,syminfo.mintick))*(1.0+math.abs(close-open)/trNow)
float activity=math.log(1.0+math.max(not na(volume) and volume>0?volume:activityProxy,0.0))
float typical=hlc3
float pRank=f_norm(typical,160)
float aRank=f_norm(activity,160)
float pVel=pRank-nz(pRank[1],pRank)
float aVel=aRank-nz(aRank[1],aRank)
float kRaw=math.sqrt(math.max(ta.stdev(pRank,20)*ta.stdev(aRank,20),0.0))*math.abs(pVel)*math.abs(aVel)
float kRank=f_norm(kRaw,160)
float pRet=math.log(typical/typical[1])
float aRet=activity-nz(activity[1],activity)
float pAcc=pRet-nz(pRet[1],pRet)
float aAcc=aRet-nz(aRet[1],aRet)
float rv=ta.stdev(pRet,20)
float tensor=(math.abs(f_z(pAcc,64))+math.abs(f_z(aAcc,64))+math.abs(f_z(rv-nz(rv[1],rv),64)))/3.0
float tensorRank=f_norm(tensor,160)
float intensityRank=f_norm(math.abs(ta.stdev(pRet,12)*ta.stdev(aRet,12)),160)
float yellowStrength=f_clamp(0.44*kRank+0.26*tensorRank+0.15*intensityRank+0.15*aRank,0.0,1.0)

float wpr9=f_wpr(9)
float wpr34=f_wpr(34)
float rsiRank=f_norm(rsi13,160)
bool bullDeep=rsiRank<=0.15 and wpr9<=-80 and wpr34<=-75
bool bearDeep=rsiRank>=0.85 and wpr9>=-20 and wpr34>=-25
int bullDeepAge=ta.barssince(bullDeep)
int bearDeepAge=ta.barssince(bearDeep)
bool bullRecent=not na(bullDeepAge) and bullDeepAge<=6
bool bearRecent=not na(bearDeepAge) and bearDeepAge<=6
bool bullRelease=bullRecent and wpr9>wpr9[1] and rsi13>=rsi13[1]
bool bearRelease=bearRecent and wpr9<wpr9[1] and rsi13<=rsi13[1]
float bullExhaust=f_clamp(1.0-rsiRank+(bullRelease?0.35:0.0),0.0,1.0)
float bearExhaust=f_clamp(rsiRank+(bearRelease?0.35:0.0),0.0,1.0)

float stretch=(close-ta.ema(close,34))/atrSafe
float stretchRank=f_norm(stretch,160)
float span=math.max(high-low,syminfo.mintick)
float lowerWick=f_clamp((math.min(open,close)-low)/span,0.0,1.0)
float upperWick=f_clamp((high-math.max(open,close))/span,0.0,1.0)
float closePos=f_clamp((close-low)/span,0.0,1.0)
float bullReject=f_clamp((1.0-stretchRank)*0.55+lowerWick*0.30+closePos*0.15,0.0,1.0)
float bearReject=f_clamp(stretchRank*0.55+upperWick*0.30+(1.0-closePos)*0.15,0.0,1.0)

float signedFlow=activity*(close>open?1.0:close<open?-1.0:0.0)
float flowOsc=ta.ema(signedFlow,5)-ta.ema(signedFlow,13)
float mf=((close-low)-(high-close))/span
float volCmf=not na(volume) and volume>0?volume:math.exp(activity)-1.0
float cmf=f_div(math.sum(mf*volCmf,20),math.sum(volCmf,20))
float bullPts=(low<low[8] and rsi13>rsi13[8]?1.0:0.0)+(low<low[13] and mh>mh[13]?1.0:0.0)+(low<low[21] and flowOsc>flowOsc[21]?0.8:0.0)+(low<low[13] and cmf>cmf[13]?0.7:0.0)
float bearPts=(high>high[8] and rsi13<rsi13[8]?1.0:0.0)+(high>high[13] and mh<mh[13]?1.0:0.0)+(high>high[21] and flowOsc<flowOsc[21]?0.8:0.0)+(high>high[13] and cmf<cmf[13]?0.7:0.0)
float qPrice=f_norm(close,64)
float qRsi=f_norm(rsi13,64)
float qdivBull=f_clamp(qRsi-qPrice,0.0,1.0)
float qdivBear=f_clamp(qPrice-qRsi,0.0,1.0)
float bullDiv=f_clamp((bullPts+0.8*qdivBull)/4.3,0.0,1.0)
float bearDiv=f_clamp((bearPts+0.8*qdivBear)/4.3,0.0,1.0)

// Causal Elliott 0-4 anchors. A pivot exists only after five right-side bars have closed.
int pivotLen=5
float ph=ta.pivothigh(high,pivotLen,pivotLen)
float pl=ta.pivotlow(low,pivotLen,pivotLen)
var pType=array.new_int()
var pPrice=array.new_float()
var pIndex=array.new_int()
if not na(ph)
    int typ=1
    int idx=bar_index-pivotLen
    if array.size(pType)>0 and array.get(pType,array.size(pType)-1)==typ
        if ph>=array.get(pPrice,array.size(pPrice)-1)
            array.set(pPrice,array.size(pPrice)-1,ph), array.set(pIndex,array.size(pIndex)-1,idx)
    else
        array.push(pType,typ),array.push(pPrice,ph),array.push(pIndex,idx)
if not na(pl)
    int typ=-1
    int idx=bar_index-pivotLen
    if array.size(pType)>0 and array.get(pType,array.size(pType)-1)==typ
        if pl<=array.get(pPrice,array.size(pPrice)-1)
            array.set(pPrice,array.size(pPrice)-1,pl), array.set(pIndex,array.size(pIndex)-1,idx)
    else
        array.push(pType,typ),array.push(pPrice,pl),array.push(pIndex,idx)
while array.size(pType)>40
    array.shift(pType),array.shift(pPrice),array.shift(pIndex)

var int bullI4=na
var float bullP0=na
var float bullP1=na
var float bullP2=na
var float bullP3=na
var float bullP4=na
var int bearI4=na
var float bearP0=na
var float bearP1=na
var float bearP2=na
var float bearP3=na
var float bearP4=na
if array.size(pType)>=5
    int n=array.size(pType)
    int t0=array.get(pType,n-5),t1=array.get(pType,n-4),t2=array.get(pType,n-3),t3=array.get(pType,n-2),t4=array.get(pType,n-1)
    if t0==-1 and t1==1 and t2==-1 and t3==1 and t4==-1
        bullP0:=array.get(pPrice,n-5),bullP1:=array.get(pPrice,n-4),bullP2:=array.get(pPrice,n-3),bullP3:=array.get(pPrice,n-2),bullP4:=array.get(pPrice,n-1),bullI4:=array.get(pIndex,n-1)
    if t0==1 and t1==-1 and t2==1 and t3==-1 and t4==1
        bearP0:=array.get(pPrice,n-5),bearP1:=array.get(pPrice,n-4),bearP2:=array.get(pPrice,n-3),bearP3:=array.get(pPrice,n-2),bearP4:=array.get(pPrice,n-1),bearI4:=array.get(pIndex,n-1)

int bullAge=na(bullI4)?0:bar_index-bullI4
int bearAge=na(bearI4)?0:bar_index-bearI4
float bullLive5=na(bullI4) or bullAge<0 or bullAge>80?na:ta.highest(high,math.max(1,bullAge+1))
float bearLive5=na(bearI4) or bearAge<0 or bearAge>80?na:ta.lowest(low,math.max(1,bearAge+1))
float bw1=not na(bullP1)?bullP1-bullP0:na
float bw3a=not na(bullP3)?bullP3-bullP2:na
float bw2=f_div(bullP1-bullP2,bw1), bw3=f_div(bw3a,bw1), bw4=f_div(bullP3-bullP4,bw3a), bw5=f_div(bullLive5-bullP4,bw1)
bool bullBase=not na(bullLive5) and bw1>0 and bw3a>0 and bw2>=0.236 and bw2<=0.886 and bw3>=0.80 and bw4>=0.146 and bw4<=0.786 and bullP3>bullP1
bool bullTrunc=bullBase and bw3>=1.45 and math.abs(bullLive5-bullP3)<=0.35*atrSafe and bullAge>=2
bool bullDiag=bullBase and bullP4<=bullP1+0.20*atrSafe and bw5>=0.20 and bw5<=1.35 and bullAge>=3
bool ellSellValid=bullBase and bullAge>=2 and bw5>=0.15 and bw5<=3.0 and (bullLive5>=bullP3 or bullTrunc or bullDiag)
float ellSellFib=ellSellValid?f_fibq(bw2,bw3,bw4,bw5):0.0

float sw1=not na(bearP1)?bearP0-bearP1:na
float sw3a=not na(bearP3)?bearP2-bearP3:na
float sw2=f_div(bearP2-bearP1,sw1), sw3=f_div(sw3a,sw1), sw4=f_div(bearP4-bearP3,sw3a), sw5=f_div(bearP4-bearLive5,sw1)
bool bearBase=not na(bearLive5) and sw1>0 and sw3a>0 and sw2>=0.236 and sw2<=0.886 and sw3>=0.80 and sw4>=0.146 and sw4<=0.786 and bearP3<bearP1
bool bearTrunc=bearBase and sw3>=1.45 and math.abs(bearLive5-bearP3)<=0.35*atrSafe and bearAge>=2
bool bearDiag=bearBase and bearP4>=bearP1-0.20*atrSafe and sw5>=0.20 and sw5<=1.35 and bearAge>=3
bool ellBuyValid=bearBase and bearAge>=2 and sw5>=0.15 and sw5<=3.0 and (bearLive5<=bearP3 or bearTrunc or bearDiag)
float ellBuyFib=ellBuyValid?f_fibq(sw2,sw3,sw4,sw5):0.0
int innerBull=0
int innerBear=0
if array.size(pIndex)>0
    for ii=0 to array.size(pIndex)-1
        int pi=array.get(pIndex,ii)
        innerBull+=not na(bullI4) and pi>bullI4?1:0
        innerBear+=not na(bearI4) and pi>bearI4?1:0
bool bullW5of5=ellSellValid and innerBull>=3 and bullAge>=5
bool bearW5of5=ellBuyValid and innerBear>=3 and bearAge>=5
bool ellSellExtreme=ellSellValid and high>=bullLive5-syminfo.mintick
bool ellBuyExtreme=ellBuyValid and low<=bearLive5+syminfo.mintick

{const}

float cbScale={scale:.12g}
float cbBias={bias:.12g}
f_cbProb(array<float> x) =>
    float sum=0.0
    int trees=array.size(cbLeaf)/16
    for t=0 to trees-1
        int leaf=0
        int sb=t*4
        for d=0 to 3
            int fi=array.get(cbFeat,sb+d)
            float br=array.get(cbBorder,sb+d)
            if array.get(x,fi)>br
                leaf+=int(math.pow(2,d))
        sum+=array.get(cbLeaf,t*16+leaf)
    float raw=cbScale*sum+cbBias
    1.0/(1.0+math.exp(-raw))

// Feature order frozen at training time: {feat_comment}
f_features(int dir,float extreme,float exhaust,float reject,float divs,float qdiv,float wick,float closeSide,bool ellV,float ellFib,float ellW5,int ellAge,bool trunc,bool diag,bool w5of5) =>
    float rside=dir==1?(50-rsi13)/50:(rsi13-50)/50
    float r8side=dir==1?(50-rsi8)/50:(rsi8-50)/50
    float r21side=dir==1?(50-rsi21)/50:(rsi21-50)/50
    float wside=dir==1?-wpr9/100:1+wpr9/100
    array.from(float(dir),dir*ret1,dir*ret3,dir*ret6,dir*ret12,atrPct,volRatio,efficiency,noise,impulse,f_clamp(rside,-1,1),f_clamp(r8side,-1,1),f_clamp(r21side,-1,1),dir*macdAtr,dir*ewoAtr,dir*zema20,dir*zema55,yellowStrength,extreme,exhaust,reject,divs,aRank,f_clamp(wside,0,1),wick,closeSide,qdiv,ellV?1.0:0.0,ellFib,f_clamp(ellW5,0,3),f_clamp(ellAge/40.0,0,2),trunc?1.0:0.0,diag?1.0:0.0,w5of5?1.0:0.0)

bool buyCandidate=(lowStrength>=0.25 and (yellowStrength>=0.42 or bullExhaust>=0.62 or bullReject>=0.62 or bullDiv>=0.25 or qdivBull>=0.25)) or (ellBuyValid and ellBuyExtreme)
bool sellCandidate=(highStrength>=0.25 and (yellowStrength>=0.42 or bearExhaust>=0.62 or bearReject>=0.62 or bearDiv>=0.25 or qdivBear>=0.25)) or (ellSellValid and ellSellExtreme)
array<float> xb=f_features(1,lowStrength,bullExhaust,bullReject,bullDiv,qdivBull,lowerWick,closePos,ellBuyValid,ellBuyFib,sw5,bearAge,bearTrunc,bearDiag,bearW5of5)
array<float> xs=f_features(-1,highStrength,bearExhaust,bearReject,bearDiv,qdivBear,upperWick,1-closePos,ellSellValid,ellSellFib,bw5,bullAge,bullTrunc,bullDiag,bullW5of5)
float pBuy=buyCandidate?f_cbProb(xb):na
float pSell=sellCandidate?f_cbProb(xs):na
bool tfOK=not strictH1 or timeframe.in_seconds()==3600
var int lastSig=na
bool coolOK=na(lastSig) or bar_index-lastSig>cooldown
bool buySignal=tfOK and barstate.isconfirmed and coolOK and buyCandidate and pBuy>=cbThreshold and (na(pSell) or pBuy>=pSell+probGap)
bool sellSignal=tfOK and barstate.isconfirmed and coolOK and sellCandidate and pSell>=cbThreshold and (na(pBuy) or pSell>=pBuy+probGap)
if buySignal or sellSignal
    lastSig:=bar_index
plotshape(buySignal,title="IMOEX CatBoost BUY",style=shape.triangleup,location=location.belowbar,color=color.lime,size=size.large,text="BUY",textcolor=color.black)
plotshape(sellSignal,title="IMOEX CatBoost SELL",style=shape.triangledown,location=location.abovebar,color=color.red,size=size.large,text="SELL",textcolor=color.white)
if showLabels and buySignal
    label.new(bar_index,low,"BUY "+str.tostring(pBuy*100,"#.0")+"%\nH{horizon}",style=label.style_label_up,color=color.lime,textcolor=color.black)
if showLabels and sellSignal
    label.new(bar_index,high,"SELL "+str.tostring(pSell*100,"#.0")+"%\nH{horizon}",style=label.style_label_down,color=color.red,textcolor=color.white)
var table dash=table.new(position.top_right,2,7,border_width=1)
if barstate.islast
    table.cell(dash,0,0,"IMOEX UNIFIED CATBOOST")
    table.cell(dash,1,0,"H{horizon}")
    table.cell(dash,0,1,"Frozen threshold")
    table.cell(dash,1,1,str.tostring(cbThreshold,"#.00"))
    table.cell(dash,0,2,"BUY probability")
    table.cell(dash,1,2,na(pBuy)?"—":str.tostring(pBuy*100,"#.0")+"%")
    table.cell(dash,0,3,"SELL probability")
    table.cell(dash,1,3,na(pSell)?"—":str.tostring(pSell*100,"#.0")+"%")
    table.cell(dash,0,4,"Yellow / regime")
    table.cell(dash,1,4,str.tostring(yellowStrength,"#.00")+" / "+str.tostring(noise,"#.00"))
    table.cell(dash,0,5,"Elliott B/S")
    table.cell(dash,1,5,(ellBuyValid?"B":"-")+" / "+(ellSellValid?"S":"-"))
    table.cell(dash,0,6,"Training")
    table.cell(dash,1,6,"{hist_from}..{hist_to}")
alertcondition(buySignal,title="IMOEX Unified BUY",message="IMOEX Unified CatBoost BUY on {{ticker}} {{interval}}")
alertcondition(sellSignal,title="IMOEX Unified SELL",message="IMOEX Unified CatBoost SELL on {{ticker}} {{interval}}")
'''


def write_report(df: pd.DataFrame, selected: StudyResult, studies: List[StudyResult], model: CatBoostClassifier):
    rows = []
    for s in studies:
        rows.append({
            "horizon": s.horizon, "threshold": s.threshold,
            "cv_auc": s.cv_auc, "cv_precision": s.cv_precision, "cv_mean_bps": s.cv_mean_bps, "cv_pf": s.cv_pf, "cv_n": s.cv_n,
            "hold_auc": s.hold_auc, "hold_precision": s.hold_precision, "hold_mean_bps": s.hold_mean_bps, "hold_pf": s.hold_pf, "hold_n": s.hold_n,
        })
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "horizon_comparison.csv", index=False)
    fi = pd.DataFrame({"feature": FEATURES, "importance": model.get_feature_importance()}).sort_values("importance", ascending=False)
    fi.to_csv(OUT / "feature_importance.csv", index=False)
    s = selected
    buy = s.oof[(s.oof.prob >= s.threshold) & (s.oof.direction == 1)]
    sell = s.oof[(s.oof.prob >= s.threshold) & (s.oof.direction == -1)]
    warning = ""
    if df.time.min() > pd.Timestamp("1999-01-15"):
        warning = f"\n**Data availability note:** official ISS H1 actually begins at {df.time.min()}. No synthetic bars were created for 1999 through the first available H1 candle.\n"
    report = f'''# IMOEX Unified Reversal CatBoost V1\n\n## Data\n\n- Requested history: 1999-01-01 through {TODAY}\n- Official MOEX H1 actually retrieved: **{df.time.min()} through {df.time.max()}**\n- H1 bars: **{len(df):,}**\n- Sources: MICEXINDEXCF legacy code + IMOEX current code, official MOEX ISS only.\n{warning}\n## Architecture\n\nOne CatBoost classifier scores side-aligned BUY and SELL reversal candidates. The candidate/feature stack combines causal Elliott Wave-5 termination, Wave-5-of-5, failed fifth/truncation, ending diagonal, 3D Yellow Cluster, adaptive 6/10/16/24/36-bar extrema, dual Williams %R + RSI exhaustion/release, percentile stretch and wick rejection, RSI/MACD/flow/CMF divergence, quantile divergence (QDIV), volatility/regime features.\n\nNo future values enter features. Future bars are used only for labels. CV is expanding chronological with purge. The final 15% is held out from horizon and threshold selection.\n\n## Selected model\n\n- Horizon: **H{s.horizon}**\n- Frozen probability threshold: **{s.threshold:.2f}**\n- OOF CV AUC: **{s.cv_auc:.4f}**\n- OOF selected precision: **{s.cv_precision:.3%}** on {s.cv_n:,} signals\n- OOF selected mean signed return: **{s.cv_mean_bps:.3f} bps**\n- OOF selected PF proxy: **{s.cv_pf:.3f}**\n- Frozen holdout AUC: **{s.hold_auc:.4f}**\n- Frozen holdout precision: **{s.hold_precision:.3%}** on {s.hold_n:,} signals\n- Frozen holdout mean signed return: **{s.hold_mean_bps:.3f} bps**\n- Frozen holdout PF proxy: **{s.hold_pf:.3f}**\n- OOF BUY precision: **{(buy.target.mean() if len(buy) else float('nan')):.3%}** ({len(buy):,})\n- OOF SELL precision: **{(sell.target.mean() if len(sell) else float('nan')):.3%}** ({len(sell):,})\n\n## Important\n\nThis is a probabilistic reversal model, not a guarantee of exact turning points. The deployment threshold is the threshold selected on expanding OOF data and then evaluated unchanged on the frozen chronological holdout.\n'''
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")


def main():
    np.random.seed(SEED)
    df = load_history()
    print("H1 range", df.time.min(), df.time.max(), "bars", len(df))
    df = build_base_features(df)
    # Avoid feature warm-up region.
    df = df.iloc[300:].reset_index(drop=True)
    studies: List[StudyResult] = []
    for h in [3, 6, 10]:
        ev = build_events(df, h)
        print("H", h, "events", len(ev), "positive", ev.target.mean())
        ev.to_csv(OUT / f"events_H{h}.csv", index=False)
        s = run_horizon(ev, h)
        studies.append(s)
        s.oof.to_csv(OUT / f"oof_H{h}.csv", index=False)
        s.folds.to_csv(OUT / f"folds_H{h}.csv", index=False)
        s.threshold_table.to_csv(OUT / f"thresholds_H{h}.csv", index=False)
    # Horizon selection uses only OOF: precision first, then mean return and sample support.
    selected = max(studies, key=lambda s: ((s.cv_precision if np.isfinite(s.cv_precision) else -1), (s.cv_mean_bps if np.isfinite(s.cv_mean_bps) else -1), math.log1p(s.cv_n)))
    final_model, js = export_model(selected.events)
    pine = generate_pine(js, selected.threshold, selected.horizon, str(df.time.min()), str(df.time.max()))
    (OUT / "IMOEX_UNIFIED_REVERSAL_CATBOOST_V1.pine").write_text(pine, encoding="utf-8")
    write_report(df, selected, studies, final_model)
    print((OUT / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
