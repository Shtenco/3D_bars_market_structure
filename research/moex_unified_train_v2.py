#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

import moex_unified_train as v1

OUT = Path("artifacts/moex_unified_v2")
OUT.mkdir(parents=True, exist_ok=True)
SEED = v1.SEED

EXTRA_FEATURES = [
    "range_atr", "body_atr", "gap_atr", "extreme100", "dist100_atr",
    "trend20_80_side", "trend55_200_side",
    "yellow_lag1", "yellow_lag3", "yellow_lag6",
    "rsi_side_lag1", "rsi_side_lag3", "macd_side_lag1", "macd_side_lag3",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
FEATURES = v1.FEATURES + EXTRA_FEATURES


def enrich_base(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    at = d.atr14.replace(0, np.nan)
    d["range_atr"] = ((d.high - d.low) / at).clip(0, 10).fillna(0)
    d["body_atr"] = ((d.close - d.open) / at).clip(-10, 10).fillna(0)
    d["gap_atr"] = ((d.open - d.close.shift()) / at).clip(-10, 10).fillna(0)
    low100 = d.low.rolling(100, min_periods=100).min()
    high100 = d.high.rolling(100, min_periods=100).max()
    d["low100_strength"] = (d.low <= low100 + 1e-12).astype(float)
    d["high100_strength"] = (d.high >= high100 - 1e-12).astype(float)
    d["dist_low100_atr"] = ((d.close - low100) / at).clip(0, 50).fillna(0)
    d["dist_high100_atr"] = ((high100 - d.close) / at).clip(0, 50).fillna(0)
    e20 = v1.ema(d.close, 20)
    e80 = v1.ema(d.close, 80)
    e55 = v1.ema(d.close, 55)
    e200 = v1.ema(d.close, 200)
    d["trend20_80"] = ((e20 - e80) / at).clip(-20, 20).fillna(0)
    d["trend55_200"] = ((e55 - e200) / at).clip(-30, 30).fillna(0)
    return d


def side_frame_v2(df: pd.DataFrame, direction: int) -> pd.DataFrame:
    z = v1.side_frame(df, direction).copy()
    long = direction == 1
    z["range_atr"] = df.range_atr
    z["body_atr"] = direction * df.body_atr
    z["gap_atr"] = direction * df.gap_atr
    z["extreme100"] = df.low100_strength if long else df.high100_strength
    z["dist100_atr"] = df.dist_low100_atr if long else df.dist_high100_atr
    z["trend20_80_side"] = direction * df.trend20_80
    z["trend55_200_side"] = direction * df.trend55_200
    z["yellow_lag1"] = df.yellow_strength.shift(1)
    z["yellow_lag3"] = df.yellow_strength.shift(3)
    z["yellow_lag6"] = df.yellow_strength.shift(6)
    rside = ((50 - df.rsi13) / 50 if long else (df.rsi13 - 50) / 50).clip(-1, 1)
    z["rsi_side_lag1"] = rside.shift(1)
    z["rsi_side_lag3"] = rside.shift(3)
    z["macd_side_lag1"] = direction * df.macd_atr.shift(1)
    z["macd_side_lag3"] = direction * df.macd_atr.shift(3)
    hour = pd.to_datetime(df.time).dt.hour + pd.to_datetime(df.time).dt.minute / 60.0
    dow = pd.to_datetime(df.time).dt.dayofweek
    z["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    z["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    z["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    z["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    # Broader but still reversal-centric candidate union.
    z["candidate"] = z.candidate | (z.extreme100 > 0.5) | ((z.reject_score > 0.72) & (z.exhaust_score > 0.60))
    return z


def build_first_passage_events(df: pd.DataFrame, horizon: int, barrier_atr: float = 0.45) -> pd.DataFrame:
    l = side_frame_v2(df, 1)
    s = side_frame_v2(df, -1)
    ev = pd.concat([l[l.candidate], s[s.candidate]], ignore_index=True).sort_values(["time", "direction"]).reset_index(drop=True)
    idx = ev.bar_index.astype(int).to_numpy()
    entry = ev.close.to_numpy(float)
    at = ev.atr14.to_numpy(float)
    direc = ev.direction.to_numpy(float)
    first_fav = np.full(len(ev), np.inf)
    first_adv = np.full(len(ev), np.inf)
    for k in range(1, horizon + 1):
        hi = df.high.shift(-k).to_numpy()[idx]
        lo = df.low.shift(-k).to_numpy()[idx]
        fav = np.where(direc > 0, (hi - entry) / at, (entry - lo) / at)
        adv = np.where(direc > 0, (entry - lo) / at, (hi - entry) / at)
        first_fav[(first_fav == np.inf) & (fav >= barrier_atr)] = k
        first_adv[(first_adv == np.inf) & (adv >= barrier_atr)] = k
    resolved = np.isfinite(first_fav) | np.isfinite(first_adv)
    non_tie = first_fav != first_adv
    ev["first_fav"] = first_fav
    ev["first_adv"] = first_adv
    ev["target"] = (first_fav < first_adv).astype(int)
    fc = df.close.shift(-horizon).to_numpy()[idx]
    ev["signed_ret_bps"] = direc * (fc / entry - 1) * 10000.0
    ev = ev[resolved & non_tie].copy()
    ev = ev.replace([np.inf, -np.inf], np.nan)
    ev = ev.dropna(subset=FEATURES + ["target", "signed_ret_bps"]).reset_index(drop=True)
    return ev


def factory(iterations=160, depth=4):
    return CatBoostClassifier(
        iterations=iterations, depth=depth, learning_rate=0.03,
        loss_function="Logloss", eval_metric="AUC", random_seed=SEED,
        l2_leaf_reg=7.0, random_strength=0.5, bootstrap_type="Bernoulli",
        subsample=0.80, allow_writing_files=False, verbose=False, thread_count=-1,
    )


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def pf(r):
    r = np.asarray(r, float)
    pos = r[r > 0].sum(); neg = -r[r < 0].sum()
    return float(pos / neg) if neg > 0 else float("inf")


def wilson_lower(successes: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = successes / n
    den = 1 + z*z/n
    cen = p + z*z/(2*n)
    adj = z * math.sqrt((p*(1-p) + z*z/(4*n))/n)
    return (cen - adj) / den


def chronological_oof_side(events: pd.DataFrame, direction: int) -> pd.DataFrame:
    e = events[events.direction == direction].sort_values("time").reset_index(drop=True)
    times = np.array(sorted(e.time.unique()))
    q = [0.42, 0.54, 0.66, 0.78, 0.90, 1.0]
    cuts = [min(int(len(times)*x), len(times)) for x in q]
    outs = []
    for fold in range(len(cuts)-1):
        a = cuts[fold]; b = cuts[fold+1]
        if a >= len(times) or b <= a:
            continue
        v0 = times[a]; v1 = times[b-1]
        tr = e[e.time < pd.Timestamp(v0) - pd.Timedelta(hours=24)].copy()
        va = e[(e.time >= v0) & (e.time <= v1)].copy()
        if len(tr) < 300 or len(va) < 60 or tr.target.nunique() < 2:
            continue
        m = factory(140, 4)
        m.fit(tr[FEATURES], tr.target)
        va["prob"] = m.predict_proba(va[FEATURES])[:,1]
        va["fold"] = fold
        outs.append(va)
    if not outs:
        raise RuntimeError(f"No valid OOF folds for direction {direction}")
    return pd.concat(outs, ignore_index=True)


def choose_threshold(oof: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    rows = []
    min_n = max(80, int(len(oof)*0.04))
    for t in np.arange(0.50, 0.701, 0.01):
        s = oof[oof.prob >= t]
        n = len(s); wins = int(s.target.sum()) if n else 0
        prec = wins/n if n else np.nan
        lower = wilson_lower(wins, n) if n else 0.0
        mean_bps = float(s.signed_ret_bps.mean()) if n else np.nan
        pfr = pf(s.signed_ret_bps) if n else np.nan
        fold_counts = s.groupby("fold").size()
        robust_folds = int((fold_counts >= 10).sum())
        objective = lower + (0.0002*max(min(mean_bps if np.isfinite(mean_bps) else -100, 200), -200))
        if n < min_n or robust_folds < 3:
            objective = -999
        rows.append(dict(threshold=float(round(t,2)), n=n, precision=prec, wilson_lower=lower, mean_bps=mean_bps, pf=pfr, robust_folds=robust_folds, objective=objective))
    tab = pd.DataFrame(rows)
    best = tab.sort_values(["objective","wilson_lower","n"], ascending=False).iloc[0]
    return float(best.threshold), tab


def evaluate_horizon(events: pd.DataFrame, horizon: int) -> Dict:
    # Architecture/threshold selection uses only the first 80% chronologically.
    split = events.time.sort_values().iloc[int(len(events)*0.80)]
    dev = events[events.time < split].copy()
    retest = events[events.time >= split].copy()
    out = {"horizon": horizon, "split": str(split)}
    combined_dev = []
    combined_re = []
    for direction, name in [(1,"buy"),(-1,"sell")]:
        oo = chronological_oof_side(dev, direction)
        th, tab = choose_threshold(oo)
        tab.to_csv(OUT/f"threshold_{name}_H{horizon}.csv", index=False)
        sel = oo[oo.prob>=th]
        out[f"{name}_threshold"] = th
        out[f"{name}_oof_auc"] = auc(oo.target,oo.prob)
        out[f"{name}_oof_n"] = len(sel)
        out[f"{name}_oof_precision"] = float(sel.target.mean()) if len(sel) else np.nan
        out[f"{name}_oof_wilson"] = wilson_lower(int(sel.target.sum()),len(sel)) if len(sel) else 0
        out[f"{name}_oof_bps"] = float(sel.signed_ret_bps.mean()) if len(sel) else np.nan
        oo["threshold"] = th
        combined_dev.append(oo)
        tr = dev[dev.direction==direction].copy()
        re = retest[retest.direction==direction].copy()
        m = factory(160,4); m.fit(tr[FEATURES],tr.target)
        re["prob"] = m.predict_proba(re[FEATURES])[:,1]
        re["threshold"] = th
        rs = re[re.prob>=th]
        out[f"{name}_retest_auc"] = auc(re.target,re.prob)
        out[f"{name}_retest_n"] = len(rs)
        out[f"{name}_retest_precision"] = float(rs.target.mean()) if len(rs) else np.nan
        out[f"{name}_retest_wilson"] = wilson_lower(int(rs.target.sum()),len(rs)) if len(rs) else 0
        out[f"{name}_retest_bps"] = float(rs.signed_ret_bps.mean()) if len(rs) else np.nan
        combined_re.append(re)
    cd = pd.concat(combined_dev,ignore_index=True)
    cr = pd.concat(combined_re,ignore_index=True)
    csel = cd[cd.prob>=cd.threshold]
    rsel = cr[cr.prob>=cr.threshold]
    out["oof_n"] = len(csel); out["oof_precision"] = float(csel.target.mean()); out["oof_wilson"] = wilson_lower(int(csel.target.sum()),len(csel)); out["oof_bps"] = float(csel.signed_ret_bps.mean()); out["oof_pf"] = pf(csel.signed_ret_bps)
    out["retest_n"] = len(rsel); out["retest_precision"] = float(rsel.target.mean()) if len(rsel) else np.nan; out["retest_wilson"] = wilson_lower(int(rsel.target.sum()),len(rsel)) if len(rsel) else 0; out["retest_bps"] = float(rsel.signed_ret_bps.mean()) if len(rsel) else np.nan; out["retest_pf"] = pf(rsel.signed_ret_bps) if len(rsel) else np.nan
    # Conservative model-selection score: OOF Wilson lower bound, then breadth/return.
    out["selection_score"] = out["oof_wilson"] + 0.0002*max(min(out["oof_bps"],200),-200)
    cd.to_csv(OUT/f"oof_H{horizon}.csv",index=False)
    cr.to_csv(OUT/f"retest_H{horizon}.csv",index=False)
    return out


def main():
    np.random.seed(SEED)
    df = v1.load_history()
    raw_from, raw_to, raw_n = df.time.min(), df.time.max(), len(df)
    df = v1.build_base_features(df)
    df = enrich_base(df).iloc[350:].reset_index(drop=True)
    results=[]
    all_events={}
    for h in [6,10,20]:
        ev=build_first_passage_events(df,h,0.45)
        all_events[h]=ev
        print(f"V2 H{h}: events={len(ev)} target={ev.target.mean():.4f} buy={sum(ev.direction==1)} sell={sum(ev.direction==-1)}")
        results.append(evaluate_horizon(ev,h))
    tab=pd.DataFrame(results).sort_values("selection_score",ascending=False)
    tab.to_csv(OUT/"comparison.csv",index=False)
    print("\n=== V2 ROBUST COMPARISON ===")
    print(tab.to_string(index=False))
    best=tab.iloc[0].to_dict(); h=int(best["horizon"])
    report=f'''# IMOEX Unified Reversal CatBoost V2 — robust first-passage research\n\nOfficial MOEX H1 retrieved: {raw_from} -> {raw_to}, {raw_n:,} raw bars. After warm-up: {len(df):,}.\n\nTarget: favorable 0.45 ATR barrier must be hit before adverse 0.45 ATR barrier. Same-bar barrier ties and unresolved events are excluded. BUY and SELL have separate CatBoost models and separately frozen thresholds.\n\nSelected H{h}: OOF n={int(best['oof_n'])}, precision={best['oof_precision']:.3%}, Wilson95 lower={best['oof_wilson']:.3%}, mean={best['oof_bps']:.3f} bps, PF={best['oof_pf']:.3f}.\nRetest n={int(best['retest_n'])}, precision={best['retest_precision']:.3%}, Wilson95 lower={best['retest_wilson']:.3%}, mean={best['retest_bps']:.3f} bps, PF={best['retest_pf']:.3f}.\nBUY threshold={best['buy_threshold']:.2f}; SELL threshold={best['sell_threshold']:.2f}.\n\nThis retest is not called pristine because V1 metrics from the end of this same history were already inspected before V2 was designed.\n'''
    (OUT/"REPORT.md").write_text(report,encoding="utf-8")
    print("\n"+report)

if __name__=="__main__":
    main()
