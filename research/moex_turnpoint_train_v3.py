#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

import moex_unified_train as v1
import moex_unified_train_v2 as v2

OUT=Path('artifacts/moex_turnpoint_v3'); OUT.mkdir(parents=True,exist_ok=True)
SEED=v1.SEED
FEATURES=v2.FEATURES+['depth72_atr','range_lag1','range_lag3','activity_lag1','activity_lag3']


def side_v3(df,direction):
    z=v2.side_frame_v2(df,direction).copy()
    at=df.atr14.replace(0,np.nan)
    if direction==1:
        z['depth72_atr']=((df.high.rolling(72,min_periods=24).max()-df.low)/at).clip(0,50).fillna(0)
    else:
        z['depth72_atr']=((df.high-df.low.rolling(72,min_periods=24).min())/at).clip(0,50).fillna(0)
    z['range_lag1']=df.range_atr.shift(1)
    z['range_lag3']=df.range_atr.shift(3)
    z['activity_lag1']=df.activity_rank.shift(1)
    z['activity_lag3']=df.activity_rank.shift(3)
    # Keep candidates broad enough that ML, not hand rules, decides precision.
    z['candidate']=z.candidate | (z.extreme_strength>=0.20) | (z.extreme100>0.5) | (z.ell_valid>0.5)
    return z


def build_events(df,horizon,rebound_atr=0.60,tol_atr=0.08):
    ev=pd.concat([side_v3(df,1).query('candidate'),side_v3(df,-1).query('candidate')],ignore_index=True)
    ev=ev.sort_values(['time','direction']).reset_index(drop=True)
    idx=ev.bar_index.astype(int).to_numpy(); d=ev.direction.to_numpy(float)
    entry=ev.close.to_numpy(float); low0=ev.low.to_numpy(float); high0=ev.high.to_numpy(float); at=ev.atr14.to_numpy(float)
    future_lows=pd.concat([df.low.shift(-k) for k in range(1,horizon+1)],axis=1)
    future_highs=pd.concat([df.high.shift(-k) for k in range(1,horizon+1)],axis=1)
    fmin=future_lows.min(axis=1).to_numpy()[idx]
    fmax=future_highs.max(axis=1).to_numpy()[idx]
    fc=df.close.shift(-horizon).to_numpy()[idx]
    long_bottom=(low0 <= fmin + tol_atr*at) & ((fmax-entry)/at >= rebound_atr)
    short_top=(high0 >= fmax - tol_atr*at) & ((entry-fmin)/at >= rebound_atr)
    ev['target']=np.where(d>0,long_bottom,short_top).astype(int)
    ev['signed_ret_bps']=d*(fc/entry-1)*10000
    ev['mfe_atr']=np.where(d>0,(fmax-entry)/at,(entry-fmin)/at)
    ev['mae_atr']=np.where(d>0,(entry-fmin)/at,(fmax-entry)/at)
    valid=np.isfinite(fc)&np.isfinite(fmin)&np.isfinite(fmax)&np.isfinite(at)&(at>0)
    ev=ev[valid].replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+['target','signed_ret_bps']).reset_index(drop=True)
    return ev


def factory():
    return CatBoostClassifier(iterations=220,depth=5,learning_rate=0.025,loss_function='Logloss',eval_metric='AUC',
        random_seed=SEED,l2_leaf_reg=8.0,random_strength=0.6,bootstrap_type='Bernoulli',subsample=0.8,
        auto_class_weights='Balanced',allow_writing_files=False,verbose=False,thread_count=-1)

def auc(y,p): return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else float('nan')
def ap(y,p): return float(average_precision_score(y,p)) if len(np.unique(y))>1 else float('nan')
def pf(r):
    r=np.asarray(r,float); pos=r[r>0].sum(); neg=-r[r<0].sum(); return float(pos/neg) if neg>0 else float('inf')
def wilson(w,n,z=1.96):
    if n<1:return 0
    p=w/n; den=1+z*z/n; return (p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/den


def oof_side(dev,direction):
    e=dev[dev.direction==direction].sort_values('time').reset_index(drop=True)
    times=np.array(sorted(e.time.unique())); outs=[]
    for fold,(q0,q1) in enumerate([(0.40,0.52),(0.52,0.64),(0.64,0.76),(0.76,0.88),(0.88,1.0)]):
        a=int(len(times)*q0); b=max(a+1,int(len(times)*q1)); b=min(b,len(times))
        v0=times[a]; v1=times[b-1]
        tr=e[e.time < pd.Timestamp(v0)-pd.Timedelta(hours=24)].copy(); va=e[(e.time>=v0)&(e.time<=v1)].copy()
        if len(tr)<400 or len(va)<80 or tr.target.nunique()<2:continue
        m=factory();m.fit(tr[FEATURES],tr.target);va['prob']=m.predict_proba(va[FEATURES])[:,1];va['fold']=fold;outs.append(va)
    if not outs:raise RuntimeError('no folds')
    return pd.concat(outs,ignore_index=True)


def choose(oof):
    rows=[]; min_n=max(80,int(len(oof)*0.03)); qs=np.linspace(0.55,0.95,17)
    thresholds=sorted(set([0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]+[float(oof.prob.quantile(q)) for q in qs]))
    baseline=float(oof.target.mean())
    for t in thresholds:
        s=oof[oof.prob>=t];n=len(s);w=int(s.target.sum()) if n else 0;p=w/n if n else np.nan;lo=wilson(w,n) if n else 0
        folds=int((s.groupby('fold').size()>=10).sum()) if n else 0
        mean=float(s.signed_ret_bps.mean()) if n else np.nan
        lift=p/baseline if n and baseline>0 else np.nan
        objective=(lo-baseline)+0.015*max(min((lift if np.isfinite(lift) else 0)-1,3),-1)+0.0001*max(min(mean if np.isfinite(mean) else -200,200),-200)
        if n<min_n or folds<3:objective=-999
        rows.append(dict(threshold=t,n=n,precision=p,wilson=lo,baseline=baseline,lift=lift,mean_bps=mean,pf=pf(s.signed_ret_bps) if n else np.nan,folds=folds,objective=objective))
    tab=pd.DataFrame(rows);best=tab.sort_values(['objective','wilson','lift'],ascending=False).iloc[0]
    return float(best.threshold),tab


def evaluate(ev,h):
    split=ev.time.sort_values().iloc[int(len(ev)*0.80)];dev=ev[ev.time<split].copy();re=ev[ev.time>=split].copy();out={'horizon':h,'split':str(split)}
    comb=[];combre=[]
    for direction,name in [(1,'buy'),(-1,'sell')]:
        oo=oof_side(dev,direction);th,tab=choose(oo);tab.to_csv(OUT/f'threshold_{name}_H{h}.csv',index=False)
        ss=oo[oo.prob>=th];base=oo.target.mean();out.update({f'{name}_th':th,f'{name}_auc':auc(oo.target,oo.prob),f'{name}_ap':ap(oo.target,oo.prob),f'{name}_base':base,f'{name}_n':len(ss),f'{name}_precision':ss.target.mean(),f'{name}_lift':ss.target.mean()/base,f'{name}_wilson':wilson(int(ss.target.sum()),len(ss)),f'{name}_bps':ss.signed_ret_bps.mean()});oo['threshold']=th;comb.append(oo)
        tr=dev[dev.direction==direction].copy();rr=re[re.direction==direction].copy();m=factory();m.fit(tr[FEATURES],tr.target);rr['prob']=m.predict_proba(rr[FEATURES])[:,1];rr['threshold']=th;rs=rr[rr.prob>=th];rbase=rr.target.mean()
        out.update({f'{name}_re_auc':auc(rr.target,rr.prob),f'{name}_re_ap':ap(rr.target,rr.prob),f'{name}_re_base':rbase,f'{name}_re_n':len(rs),f'{name}_re_precision':rs.target.mean() if len(rs) else np.nan,f'{name}_re_lift':(rs.target.mean()/rbase if len(rs) and rbase>0 else np.nan),f'{name}_re_wilson':wilson(int(rs.target.sum()),len(rs)) if len(rs) else 0,f'{name}_re_bps':rs.signed_ret_bps.mean() if len(rs) else np.nan});combre.append(rr)
    co=pd.concat(comb);cr=pd.concat(combre);s=co[co.prob>=co.threshold];r=cr[cr.prob>=cr.threshold]
    out.update({'oof_n':len(s),'oof_precision':s.target.mean(),'oof_baseline':co.target.mean(),'oof_lift':s.target.mean()/co.target.mean(),'oof_wilson':wilson(int(s.target.sum()),len(s)),'oof_bps':s.signed_ret_bps.mean(),'oof_pf':pf(s.signed_ret_bps),'re_n':len(r),'re_precision':r.target.mean() if len(r) else np.nan,'re_baseline':cr.target.mean(),'re_lift':(r.target.mean()/cr.target.mean() if len(r) else np.nan),'re_wilson':wilson(int(r.target.sum()),len(r)) if len(r) else 0,'re_bps':r.signed_ret_bps.mean() if len(r) else np.nan,'re_pf':pf(r.signed_ret_bps) if len(r) else np.nan})
    out['score']=(out['oof_wilson']-out['oof_baseline'])+0.02*(out['oof_lift']-1)+0.0001*max(min(out['oof_bps'],200),-200)
    co.to_csv(OUT/f'oof_H{h}.csv',index=False);cr.to_csv(OUT/f'retest_H{h}.csv',index=False)
    return out


def main():
    df=v1.load_history();raw_from,raw_to,raw_n=df.time.min(),df.time.max(),len(df)
    df=v1.build_base_features(df);df=v2.enrich_base(df).iloc[350:].reset_index(drop=True)
    results=[]
    for h in [6,10,20]:
        ev=build_events(df,h);print(f'V3 H{h} events={len(ev)} positives={ev.target.mean():.4f}');results.append(evaluate(ev,h))
    tab=pd.DataFrame(results).sort_values('score',ascending=False);tab.to_csv(OUT/'comparison.csv',index=False);print('\n=== V3 DIRECT TURNPOINT ===');print(tab.to_string(index=False))
    b=tab.iloc[0]
    rep=f'''# IMOEX Direct Turn-Point CatBoost V3\n\nOfficial H1: {raw_from} -> {raw_to}; {raw_n:,} raw bars.\n\nPositive label = current candidate remains the local high/low for H bars within 0.08 ATR and produces at least 0.60 ATR rebound/rejection. This directly models a turning point rather than generic direction.\n\nSelected H{int(b.horizon)}: OOF {int(b.oof_n)} signals, precision {b.oof_precision:.3%} vs baseline {b.oof_baseline:.3%} (lift {b.oof_lift:.2f}x), Wilson95 lower {b.oof_wilson:.3%}, mean {b.oof_bps:.2f} bps, PF {b.oof_pf:.3f}.\nRetest {int(b.re_n)} signals, precision {b.re_precision:.3%} vs baseline {b.re_baseline:.3%} (lift {b.re_lift:.2f}x), Wilson95 lower {b.re_wilson:.3%}, mean {b.re_bps:.2f} bps, PF {b.re_pf:.3f}.\nBUY th={b.buy_th:.4f}; SELL th={b.sell_th:.4f}.\n''';(OUT/'REPORT.md').write_text(rep,encoding='utf-8');print('\n'+rep)

if __name__=='__main__':main()
