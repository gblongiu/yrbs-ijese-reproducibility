"""Conditional paired public-design cluster sensitivity, not full design inference."""
import runpy
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'revision_r1/outputs/tables'
SEED=20260918
REPS=2000

def multipliers(design, rng):
    """Rao-Wu n_h-1 resampling; one common cluster multiplier for every endpoint."""
    mult=pd.Series(0.,index=design.index)
    for _,group in design.groupby('stratum',sort=True):
        psus=sorted(group.psu.unique())
        nh=len(psus)
        if nh<2:raise ValueError('Singleton stratum: no automatic certainty assumption')
        count=np.bincount(rng.integers(0,nh,size=nh-1),minlength=nh)
        lookup=dict(zip(psus,nh/(nh-1)*count))
        mult.loc[group.index]=group.psu.map(lookup)
    return mult.to_numpy()

def scores(y,p,w):
    return np.array([roc_auc_score(y,p,sample_weight=w),average_precision_score(y,p,sample_weight=w),brier_score_loss(y,p,sample_weight=w)])

def main():
    r=runpy.run_path(str(ROOT/'revision_r1/scripts/run_2023_revision_analyses.py'))
    build,week=r['load_project_modules']()
    df=r['load_modeling_table'](build)
    tr,te,fold=r['load_frozen_indices'](df,week)
    pred=pd.read_csv(OUT/'matched_hgb_2023_holdout_predictions_seed2026.csv')
    assert np.array_equal(pred.row_index,te)
    design=df[['stratum','psu']]
    records=[]
    for h,g in design.groupby('stratum'):
        th=df.loc[te].query('stratum == @h')
        records.append(dict(stratum=h,full_analytic_psus=g.psu.nunique(),evaluation_psus=th.psu.nunique(),evaluation_n=len(th)))
    pd.DataFrame(records).to_csv(OUT/'public_design_structure.csv',index=False)
    y=pred.y_true.to_numpy()
    w=pred.weight.to_numpy()
    states={'uncalibrated':('baseline_none','expanded_none'),'platt':('baseline_platt','expanded_platt')}
    rng=np.random.default_rng(SEED)
    draws=[]
    for b in range(REPS):
        wb=w*multipliers(design,rng)[te]
        for state,(c0,c1) in states.items():
            a=scores(y,pred[c0].to_numpy(),wb)
            e=scores(y,pred[c1].to_numpy(),wb)
            delta=(e-a)*[1,1,-1]
            draws.append(dict(replicate=b,calibration=state,roc_auc_change=delta[0],average_precision_change=delta[1],brier_improvement=delta[2]))
    draws=pd.DataFrame(draws)
    draws.to_csv(OUT/'public_design_bootstrap_draws.csv',index=False)
    rows=[]
    for state,(c0,c1) in states.items():
        a=scores(y,pred[c0].to_numpy(),w);e=scores(y,pred[c1].to_numpy(),w)
        for j,key in enumerate(['roc_auc_change','average_precision_change','brier_improvement']):
            v=draws.loc[draws.calibration==state,key]
            rows.append(dict(calibration=state,metric=key,baseline=a[j],expanded=e[j],point_difference=(e-a)[j]*([1,1,-1][j]),ci95_low=v.quantile(.025),ci95_high=v.quantile(.975),replicates=REPS,seed=SEED))
    pd.DataFrame(rows).to_csv(OUT/'public_design_paired_sensitivity.csv',index=False)
    print('Public-design sensitivity complete',flush=True)

if __name__=='__main__':main()
