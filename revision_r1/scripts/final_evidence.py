"""Regenerate descriptive context and verify frozen-pipeline identity."""
import hashlib
import json
import runpy
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'revision_r1/outputs'

def main():
    r=runpy.run_path(str(ROOT/'revision_r1/scripts/run_2023_revision_analyses.py'))
    b,w=r['load_project_modules']();df=r['load_modeling_table'](b)
    tr,te,f=r['load_frozen_indices'](df,w)
    counts=[dict(wave=2023,scope='analytic',n=len(df),events=int(df.y_qn26.sum()),non_events=int((df.y_qn26==0).sum())),
            dict(wave=2023,scope='training',n=len(tr),events=int(df.loc[tr,'y_qn26'].sum()),non_events=int((df.loc[tr,'y_qn26']==0).sum())),
            dict(wave=2023,scope='internal',n=len(te),events=int(df.loc[te,'y_qn26'].sum()),non_events=int((df.loc[te,'y_qn26']==0).sum()))]
    ext=pd.read_csv(OUT/'external/yrbs2021_external_predictions.csv')
    counts.append(dict(wave=2021,scope='external',n=len(ext),events=int(ext.y_qn26.sum()),non_events=int((ext.y_qn26==0).sum())))
    pd.DataFrame(counts).to_csv(OUT/'tables/sample_counts.csv',index=False)
    raw=pd.read_excel(ROOT/'data/raw/YRBS_2023_MH_subset.xlsx')
    desc=[]
    for q in ['QN26','QN24','QN25']:
        valid=raw[q].isin([1,2]);y=(raw.loc[valid,q]==1).astype(int)
        desc.append(dict(wave=2023,variable=q,n=int(valid.sum()),empirical_prevalence=y.mean(),cdc_weighted_prevalence=np.average(y,weights=raw.loc[valid,'weight'])))
    desc.append(dict(wave=2021,variable='QN25_outcome',n=len(ext),empirical_prevalence=ext.y_qn26.mean(),cdc_weighted_prevalence=np.average(ext.y_qn26,weights=ext.weight)))
    pd.DataFrame(desc).to_csv(OUT/'tables/descriptive_prevalence.csv',index=False)
    training_prevalence=float(df.loc[tr,'y_qn26'].mean())
    internal_y=df.loc[te,'y_qn26'].to_numpy(dtype=int)
    null_brier=float(np.mean((internal_y-training_prevalence)**2))
    matched=pd.read_csv(OUT/'metrics/matched_hgb_2023_metrics_seed2026.csv')
    selected=matched.query("feature_set == 'expanded' and calibration == 'platt'").iloc[0]
    pd.DataFrame([dict(
        reference='training-prevalence constant probability',
        training_prevalence=training_prevalence,
        n_internal=len(internal_y),
        null_brier=null_brier,
        selected_model_brier=float(selected.brier),
        brier_skill_score=1-float(selected.brier)/null_brier,
    )]).to_csv(OUT/'tables/null_brier_reference_2023.csv',index=False)
    pred=pd.read_csv(OUT/'tables/matched_hgb_2023_holdout_predictions_seed2026.csv')
    records=[]
    for label in ['baseline','expanded']:
        path=OUT/f'models/{label}_platt_2023_frozen.joblib'
        fitted=joblib.load(path)
        for wave,data,target,col in [(2023,df.loc[te],pred,f'{label}_platt'),(2021,ext,ext,f'{label}_platt_2023_frozen')]:
            x=w['prepare_feature_frame'](data,fitted['features'],['q1','q2','q3','raceeth'])
            actual=r['apply_platt'](fitted['calibrator'],np.clip(fitted['estimator'].predict_proba(x)[:,1],r['EPS'],1-r['EPS']))
            error=float(np.max(np.abs(actual-target[col].to_numpy())))
            assert error<1e-12
            records.append(dict(feature_set=label,wave=wave,n=len(data),maximum_absolute_prediction_difference=error,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pd.DataFrame(records).to_csv(OUT/'tables/frozen_model_identity.csv',index=False)
    selection=pd.read_csv(OUT/'tables/training_side_candidate_selection_seed2026.csv')
    claims={'platt_vs_none_cv_brier_margin':float(selection.iloc[1].brier-selection.iloc[0].brier),
            'raw_2023_records':20103,'excluded_2023_outcomes':240,
            'raw_2021_records':17232,'excluded_2021_outcomes':271,
            'public_design_strata':int(df.stratum.nunique()),'public_design_psus':int(df.psu.nunique())}
    (OUT/'derived_claim_values.json').write_text(json.dumps(claims,indent=2),encoding='utf-8')
    versions={}
    import importlib.metadata,platform,sys
    for p in ['numpy','pandas','scikit-learn','scipy','matplotlib','pyarrow','openpyxl','joblib','statsmodels','pytest']:
        versions[p]=importlib.metadata.version(p)
    (OUT/'execution_environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),packages=versions),indent=2),encoding='utf-8')
    print('Descriptive outputs and frozen model identity verified.')

if __name__=='__main__':main()
