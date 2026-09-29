"""Recompute the retained candidate set using the original fold-safe procedure."""
import runpy
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = runpy.run_path(str(ROOT/'revision_r1/scripts/run_2023_revision_analyses.py'))

def main():
    build, w = R['load_project_modules']()
    df = R['load_modeling_table'](build)
    tr, te, fold = R['load_frozen_indices'](df,w)
    params=R['load_hgb_params']()
    fs=w['derive_feature_sets'](df.columns)
    records=[]
    for model, featureset, features, cal in [
        ('logreg','baseline',fs.baseline_features,'none'),
        ('logreg','full',fs.full_features,'none'),
        ('hgb','baseline',fs.baseline_features,'none'),
        ('hgb','full',fs.full_features,'none'),
        ('hgb','full',fs.full_features,'platt'),
        ('hgb','full',fs.full_features,'isotonic')]:
        print(model,featureset,cal,flush=True)
        x=w['prepare_feature_frame'](df,features,['q1','q2','q3','raceeth'])
        y=df.y_qn26.astype(int)
        def factory():
            return w['build_estimator'](model_name=model,preprocessor=w['build_preprocessor'](features,['q1','q2','q3','raceeth']),seed=2026,tuned_hgb_params=params)
        folds=w['run_cv_metrics_with_leakage_safe_calibration'](estimator_factory=factory,x_train=x.loc[tr],y_train=y.loc[tr],fold_id=fold,calibration_method=cal,seed=2026)
        stem=f'seed2026_{model}_{featureset}_{cal}'
        folds.to_csv(ROOT/f'revision_r1/outputs/tables/cv_folds_{stem}.csv',index=False)
        for scope,metrics in [('cv',w['aggregate_cv_metrics'](folds)),('test',w['run_test_metrics_with_train_only_calibration'](estimator_factory=factory,x_train=x.loc[tr],y_train=y.loc[tr],x_test=x.loc[te],y_test=y.loc[te],fold_id=fold,calibration_method=cal))]:
            path=ROOT/f'outputs/metrics/metrics_{scope}_{stem}.csv'
            old=pd.read_csv(path).iloc[0].to_dict()
            for k,v in metrics.items():
                if k in old:records.append(dict(file=path.name,metric=k,retained=old[k],regenerated=v,absolute_difference=abs(float(old[k])-v)))
            old.update(metrics)
            pd.DataFrame([old]).to_csv(path,index=False)
    pd.DataFrame(records).to_csv(ROOT/'revision_r1/outputs/tables/regeneration_comparison.csv',index=False)
    R['select_from_training_cv']()

if __name__=='__main__':main()
