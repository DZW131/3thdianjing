"""Fixed-object conditional classification: existing baseline vs grouped fusion."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support,f1_score,confusion_matrix

def main():
    p=argparse.ArgumentParser()
    for k in ['baseline','states','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Use fresh output')
    a.output.mkdir(parents=True)
    ref=[r for r in csv.DictReader((a.states/'oof_predictions.csv').open()) if r['experiment']=='appearance_control']
    b=list(csv.DictReader((a.baseline/'baseline_objects.csv').open()))
    def key(r):return(r['sample_id'],r['component_id'],r['class_id'])
    lookup={key(r):r for r in b};assert len(lookup)==len(ref)==len(b)==101
    b=[lookup[key(r)] for r in ref]
    assert all(r['fold']==s['fold'] and r['case_id']==s['case_id'] for r,s in zip(ref,b))
    labels=np.array([int(r['class_id']) for r in ref]);folds=np.array([int(r['fold']) for r in ref]);local=np.load(a.states/'local_states.npz')
    columns=[f'{prefix}_{cl}' for prefix in ['prob','raw_fraction','post_fraction'] for cl in range(5)]
    x=np.array([[float(r[c]) for c in columns] for r in b]);assert np.isfinite(x).all()
    preds={'baseline_raw':np.array([int(r['raw_class']) for r in b]),'baseline_post':np.array([int(r['post_class']) for r in b])}
    experiments={'baseline_information_calibrator':x,'baseline_information_plus_local_mean':np.c_[x,local['mean']],'baseline_information_plus_local_mean_std':np.c_[x,local['mean'],local['std']]}
    for name,features in experiments.items():
        pred=np.zeros(len(labels),dtype=int)
        for fold in np.unique(folds):
            tr=folds!=fold;te=~tr
            assert not set(r['case_id'] for r,t in zip(ref,tr) if t)&set(r['case_id'] for r,t in zip(ref,te) if t)
            clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,random_state=42))
            clf.fit(features[tr],labels[tr]);pred[te]=clf.predict(features[te]);joblib.dump(clf,a.output/f'{name}_fold{fold}.joblib')
        preds[name]=pred
    results=[];rows=[];detail=[]
    for name,pred in preds.items():
        pp,rr,ff,nn=precision_recall_fscore_support(labels,pred,labels=[1,2,3],zero_division=0)
        results.append(dict(experiment=name,macro_f1=float(ff.mean()),moderate_precision=float(pp[1]),moderate_recall=float(rr[1]),moderate_f1=float(ff[1]),n=len(labels)))
        for cl,pr,re,f,n in zip([1,2,3],pp,rr,ff,nn):detail.append(dict(experiment=name,class_id=cl,precision=float(pr),recall=float(re),f1=float(f),n=int(n)))
        # Keep background/autophagy predictions as errors, never remap them into H/M/S.
        np.savetxt(a.output/f'{name}_confusion.csv',confusion_matrix(labels,pred,labels=[0,1,2,3,4]),delimiter=',',fmt='%d')
        rows.extend(dict(r,experiment=name,prediction=int(pr)) for r,pr in zip(ref,pred))
    for filename,data in [('comparison.csv',results),('per_class.csv',detail),('oof_predictions.csv',rows)]:
        with (a.output/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
    (a.output/'results.json').write_text(json.dumps(dict(results=results,n=len(labels),baseline_provenance=json.loads((a.baseline/'provenance.json').read_text()),feature_columns=columns,limitations=['GT regions define evaluation objects; conditional classification not detection','baseline and texture-classifier training overlap not excluded; direct outputs not OOF','only auxiliary fusion is trained with held-out case folds','same folds reused for exploratory comparisons, not independent confirmation','candidate states lack cardiac GT validation','not a production model replacement']),indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
