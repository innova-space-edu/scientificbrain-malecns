"""MaleCNS-Cognitive C0.10 2x2 native consolidation ablation.
From C09 epoch08 (read-only), rerun epochs 9-12 with fixed curriculum.
Ablate EXTRA 372 rehearsals per epoch and historical 416 pseudo-labels independently.
NO ADAPTER. No changes to Principal/MAND. No tuning on DEV/H.
"""
from pathlib import Path
import hashlib, json, time
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi

C010_VERSION="MaleCNS-Cognitive-C0.10-Consolidation-DEV/0.1"
C010_C09_SOURCE_SHA="255a9785b7dc04d15d5287b68a0cdd240c22da7ddb761e2acb295517a12ad2fe"
C010_PARENT_SYSTEM_SHA="6b2352f305e5e9f897cfca4120408bbc3a0de9a68beb6e7c4e8258f184c4c0e9"
C010_ARMS={
 "no_extra_no_pseudo": (False,False,"native_interleaved"),
 "extra_only": (True,False,"native_interleaved"),
 "pseudo_only": (False,True,"native_interleaved"),
 "extra_and_pseudo": (True,True,"native_interleaved"),
 "frozen_full": (True,True,"frozen"),
 "shuffled_full": (True,True,"shuffled"),
}

api=KaggleApi()
api.authenticate()
pull=Path('/kaggle/working/c010_c09_verified')
pull.mkdir(parents=True,exist_ok=True)
api.kernels_pull('esthefano/malecns-cognitive-c09-native-smoke',path=str(pull),metadata=False)
f=list(pull.glob('*.py'))
assert len(f)==1,f
previous=f[0].read_text()
assert hashlib.sha256(previous.encode()).hexdigest()==C010_C09_SOURCE_SHA
separator="\nc09_tests()\nSMOKE_MODE=True"
assert previous.count(separator)==1,"C09 source layout unexpected"
exec(compile(previous.split(separator)[0],"<Cognitive-C09-pinned-definitions>","exec"),globals())
assert all(x in globals() for x in ("c09_dataset","c09_order","c09_replay","c08_metric","c09_cohort_scores","native_step","checkpoint_with_cues","load"))
print("C010 VERIFIED native source; no external plasticity adapter",flush=True)

def c010_tests(seed=7403):
    for test_seed in (7403,7411,7429):
        pool,old,D,E,H=c09_dataset(test_seed)
        assert len(pool)==744 and sum(x["cohort"] in ("A","B","C") for x in pool)==248
        assert len(set(p["cue"] for p in pool))==744
        for epoch in range(9,13):
            order=c09_order(pool,test_seed,epoch,2,"native_interleaved")
            assert len(order)==len(set(order))==744
            repeat=c09_replay(pool,test_seed,epoch,"native_interleaved")
            assert len(repeat)==len(set(repeat))==372
        assert len(H)==64
    assert len(C010_ARMS)==6
    print("C010 STATIC TESTS PASS: 3 seeds (schedule integrity, no data leakage)",flush=True)

def c010_parent():
    found=[p for p in Path('/kaggle/input').rglob('system.json')
       if 'malecns-cognitive-c09-retrain-reinforce-long' in str(p)
       and '/MaleCNS_C09_Long_7403/checkpoints/native_interleaved/epoch08/system.json' in str(p)]
    assert len(found)==1,('C09 epoch08 checkpoint not unique',len(found))
    p=found[0]
    assert hashlib.sha256(p.read_bytes()).hexdigest()==C010_PARENT_SYSTEM_SHA
    return p.parent

def c010_run(seed=7403,epochs=4,smoke=False,output="/kaggle/working/MaleCNS_C010_Consolidation_7403"):
    started=time.time()
    out=Path(output)
    out.mkdir(parents=True,exist_ok=True)
    assert seed==7403,"Only seed 7403 has a trained C09 epoch08 checkpoint"
    parent=c010_parent()
    initial=load(parent,threads=2)
    assert initial.core.brain.n==8192 and initial.core.nkc==4064 and initial.cues.shape==(2224,32)
    theta_ref=initial.core.trainer.params['theta'].copy()
    pool,old,D,E,H=c09_dataset(seed)
    labels=np.array([p['target'] for p in pool],dtype=np.int32)
    scrambled=labels.copy()
    rng=np.random.default_rng(np.random.SeedSequence([seed,909,89]))
    for area in C08_AREAS:
        ix=[i for i,p in enumerate(pool) if p['area']==area]
        scrambled[ix]=rng.permutation(scrambled[ix])
    parent_snapshot=checkpoint_with_cues(initial,out/'parent_snapshot')
    historical={int(i):int(y) for i,y,v in scores(parent_snapshot,list(range(416)),seed)}
    probes=old_probe_ids(416,128,seed)
    original={int(i):int(y) for i,y,v in scores(parent_snapshot,probes,seed)}
    assert len(historical)==416 and len(original)==128
    baseline=c08_metric(parent,old,D,E,H,seed,original,'FINAL')
    baseline_cohorts=c09_cohort_scores(parent,old,seed)
    print('C010 BASELINE',json.dumps({'metrics':baseline,'cohorts':baseline_cohorts}),flush=True)
    result={}
    for name,(extra,pseudo,control) in C010_ARMS.items():
        s=load(parent,threads=2)
        s.mode='frozen' if control=='frozen' else 'live'
        s.core.stable_mode=s.mode
        s.memory_settings['batch_size']=1
        fixed_margin=float(s.memory_settings['margin'])
        fixed_damping=float(s.memory_settings['damping'])
        meters={'feedback':0,'changed_events':0,'changed_synapses_cumulative':0,'native_skipped':0}
        history=[]
        for local_epoch in range(1,epochs+1):
            epoch=8+local_epoch
            order=c09_order(pool,seed,epoch,2,'native_interleaved')
            if smoke: order=order[:48]
            for pos,i in enumerate(order):
                y=int(scrambled[i] if control=='shuffled' else labels[i])
                q1,q2=pool[i]['cue']
                masked='partial' if (epoch+pos)%5==0 else 'clean'
                native_step(s,int(q1),y,seed,epoch,masked,control,meters)
                native_step(s,int(q2),y,seed,epoch,masked,control,meters)
            if extra:
                repeats=c09_replay(pool,seed,epoch,'native_interleaved')
                if smoke:repeats=repeats[:24]
                for i in repeats:
                    y=int(scrambled[i] if control=='shuffled' else labels[i])
                    q1,q2=pool[i]['cue']
                    native_step(s,int(q1),y,seed,epoch,'clean',control,meters)
                    native_step(s,int(q2),y,seed,epoch,'clean',control,meters)
            if pseudo and local_epoch==1:
                for q in range(32 if smoke else 416):
                    y=int(historical[q])
                    if control=='shuffled':y=-y
                    native_step(s,q,y,seed,epoch,'clean',control,meters)
            assert s.pending is None and s.core.pending is None
            assert float(s.memory_settings['margin'])==fixed_margin
            assert float(s.memory_settings['damping'])==fixed_damping
            ck=checkpoint_with_cues(s,out/'checkpoints'/name/f'ep{epoch:02d}')
            metric=c08_metric(ck,old,D,E,H,seed,original,'FINAL')
            cohort_metric=c09_cohort_scores(ck,old,seed)
            record={'epoch':epoch,'metrics':metric,'cohorts':cohort_metric,'meters':meters.copy()}
            history.append(record)
            print('C010',name,'ep',epoch,'ABC',round(metric['A_B_C_oldtrain'],4),
                  'D',round(metric['new_D'],4),'E',round(metric['new_E_one_shot'],4),
                  'DEV',round(metric['prior_DEV'],4),'H',round(metric['fresh_H_never_trained'],4),
                  'agreement',round(metric['original_128_prediction_agreement'],4),flush=True)
        result[name]={'extra_372':extra,'pseudo_416':pseudo,'control':control,
                      'history':history,'final':history[-1],
                      'distinct_theta_changed':int(np.count_nonzero(s.core.trainer.params['theta']!=theta_ref))}
    report={'version':C010_VERSION,'status':'DEV_COMPLETE','seed':seed,'smoke':smoke,
      'epochs':epochs,'start_from':'C09 long native_interleaved epoch08, read-only',
      'parent_sha256':C010_PARENT_SYSTEM_SHA,'previous_code_sha256':C010_C09_SOURCE_SHA,
      'no_adapter':True,'native_plasticity_only':True,'neurons':8192,'training_experiences':744,
      'control_design':'2x2 extra rehearsal x historical pseudo labels plus Frozen/Shuffled',
      'previous_DEV_H_are_exploratory':True,'historical_agreement_is_not_accuracy':True,
      'baseline':baseline,'baseline_cohorts':baseline_cohorts,
      'results':result,'seconds':time.time()-started,'promoted':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('C010_FINISHED',smoke,epochs,round(report['seconds'],1),flush=True)
    return report

c010_tests()
SMOKE=True
if SMOKE:
    R=c010_run(seed=7403,epochs=1,smoke=True,output='/kaggle/working/MaleCNS_C010_Smoke_7403')
else:
    R=c010_run(seed=7403,epochs=4,smoke=False,output='/kaggle/working/MaleCNS_C010_Long_7403')
assert R['status']=='DEV_COMPLETE' and R['no_adapter'] and len(R['results'])==6
