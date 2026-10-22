"""MaleCNS-Cognitive C0.12 SHORT CONTINUOUS: three full epochs, no adapter.
Parent C0.11 continuous/final, complete checkpoints and final-only novel J test.
No pseudo-labels, no extra rehearsal, no selection mid-training.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
C012_VERSION="MaleCNS-Cognitive-C0.12-ShortContinuous-DEV/0.1"
C012_PARENT_SHA="888872be4c5b12b59414c7a98f908fad65b542c00f9a5cf5c8df34a53199e11e"
C012_PINNED_C011_CODE="c5bc7ad4fa3d533f5fb7a7917ae48a49504a1d0b7898daed92f6c2ba0fc15e52"
api=KaggleApi();api.authenticate()
cache=Path('/kaggle/working/c012_verified_c011');cache.mkdir(parents=True,exist_ok=True)
api.kernels_pull('esthefano/malecns-cognitive-c011-micro-smoke',path=str(cache),metadata=False)
pieces=list(cache.glob('*.py'));assert len(pieces)==1,pieces
code=pieces[0].read_text()
assert hashlib.sha256(code.encode()).hexdigest()==C012_PINNED_C011_CODE,'Previous version changed: fail closed'
tail='\nc011_tests()\nSMOKE=True'
assert code.count(tail)==1
exec(compile(code.split(tail)[0],'<Cognitive-C011-native-definitions>','exec'),globals())
need=('c09_dataset','c09_order','c011_train_one','c011_score_holdout',
      'c011_eval','c011_new_holdouts','checkpoint_with_cues','load','scores','encode_corpus')
assert all(n in globals() for n in need)
print('C012 NATIVE C011 CODE VERIFIED (no external adapter)',flush=True)

def c012_parent():
    matched=[p for p in Path('/kaggle/input').rglob('system.json')
        if 'malecns-cognitive-c011-micro-long' in str(p)
        and '/MaleCNS_C011_Long_7403/checkpoints/continuous/final/system.json' in str(p)]
    assert len(matched)==1,('C012 expected exactly one C011 continuous checkpoint',len(matched))
    got=hashlib.sha256(matched[0].read_bytes()).hexdigest()
    assert got==C012_PARENT_SHA,('C012 parent SHA mismatch',got)
    return matched[0].parent

def c012_new_J():
    """64 new binary held-out problems; only evaluated at end, never used for feedback or checkpoint selection."""
    chars='ρστυφχψω'
    assert len(chars)==8 and len(set(chars))==8
    rows=[]
    for i in range(8):
        a,b=chars[i],chars[(i+1)%8]
        rows.append(dict(area='symbols',op='equal',args=[a,a],target=1,cohort='J_final'))
        rows.append(dict(area='symbols',op='equal',args=[a,b],target=-1,cohort='J_final'))
    for i in range(12):
        a=3100+3*i;b=53+5*i
        rows.append(dict(area='math',op='add',args=[a,b,a+b],target=1,cohort='J_final'))
        rows.append(dict(area='math',op='add',args=[a,b,a+b+2+(i%5)],target=-1,cohort='J_final'))
    for i in range(12):
        a=chars[i%8];b=chars[(i+1+i//8)%8];c=chars[(i+3+2*(i//8))%8]
        assert len({a,b,c})==3
        rows.append(dict(area='logic',op='transitivity',args=[a,b,c,a,c],target=1,cohort='J_final'))
        rows.append(dict(area='logic',op='transitivity',args=[a,b,c,c,a],target=-1,cohort='J_final'))
    assert len(rows)==64 and sum(x['target']==1 for x in rows)==32
    signatures=[(r['area'],r['op'],tuple(r['args'])) for r in rows]
    assert len(set(signatures))==64
    return rows

def c012_static_tests():
    J=c012_new_J()
    F,G=c011_new_holdouts()
    sig=lambda r:(r['area'],r['op'],tuple(r['args']))
    assert len({sig(r) for r in F+G+J})==192
    for seed in (7403,7411,7429):
        pool,old,D,E,H=c09_dataset(seed)
        assert len(pool)==744
        for epoch in (9,10,11):
            order=c09_order(pool,seed,epoch,2,'native_interleaved')
            assert len(order)==len(set(order))==744
    print('C012 STATIC TESTS: PASS (3 schedule seeds; 64 final-only J)',flush=True)

def c012_run(seed=7403,epochs=3,smoke=False,output='/kaggle/working/MaleCNS_C012_Long_7403'):
    started=time.time()
    assert seed==7403,'Only 7403 has a verified trained parent checkpoint'
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    parent=c012_parent()
    initial=load(parent,threads=2)
    assert initial.core.brain.n==8192 and initial.core.nkc==4064
    assert initial.cues.shape==(2480,32),initial.cues.shape
    initial_theta=initial.core.trainer.params['theta'].copy()
    pool,old,D,E,H=c09_dataset(seed)
    F,G=c011_new_holdouts()
    J=c012_new_J()
    j_code=encode_corpus(J,initial.core.nkc)
    assert j_code.shape==(128,32)
    all_cues={tuple(sorted(map(int,v))) for v in initial.cues}
    assert len(all_cues)==2480
    for cue in j_code:
        sig=tuple(sorted(map(int,cue)))
        assert sig not in all_cues,'J overlaps earlier memory: reject'
        all_cues.add(sig)
    initial.cues=np.vstack([initial.cues,j_code])
    assert initial.cues.shape==(2608,32)
    prepared=checkpoint_with_cues(initial,out/'immutable_parent_snapshot')
    refids=old_probe_ids(416,128,seed)
    old_predictions={int(i):int(y) for i,y,v in scores(prepared,refids,seed)}
    assert len(old_predictions)==128
    frozen=c011_eval(prepared,pool,old,D,E,H,F,seed,old_predictions,2224)
    print('C012 BASELINE',json.dumps(frozen),flush=True)
    meters={'feedback':0,'changed_events':0,'changed_synapses_cumulative':0,'native_skipped':0}
    records=[]
    # One and only one live model is carried across all epochs without restoring
    # checkpoints or selecting by any metric.
    model=load(prepared,threads=2)
    for cycle in range(9,9+epochs):
        used=c011_train_one(model,pool,seed,cycle,smoke,meters)
        assert used==(32 if smoke else 744)
        saved=checkpoint_with_cues(model,out/'checkpoints'/f'epoch{cycle:02d}')
        metric=c011_eval(saved,pool,old,D,E,H,F,seed,old_predictions,2224)
        records.append(dict(epoch=cycle,examples=used,metrics=metric,meters=meters.copy(),checkpoint=str(saved)))
        print('C012 LIVE EPOCH',cycle,'macro',round(metric['training_macro_mean'],4),
              'F',round(metric['validation_F']['accuracy'],4),
              'old',round(metric['historical_agreement'],4),
              'previous_DEV',round(metric['previous_DEV_exploratory'],4),flush=True)
    # Fresh J is read at the end only. Baseline and final scores both
    # computed here, after all training; not used to tune or select checkpoints.
    frozenJ=c011_score_holdout(prepared,2480,J,seed)
    last=records[-1]['checkpoint']
    finalJ=c011_score_holdout(last,2480,J,seed)
    priorG=c011_score_holdout(last,2352,G,seed)
    theta_changed=int(np.count_nonzero(model.core.trainer.params['theta']!=initial_theta))
    report=dict(version=C012_VERSION,status='DEV_COMPLETE',seed=seed,smoke=bool(smoke),
      epochs=epochs,source_C011_sha256=C012_PINNED_C011_CODE,
      read_only_parent_sha256=C012_PARENT_SHA,parent='C011 continuous final (8 cycles)',
      neurons=8192,train_experiences=744,no_plasticity_adapter=True,
      only_native_synaptic_feedback=True,no_extra_rehearsals=True,no_pseudo_labels=True,
      sequential_epochs_without_reload=True,checkpoint_per_epoch=True,
      final_J_not_trained_not_selected=True,previous_F_G_H_DEV_exploratory=True,
      baseline=frozen,records=records,final=records[-1]['metrics'],
      new_J_baseline_evaluated_at_end=frozenJ,new_J_final=finalJ,
      prior_G_exploratory=priorG,distinct_theta_changed=theta_changed,
      meters=meters,elapsed_seconds=time.time()-started,promoted=False)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('C012 DONE',seed,epochs,'smoke',smoke,'J',frozenJ['accuracy'],finalJ['accuracy'],
          'seconds',round(report['elapsed_seconds'],1),flush=True)
    return report

c012_static_tests()
short=c012_run(seed=7403,epochs=1,smoke=True,output='/kaggle/working/MaleCNS_C012_EmbeddedSmoke_7403')
assert short['status']=='DEV_COMPLETE' and short['no_plasticity_adapter'] and short['train_experiences']==744
assert len(short['records'])==1
print('C012 SMOKE GATE PASSED: 3 full uninterrupted epochs',flush=True)
full=c012_run(seed=7403,epochs=3,smoke=False,output='/kaggle/working/MaleCNS_C012_Long_7403')
assert full['status']=='DEV_COMPLETE' and not full['smoke']
assert len(full['records'])==3 and full['records'][-1]['epoch']==11
assert sum(item['examples'] for item in full['records'])==2232
print('C012 THREE CONTINUOUS EPOCHS COMPLETE',round(full['elapsed_seconds'],1),flush=True)
