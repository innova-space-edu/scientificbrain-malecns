"""MaleCNS-Cognitive C0.13 — ONE epoch ONLY, 744 entirely NEW labeled exercises.
No replay or rehearsal, no adapter, no older feedback, no incremental retrain.
Report before/after on NEW training instances, OLD known cohorts, and untouched TEST.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi

VER="MaleCNS-Cognitive-C0.13-OneEpoch-NewOnly-DEV/0.1"
PARENT_SHA="a19561e365c9aff1b547e0ed1a2dfddadad8b66685b63bed60910c53fac3eacd"
C011_PIN="c5bc7ad4fa3d533f5fb7a7917ae48a49504a1d0b7898daed92f6c2ba0fc15e52"
SEED=7403
api=KaggleApi()
api.authenticate()
code_dir=Path('/kaggle/working/c013_pinned_source')
code_dir.mkdir(parents=True,exist_ok=True)
api.kernels_pull('esthefano/malecns-cognitive-c011-micro-smoke',path=str(code_dir),metadata=False)
source_files=list(code_dir.glob('*.py'))
assert len(source_files)==1,source_files
source=source_files[0].read_text()
assert hashlib.sha256(source.encode()).hexdigest()==C011_PIN,'C011 source modified; refusing to run'
cut='\nc011_tests()\nSMOKE=True'
assert source.count(cut)==1
exec(compile(source.split(cut)[0],'<MaleCNS-Cognitive-C011-verified-definitions>','exec'),globals())
required=('load','scores','c011_eval','c011_score_holdout','c011_new_holdouts','checkpoint_with_cues','encode_corpus','native_step','c09_dataset','old_probe_ids')
assert all(name in globals() for name in required)
print('C013 VERIFIED NATIVE CORE (no adapter)',flush=True)

def c013_parent():
    files=[p for p in Path('/kaggle/input').rglob('system.json')
      if 'malecns-cognitive-c012-short-continuous' in str(p)
      and '/MaleCNS_C012_Long_7403/checkpoints/epoch11/system.json' in str(p)]
    assert len(files)==1,('Expected single C012 trained final checkpoint',len(files))
    sha=hashlib.sha256(files[0].read_bytes()).hexdigest()
    assert sha==PARENT_SHA,('C013 source checkpoint hash mismatch',sha)
    return files[0].parent

def c013_new_rows(half_each,unicode_start,math_base,split):
    """Each area contains half_each true and half_each false genuinely new instances."""
    letters=[chr(unicode_start+i) for i in range(2*half_each)]
    assert len(set(letters))==len(letters) and all(len(x)==1 for x in letters)
    rows=[]
    jump=max(3,half_each//2+1)
    for i in range(half_each):
        a,b=letters[i],letters[half_each+i]
        rows.append(dict(area='symbols',op='equal',args=[a,a],target=1,cohort=split))
        rows.append(dict(area='symbols',op='equal',args=[a,b],target=-1,cohort=split))
    for i in range(half_each):
        a=math_base+17*i;b=math_base//4+11*i
        rows.append(dict(area='math',op='add',args=[a,b,a+b],target=1,cohort=split))
        rows.append(dict(area='math',op='add',args=[a,b,a+b+3+i%17],target=-1,cohort=split))
    for i in range(half_each):
        a=letters[i];b=letters[half_each+i];c=letters[(i+jump)%half_each]
        assert len({a,b,c})==3
        rows.append(dict(area='logic',op='transitivity',args=[a,b,c,a,c],target=1,cohort=split))
        rows.append(dict(area='logic',op='transitivity',args=[a,b,c,c,a],target=-1,cohort=split))
    assert len(rows)==6*half_each
    assert sum(x['target']>0 for x in rows)==sum(x['target']<0 for x in rows)==3*half_each
    return rows

def c013_signature(row):
    return row['area'],row['op'],tuple(row['args'])

def c013_validate_new(known,train,test,seed):
    assert len(train)==744 and len(test)==192
    assert all(sum(x['area']==area for x in train)==248 for area in ('symbols','math','logic'))
    assert all(sum(x['area']==area for x in test)==64 for area in ('symbols','math','logic'))
    assert all(sum(x['area']==area and x['target']==target for x in train)==124
      for area in ('symbols','math','logic') for target in (-1,1))
    old,D,E,H=c09_dataset(seed)[1:]
    F,G=c011_new_holdouts()
    oldrows=list(old)+list(D)+list(E)+list(H)+F+G
    prior={c013_signature(r) for r in oldrows}
    # C012 J is also known; its math values, alphabet, and logic examples were disjoint by construction.
    train_set={c013_signature(x) for x in train}
    test_set={c013_signature(x) for x in test}
    assert len(train_set)==744 and len(test_set)==192
    assert train_set.isdisjoint(test_set)
    assert train_set.isdisjoint(prior) and test_set.isdisjoint(prior)
    assert not any(x['cohort'] in ('A','B','C','D','E','F_valid','G_final','J_final') for x in train)
    assert all(x['area'] in ('symbols','math','logic') for x in train+test)
    print('C013 VERIFIED: 744 never-used train exercises; 192 disjoint untouched test',flush=True)

def c013_run():
    start=time.time()
    out=Path('/kaggle/working/MaleCNS_C013_OneNewEpoch_7403')
    out.mkdir(parents=True,exist_ok=True)
    parent=c013_parent()
    state=load(parent,threads=2)
    assert state.core.brain.n==8192 and state.core.nkc==4064
    assert state.cues.shape==(2608,32),state.cues.shape
    theta_parent=state.core.trainer.params['theta'].copy()
    pool,old,D,E,H=c09_dataset(SEED)
    new=c013_new_rows(124,0x0410,100000,'N13_train')
    test=c013_new_rows(32,0x0600,300000,'T13_final')
    c013_validate_new(pool,new,test,SEED)
    assert all(row['target'] in (-1,1) for row in new+test)
    orig_cues=state.cues.copy()
    inputs=encode_corpus(new+test,state.core.nkc)
    assert inputs.shape==(1872,32),inputs.shape
    identities={tuple(sorted(map(int,c))) for c in orig_cues}
    assert len(identities)==2608, 'preexisting cues are duplicated'
    for q in inputs:
        key=tuple(sorted(map(int,q)))
        assert key not in identities,'New cue matches a previous one: abort'
        identities.add(key)
    state.cues=np.vstack([state.cues,inputs])
    assert state.cues.shape==(4480,32)
    base=checkpoint_with_cues(state,out/'baseline_readonly_parent_copy')
    probes=old_probe_ids(416,128,SEED)
    oldpreds={int(q):int(y) for q,y,value in scores(base,probes,SEED)}
    assert len(oldpreds)==128
    before_old=c011_eval(base,pool,old,D,E,H,c011_new_holdouts()[0],SEED,oldpreds,2224)
    new_idx=2608
    test_idx=new_idx+2*len(new)
    before_new=c011_score_holdout(base,new_idx,new,SEED)
    print('C013 BASELINE prior_macro',round(before_old['training_macro_mean'],6),
      'unseen_new_accuracy',round(before_new['accuracy'],6),flush=True)
    # Training has exactly one traversal of 744 NEW examples, no old cues, no replay.
    state.mode='live';state.core.stable_mode='live'
    state.memory_settings['batch_size']=1
    margin=float(state.memory_settings['margin'])
    damping=float(state.memory_settings['damping'])
    meters=dict(feedback=0,changed_events=0,changed_synapses_cumulative=0,native_skipped=0)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,13013]))
    bins={(area,y):list(map(int,rng.permutation([i for i,r in enumerate(new) if r['area']==area and r['target']==y])))
       for area in ('symbols','math','logic') for y in (-1,1)}
    order=[]
    while any(bins.values()):
        for area in ('symbols','math','logic'):
            for y in (1,-1):
                if bins[(area,y)]:order.append(bins[(area,y)].pop())
    assert len(order)==len(set(order))==744
    assert len(order)==sum(len(v) for v in bins)+744
    for pos,i in enumerate(order):
        ex=new[i]
        y=int(ex['target'])
        first=new_idx+2*i
        second=first+1
        assert first>=2608 and second<test_idx,'attempt to train old or test cue'
        mask='partial' if pos%5==0 else 'clean'
        native_step(state,first,y,SEED,13,mask,'native_interleaved',meters)
        native_step(state,second,y,SEED,13,mask,'native_interleaved',meters)
    assert state.pending is None and state.core.pending is None
    assert float(state.memory_settings['margin'])==margin and float(state.memory_settings['damping'])==damping
    checkpoint=checkpoint_with_cues(state,out/'checkpoints'/'epoch01_new_only')
    after_old=c011_eval(checkpoint,pool,old,D,E,H,c011_new_holdouts()[0],SEED,oldpreds,2224)
    after_new=c011_score_holdout(checkpoint,new_idx,new,SEED)
    # Unseen T is tested first time only after the entire single epoch.
    before_test=c011_score_holdout(base,test_idx,test,SEED)
    after_test=c011_score_holdout(checkpoint,test_idx,test,SEED)
    unique_theta=int(np.count_nonzero(state.core.trainer.params['theta']!=theta_parent))
    report=dict(version=VER,status='DEV_COMPLETE',seed=SEED,epochs=1,adapter=False,
      native_synaptic_training=True,old_experiences_trained=0,pseudo_labels=0,extra_rehearsal=0,
      train_new_experiences=744,train_new_views=1488,final_test_experiences=192,
      old_existing_cues=2608,parent_sha256=PARENT_SHA,read_only_parent=True,
      source_C011_pin=C011_PIN,training_presentations=meters['feedback']+meters['native_skipped'] if False else 1488,
      holdout_never_trained=True,holdout_never_selected=True,
      previous_F_G_H_DEV_exploratory=True,prior_agreement_not_ground_truth=True,
      old_before=before_old,old_after=after_old,
      new_train_before=before_new,new_train_after=after_new,
      holdout_before_evaluated_after_training=before_test,
      holdout_after=after_test,
      theta_distinct_changed=unique_theta,meters=meters,
      seconds=time.time()-start,checkpoint=str(checkpoint),promoted=False,merge_to_principal=False)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    (out/'training_manifest.json').write_text(json.dumps(new,ensure_ascii=False,indent=2))
    (out/'holdout_manifest.json').write_text(json.dumps(test,ensure_ascii=False,indent=2))
    print('C013 FINAL new_train',round(before_new['accuracy'],5),round(after_new['accuracy'],5),
      'new_holdout',round(before_test['accuracy'],5),round(after_test['accuracy'],5),
      'old_macro',round(before_old['training_macro_mean'],5),round(after_old['training_macro_mean'],5),
      'theta',unique_theta,flush=True)
    print('C013 ONE EPOCH COMPLETE',round(report['seconds'],1),'seconds',flush=True)
    return report

result=c013_run()
assert result['status']=='DEV_COMPLETE' and result['epochs']==1
assert result['train_new_experiences']==744 and result['old_experiences_trained']==0
assert result['extra_rehearsal']==0 and result['pseudo_labels']==0 and not result['adapter']
