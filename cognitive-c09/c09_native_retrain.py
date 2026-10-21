"""MaleCNS-Cognitive C0.9: all known labeled experiences, then reinforcement; native plasticity ONLY.
Full training: 8 complete epochs over A/B/C/D/E (744 labels) + 4 reinforcement epochs.
Read-only parent = Cognitive C0.8 native_interleaved final. Does NOT touch Principal or MAND.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate()
cache=Path('/kaggle/working/c09_c08_pinned');cache.mkdir(parents=True,exist_ok=True)
api.kernels_pull('esthefano/malecns-cognitive-c08-threephases-smoke',path=str(cache),metadata=False)
f=list(cache.glob('*.py'))
assert len(f)==1
source=f[0].read_text()
C08_SOURCE_HASH='562675750ee8087b1e83795bb3e858c149fd8ee407b8d956f9d428168067366c'
assert hashlib.sha256(source.encode()).hexdigest()==C08_SOURCE_HASH
cut='\nc08_tests()'
assert source.count(cut)==1
exec(compile(source.split(cut,1)[0],'<Cognitive-C08-native-definitions>','exec'),globals())
assert all(name in globals() for name in ('c08_material','native_step','load','scores','checkpoint_with_cues','c08_metric'))
print('C09 CORE LOADED: C08 exact pinned source, no adapter',flush=True)

C09_VERSION='MaleCNS-Cognitive-C0.9-Retrain-Reinforce-DEV/0.1'
C09_MODES=('native_interleaved','native_uniform','frozen','shuffled')
C09_COUNTS={'A':72,'B':80,'C':96,'D':248,'E':248}
C09_REPLAY_QUOTA={'A':30,'B':55,'C':60,'D':100,'E':127}
assert sum(C09_COUNTS.values())==744 and sum(C09_REPLAY_QUOTA.values())==372

def c09_dataset(seed):
    old,D,E,H=c08_material(seed)
    train_inds=[i for i,r in enumerate(old) if r['split']=='train']
    assert len(train_inds)==248 and len(D)==len(E)==248 and len(H)==64
    pool=[]
    for i in train_inds:
        r=old[i]
        pool.append({'cohort':r['cohort'],'area':r['area'],'target':int(r['target']),
                     'cue':(416+2*i,417+2*i),'experience':('C07',i)})
    for key,rows,offset in [('D',D,1104),('E',E,1104+2*248)]:
        for i,r in enumerate(rows):
            pool.append({'cohort':key,'area':r['area'],'target':int(r['target']),
                         'cue':(offset+2*i,offset+2*i+1),'experience':(key,i)})
    assert len(pool)==744 and {x['cohort'] for x in pool}==set(C09_COUNTS)
    for group,n in C09_COUNTS.items():
        assert sum(x['cohort']==group for x in pool)==n,(group,n)
    assert len({x['cue'] for x in pool})==744
    assert all(set(x['cue']).isdisjoint(set(range(416))) for x in pool)
    assert max(q for x in pool for q in x['cue'])<2224
    return pool,old,D,E,H

def c09_order(pool,seed,epoch,part,mode):
    rng=np.random.default_rng(np.random.SeedSequence([seed,909,epoch,part]))
    ids=list(range(len(pool)))
    if mode=='native_uniform':
        return list(map(int,rng.permutation(ids)))
    groups={}
    for cohort in C09_COUNTS:
        for area in C08_AREAS:
            for y in (-1,1):
                cand=[i for i in ids if pool[i]['cohort']==cohort and pool[i]['area']==area and pool[i]['target']==y]
                groups[(cohort,area,y)]=list(map(int,rng.permutation(cand)))
    output=[]
    while any(groups.values()):
        for area in C08_AREAS:
            for y in (-1,1):
                for cohort in C09_COUNTS:
                    g=groups[(cohort,area,y)]
                    if g:output.append(g.pop())
    assert len(output)==744 and len(set(output))==744
    return output

def c09_replay(pool,seed,ep,mode,limit=None):
    rng=np.random.default_rng(np.random.SeedSequence([seed,909,55,ep]))
    if mode=='native_uniform':
        picks=list(map(int,rng.choice(744,size=372,replace=False)))
    else:
        picks=[]
        for cohort,n in C09_REPLAY_QUOTA.items():
            candidates=[i for i,r in enumerate(pool) if r['cohort']==cohort]
            picks.extend(map(int,rng.choice(candidates,size=n,replace=False)))
        rng.shuffle(picks)
    assert len(picks)==len(set(picks))==372
    return picks if limit is None else picks[:limit]

def c09_cohort_scores(ck,old,seed):
    idx=[i for i,r in enumerate(old) if r['split']=='train']
    q=[416+2*i+j for i in idx for j in (0,1)]
    values={int(i):float(v) for i,y,v in scores(ck,q,seed)}
    results={}
    for cohort in ('A','B','C'):
        relevant=[i for i in idx if old[i]['cohort']==cohort]
        results[cohort]=float(np.mean([(1 if (values[416+2*i]+values[417+2*i])>=0 else -1)==old[i]['target'] for i in relevant]))
    return results

def c09_tests():
    for seed in (7403,7411,7429):
        pool,old,D,E,H=c09_dataset(seed)
        for mode in C09_MODES:
            assert len(c09_order(pool,seed,1,1,mode))==744
            assert len(c09_replay(pool,seed,1,mode))==372
        assert len(set(x['experience'] for x in pool))==744
        assert sum(x['target']>0 for x in pool)==sum(x['target']<0 for x in pool)
    print('C09 STATIC INTEGRITY PASSED (3 seeds, 744 distinct true TRAIN labels)',flush=True)

def c09_run(seed=7403,retrain_epochs=8,reinforce_epochs=4,smoke=False,output='/kaggle/working/MaleCNS_C09_Retrain_Reinforce'):
    started=time.time()
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    p=[p for p in Path('/kaggle/input').rglob('system.json')
       if 'malecns-cognitive-c08-native-double-retrain-novel' in str(p)
       and '/MaleCNS_C08_Native_Long_7403/' in str(p)
       and '/native_interleaved/final/' in str(p)]
    assert len(p)==1,('Expected only full C08 final checkpoint',len(p))
    parent=p[0].parent
    parenthash=hashlib.sha256(p[0].read_bytes()).hexdigest()
    pool,old,D,E,H=c09_dataset(seed)
    parentstate=load(parent,threads=2)
    assert parentstate.core.brain.n==8192 and parentstate.core.nkc==4064
    assert parentstate.cues.shape==(2224,32),parentstate.cues.shape
    theta0=parentstate.core.trainer.params['theta'].copy()
    initial=checkpoint_with_cues(parentstate,out/'checkpoint_C08_parent_copy')
    probes=old_probe_ids(416,128,seed)
    oldbaseline={int(i):int(pred) for i,pred,v in scores(initial,probes,seed)}
    pseudo416={int(i):int(pred) for i,pred,v in scores(initial,list(range(416)),seed)}
    assert len(oldbaseline)==128 and len(pseudo416)==416
    labels=np.array([x['target'] for x in pool],dtype=np.int32)
    scrambled=labels.copy()
    rng=np.random.default_rng(np.random.SeedSequence([seed,909,89]))
    for area in C08_AREAS:
        indices=[i for i,x in enumerate(pool) if x['area']==area]
        scrambled[indices]=rng.permutation(scrambled[indices])
    checkpoints=[]
    results={}
    for mode in C09_MODES:
        state=load(initial,threads=2)
        state.mode='frozen' if mode=='frozen' else 'live'
        state.core.stable_mode=state.mode
        state.memory_settings['batch_size']=1
        margin=float(state.memory_settings['margin']);damping=float(state.memory_settings['damping'])
        counts=dict(feedback=0,changed_events=0,changed_synapses_cumulative=0,native_skipped=0)
        history=[]
        initm=c08_metric(initial,old,D,E,H,seed,oldbaseline,'FINAL')
        history.append(dict(stage='INITIAL',epoch=0,**initm))
        total=retrain_epochs+reinforce_epochs
        for ep in range(1,total+1):
            stage='RETRAIN_ALL' if ep<=retrain_epochs else 'REINFORCE'
            order=c09_order(pool,seed,ep,1 if ep<=retrain_epochs else 2,mode)
            if smoke:order=order[:60]
            for pos,i in enumerate(order):
                y=int(scrambled[i] if mode=='shuffled' else labels[i])
                cue1,cue2=pool[i]['cue']
                kind='partial' if (ep+pos)%5==0 else 'clean'
                native_step(state,int(cue1),y,seed,ep,kind,mode,counts)
                native_step(state,int(cue2),y,seed,ep,kind,mode,counts)
            if stage=='REINFORCE':
                repeats=c09_replay(pool,seed,ep,mode,limit=24 if smoke else None)
                for pos,i in enumerate(repeats):
                    y=int(scrambled[i] if mode=='shuffled' else labels[i])
                    q1,q2=pool[i]['cue']
                    native_step(state,int(q1),y,seed,ep,'clean',mode,counts)
                    native_step(state,int(q2),y,seed,ep,'clean',mode,counts)
                if ep==retrain_epochs+1:
                    # One review of 416 historical pseudo-labels. Do not count these
                    # as verified historical correctness; preserve C08 baseline predictions.
                    for k,q in enumerate(range(32 if smoke else 416)):
                        y=-pseudo416[q] if mode=='shuffled' else pseudo416[q]
                        native_step(state,int(q),int(y),seed,ep,'clean',mode,counts)
            assert state.pending is None and state.core.pending is None
            assert state.memory_settings['margin']==margin and state.memory_settings['damping']==damping
            if ep==retrain_epochs or ep==total or (not smoke and ep in (4,10)):
                ck=checkpoint_with_cues(state,out/'checkpoints'/mode/f'epoch{ep:02d}')
                m=c08_metric(ck,old,D,E,H,seed,oldbaseline,'FINAL')
                cm=c09_cohort_scores(ck,old,seed)
                item=dict(stage=stage,epoch=ep,metrics=m,cohorts=cm,counts=counts.copy())
                history.append(item)
                print('C09',mode,stage,'ep',ep,'ABC',round(m['A_B_C_oldtrain'],4),'D',round(m['new_D'],4),'E',round(m['new_E_one_shot'],4),'DEV',round(m['prior_DEV'],4),'H',round(m['fresh_H_never_trained'],4),'old_stable',round(m['original_128_prediction_agreement'],4),flush=True)
        assert history[-1]['epoch']==total
        results[mode]={'baseline':history[0],'history':history[1:],'final':history[-1],
                       'distinct_theta_changed':int(np.count_nonzero(state.core.trainer.params['theta']!=theta0))}
    report=dict(schema='C09-NATIVE-DEV-1',version=C09_VERSION,status='DEV_COMPLETE',seed=seed,smoke=bool(smoke),
                retrain_epochs=retrain_epochs,reinforce_epochs=reinforce_epochs,
                original_A_B_C=248,D=248,E=248,known_true_labelled_items=744,
                pseudo_labelled_historical_items=416,independent_H=64,
                no_plasticity_adapter=True,only_native_synaptic_feedback=True,
                source_c08_sha256=C08_SOURCE_HASH,parent_system_sha256=parenthash,
                parent_checkpoint_readonly=True,physical_neurons=8192,
                revised_old_experience_accuracy_is_not_historical_truth=True,
                heldout_H_exploratory_previously_inspected=True,promoted=False,
                runtime_seconds=time.time()-started,results=results)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('C09_DONE',seed,'smoke',smoke,'seconds',round(report['runtime_seconds'],1),flush=True)
    return report

c09_tests()
SMOKE_MODE=True
if SMOKE_MODE:
    REPORT=c09_run(seed=7403,retrain_epochs=1,reinforce_epochs=1,smoke=True,output='/kaggle/working/MaleCNS_C09_Smoke_7403')
else:
    REPORT=c09_run(seed=7403,retrain_epochs=8,reinforce_epochs=4,smoke=False,output='/kaggle/working/MaleCNS_C09_Long_7403')
assert REPORT['status']=='DEV_COMPLETE'
