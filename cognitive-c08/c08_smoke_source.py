"""Cognitive C0.8 Kaggle bootstrap: verified original C07 source, C07 native ep24 output."""
from pathlib import Path
import hashlib
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate()
d=Path('/kaggle/working/c08_pinned_c07');d.mkdir(parents=True,exist_ok=True)
api.kernels_pull('esthefano/malecns-cognitive-c07-native-smoke',path=str(d),metadata=False)
fs=list(d.glob('*.py'));assert len(fs)==1
old_script=fs[0].read_text()
assert hashlib.sha256(old_script.encode()).hexdigest()=='f4c3bb0237bb5b3aaecc28402de4b9c946c0178b86b613ed2ab78c166c2c9062'
cut="result=run_native(seed=7403,epochs=3,"
assert old_script.count(cut)==1
old_script=old_script.split(cut,1)[0]
exec(compile(old_script,'<C07-PINNED-PYTHON-DEFINITIONS-ONLY>','exec'),globals())


# MaleCNS-Cognitive C0.8 — staged rehearsal then novel one-shot, no plasticity adapter.
# Runtime prerequisites: own C07 native source loaded; C07 pretrained checkpoint read-only.
from pathlib import Path
import numpy as np, itertools, json, hashlib, time
from circuit_budget import load

C08_VERSION='MaleCNS-Cognitive-C0.8-NativeThreePhases-DEV/0.1'
C08_MODES=('native_interleaved','native_uniform','frozen','shuffled')
C08_BASE_CUES=1104
C08_PREVIOUS_CUES=416
C08_AREAS=('symbols','math','logic')

def c08_rows(seed, phase):
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),808,{'D':11,'E':13,'H':17}[phase]]))
    if phase=='D':
        chars='efghijklmn';a_low=20;nsym=10;nmath=52;nlogic=62
    elif phase=='E':
        chars='opqrstuvwx';a_low=40;nsym=10;nmath=52;nlogic=62
    elif phase=='H':
        chars='yz01234567';a_low=60;nsym=8;nmath=12;nlogic=12
    else: raise ValueError('Unexpected phase')
    result=[]
    for s in chars[:nsym]:
        result.append(dict(area='symbols',op='equal',args=[s,s],target=1,cohort=phase))
    neg=list(itertools.permutations(chars,2));rng.shuffle(neg)
    for a,b in neg[:nsym]:
        result.append(dict(area='symbols',op='equal',args=[a,b],target=-1,cohort=phase))
    ab=list(itertools.product(range(a_low,a_low+16),range(20)))
    rng.shuffle(ab)
    for a,b in ab[:nmath]:
        result.append(dict(area='math',op='add',args=[a,b,a+b],target=1,cohort=phase))
        result.append(dict(area='math',op='add',args=[a,b,a+b+int(rng.integers(1,5))],target=-1,cohort=phase))
    combos=list(itertools.permutations(chars,3));rng.shuffle(combos)
    for a,b,c in combos[:nlogic]:
        result.append(dict(area='logic',op='transitivity',args=[a,b,c,a,c],target=1,cohort=phase))
        result.append(dict(area='logic',op='transitivity',args=[a,b,c,c,a],target=-1,cohort=phase))
    expected=64 if phase=='H' else 248
    assert len(result)==expected,(phase,len(result))
    for area in C08_AREAS:
        yes=sum(1 for r in result if r['area']==area and r['target']==1)
        no=sum(1 for r in result if r['area']==area and r['target']==-1)
        assert yes==no and yes>0,(phase,area,yes,no)
    return result

def c08_material(seed):
    old=make_corpus(seed)+fresh_transfer()
    assert len(old)==344
    D=c08_rows(seed,'D');E=c08_rows(seed,'E');H=c08_rows(seed,'H')
    union=old+D+E+H
    signature=lambda r:(r['area'],r['op'],tuple(r['args']))
    assert len(union)==904
    assert len({signature(r) for r in union})==len(union),'Duplicate across old/D/E/heldout'
    assert sum(r['split']=='train' for r in old)==248
    assert [len(D),len(E),len(H)]==[248,248,64]
    return old,D,E,H

def c08_training_pool(old,D):
    seen=[dict(r,cohort=r.get('cohort','A')) for r in old if r['split']=='train']+D
    assert len(seen)==496
    return seen

def c08_order(pool,seed,phase,ep,mode):
    ids=list(range(len(pool)))
    rng=np.random.default_rng(np.random.SeedSequence([seed,808,phase,ep]))
    if mode=='native_uniform':
        return list(map(int,rng.permutation(ids)))
    groups={}
    for cohort in ('A','B','C','D'):
        for area in C08_AREAS:
            for sign in (1,-1):
                indices=[i for i in ids if pool[i]['cohort']==cohort and pool[i]['area']==area and pool[i]['target']==sign]
                groups[(cohort,area,sign)]=list(map(int,rng.permutation(indices)))
    output=[]
    while any(groups.values()):
        for area in C08_AREAS:
            for sign in (1,-1):
                for cohort in ('A','B','C','D'):
                    g=groups[(cohort,area,sign)]
                    if g:output.append(g.pop())
    assert len(output)==len(ids) and set(output)==set(ids)
    return output

def c08_metric(checkpoint,old,D,E,H,seed,old_baseline,phase):
    # Inference-only. The labels DEV and H are never used for feedback or replay.
    queries={}
    original_train=[i for i,r in enumerate(old) if r['split']=='train']
    prior_dev=[i for i,r in enumerate(old) if r['split']=='test']
    prior_stress=[i for i,r in enumerate(old) if r['split']=='stress']
    old_transfer=[i for i,r in enumerate(old) if r['split']=='transfer']
    assert len(original_train)==248 and [len(prior_dev),len(prior_stress),len(old_transfer)]==[32,32,32]
    queries['A_B_C_oldtrain']=[(416+2*i+j,old[i]['target']) for i in original_train for j in (0,1)]
    queries['prior_DEV']=[(416+2*i+j,old[i]['target']) for i in prior_dev for j in (0,1)]
    queries['prior_stress']=[(416+2*i+j,old[i]['target']) for i in prior_stress for j in (0,1)]
    queries['prior_transfer']=[(416+2*i+j,old[i]['target']) for i in old_transfer for j in (0,1)]
    queries['new_D']=[(C08_BASE_CUES+2*i+j,r['target']) for i,r in enumerate(D) for j in (0,1)]
    if phase=='FINAL':
        queries['new_E_one_shot']=[(C08_BASE_CUES+2*(248+i),r['target']) for i,r in enumerate(E)]
        queries['new_E_second_view_untrained']=[(C08_BASE_CUES+2*(248+i)+1,r['target']) for i,r in enumerate(E)]
        queries['fresh_H_never_trained']=[(C08_BASE_CUES+2*(496+i)+j,r['target']) for i,r in enumerate(H) for j in (0,1)]
    allids=sorted(set(q for data in queries.values() for q,y in data))
    predictions=scores(checkpoint,allids,seed)
    raw={int(i):float(v) for i,ans,v in predictions}
    def a(v):return 1 if v>=0 else -1
    result={}
    for group,obs in queries.items():
        byquestion={}
        for q,y in obs:
            # Pair first + second views only within a single logical experience.
            question=q//2 if q>=416 else q
            byquestion.setdefault(question,[]).append((raw[q],int(y)))
        result[group]=float(np.mean([a(np.mean([v for v,y in item]))==item[0][1] for item in byquestion.values()]))
    old_ids=list(old_baseline)
    hist=scores(checkpoint,old_ids,seed)
    result['original_128_prediction_agreement']=float(np.mean([int(ans)==old_baseline[int(q)] for q,ans,v in hist]))
    return result

def c08_run(seed=7403,phase1=10,phase2=6,phase3_items=248,smoke=False,output='/kaggle/working/MaleCNS_C08_Native'):
    from pathlib import Path
    import shutil
    stamp=time.time();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    candidates=[p for p in Path('/kaggle/input').rglob('system.json') if 'native_balanced' in str(p) and 'ep24' in str(p) and 'malecns-cognitive-c07-native-24epoch' in str(p)]
    assert len(candidates)==1,('Expected exactly one Cognitive C07 native checkpoint',len(candidates))
    trained_ck=candidates[0].parent
    source_hash=hashlib.sha256(candidates[0].read_bytes()).hexdigest()
    s=load(trained_ck,threads=2)
    assert s.core.brain.n==8192 and s.core.nkc==4064 and s.cues.shape==(1104,32)
    theta0=s.core.trainer.params['theta'].copy()
    old,D,E,H=c08_material(seed)
    cues=encode_corpus(D+E+H,s.core.nkc)
    assert cues.shape==(1120,32)
    prior={tuple(sorted(map(int,cue))) for cue in s.cues}
    assert len(prior)==len(s.cues)
    for code in cues:
        sig=tuple(sorted(map(int,code)))
        assert sig not in prior,'New cue collides with C0.7 existing cue'
        prior.add(sig)
    s.cues=np.vstack([s.cues,cues])
    initial=checkpoint_with_cues(s,out/'initial_checkpoint')
    history_ids=old_probe_ids(416,128,seed)
    old_answers={int(q):int(y) for q,y,v in scores(initial,history_ids,seed)}
    labels=[r['target'] for r in c08_training_pool(old,D)]
    pool=c08_training_pool(old,D)
    # Native feedback uses true TRAIN labels except shuffled.
    rng=np.random.default_rng(np.random.SeedSequence([seed,808,72]))
    scrambled=np.array(labels,dtype=np.int32)
    for area in C08_AREAS:
        idx=[i for i,r in enumerate(pool) if r['area']==area]
        scrambled[idx]=rng.permutation(scrambled[idx])
    # The original 248 Cognitive TRAIN rows are scattered among 344 old rows.
    original_ids=[i for i,r in enumerate(old) if r['split']=='train']
    assert len(original_ids)==248
    ids=[(416+2*original_ids[i],416+2*original_ids[i]+1) for i in range(248)]
    ids += [(1104+2*i,1104+2*i+1) for i in range(248)]
    assert len(ids)==496
    # Statically predetermined order; no feedback-driven schedule or adaptive params.
    outm={}
    for mode in C08_MODES:
        system=load(initial,threads=2)
        system.mode='frozen' if mode=='frozen' else 'live'
        system.core.stable_mode=system.mode
        system.memory_settings['batch_size']=1
        base_margin=float(system.memory_settings['margin'])
        base_damping=float(system.memory_settings['damping'])
        meters=dict(feedback=0,changed_events=0,changed_synapses_cumulative=0,native_skipped=0,novel_unique=0)
        trace=[]
        for ep in range(1,phase1+phase2+1):
            part='DOUBLE' if ep<=phase1 else 'RETRAIN_ALL'
            within=ep if ep<=phase1 else ep-phase1
            order=c08_order(pool,seed,1 if part=='DOUBLE' else 2,within,mode)
            assert len(order)==496
            for position,i in enumerate(order):
                y=int(scrambled[i]) if mode=='shuffled' else int(labels[i])
                # Distinct native views of the same experience, no external adaptive credit.
                masked='partial' if (ep+position)%5==0 else 'clean'
                for q in ids[i]:
                    native_step(system,int(q),y,seed,ep,masked,mode,meters)
            if part=='RETRAIN_ALL' and within==1:
                # One complete historical pass. The 416 original cue labels are self-distilled
                # from C07 and NEVER reported as verified ground truth.
                for k,q in enumerate(range(416)):
                    native_step(system,q,old_answers[q] if q in old_answers else 0,seed,ep,'clean',mode,meters) if q in old_answers else None
            assert system.pending is None and system.core.pending is None
            assert system.memory_settings['margin']==base_margin and system.memory_settings['damping']==base_damping
            if ep==phase1 or ep==phase1+phase2 or (not smoke and ep in (5,10,13)):
                ck=checkpoint_with_cues(system,out/'checkpoints'/mode/f'epoch{ep:02d}')
                met=c08_metric(ck,old,D,E,H,seed,old_answers,'PRE')
                trace.append(dict(phase=part,epoch=ep,**met,**meters.copy()))
                print('C08',seed,mode,part,ep,'old',round(met['A_B_C_oldtrain'],4),'D',round(met['new_D'],4),'DEV',round(met['prior_DEV'],4),'old_stability',round(met['original_128_prediction_agreement'],4),flush=True)
                assert met['original_128_prediction_agreement']>=.90,'Retention guard <90%'
        # Third phase: 248 NEW unique experiences, exactly one cue and one feedback
        # each; no previous data, no legacy rehearsal, no replay, no second view.
        novel=list(range(248))
        nrng=np.random.default_rng(np.random.SeedSequence([seed,808,3]))
        nrng.shuffle(novel)
        trained=novel[:phase3_items]
        assert len(trained)==len(set(trained))
        for pos,i in enumerate(trained):
            val=E[i]['target']
            if mode=='shuffled':val=-val
            native_step(system,1104+2*(248+i),int(val),seed,phase1+phase2+1,'clean',mode,meters)
            meters['novel_unique']+=1
        assert system.pending is None and system.core.pending is None
        assert system.memory_settings['margin']==base_margin and system.memory_settings['damping']==base_damping
        final=checkpoint_with_cues(system,out/'checkpoints'/mode/'final')
        met=c08_metric(final,old,D,E,H,seed,old_answers,'FINAL')
        trace.append(dict(phase='FINAL_NEW_ONLY',trained_unique=phase3_items,**met,**meters.copy()))
        print('C08',seed,mode,'FINAL','old',round(met['A_B_C_oldtrain'],4),'D',round(met['new_D'],4),'E',round(met['new_E_one_shot'],4),'E_untrained_secondview',round(met['new_E_second_view_untrained'],4),'H_heldout',round(met['fresh_H_never_trained'],4),'DEV',round(met['prior_DEV'],4),'old_stability',round(met['original_128_prediction_agreement'],4),flush=True)
        outm[mode]=dict(final=trace[-1],trace=trace,distinct_theta_changed=int(np.count_nonzero(system.core.trainer.params['theta']!=theta0)))
    result=dict(schema='C08-NATIVE-DEV-1',version=C08_VERSION,status='DEV_COMPLETE',seed=seed,smoke=bool(smoke),no_plasticity_adapter=True,only_native_synaptic_feedback=True,checkpoint_provenance='C07 native_balanced epoch24',parent_system_sha256=source_hash,parent_read_only=True,neurons=8192,original_train=248,phase1_novel_D=248,phase3_novel_E=248,independent_holdout_H=64,phase1_epochs=phase1,phase2_epochs=phase2,phase3_once=phase3_items,phase2_historical_pseudolabel_probes=len(old_answers),time_seconds=time.time()-stamp,results=outm,promoted=False)
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('C08 COMPLETE',seed,result['status'],'seconds',round(result['time_seconds'],1),flush=True)
    return result

def c08_tests():
    for seed in (7403,7411,7429):
        old,D,E,H=c08_material(seed)
        assert len(c08_training_pool(old,D))==496
        assert len(set((r['area'],r['op'],tuple(r['args'])) for r in old+D+E+H))==904
        for mode in C08_MODES:
            order=c08_order(c08_training_pool(old,D),seed,1,1,mode)
            assert len(order)==len(set(order))==496
        assert len([r for r in E if r['cohort']=='E'])==248
    print('C08 LOCAL INTEGRITY: PASS',flush=True)

c08_tests()
SMOKE_FLAG = True
if SMOKE_FLAG:
    report=c08_run(seed=7403,phase1=1,phase2=1,phase3_items=16,smoke=True,output='/kaggle/working/MaleCNS_C08_Native_Smoke_7403')
else:
    report=c08_run(seed=7403,phase1=10,phase2=6,phase3_items=248,smoke=False,output='/kaggle/working/MaleCNS_C08_Native_Long_7403')
assert report['status']=='DEV_COMPLETE'
