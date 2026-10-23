"""MaleCNS-Cognitive C0.11 -- eight short training cycles versus eight continuous epochs.
NO adapter, no dynamic plasticity parameter updates, CPU, checkpoint preservation.
This module executes a small smoke run. The separate SHA-pinned launcher gates the long run.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi

C011_VERSION="MaleCNS-Cognitive-C0.11-Microcycles-DEV/0.1"
C011_C010_SOURCE_SHA="f7bfc6af90213dc37e417769def157d04398e089d3cf8fc68504702230e0f384"
C011_PARENT_HASH="35bf6a8fbdcec1e83b1dfacfd232028d30d43403a1e662b8aa92d7044463adb6"
api=KaggleApi()
api.authenticate()
source_dir=Path('/kaggle/working/c011_source_pin')
source_dir.mkdir(parents=True,exist_ok=True)
api.kernels_pull('esthefano/malecns-cognitive-c010-ablation-smoke',path=str(source_dir),metadata=False)
files=list(source_dir.glob('*.py'))
assert len(files)==1,files
c010_code=files[0].read_text()
assert hashlib.sha256(c010_code.encode()).hexdigest()==C011_C010_SOURCE_SHA
splitter='\nc010_tests()\nSMOKE=True'
assert c010_code.count(splitter)==1,'C0.10 pinned source format changed'
exec(compile(c010_code.split(splitter)[0],'<Cognitive-C010-source-definitions>','exec'),globals())
required=('c09_dataset','c09_order','native_step','checkpoint_with_cues','load','scores','c08_metric','c09_cohort_scores','encode_corpus')
assert all(v in globals() for v in required)
print('C011 CORE verified, C0.10 native only',flush=True)

def c011_parent():
    found=[p for p in Path('/kaggle/input').rglob('system.json')
       if 'malecns-cognitive-c0-10-corrected-full-factorial' in str(p)
       and '/MaleCNS_C010_Long_7403/checkpoints/no_extra_no_pseudo/ep12/system.json' in str(p)]
    assert len(found)==1,('C011 source C010 checkpoint ambiguity',len(found))
    got=hashlib.sha256(found[0].read_bytes()).hexdigest()
    global C011_PARENT_HASH
    if C011_PARENT_HASH=="CAPTURE_EXACT_SHA_AT_PREFLIGHT":
        C011_PARENT_HASH=got
        print("C011_READONLY_PARENT_SHA256",got,flush=True)
    assert got==C011_PARENT_HASH,('C011 parent changed',got)
    return found[0].parent

def c011_new_holdouts():
    """Two disjoint, balanced symbolic/math/logic sets, never fed back."""
    def rows(prefix,base,alphabet):
        assert len(alphabet)==8 and all(len(c)==1 for c in alphabet)
        out=[]
        for i in range(8):
            a=alphabet[i]
            b=alphabet[(i+1)%8]
            out.append(dict(area='symbols',op='equal',args=[a,a],target=1,cohort=prefix))
            out.append(dict(area='symbols',op='equal',args=[a,b],target=-1,cohort=prefix))
        for i in range(12):
            a=base+3*i;b=17+5*i
            out.append(dict(area='math',op='add',args=[a,b,a+b],target=1,cohort=prefix))
            out.append(dict(area='math',op='add',args=[a,b,a+b+2+(i%5)],target=-1,cohort=prefix))
        for i in range(12):
            a=alphabet[i%8];b=alphabet[(i+1+i//8)%8];c=alphabet[(i+3+2*(i//8))%8]
            assert len({a,b,c})==3
            out.append(dict(area='logic',op='transitivity',args=[a,b,c,a,c],target=1,cohort=prefix))
            out.append(dict(area='logic',op='transitivity',args=[a,b,c,c,a],target=-1,cohort=prefix))
        assert len(out)==64
        assert sum(r['target']==1 for r in out)==32
        return out
    F,G=rows('F_valid',1000,'αβγδεζηθ'),rows('G_final',2000,'ικλμνξοπ')
    signatures=[(r['area'],r['op'],tuple(r['args'])) for r in F+G]
    assert len(set(signatures))==128
    return F,G

def c011_tests():
    for seed in (7403,7411,7429):
        pool,old,D,E,H=c09_dataset(seed)
        assert len(pool)==744
        assert len(set(q for x in pool for q in x['cue']))==1488
        for cycle in range(1,9):
            a=c09_order(pool,seed,cycle,2,'native_interleaved')
            assert len(a)==len(set(a))==744
    F,G=c011_new_holdouts()
    assert len(F)==len(G)==64
    print('C011 static tests passed: schedule/holdout, three generation seeds',flush=True)

def c011_score_holdout(ck,cues_start,rows,seed):
    pairs=[(cues_start+2*i+j,int(r['target'])) for i,r in enumerate(rows) for j in (0,1)]
    vals={int(k):float(v) for k,y,v in scores(ck,[a for a,_ in pairs],seed)}
    assert len(vals)==2*len(rows)
    correct=[]
    by_area={}
    for i,r in enumerate(rows):
        q=cues_start+2*i
        pred=1 if (vals[q]+vals[q+1])>=0 else -1
        good=int(pred==r['target'])
        correct.append(good)
        by_area.setdefault(r['area'],[]).append(good)
    return {'accuracy':float(np.mean(correct)),
            'by_area':{k:float(np.mean(v)) for k,v in by_area.items()},
            'n':len(rows)}

def c011_eval(ck,pool,old,D,E,H,F,seed,reference,f_index):
    metrics=c08_metric(ck,old,D,E,H,seed,reference,'FINAL')
    cohorts=c09_cohort_scores(ck,old,seed)
    val=c011_score_holdout(ck,f_index,F,seed)
    training_summary={**cohorts,'D':metrics['new_D'],'E':metrics['new_E_one_shot']}
    mean=float(np.mean(list(training_summary.values())))
    # Metrics include previous DEV/H for reporting only, never for selection.
    return {'training_cohorts':training_summary,'training_macro_mean':mean,
            'validation_F':val,
            'historical_agreement':metrics['original_128_prediction_agreement'],
            'previous_DEV_exploratory':metrics['prior_DEV'],
            'previous_H_exploratory':metrics['fresh_H_never_trained']}

def c011_accept(candidate,incumbent,baseline):
    """Precommitted checkpoint selector; no leakage from G, prior DEV or H."""
    old=candidate['training_cohorts'];prev=incumbent['training_cohorts']
    base=baseline['training_cohorts']
    floor={'A':.03,'B':.03,'C':.03,'D':.04,'E':.04}
    if any(old[k] < base[k]-floor[k] for k in floor):return False,'baseline_cohort_floor'
    if candidate['validation_F']['accuracy'] < incumbent['validation_F']['accuracy']-.03125:
        return False,'validation_regression'
    if candidate['historical_agreement'] < .95:return False,'historical_regression'
    # Accept when macro train increases by at least ~0.1 percentage points and
    # validation is not notably worse. Do not use G or DEV/H for decisions.
    if candidate['training_macro_mean'] > incumbent['training_macro_mean']+.001:
        return True,'improved_training_macro_guarded'
    return False,'no_macro_improvement'

def c011_train_one(state,pool,seed,cycle,smoke,meters):
    state.mode='live';state.core.stable_mode='live'
    state.memory_settings['batch_size']=1
    order=c09_order(pool,seed,cycle,2,'native_interleaved')
    if smoke:order=order[:32]
    margin=float(state.memory_settings['margin']);damping=float(state.memory_settings['damping'])
    for pos,i in enumerate(order):
        x=pool[i];a,b=x['cue'];y=int(x['target'])
        masked='partial' if (cycle+pos)%5==0 else 'clean'
        native_step(state,int(a),y,seed,cycle,masked,'native_interleaved',meters)
        native_step(state,int(b),y,seed,cycle,masked,'native_interleaved',meters)
    assert state.pending is None and state.core.pending is None
    assert float(state.memory_settings['margin'])==margin
    assert float(state.memory_settings['damping'])==damping
    return len(order)

def c011_run(seed=7403,n_cycles=8,smoke=False,output='/kaggle/working/MaleCNS_C011_Long_7403'):
    t=time.time();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    assert seed==7403,'Only seed 7403 parent trained; other seeds static tests only'
    parent=c011_parent()
    s=load(parent,threads=2)
    assert s.core.brain.n==8192 and s.core.nkc==4064 and s.cues.shape==(2224,32)
    theta0=s.core.trainer.params['theta'].copy()
    pool,old,D,E,H=c09_dataset(seed)
    F,G=c011_new_holdouts()
    extra=encode_corpus(F+G,s.core.nkc)
    assert extra.shape==(256,32),extra.shape
    seen={tuple(sorted(map(int,c))) for c in s.cues}
    assert len(seen)==2224
    for c in extra:
        sig=tuple(sorted(map(int,c)))
        assert sig not in seen,'C011 validation/holdout cue collision; fail-closed'
        seen.add(sig)
    s.cues=np.vstack((s.cues,extra))
    assert s.cues.shape==(2480,32)
    # Exact same F/G interfaces for all arms and cycles, no feedback on indices >=2224.
    f_idx=2224;g_idx=2224+128
    prepared=checkpoint_with_cues(s,out/'immutable_parent_copy')
    probes=old_probe_ids(416,128,seed)
    history={int(q):int(ans) for q,ans,v in scores(prepared,probes,seed)}
    assert len(history)==128
    initial=c011_eval(prepared,pool,old,D,E,H,F,seed,history,f_idx)
    print('C011 START',json.dumps(initial),flush=True)
    results={}
    for arm in ('micro_select','micro_always','continuous'):
        meters={'feedback':0,'changed_events':0,'changed_synapses_cumulative':0,'native_skipped':0}
        checkpoints=[];records=[];accepted=0
        chosen_path=prepared
        if arm=='continuous':
            state=load(prepared,threads=2)
        for cycle in range(1,n_cycles+1):
            if arm!='continuous':
                state=load(chosen_path,threads=2)
            count=c011_train_one(state,pool,seed,cycle,smoke,meters)
            path=checkpoint_with_cues(state,out/'checkpoints'/arm/f'cycle{cycle:02d}')
            checkpoints.append(str(path))
            if arm=='continuous':
                # No interim validation or reload to keep a true continuous control.
                continue
            metric=c011_eval(path,pool,old,D,E,H,F,seed,history,f_idx)
            incumbent=initial if accepted==0 else records[next(i for i in range(len(records)-1,-1,-1) if records[i]['accepted'])]['metrics']
            if arm=='micro_always':
                accept=True;reason='always_continue'
            else:
                accept,reason=c011_accept(metric,incumbent,initial)
            if accept:
                chosen_path=path;accepted+=1
            records.append({'cycle':cycle,'examples':count,'accepted':accept,
                            'reason':reason,'metrics':metric,'checkpoint':str(path)})
            print('C011',arm,'cycle',cycle,'selected',accept,'macro',round(metric['training_macro_mean'],4),
                  'F',round(metric['validation_F']['accuracy'],4),'old',round(metric['historical_agreement'],4),flush=True)
            del state
        if arm=='continuous':
            final=checkpoint_with_cues(state,out/'checkpoints'/arm/'final')
            chosen_path=final
            final_metric=c011_eval(final,pool,old,D,E,H,F,seed,history,f_idx)
            records=[{'cycle':n_cycles,'continuous_final':True,'metrics':final_metric,'checkpoint':str(final)}]
            del state
        else:
            final_metric=c011_eval(chosen_path,pool,old,D,E,H,F,seed,history,f_idx)
        # G evaluated exactly once per ARM at final, never used to select paths.
        Gmetric=c011_score_holdout(chosen_path,g_idx,G,seed)
        finstate=load(chosen_path,threads=2)
        unique=int(np.count_nonzero(finstate.core.trainer.params['theta']!=theta0))
        results[arm]={'final':final_metric,'sealed_test_G':Gmetric,
                      'accepted_cycles':accepted,'history':records,
                      'theta_unique_changed':unique,'meters':meters,'selected_checkpoint':str(chosen_path)}
        print('C011 ARM COMPLETE',arm,'final',round(final_metric['training_macro_mean'],4),
              'F',round(final_metric['validation_F']['accuracy'],4),
              'G',round(Gmetric['accuracy'],4),'changed',unique,flush=True)
    frozenG=c011_score_holdout(prepared,g_idx,G,seed)
    report={'version':C011_VERSION,'status':'DEV_COMPLETE','seed':seed,'smoke':smoke,
      'cycles':n_cycles,'examples_per_cycle':32 if smoke else 744,
      'original_examples_total':744,'parent':'C0.10 no_extra_no_pseudo ep12',
      'parent_sha256':C011_PARENT_HASH,'parent_unchanged':True,
      'neurons':8192,'adapter':False,'only_native_feedback':True,
      'validation_F_64_never_training':True,'final_G_64_never_training_never_selected':True,
      'known_earlier_H_DEV_exploratory':True,
      'checkpoint_selector_uses_only_known_train_and_validation_F':True,
      'eight_independent_training_cycles_with_checkpoint_reload':True,
      'continuous_control_same_order_and_budget':True,
      'initial':initial,'baseline_G_evaluated_at_final':frozenG,
      'results':results,'elapsed_seconds':time.time()-t,
      'promoted':False,'merge_to_principal':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('C011 DEV COMPLETE',n_cycles,round(report['elapsed_seconds'],1),flush=True)
    return report

c011_tests()
SMOKE=True
if SMOKE:
    r=c011_run(n_cycles=1,smoke=True,output='/kaggle/working/MaleCNS_C011_Smoke_7403')
else:
    r=c011_run(n_cycles=8,smoke=False,output='/kaggle/working/MaleCNS_C011_Long_7403')
assert r['status']=='DEV_COMPLETE' and not r['adapter']
