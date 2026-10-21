"""MaleCNS-Cognitive C0.8 native long: SHA-pinned source, integration gate and three-stage high-load training. NO adapter."""
from pathlib import Path
import hashlib
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate()
d=Path('/kaggle/working/c08_verified_runner');d.mkdir(parents=True,exist_ok=True)
api.kernels_pull('esthefano/malecns-cognitive-c08-threephases-smoke',path=str(d),metadata=False)
sources=list(d.glob('*.py'));assert len(sources)==1,sources
source=sources[0].read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='562675750ee8087b1e83795bb3e858c149fd8ee407b8d956f9d428168067366c', 'Refuse any changed C0.8 smoke source'
def patch(needle,replacement):
 global source
 assert source.count(needle)==1, ('Expected unique code section',needle[:100],source.count(needle))
 source=source.replace(needle,replacement)
patch("old_answers={int(q):int(y) for q,y,v in scores(initial,history_ids,seed)}",
"""legacy_all={int(q):int(y) for q,y,v in scores(initial,list(range(416)),seed)}
    assert len(legacy_all)==416
    old_answers={q:legacy_all[q] for q in history_ids}""")
patch("native_step(system,q,old_answers[q] if q in old_answers else 0,seed,ep,'clean',mode,meters) if q in old_answers else None",
      "native_step(system,q,legacy_all[q],seed,ep,'clean',mode,meters)")
patch("phase2_historical_pseudolabel_probes=len(old_answers)",
      "phase2_historical_pseudolabel_probes=len(legacy_all)")
patch("queries['A_B_C_oldtrain']=[(416+2*i+j,old[i]['target']) for i in original_train for j in (0,1)]",
"""queries['A_B_C_oldtrain']=[(416+2*i+j,old[i]['target']) for i in original_train for j in (0,1)]
    for cohort in ('A','B','C'):
        group=[i for i in original_train if old[i]['cohort']==cohort]
        assert group
        queries['old_train_'+cohort]=[(416+2*i+j,old[i]['target']) for i in group for j in (0,1)]""")
marker="SMOKE_FLAG = True"
assert source.count(marker)==1
# Reuse exact source; the first training is a shorter smoke with all three phases.
source=source.replace(marker,"SMOKE_FLAG = True")
start="if SMOKE_FLAG:\n    report=c08_run(seed=7403,phase1=1,phase2=1,phase3_items=16,smoke=True,output='/kaggle/working/MaleCNS_C08_Native_Smoke_7403')\nelse:\n    report=c08_run(seed=7403,phase1=10,phase2=6,phase3_items=248,smoke=False,output='/kaggle/working/MaleCNS_C08_Native_Long_7403')\nassert report['status']=='DEV_COMPLETE'"
assert source.count(start)==1
replace="""report_smoke=c08_run(seed=7403,phase1=1,phase2=1,phase3_items=16,smoke=True,output='/kaggle/working/MaleCNS_C08_Native_Embedded_Smoke_7403')
assert report_smoke['status']=='DEV_COMPLETE'
assert report_smoke['no_plasticity_adapter'] is True
assert report_smoke['phase2_historical_pseudolabel_probes']==416
assert all(v['final']['novel_unique']==16 for v in report_smoke['results'].values())
print('C08 SMOKE GATE PASSED; BEGIN FULL 10+6+248',flush=True)
report=c08_run(seed=7403,phase1=10,phase2=6,phase3_items=248,smoke=False,output='/kaggle/working/MaleCNS_C08_Native_Long_7403')
assert report['status']=='DEV_COMPLETE'
assert report['phase2_historical_pseudolabel_probes']==416
assert all(v['final']['novel_unique']==248 for v in report['results'].values())
print('C08 LONG THREE PHASES DONE',round(report['time_seconds'],1),flush=True)"""
source=source.replace(start,replace)
assert source.count('SMOKE_FLAG = True')==1
assert source.count('native_step(system,q,legacy_all[q]')==1
assert source.count("model.feedback(")==0, 'Only native_step defined in pinned C07 source does feedback'
print('C08 FULL SOURCE SHA256',hashlib.sha256(source.encode()).hexdigest(),flush=True)
exec(compile(source,'<MaleCNS-Cognitive-C08-no-adapter-long>','exec'),{'__name__':'__main__'})
