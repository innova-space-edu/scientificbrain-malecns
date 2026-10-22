"""Run exact Cognitive C0.11 SHA-verified Kaggle source: smoke gate, then 8 microcycles/8 continuous.
Only Cognitive; no adapter. CPU-only and preserve parent checkpoints.
"""
from pathlib import Path
import hashlib
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate()
d=Path('/kaggle/working/c011_verified_source');d.mkdir(parents=True,exist_ok=True)
api.kernels_pull('esthefano/malecns-cognitive-c011-micro-smoke',path=str(d),metadata=False)
files=list(d.glob('*.py'));assert len(files)==1,files
source=files[0].read_text()
expected='047ec372ddb83b06ec683affa51da5f70649e2eee204d9c52d4d802991bdd418'
assert hashlib.sha256(source.encode()).hexdigest()==expected,'C011 source changed. Abort.'
old="""c011_tests()
SMOKE=True
if SMOKE:
    r=c011_run(n_cycles=1,smoke=True,output='/kaggle/working/MaleCNS_C011_Smoke_7403')
else:
    r=c011_run(n_cycles=8,smoke=False,output='/kaggle/working/MaleCNS_C011_Long_7403')
assert r['status']=='DEV_COMPLETE' and not r['adapter']"""
assert source.count(old)==1,"Expected C011 smoke tail missing"
new="""c011_tests()
gate=c011_run(n_cycles=1,smoke=True,output='/kaggle/working/MaleCNS_C011_Internal_Smoke_7403')
assert gate['status']=='DEV_COMPLETE' and not gate['adapter']
assert gate['validation_F_64_never_training'] and gate['final_G_64_never_training_never_selected']
assert all(int(v['history'][-1]['cycle'])==1 for v in gate['results'].values())
print('C011 SMOKE GATE PASSED; START EIGHT FULL MICROCYCLES',flush=True)
r=c011_run(n_cycles=8,smoke=False,output='/kaggle/working/MaleCNS_C011_Long_7403')
assert r['status']=='DEV_COMPLETE' and not r['adapter'] and r['cycles']==8
assert len(r['results'])==3
assert all(v['history'][-1]['cycle']==8 for v in r['results'].values())
print('C011 EIGHT MICRO-CYCLE EXPERIMENT COMPLETED',round(r['elapsed_seconds'],1),flush=True)"""
source=source.replace(old,new)
print('C011_LONG_EXECUTED_SHA',hashlib.sha256(source.encode()).hexdigest(),flush=True)
exec(compile(source,'<MaleCNS-Cognitive-C011-8-native-microcycles>','exec'),{'__name__':'__main__'})
