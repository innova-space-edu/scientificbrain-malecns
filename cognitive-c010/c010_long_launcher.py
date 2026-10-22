"""Cognitive C0.10: run exact verified smoke source, require smoke PASS, then 4 full factorial epochs.
No external adapter and no modification of parent C09 or Principal.
"""
from pathlib import Path
import hashlib
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi()
api.authenticate()
d=Path("/kaggle/working/c010_verified_remote")
d.mkdir(parents=True,exist_ok=True)
api.kernels_pull("esthefano/malecns-cognitive-c010-ablation-smoke",path=str(d),metadata=False)
files=list(d.glob("*.py"))
assert len(files)==1,files
source=files[0].read_text()
expected="727a35c7c63d5f7904566a70125e4274c4b61dbb79f467ebe021e2f2432b870e"
assert hashlib.sha256(source.encode()).hexdigest()==expected,"C010 smoke source mismatch; fail closed"
needle="""c010_tests()
SMOKE=True
if SMOKE:
    R=c010_run(seed=7403,epochs=1,smoke=True,output='/kaggle/working/MaleCNS_C010_Smoke_7403')
else:
    R=c010_run(seed=7403,epochs=4,smoke=False,output='/kaggle/working/MaleCNS_C010_Long_7403')
assert R['status']=='DEV_COMPLETE' and R['no_adapter'] and len(R['results'])==6"""
assert source.count(needle)==1
replacement="""c010_tests()
gate=c010_run(seed=7403,epochs=1,smoke=True,output='/kaggle/working/MaleCNS_C010_Gated_Smoke_7403')
assert gate['status']=='DEV_COMPLETE' and gate['no_adapter'] and len(gate['results'])==6
assert all(v['final']['epoch']==9 for v in gate['results'].values())
print('C010 SMOKE GATE PASSED; BEGIN 2x2 FULL EXPERIMENT',flush=True)
R=c010_run(seed=7403,epochs=4,smoke=False,output='/kaggle/working/MaleCNS_C010_Long_7403')
assert R['status']=='DEV_COMPLETE' and R['no_adapter']
assert len(R['results'])==6 and all(v['final']['epoch']==12 for v in R['results'].values())
print('C010 FULL FACTORIAL FINISHED',round(R['seconds'],1),flush=True)"""
source=source.replace(needle,replacement)
print('C010_LONG_SHA256',hashlib.sha256(source.encode()).hexdigest(),flush=True)
exec(compile(source,'<Cognitive-C010-native-consolidation>','exec'),{'__name__':'__main__'})
