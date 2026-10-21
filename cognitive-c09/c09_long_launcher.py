"""MaleCNS Cognitive C0.9 long run. Pull exact owned Kaggle smoke source by pinned SHA; run smoke gate then 8+4 epochs."""
from pathlib import Path
import hashlib
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate()
d=Path('/kaggle/working/c09_original_verified');d.mkdir(parents=True,exist_ok=True)
api.kernels_pull('esthefano/malecns-cognitive-c09-native-smoke',path=str(d),metadata=False)
files=list(d.glob('*.py'));assert len(files)==1
source=files[0].read_text()
expected='255a9785b7dc04d15d5287b68a0cdd240c22da7ddb761e2acb295517a12ad2fe'
assert hashlib.sha256(source.encode()).hexdigest()==expected,'C09 source changed since validation'
old="""c09_tests()
SMOKE_MODE=True
if SMOKE_MODE:
    REPORT=c09_run(seed=7403,retrain_epochs=1,reinforce_epochs=1,smoke=True,output='/kaggle/working/MaleCNS_C09_Smoke_7403')
else:
    REPORT=c09_run(seed=7403,retrain_epochs=8,reinforce_epochs=4,smoke=False,output='/kaggle/working/MaleCNS_C09_Long_7403')
assert REPORT['status']=='DEV_COMPLETE'
"""
assert source.count(old)==1,'Missing expected smoke control'
new="""c09_tests()
gate=c09_run(seed=7403,retrain_epochs=1,reinforce_epochs=1,smoke=True,output='/kaggle/working/MaleCNS_C09_Gated_Smoke_7403')
assert gate['status']=='DEV_COMPLETE' and gate['no_plasticity_adapter'] is True
assert all(int(v['final']['epoch'])==2 for v in gate['results'].values())
print('C09_SMOKE_PASSED: starting full 8+4 phases, all 744 experiences, no adapter',flush=True)
REPORT=c09_run(seed=7403,retrain_epochs=8,reinforce_epochs=4,smoke=False,output='/kaggle/working/MaleCNS_C09_Long_7403')
assert REPORT['status']=='DEV_COMPLETE' and REPORT['known_true_labelled_items']==744
assert all(int(v['final']['epoch'])==12 for v in REPORT['results'].values())
print('C09_TRAIN_REINFORCE_COMPLETE',round(REPORT['runtime_seconds'],1),flush=True)
"""
source=source.replace(old,new)
print('C09_PINNED_FULL_SOURCE_SHA256',hashlib.sha256(source.encode()).hexdigest(),flush=True)
exec(compile(source,'<Cognitive-C09-native-full-retraining>','exec'),{'__name__':'__main__'})
