# MaleCNS-Cognitive C0 — independent V62 development

Independent cognitive pilot; MaleCNS main branch and all parent checkpoints stay unchanged.

Source: original MaleCNS_V62(1).zip + inherited MaleCNS core verified against V62 protocol hashes.

Parent candidate: V62 DEV seed 20292111 balanced epoch 20:
gs://scientificbrain-compute-artifacts-260133939682/malecns/v62-20261019/dev/20292111_balanced/recovery_epoch20.zip

C0 goal: learn structured symbolic comparisons, small mathematical equalities and transitivity using real 8192-neuron MaleCNS circuitry with 32-KC input and binary readout. Pilot includes 104 examples (72 TRAIN, 32 DEV test) and LIVE / frozen / shuffled controls.

V62 protected checkpoint is NOT drop-in compatible with the new C0 cue IDs: its regulator accepts only original 0..415 groups; evaluate protected separately, without bypassing safeguards.

V62 results are DEV only, not validated final. This branch must not automatically merge into main or read held-out final evaluation. No checkpoint, private weights, Kaggle token or other secrets are committed.

Implementation bundle (not yet committed here): MaleCNS_Cognitive_C0_V62_preparado.zip in the originating ChatGPT task. Import notebook inside the bundle into Kaggle with a PRIVATE checkpoint dataset. Before any training, verify checkpoint provenance and hash; record independent metric results. 
