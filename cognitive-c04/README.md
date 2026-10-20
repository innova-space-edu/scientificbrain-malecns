# MaleCNS-Cognitive C0.4 (isolated DEV)

Cognitive-only. No MAND/Principal code, model checkpoint, or branches merged. Parent: frozen V62 balanced 8192-neuron model.

Complete C0.4 native implementation in ChatGPT ZIP: c04_credit.py, c04_experiment.py, test_c04.py, plus verified Kaggle launchers. Six epochs, seed list 7403/7411/7429, 32 -> 64 KC logical cue slots, four controls adaptive_live/fixed_live/frozen/shuffled. Per-domain bounded native margin/damping, 12 TRAIN replay/epoch balanced by area and class, 32 historical pseudo-labeled replay/epoch. Static tests passed all three seeds. Native weight changes only through original V62 projection.

Abbreviated Kaggle CPU pilot derived by SHA-pinned patch of C0.3: https://www.kaggle.com/code/esthefano/malecns-cognitive-c04-credit-pilot. This pilot is related but is not the exact full standalone C04 experiment.py runner. The ZIP source is not yet committed in this PR.

Scientific gate: 3 completed seeds, Adaptive DEV >=65%, >=5pp over each matched control and legacy prediction stability >=95%; note that legacy predictions are unverified, not ground truth. No DEV-driven tuning, no heldout final, no physical neuron growth, no main branch promotion.