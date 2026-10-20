# MaleCNS Adaptive Neural Development — MAND V62 DEV

**Status:** V62 source inspection complete; two read-only Kaggle diagnostic jobs completed; metaplasticity and grouped homeostasis controller PROPOSAL tested locally (8 tests). NOT active in V62 LIVE training. NO main-branch, checkpoint or topology changes.

**Source:** MaleCNS_V62(1).zip, native V62 seed 20292111 balanced epoch20 checkpoint. Dataset private at esthefano/malecns-v62-cognitive-base. SHA256 checkpoint.npz: 35c6930133aa263ba298968393e40c124b376736cacb35a79b31c9179c1be367.

**Actual V62 architecture:** 8192 neurons; 1,612,858 edges; 241,529 plastic synapses; 416 learned cues; 1,619 circuit-budget groups. V62 already has an importance-based Jacobian regulator, local credit and physical/trust projections. Dynamics trainer uses grouped gain/bias/alpha. Synchronization creates W_fast from theta and resets W_slow to zero: do not directly add independent multiscale consolidation without revising this. Growth requires schema migration and identity checks. V62 protected regulator only supports historical 0..415 cues.

**Audit jobs:** https://www.kaggle.com/code/esthefano/malecns-mand-v62-diagnostico and https://www.kaggle.com/code/esthefano/malecns-mand-v62-metrics

Baseline: 10,224 activity observations; 4,026 zero-activity neurons, 7,434 below 1%, 101 above 90%; 70.467% theta approx zero, 2.957% theta near bounds; RMS displacement from anchor 0.116064; 47 local reference circuits above 95% budget. Activity alone does NOT establish dispensable neurons, and balanced mode can exceed nonbinding local reference caps.

**Controlled rollout:** (1) native V62 audit COMPLETE; (2) metaplasticity proposal with optional projector hook and 3-seed LIVE vs balanced/frozen/shuffled DEV comparisons PENDING; (3) homeostasis/functional specialist protection PENDING; (4) conservative replay + fast/slow round-trip redesign PENDING; (5) reversible rewiring and circuit_budget migration PENDING; (6) neuron expansion NOT AUTHORIZED until all gates pass. Held-out final tests sealed throughout. No automatic promotion or merging into main.

User-facing package: MaleCNS_MAND_V62_DEV_plan_y_prototipo.zip in ChatGPT, containing src/mand_v62.py and 8 tests; the branch is a documentation/protocol anchor pending experimental integration.
