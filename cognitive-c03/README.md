# MaleCNS-Cognitive C0.3 Adaptive — isolated DEV protocol

**Isolation**: Cognitive only. Do not merge MAND/Principal experiments, checkpoints or implementation.

**Lineage**: C0.2 -> C0.3, with the same frozen V62 balanced 8,192-neuron checkpoint and 4,064 KC population. Only the Cognitive new-cue coding, TRAIN-only replay and plasticity-control policy change.

**Why not physically expand 8,192 neurons yet?** The native system requires 32 KCs per cue and a valid 8,192x8,192 CSR circuit. C0.3 tests an expandable multi-view encoding (32,64,96,128 logical cue slots, in multiples of 32). Physical 9,216+ neuron growth requires migration of the sparse graph, neuron IDs/types, checkpoint schema, output readout, edge groups, sign constraints, projection/trust budget and old-knowledge tests. Such structural growth is BLOCKED.

**Plasticity adapter**: a bounded, opt-in controller tunes native margin/damping by *TRAIN-only* error/update EMAs, and uses a TRAIN-only replay-risk buffer. All synaptic writes remain inside the original circuit_budget native projector. Historical labels are inherited self-distillation, so old prediction stability != truth accuracy.

**CPU Kaggle mini-pilot**: https://www.kaggle.com/code/esthefano/malecns-cognitive-c03-adaptive-pilot (two epochs; logical width 32 -> 64; seed 7403; modes adaptive_live, fixed_live, frozen, shuffled; original parent read-only). Compare all controls before claiming improvement. Independent test seeds and delayed memory assessments are required.

The complete Python prototype, tests and a separate experimental launcher were delivered in ChatGPT as `MaleCNS_Cognitive_C03_Adaptive_DEV.zip`. This branch currently contains protocol/documentation; the ZIP source is NOT committed here yet.

**Do not merge/promote** until code review, 3-seed native DEV, LIVE >=5pp over all controls, historical prediction stability >=95%, DEV >=65%, and sealed heldout untouched.
