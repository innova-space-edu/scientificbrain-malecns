# MaleCNS-Cognitive C0.2 — DEV evaluation (2026-10-20)

**Isolation:** Cognitive only. V62 balanced parent read-only. MaleCNS Principal and MAND unchanged. No heldout final, no promotion, no merge.

Same cognitive seed 7403; 104 samples (72 TRAIN / 32 DEV), four epochs, LIVE/frozen/shuffled; legacy historical prediction stability tests on 128 probes.

| Run | LIVE TRAIN | LIVE DEV | frozen DEV | shuffled DEV | old prediction stability |
|---|---:|---:|---:|---:|---:|
| C0.1 reference | 59.72% | 40.625% | 46.875% | 46.875% | 100% |
| C0.2 compositional encoder | 56.944% | 46.875% | 46.875% | 46.875% | 100% |
| C0.2-Hybrid (C0.1 symbols, C0.2 math/logic) | 55.556% | 50.0% | 50.0% | 53.125% | 100% |

Per-domain LIVE DEV: C0.1 symbols 37.5%, math 33.33%, logic 50%; C0.2 symbols 12.5%, math 50%, logic 66.67%; Hybrid symbols 37.5%, math 50%, logic 58.33%. Small denominators (8/12/12), one seed, repeated DEV exploration; none qualifies as validated cognition.

Physical C0.2 updates: LIVE 17,674 distinct synapses, 158,961 cumulative updates, 253/384 feedback calls with nonzero change. Shuffled 17,002 distinct, 164,259 cumulative, 264/384 effective calls. LIVE is no better than controls despite changes.

**Decision: do not promote.** Next Cognitive-specific investigations: label-conditioned native credit direction, small-margin readout and error normalization using TRAIN data only; protect original 416-cue predictions; comparison against frozen/shuffled at >=3 independent seeds before claiming gains. Do not copy MAND code without focused tests. No neuron expansion justified.

Sources: https://www.kaggle.com/code/esthefano/malecns-cognitive-c02-v62 ; https://www.kaggle.com/code/esthefano/malecns-cognitive-c02-hybrid-v62 ; https://www.kaggle.com/code/esthefano/malecns-c02-weights-audit
