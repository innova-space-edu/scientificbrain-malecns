# MaleCNS-Cognitive C0.1 — focused rehearsal pilot

Status: local checks passed; Kaggle run initiated (not yet promoted). Remains isolated from MaleCNS Principal.

## Grounded predecessor C0
- Kaggle previous C0: LIVE train 51.39%, DEV test 37.50%; frozen DEV 46.88%; shuffled DEV 43.75%.
- Historical **prediction stability** on 128 probes = 100%; not independently verified old-task ground-truth accuracy.
- Existing baseline C0 failed 3 of 4 exploratory criteria; no evidence of cognitive transfer.

## C0.1 hypotheses
- Reduce non-semantic shared KC padding; make value and role features distinct while preserving 32 KC per input and 8192 real V62 neurons.
- Group-balanced curriculum TRAIN rehearsal interleaved by domain and +/- labels, two clean epochs followed by degraded cues.
- Historical replay stays at 32 per epoch, original budget/trust ceilings unchanged.
- Record whether feedback modifies synapses; frozen and shuffled remain independent controls, with no DEV feedback.
- No extra learner/LLM, no cached answers, no label-dependent encoding, no held-out final use.

## Artifacts
- Kaggle: https://www.kaggle.com/code/esthefano/malecns-cognitive-c01-v62
- User-provided C0.1 package SHA256: `431ec76ba2573a597739d06336c1ca94c6b77d98caa1fec76b135501619b60ff`
- Revised standalone runner SHA256: `c28b63ae56db821035b1b8c5e0e6616082b5eb9fba98d2003b59be0a42b908bc`
- An observed metric improvement needs LIVE outperforming both controls on new combinations while retaining ≥95% old predictions. Do not merge or promote based solely on local encoding overlap.

DEV-only. At least 3 independent seeds and reserved final evaluation are required for scientific validation. The V62 parent is immutable.
