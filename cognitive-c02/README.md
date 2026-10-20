# MaleCNS-Cognitive C0.2 — independent DEV

This branch contains only Cognitive experiments. No MAND or MaleCNS Principal code changes or training state are imported or merged.

## Baseline
C0.1 Kaggle seed 7403: LIVE TRAIN 59.72%, DEV TEST 40.625%, FROZEN DEV 46.875%, shuffled DEV 46.875%, historical prediction stability 100%. This is NOT validation.

## C0.2 hypothesis
Keep the same 8,192-neuron V62 balanced parent checkpoint (dataset esthefano/malecns-v62-cognitive-base, source seed 20292111 epoch20). Change only input representation and shuffled control: 32 distinct KC indices per cue derived from role/value, number bins/parity/thresholds/comparisons, symbolic identity and two logic premises. No target truth, addition output, example ID, or TEST label is encoded. Retain same curriculum and C0.1 rehearsal schedule.

Data: 416 original cues; 104 cognitive examples: 72 TRAIN / 32 DEV, 3 domains (symbols 24, math 40, logic 40); four epochs; 32 old rehearsal cues per epoch, 128 original historical prediction probes. Controls LIVE, frozen, shuffled. Shuffled is now permuted separately within each domain (same domain priors).

## Kaggle
Private notebook: https://www.kaggle.com/code/esthefano/malecns-cognitive-c02-v62
Standalone notebook and source in ChatGPT download package MaleCNS_Cognitive_C02_V62_preparado.zip. Local invariant tests: 3 seeds, 104 nonduplicated cues, no target/split leakage, 32 KCs/cue, 72/32 distribution. The branch is a protocol record, pending code review and results.

## Rules
- No merge with Principal/MAND; never alter the V62 parent checkpoint.
- No heldout final access; no automatic promotion.
- LIVE should outperform matched frozen and shuffled; historical prediction stability >=95% does not prove old-label accuracy.
- Follow up with independent seeds and separate statistical evaluation; never tune against heldout final.
