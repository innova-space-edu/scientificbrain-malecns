# MaleCNS-Cognitive C0.6 — LongStream DEV

**Exclusive Cognitive experiment.** No MaleCNS Principal/MAND checkpoints or training transferred; V62 balanced is a pinned read-only neural dependency.

**Proposal implemented as reproducible local Python files**: c06_runner.py, test_c06.py, protocol.json, README.md. Local tests pass across 7403, 7411 and 7429 for collision-free 64-KC two-view input, 248 TRAIN (72+176), 32 prior DEV and 32 fresh stress DEV reserved from feedback. Cohort A foundation, B expansion, C challenge. 24 continuous epochs; clean/easy reminders interleaved with hard partial cues; 4 matched controls Adaptive/Fixed/Frozen/Shuffled. Checkpoints/retention every 4 epochs; monitor TRAIN truth by cohort, original DEV, unseen stress DEV and baseline prediction stability. No original checkpoint writes, physical growth, GPU requirement, or promotion.

**Execution status 2026-10-21**: automated 3-epoch Kaggle smoke https://www.kaggle.com/code/esthefano/malecns-cognitive-c06-smoke-dev did NOT run training: it failed during decompression of transmitted source. No 24-epoch job has been launched yet. Full transparent offline Kaggle notebook generated in ChatGPT: MaleCNS_Cognitive_C06_LongStream_Kaggle.ipynb (standalone, embeds Cognitive V62 source; attach private dataset esthefano/malecns-v62-cognitive-base), and accompanying MaleCNS_Cognitive_C06_LongStream_DEV.zip. It runs 3-epoch integration then 24-epoch long run only if smoke returns COMPLETE.

**Evidence rule**: no claims of efficacy until matched training completes, DEV and true labeled TRAIN delayed recall are analyzed, adapter differentiates causally from Fixed, and historical stability is measured. Old 416 cues have pseudo-labels derived from original predictions; this is stability, not measured correctness.

**Operational guard**: The Kaggle notebook, not GitHub PR, currently contains the full C0.6 executable code. Do not merge this PR until its code is audited and complete multiseed validations exist.
