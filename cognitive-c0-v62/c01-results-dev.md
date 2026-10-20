# MaleCNS-Cognitive C0.1 — observed DEV result (2026-10-20)

Completed one independent Kaggle CPU execution on **V62 balanced, seed 20292111, epoch 20**, 8192 neurons, 416 original associations, 4 Cognitive C0.1 epochs, seed 7403.

| Metric | Earlier C0 | C0.1 |
| --- | ---: | ---: |
| LIVE train | 51.39% | 59.72% |
| LIVE DEV test | 37.50% | 40.625% |
| Frozen DEV test | 46.875% | 46.875% |
| Shuffled DEV test | 43.75% | 46.875% |
| Inherited-prediction stability / 128 cues | 100% | 100% |

C0.1 LIVE test by domain: symbols 37.50%; math 33.33%; logic 50.00%.
C0.1 distinct modified existing synapses: LIVE **23,615**; shuffled **23,560**. Feedback operations with nonzero synapse change: LIVE **218 / 384**; shuffled **225 / 384**. Thus there is genuine weight plasticity, but no demonstrated beneficial generalization: shuffled and frozen both beat LIVE on unseen problems. 
Original checkpoint unchanged. No final held-out opened, no promotion. Only 1 experiment seed was measured.

Decision: **Do not merge to main or scale neurons / repetitions from this alone.** Investigate representation/relational learning and area interference; next work should isolate easier symbol operations, use new compositional splits, and verify multiple independent seeds before interpreting progress.

[Experimental run](https://www.kaggle.com/code/esthefano/malecns-cognitive-c01-v62).
