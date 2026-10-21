# MaleCNS-Cognitive C0.7 — Native learning ONLY

**Research scope**: MaleCNS-Cognitive. MAND/Principal code and checkpoints must not be imported or merged. Native V62 balanced checkpoint is a read-only scientific dependency (8192 neurons).

**No plasticity adapter**: C0.7 removes C0.6's make_adapter, adaptive_params, EMA-derived margin/damping control, error-based replay routing and the external confidence feedback gate. It calls the native V62 feedback method for *every* labeled TRAIN experience with fixed native hyperparameters. External labels remain needed for supervised training; it is not unsupervised learning.

**Experiment**: 248 labeled TRAIN examples across A(72), B(80), C(96); 32 old DEV, 32 reused exploratory stress DEV, 32 fresh end-only transfer items. Encodes each example as two 32-KC native cues; model remains 8192 neurons / 4064 KC population. 24 continuous epochs. Checkpoints every 4 epochs plus phase transitions.

**Matched controls**: native_balanced (fixed replay A16/B32/C16 after epoch8), native_uniform (same number of repeats sampled uniformly), frozen, shuffled. All receive 64 repeated TRAIN examples per epoch, fixed 32 old pseudo-label rehearsal items/epoch and equal easy/hard exposure. Frozen receives the stimuli but no feedback. Pseudo-label old prediction stability != ground truth historical accuracy. New transfer items are never used as training feedback and only scored at epoch24.

**Local validation**: unit suite passed for seeds 7403, 7411, 7429: corpus partitions 248+32+32+32, 688 unique 32-KC cues, no target/split leaking into encoded input, stable schedules, B quota verified, adapter functions absent. No claim of learning efficacy yet.

**Real Kaggle**:
- https://www.kaggle.com/code/esthefano/malecns-cognitive-c07-native-smoke
- https://www.kaggle.com/code/esthefano/malecns-cognitive-c07-native-24epoch

The full standalone code is distributed as `MaleCNS_Cognitive_C07_Native_DEV.zip` plus `MaleCNS_Cognitive_C07_Native_Kaggle.ipynb` in ChatGPT. PR documentation only until full source is reviewed and committed.

**Do not merge/promote** before the 24-epoch evidence, multi-seed replication, native-vs-native comparison, independent transfer and verified historical memory. No GPU or physical neuron growth performed.