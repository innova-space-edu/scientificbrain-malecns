# MaleCNS BioEngine v1

## Goal

BioEngine v1 is a compatibility-first layer around the MaleCNS experiments.
It does not replace the validated sparse reservoir before reproducing it.

The first gate compares the historical ConnectomeReservoir with
BioEngine CompatReservoir on the same graph, sign vector, external inputs
and seeds.

## Baseline dynamics preserved

The archived notebook implements the recurrent update as:

    signed = state * sign
    recurrent = A.T.dot(signed) / max(in_strength, 1)
    proposal = tanh(gain * recurrent + external)
    state += leak * (proposal - state)

The v0.4-v0.8 experiments used leak=0.2, gain=1.2 and
sign_mode="biological_fast".

## New decomposition

BioEngine separates:

- fast_sign: simplified fast transmission used by compatibility experiments.
- neuromodulation: DA, 5-HT and OA channels.
- eligibility: temporal credit only for selected plastic edges.
- plastic update: rule-specific raw update.
- plastic budget: common RMS update magnitude for fair comparisons.
- fast/slow weights: online adaptation vs consolidated memory.

The experimental split is:

    WHERE -> plastic edge mask
    WHEN  -> modulation / reward / event
    HOW   -> Hebb, BCM, temporal, three-factor, ...

## Sparse plastic state

The MaleCNS graph has about 25.6M edges. Extra float32 state for every edge
costs roughly 100 MB per variable. Eligibility, fast weights and slow weights
are therefore allocated only for selected plastic edges.

## Brain-AI reuse boundary

Recovered Brain-AI modules contain useful higher-level ideas: dopamine/reward,
adjustable thresholds, episodic memory, consolidation, spreading activation,
homeostatic maintenance, pruning and LoRA/model adaptation.

They are not treated as a biological Drosophila implementation. BioEngine owns
neuronal and synaptic plasticity. Brain-AI remains a future cognitive layer
above the biological core.

## Colab usage

From a runtime that has the original MaleCNS project in Drive:

    !git clone -b bioengine-v1-colab https://github.com/innova-space-edu/scientificbrain-malecns.git
    %cd scientificbrain-malecns
    !pip install -q numpy scipy pandas

Run the short compatibility gate:

    !python experiments/p0_compatibility_gate.py --project /content/drive/MyDrive/malecns_ai_v0_1

Run the plasticity primitive smoke test:

    !python experiments/p1_plasticity_primitives_smoke.py

Only after P0 passes should the Plasticity Benchmark be connected to the
full MaleCNS simulation.
