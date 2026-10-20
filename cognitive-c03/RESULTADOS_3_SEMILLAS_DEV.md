# Cognitive C0.3 Adaptive — tres semillas DEV (2026-10-20)

CPU privado Kaggle, 2 épocas, 32→64 KC efectivos con dos vistas nativas de 32; 8.192 neuronas originales, ninguna añadida. Dataset V62 balanced montado como input privado, checkpoints de trabajo separados. Tres semillas evaluadas con 72 TRAIN y 32 DEV por semilla; stability antigua sobre 128 *predicciones* originales, sin verificar etiquetas históricas.

| Seed | Adaptive LIVE DEV | Fixed LIVE DEV | Frozen DEV | Shuffled DEV | Old prediction stability |
|---:|---:|---:|---:|---:|---:|
| 7403 | 56.25% | 53.125% | 43.75% | 43.75% | 100% |
| 7411 | 37.50% | 37.50% | 37.50% | 40.625% | 100% |
| 7429 | 56.25% | 50.00% | 56.25% | 46.875% | 100% |
| **Mean** | **50.00%** | **46.875%** | **45.833%** | **43.75%** | **100%** |

**Decision: NOT VALIDATED / DO NOT PROMOTE OR MERGE.** Mean adaptive advantage over Fixed is only +3.125 pp; seed 7411 ties Fixed and loses to Shuffled, and seed 7429 ties Frozen. 32 DEV questions/seed are insufficient for strong effect claims. Width expansion itself is controlled across modes; it does NOT demonstrate physical neuron growth. No tests with truly independent historical labels or sealed final heldout.

Real Kaggle runs:
- https://www.kaggle.com/code/esthefano/malecns-cognitive-c03-adaptive-pilot
- https://www.kaggle.com/code/esthefano/malecns-cognitive-c03-seed-7411
- https://www.kaggle.com/code/esthefano/malecns-cognitive-c03-seed-7429

Next Cognitive-only work: study binary readout margin and ensemble consistency, diagnose credit sign/direction, expand delayed-recall probes using ground-truth labels if available, make fresh sealed evaluation tasks and longer streams. Do not import MAND code wholesale. Native 8,192→N growth needs circuit_budget.npz migration, CSR graph and optimizer map, neuron identities, connection signs, constraints and exact restore/regression tests; currently blocked.

An offline Python C0.3 prototype (different, more general than the abbreviated Kaggle pilot) and tests are provided in ChatGPT ZIP MaleCNS_Cognitive_C03_Adaptive_DEV.zip. The PR contains documentation and results, not its whole source implementation.