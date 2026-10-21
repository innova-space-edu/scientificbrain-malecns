# MaleCNS-Cognitive C0.8 — entrenamiento en tres fases, sin adaptador

Exclusivamente Cognitive. NO importar ni fusionar con MaleCNS Principal ni MAND.

Checkpoint inicial: Kaggle Cognitive C0.7 native_balanced época 24, 8192 neuronas, 1104 entradas. Se conserva intacto en lectura.
Solo retroalimentación sináptica nativa V62. No existe control externo de margen, damping, plasticidad ni selector de repasos por errores.

Fase 1: 248 TRAIN vistos en C0.7 + 248 NUEVOS D = 496 experiencias, diez épocas de intercalación.
Fase 2: repasar las 496 experiencias durante seis épocas y efectuar una revisión de las 416 claves históricas usando predicciones heredadas como pseudoetiquetas, NO verdad de terreno.
Fase 3: 248 NUEVOS E, una única presentación y una única retroalimentación nativa por problema, sin repeticiones.
Evaluación final: 64 H inéditos jamás usados en feedback; también 32 DEV, 32 STRESS y 32 transfer C0.7 exploratorios, memoria A/B/C y D/E.
Controles: native_interleaved, native_uniform, frozen, shuffled con presupuestos iguales.
Primera y segunda fases tienen dos vistas nativas de 32 KC por experiencia; fase final usa solo una.

Smoke Kaggle: https://www.kaggle.com/code/esthefano/malecns-cognitive-c08-threephases-smoke
Preflight de checkpoint: https://www.kaggle.com/code/esthefano/malecns-cognitive-c08-checkpoint-preflight
El entrenamiento completo se inicia tras pasar smoke. No promover ni fusionar hasta verificar precisión, retención y control causal.
El notebook remoto contiene el código ejecutable; este PR se usa inicialmente como registro del protocolo.