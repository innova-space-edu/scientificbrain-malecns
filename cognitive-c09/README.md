# MaleCNS-Cognitive C0.9 — Reentrenamiento nativo y refuerzo

Sólo MaleCNS-Cognitive; NO fusionar con MaleCNS Principal o MAND. Sin adaptador externo.

Checkpoint fuente: C0.8 entrenamiento completo, `native_interleaved/final`, de solo lectura, 8192 neuronas.

TRAIN: 744 ejemplos reales etiquetados: A72, B80, C96, D248, E248.

- Reentrenamiento completo: 8 épocas con todos los 744 ejemplos, dos vistas KC por problema.
- Refuerzo: 4 épocas con los 744 y 372 repeticiones adicionales por época; cuotas fijas A30, B55, C60, D100, E127.
- Memoria heredada: 416 claves repasadas una vez con pseudoetiquetas (no verdad histórica).
- Controles: native_interleaved, native_uniform, frozen, shuffled.
- Evaluación: cohortes A/B/C/D/E, DEV anterior, H (64 experiencias nunca entrenadas, pero ya examinadas), acuerdo de 128 predicciones históricas.

Fuente: `c09_native_retrain.py`; lanzador con verificación SHA y prueba corta: `c09_long_launcher.py`.

Kaggle: https://www.kaggle.com/code/esthefano/malecns-cognitive-c09-native-smoke y https://www.kaggle.com/code/esthefano/malecns-cognitive-c09-retrain-reinforce-long.

Estado inicial: trabajo publicado y en curso. No afirmar mejoras hasta obtener resultados completos. No promover ni fusionar con Principal.
