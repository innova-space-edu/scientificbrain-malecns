# MaleCNS-Cognitive C0.10 — consolidación sin adaptador

## Pregunta científica
¿Por qué C0.9 mejoró experiencias D/E pero deterioró aprendizajes A/B/C y DEV durante las épocas de refuerzo? Determinar de forma controlada si la interferencia se debe al volumen adicional de repeticiones, a las pseudoetiquetas históricas, o a ambos.

## Punto de partida reproducible
- Solo MaleCNS-Cognitive. Nunca fusionar con Principal/MAND.
- **Checkpoint exacto:** Kaggle `esthefano/malecns-cognitive-c09-retrain-reinforce-long/MaleCNS_C09_Long_7403/checkpoints/native_interleaved/epoch08`.
- SHA-256 del `system.json`: `6b2352f305e5e9f897cfca4120408bbc3a0de9a68beb6e7c4e8258f184c4c0e9`.
- 8.192 neuronas, población KC 4.064, 2.224 cues almacenados (32 KC cada uno). Padre de solo lectura, checkpoints nuevos por control.
- Código C0.9 heredado por Kaggle, con hash `255a9785b7dc04d15d5287b68a0cdd240c22da7ddb761e2acb295517a12ad2fe`.
- **Ningún adaptador.** Aprendizaje exclusivamente por `native_step` del núcleo V62, sin márgenes, amortiguaciones o selección de repaso aprendidas dinámicamente.

## Diseño factorial 2 × 2 (épocas 9–12)
| Modalidad | 372 repeticiones adicionales por época | Una revisión de 416 pseudoetiquetas antiguas |
|---|---|---|
| no_extra_no_pseudo | No | No |
| extra_only | Sí | No |
| pseudo_only | No | Sí |
| extra_and_pseudo | Sí | Sí |
| frozen_full | Sí, sin modificar pesos | Sí, sin modificar pesos |
| shuffled_full | Sí, con etiquetas alteradas | Sí, con etiquetas alteradas |

Las seis ramas comparten el mismo checkpoint inicial y el mismo orden de 744 experiencias de entrenamiento A72/B80/C96/D248/E248 en cada época. Las repeticiones y pseudoetiquetas constituyen las únicas intervenciones factoriales. Las 416 pseudoetiquetas son predicciones del checkpoint de la época 8 y **no representan verdad histórica certificada**.

Los resultados se miden al concluir cada época: A/B/C, D, E, DEV anterior, estrés, transferencias anteriores, 64 H ya examinados y acuerdo con 128 predicciones históricas del padre. Frozen y shuffled son controles negativos. Las pruebas sobre DEV/H son exploratorias, no deben usarse para seleccionar parámetros.

## Archivos y ejecución
- `c010_ablation.py`: definición completa, controles y pruebas de integridad para 7403/7411/7429. Solo 7403 tiene checkpoint de época 8; las otras dos semillas comprueban la composición y el calendario, **no son ejecuciones entrenadas independientes**.
- `c010_long_launcher.py`: reutiliza el código Kaggle verificado mediante SHA-256 y realiza smoke obligatorio antes del experimento de cuatro épocas.
- [Validación del checkpoint](https://www.kaggle.com/code/esthefano/malecns-cognitive-c010-parent-check)
- [Prueba corta](https://www.kaggle.com/code/esthefano/malecns-cognitive-c010-ablation-smoke)
- [Entrenamiento completo 2x2](https://www.kaggle.com/code/esthefano/malecns-cognitive-c010-full-factorial)

Estado de la implementación: lanzada en Kaggle; resultados finales por verificar. No hay promoción ni fusión, y se preservan los checkpoints C0.8/C0.9 originales.
