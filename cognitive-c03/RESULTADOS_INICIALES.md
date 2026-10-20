# C0.3 Cognitive Adaptive — resultados DEV iniciales

**Solo Cognitive**: no integrar MAND ni MaleCNS Principal. Checkpoint V62 balanced 20292111 epoch20 montado como dataset privado y copiado en Kaggle. 8192 neuronas físicas sin cambios.

## Protocolo mini-piloto
C0.2 compositional input encoder; 104 problemas (72 TRAIN /32 DEV), 128 sondas de estabilidad de predicciones históricas (no etiquetas verificadas). Semilla 7403. Dos épocas, 32 KC por problema en época 1 y 64 KC efectivos mediante dos vistas secuenciales de 32 en época 2. Cuatro controles: adaptive_live, fixed_live (misma capacidad pero sin ajuste de margen/damping), frozen y shuffled. Todos usan 32 repasos antiguos y 8 repasos TRAIN por época, con selección por dificultad para adaptive.

## Semilla 7403: ejecución completada
| Control | TRAIN final | DEV final | estabilidad vieja |
|---|---:|---:|---:|
| adaptive_live | 54.167% | 56.25% | 100% |
| fixed_live | 54.167% | 53.125% | 100% |
| frozen | 54.167% | 43.75% | 100% |
| shuffled | 52.778% | 43.75% | 100% |

La ventaja sobre fixed_live es **3.125 pp** (1 de 32 ítems), menor que el umbral preregistrado de 5 pp. El resultado es exploratorio, no válido para promoción.

Kaggle: https://www.kaggle.com/code/esthefano/malecns-cognitive-c03-adaptive-pilot

Semillas adicionales 7411 y 7429 solicitadas, aún no evaluadas al crear este informe: https://www.kaggle.com/code/esthefano/malecns-cognitive-c03-seed-7411 y https://www.kaggle.com/code/esthefano/malecns-cognitive-c03-seed-7429

## Importante
La expansión 32→64 es de **ranuras KC lógicas mediante vistas distintas**, NO crecimiento anatómico 8192→N. No se añade sinapsis o nodo físico y no se rompe la anchura 32 del checkpoint. Se necesitan más semillas, curva de retención a largo plazo, estrés por áreas y confirmación de etiquetas antiguas antes de afirmar aprendizaje superior. El prototipo completo C0.3 se entrega en el ZIP separado de la conversación; este piloto es abreviado y no sustituye el código de producción.
