# MaleCNS-Cognitive C0.8 — resultados Kaggle (semilla 7403)

El trabajo [malecns-cognitive-c08-native-double-retrain-novel](https://www.kaggle.com/code/esthefano/malecns-cognitive-c08-native-double-retrain-novel) terminó en estado **DEV_COMPLETE**. Tiempo del ensayo largo: 7437,2 s (~124 min), además de la prueba interna de integración de ~1618,1 s. Sin adaptador, sin GPU y sin crecimiento físico del conectoma.

## Fases implementadas
1. Ampliación: C0.7 TRAIN 248 + 248 nuevos D = 496, durante 10 épocas.
2. Repaso: los mismos 496 en seis épocas, incluyendo una pasada por las 416 memorias históricas con **pseudoetiquetas** obtenidas de la versión anterior (no verdad histórica).
3. Nuevas E: 248 experiencias únicas mostradas una sola vez por pregunta; sin repaso.
4. Evaluación separada: 64 nuevos H reservados, más DEV/estrés de versiones anteriores.

## Métricas finales
| Evaluación | Intercalado nativo | Uniforme nativo | Frozen | Shuffled |
|---|---:|---:|---:|---:|
| Conocimientos anteriores (248) | 72,58 % | 71,77 % | 72,58 % | 60,08 % |
| D (248) | 59,27 % | 60,48 % | 47,58 % | 49,19 % |
| E una presentación (248) | 55,24 % | 55,24 % | 50,81 % | 54,03 % |
| E vista no entrenada | 51,61 % | 50,81 % | 51,21 % | 50,41 % |
| H inéditos (64) | 50,00 % | 53,13 % | 48,44 % | 53,13 % |
| DEV previo (32) | 46,88 % | 43,75 % | 50,00 % | 50,00 % |
| Predicciones históricas preservadas (128 sondas) | 96,88 % | 96,88 % | 100,00 % | 99,22 % |

La precisión D mejora sobre Frozen, pero en H apenas supera el azar en esta semilla. **No está demostrado que mejore la generalización.** La retención histórica mide coincidencia con predicciones del padre, no precisión real.

## Límites
- **Solo una semilla completa, 7403**. Falta repetir 7411 y 7429 con checkpoints equivalentes.
- DEV/estrés anterior se revisó en otras versiones, por tanto es exploratorio; H no participa en feedback.
- E se mide después de una exposición con retroalimentación: es aprendizaje de un ejemplo, no generalización independiente.
- No se promueve C0.8 a estable ni se mezcla con MaleCNS Principal.
- El checkpoint completo y estados binarios permanecen en Kaggle privado; GitHub conserva fuente, resultados JSON, protocolo y las rutas de procedencia.

Fuentes primarias: [prueba de integración](https://www.kaggle.com/code/esthefano/malecns-cognitive-c08-threephases-smoke) y [24 fases relativas/entrenamiento total](https://www.kaggle.com/code/esthefano/malecns-cognitive-c08-native-double-retrain-novel).
