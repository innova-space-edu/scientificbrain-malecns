# MaleCNS-Cognitive C0.11 — resultados oficiales (semilla 7403)

**Ejecución final completada**: Kaggle [C0.11 ocho microciclos](https://www.kaggle.com/code/esthefano/malecns-cognitive-c011-micro-long), estado `DEV_COMPLETE`, ocho épocas completas, tres brazos, fuente original C0.10 `no_extra_no_pseudo/ep12` de solo lectura. Sin adaptador; 8.192 neuronas.

## Procedencia y verificación
- Informe íntegro en Kaggle: `MaleCNS_C011_Long_7403/report.json`.
- Hash SHA-256 del JSON final: `31ce9247e83c4f918caaea8e34848cbb0a8b61c76c57010a7201fd9d8b219a2b` (21.177 bytes).
- Hash del código completo ejecutado: `578862b22a97205b7b1cdf4ead9c58467fdf22548ecec592cac2789b54338c58`.
- Hash del checkpoint inicial C0.10: `35bf6a8fbdcec1e83b1dfacfd232028d30d43403a1e662b8aa92d7044463adb6`.
- Tiempo de ejecución completo: 6.611,94 segundos, además de la integración interna.

## Comparación de las tres estrategias
| Indicador | C0.10 base | Microciclos con selección | Microciclos siempre aceptados | Continuo |
|---|---:|---:|---:|---:|
| Macro TRAIN A/B/C/D/E | 69,00 % | 69,45 % | 69,59 % | 69,59 % |
| Validación F (64 nuevos) | 48,44 % | 48,44 % | 51,56 % | 51,56 % |
| Evaluación final G (64 nuevos) | 45,31 % | 45,31 % | 43,75 % | 43,75 % |
| DEV anterior (exploratorio) | 46,88 % | 43,75 % | 43,75 % | 43,75 % |
| H anterior (exploratorio) | 51,56 % | 53,13 % | 51,56 % | 51,56 % |
| Acuerdo de 128 predicciones históricas | 100 % | 100 % | 99,22 % | 99,22 % |
| Conexiones `theta` distintas modificadas | — | 55.118 | 56.718 | 56.718 |
| Ciclos cuyo checkpoint se adoptó | — | 1/8 | 8/8 | No aplica |

### Exactitud por cohorte TRAIN
| Cohorte | Inicio C0.10 | Selección | Siempre / continuo |
|---|---:|---:|---:|
| A | 79,17 % | 79,17 % | 76,39 % |
| B | 72,50 % | 72,50 % | 70,00 % |
| C | 68,75 % | 69,79 % | 72,92 % |
| D | 66,94 % | 67,34 % | 67,74 % |
| E | 57,66 % | 58,47 % | 60,89 % |

## Comportamiento de la selección
Solo se aceptó el **ciclo 2**, que pasó los criterios de mejora de macro TRAIN y protección de cohortes. Rechazos: ciclos 1, 4, 5, 6 y 8 por pérdida frente a límites predefinidos, y ciclos 3 y 7 por ausencia de mejora macro suficiente.

La estrategia `micro_always` y el control `continuous` obtuvieron **exactamente los mismos resultados**, cambios sinápticos y contadores de feedback, señal de que guardar y reanudar correctamente el estado no añadió aprendizaje por sí mismo.

## Interpretación y límites
- Los microciclos con selección conservaron mejor A y B; los ciclos siempre aceptados mejoraron más C/D/E a costa de A/B.
- F mejoró ligeramente en siempre/continuo, pero **G no mejoró en ninguna trayectoria**. Los 64 ítems G son binarios equilibrados; la precisión se mantiene alrededor del azar.
- El DEV previo descendió desde el padre en las tres variantes. No declarar mejora de generalización.
- Las diferencias F/G equivalen a uno o dos aciertos; tamaño insuficiente para inferir superioridad robusta. Solo se entrenó la semilla 7403; las demás pasaron pruebas estáticas del currículo.
- F se utilizó para seleccionar; **no es test ciego**. G nunca se usó en la selección y se evaluó al final.
- Sin promoción del modelo, sin fusión con Principal/MAND, sin alterar el checkpoint base.

**Decisión de investigación**: conservar las tres ramas y los ocho checkpoints individuales; el método de guardado/reanudación funciona. Antes de agregar más microciclos, mejorar generalización mediante representación, conjuntos verdaderamente nuevos y replicaciones independientes, y estudiar un criterio multiobjetivo de selección que no se ajuste excesivamente a TRAIN.
