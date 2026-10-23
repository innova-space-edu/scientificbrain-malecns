# MaleCNS-Cognitive C0.13 — análisis oficial de una época exclusivamente nueva

**Experimento finalizado y auditado.** Kaggle `esthefano/malecns-cognitive-c013-new-only` versión corregida completó 1 única época LIVE sobre **744 ejercicios inéditos**, divididos por igual entre símbolos, matemática y lógica. No se entrenó ningún cue antiguo (0), ni se usaron repasos, pseudoetiquetas ni adaptador. Padre C0.12 `epoch11` preservado sin cambios.

## Trazabilidad
- [Notebook de entrenamiento](https://www.kaggle.com/code/esthefano/malecns-cognitive-c013-new-only); [auditoría JSON separada](https://www.kaggle.com/code/esthefano/malecns-cognitive-c013-final-audit).
- Artefacto validado `MaleCNS_C013_OneNewEpoch_7403/report.json`, SHA256 `f8fd8a287123098e2385e943ee74d5b13521d12089fa95687d054b641586bd76`, 2.914 bytes.
- Checkpoint padre C0.12 `MaleCNS_C012_Long_7403/checkpoints/epoch11/system.json`, SHA256 `a19561e365c9aff1b547e0ed1a2dfddadad8b66685b63bed60910c53fac3eacd`.
- `training_manifest.json` de 744 experiencias y `holdout_manifest.json` de 192 verificados: todas firmas únicas y disjuntas; estímulos nuevos no entrenados anteriormente. Una única semilla entrenada: 7403.
- 8.192 neuronas físicas. Registro máximo de **4.096 cues**: 2.608 heredados y 1.488 nuevos. Los 384 cues del holdout se evaluaron sobre copias diagnósticas con registro reducido a 2.992, sin modificarse el entrenamiento. No se registraron cues de prueba para feedback.
- 1.488 exposiciones (dos vistas únicas por ejercicio), 1.267 eventos efectivos de actualización, 221 sin actualización, 845.726 cambios de sinapsis acumulados, 87.331 parámetros de `theta` distintos modificados. Duración 606,87 s.

## Comparación antes/después

| Evaluación | Antes | Después | Delta |
|---|---:|---:|---:|
| **744 ejemplos nuevos enseñados** | 49,46 % | 51,34 % | +1,88 puntos |
| **192 ejemplos nuevos NO enseñados** | 46,88 % | 48,44 % | +1,56 puntos |
| **Macro de cohortes antiguas A/B/C/D/E** | 69,81 % | 66,11 % | **−3,70 puntos** |
| Validación histórica F (exploratoria) | 51,56 % | 48,44 % | −3,12 puntos |
| DEV anterior (exploratorio) | 43,75 % | 40,63 % | −3,12 puntos |
| H anterior (exploratorio) | 51,56 % | 48,44 % | −3,12 puntos |
| Acuerdo con 128 predicciones históricas | 100,00 % | 100,00 % | 0 puntos |

### Cohortes A-E preexistentes
| Cohorte | Antes | Después | Delta |
|---|---:|---:|---:|
| A | 75,00 % | 73,61 % | −1,39 pp |
| **B** | 72,50 % | **61,25 %** | **−11,25 pp** |
| C | 72,92 % | 71,88 % | −1,04 pp |
| D | 67,74 % | 65,32 % | −2,42 pp |
| E | 60,89 % | 58,47 % | −2,42 pp |

### Aprendizaje por área en el material enteramente nuevo
| Área | 248 entrenados (antes) | 248 entrenados (después) | 64 reservados (antes) | 64 reservados (después) |
|---|---:|---:|---:|---:|
| Símbolos | 44,76 % | 48,39 % | 45,31 % | 46,88 % |
| Matemáticas | 51,61 % | 53,63 % | 45,31 % | 48,44 % |
| Lógica | 52,02 % | 52,02 % | 50,00 % | 50,00 % |

## Interpretación

La única época ocasionó **modificaciones sinápticas verdaderas**, pero el beneficio observable es pequeño: +14 aciertos sobre los 744 ejercicios nuevos enseñados y +3 sobre los 192 reservados. La prueba reservada de 192 ejercicios continúa cerca del 50 % esperado por azar en clasificación binaria balanceada. El comportamiento no valida generalización funcional.

En paralelo, **todas las cohortes antiguas A-E retrocedieron**, especialmente B (−11,25 pp). El acuerdo del 100 % con 128 predicciones históricas solo muestra que esas 128 predicciones se conservaron; no demuestra ausencia de olvido en las tareas antiguas con verdad de referencia.

**Conclusión provisional:** los resultados son compatibles con interferencia/olvido catastrófico al incorporar nuevos casos sin repaso. No se ha demostrado que el mecanismo sea el único responsable ni que el efecto sea reproducible: solo una semilla. La gran plasticidad de `theta` no se tradujo en aprendizaje nuevo suficientemente estable.

La etiqueta «todo nuevo» significa instancias de igualdad, suma y transitividad nunca entrenadas, no operadores cognitivos nuevos. Se debe mantener el checkpoint C0.12 como referencia y el C0.13 como evidencia experimental, sin promocionar ni fusionar con Principal.

## Resultado científico / siguiente hipótesis

**No iniciar más épocas automáticamente.** Antes de un nuevo entrenamiento contrastar, en ramas independientes y pruebas reservadas, una protección nativa de cambios sinápticos importantes y una evaluación de interferencia por cohorte, manteniendo como opción de control esta condición *sin repaso*. Evitar introducir un adaptador permanente o reutilizar los tests J/T para selección de hiperparámetros.
