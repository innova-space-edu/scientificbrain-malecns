# MaleCNS-Cognitive C0.10 — evaluación final del factorial 2×2

**Estado:** `DEV_COMPLETE`, una semilla entrenada (7403), época final 12. Kaggle completó la integración y las 4 épocas experimentales. **Sin adaptador**, solo plasticidad nativa. Checkpoint padre: C0.9 `native_interleaved/epoch08` de solo lectura. **No promover ni fusionar con MaleCNS Principal**.

## Fuente oficial
- [Kaggle C0.10 completo](https://www.kaggle.com/code/esthefano/malecns-cognitive-c0-10-corrected-full-factorial).
- JSON oficial dentro de Kaggle: `MaleCNS_C010_Long_7403/report.json`, SHA-256 `ee66bfe315a626a7f2af32eb1f4a0d3cb1fb8186c408e2bade32b70e54efbb99` (26.139 bytes).
- Código ejecutado, SHA-256: `e70474d174470f50668e5dc6d1597b38a9e0aa80a81502104b685a005da4bee4`.
- Duración de las cuatro épocas: 7.516,9 segundos. Revisión y verificación JSON por Kaggle de solo lectura completada.
- Base antes de la época 9: ABC 71,37%; D 64,52%; E 58,47%; DEV 43,75%; H 51,56%; 128 predicciones históricas de referencia 100%.

## Resultados época 12 (porcentajes)

| Modalidad | ABC TRAIN | D TRAIN | E TRAIN | DEV anterior | H exploratorio | Estabilidad histórica |
|---|---:|---:|---:|---:|---:|---:|
| Sin repetición adicional ni pseudoetiquetas | **72,98%** | 66,94% | 57,66% | **46,88%** | 51,56% | 98,44% |
| Solo 372 repeticiones adicionales/época | 70,16% | 62,90% | 58,06% | 43,75% | 50,00% | 98,44% |
| Solo 416 pseudoetiquetas históricas | 72,58% | **67,34%** | 57,66% | **46,88%** | 51,56% | **99,22%** |
| Ambos mecanismos | 70,56% | 62,50% | 58,47% | 43,75% | 50,00% | 98,44% |
| Frozen | 71,37% | 64,52% | 58,47% | 43,75% | 51,56% | 100,00% |
| Shuffled | 59,68% | 61,29% | 52,42% | 40,63% | 48,44% | 96,09% |

## Cohortes de la variante sin repeticiones extra ni pseudoetiquetas

| Cohorte | Epoch 8 padre | Epoch 12 |
|---|---:|---:|
| A | 76,39% | 79,17% |
| B | 71,25% | 72,50% |
| C | 67,71% | 68,75% |

Memoria sináptica de esta rama: 56.534 conexiones distintas con cambios de `theta`, 5.417 eventos de plasticidad y 3.642.338 cambios sinápticos acumulados.

## Inferencia provisional
Dentro de esta semilla y este currículo, **las 372 repeticiones adicionales por época redujeron ABC, D y DEV** con o sin pseudoetiquetas. El uso de 416 pseudoetiquetas sin las repeticiones extra dejó resultados parecidos, con ligera mayor estabilidad de las predicciones antiguas y precisión D (+0,40 puntos respecto a no usar ninguno). El resultado no demuestra que las pseudoetiquetas sean seguras en todos los contextos: son respuestas del propio modelo, no etiquetas verificadas.

No existe mejora confirmada en la transferencia H. H y DEV son exploratorios y ya se habían inspeccionado en experimentos anteriores; E fue entrenado en ambas vistas en C0.9, por lo que `new_E_second_view_untrained` es un nombre histórico incorrecto para esta etapa.

## Próximo experimento recomendado
Repetir las cuatro ramas 2×2 con **al menos tres semillas independientes que realmente tengan checkpoint C0.9 época 8**; actualmente las semillas 7411 y 7429 solo pasaron pruebas estáticas del generador. Preparar nueva transferencia sellada no usada en revisiones, evaluar precisión y olvido por cohorte antes de seleccionar checkpoint. La mejor alternativa provisional para consolidación es `no_extra_no_pseudo`, no una promoción estable.
