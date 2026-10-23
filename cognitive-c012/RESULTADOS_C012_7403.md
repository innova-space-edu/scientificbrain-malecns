# MaleCNS-Cognitive C0.12 — resultados verificados, 3 épocas continuas

**Estado:** `DEV_COMPLETE`. Semilla entrenada **7403**; tres épocas consecutivas completas (9, 10 y 11) sobre 744 experiencias, dos vistas cada una, sin adaptador, sin repeticiones adicionales ni pseudoetiquetas. Solo rama **Cognitive**; no se modificó Principal/MAND.

## Evidencia reproducible
- Kaggle: https://www.kaggle.com/code/esthefano/malecns-cognitive-c012-short-continuous
- Informe íntegro `MaleCNS_C012_Long_7403/report.json` (no confundir con EmbeddedSmoke), **SHA-256 `42696d01b79c4b0688eff51a1a21046986f239d7bb55a1df61df01560c309497`**, 5.323 bytes. Auditoría en notebook `esthefano/malecns-cognitive-c012-final-report-audit`.
- Checkpoint anterior `C011 continuous/final`, SHA-256 `888872be4c5b12b59414c7a98f908fad65b542c00f9a5cf5c8df34a53199e11e`, nunca sobreescrito.
- Integración smoke interna pasó. Se guardaron checkpoints en `MaleCNS_C012_Long_7403/checkpoints/epoch09`, `epoch10` y `epoch11`. No se recargó el modelo entre épocas.
- Tiempo de ejecución de las tres épocas e instrumentación: 992,44 segundos. 8.192 neuronas sin ampliación. 55.959 conexiones distintas con cambios `theta`, 4.043 eventos efectivos y 2.721.737 modificaciones sinápticas acumuladas.

## Evolución verificable

| Métrica | Padre inicial | Época 9 | Época 10 | Época 11 final |
|---|---:|---:|---:|---:|
| A TRAIN | 76,39 % | 76,39 % | 75,00 % | 75,00 % |
| B TRAIN | 70,00 % | 72,50 % | 65,00 % | 72,50 % |
| C TRAIN | 72,92 % | 72,92 % | 72,92 % | 72,92 % |
| D TRAIN | 67,74 % | 70,16 % | 64,52 % | 67,74 % |
| E TRAIN | 60,89 % | 59,68 % | 61,29 % | 60,89 % |
| Macro TRAIN A/B/C/D/E | 69,59 % | **70,33 %** | 67,74 % | 69,81 % |
| Validación F (64; vista antes) | 51,56 % | 48,44 % | 51,56 % | 51,56 % |
| DEV antiguo, exploratorio | 43,75 % | 46,88 % | 43,75 % | 43,75 % |
| H antiguo, exploratorio | 51,56 % | 51,56 % | 51,56 % | 51,56 % |
| Acuerdo 128 respuestas históricas | 100,00 % | 99,22 % | 98,44 % | 99,22 % |

### Test J final (64 binarios, nunca usados como feedback ni criterio de selección)

| Categoría | Padre evaluado al final | C0.12 época 11 |
|---|---:|---:|
| Símbolos (16) | 56,25 % | 56,25 % |
| Matemáticas (24) | 45,83 % | 41,67 % |
| Lógica (24) | 54,17 % | 54,17 % |
| Total (64) | **51,56 %** | **50,00 %** |

El cambio total en J es de **un acierto menos en 64**, por lo que no constituye evidencia de mejora o deterioro robusto. El test G anterior terminó en 43,75 % y es **exploratorio**.

## Interpretación
- Primera época: macro TRAIN +0,74 puntos respecto al padre, debido sobre todo a B y D; pero F bajó y J no se evaluó hasta el final.
- Segunda época: caída fuerte en B y D. Tercera época: recupera B/D, pero la mejora neta vs padre final es pequeña (+0,22 puntos macro).
- El resultado **no confirma que tres épocas sean mejores que una**, porque no se observó J en la primera época. Se conservaron los tres checkpoints, por lo que pueden someterse en el futuro a un mismo test de confirmación prediseñado que no se use para elegir checkpoints.
- Las 64 pruebas de J son sintéticas y el rendimiento ronda el azar; no se debe declarar capacidad general de lenguaje, lógica o matemáticas.
- La observación de alta plasticidad sináptica, sin mejora de transferencia, sugiere que nuevas repeticiones del mismo currículo probablemente no resolverán por sí solas el problema de representación/generalización. Es una hipótesis de investigación, no prueba de mecanismo único.
- Sólo existe una semilla efectivamente entrenada; las tres semillas en pruebas estáticas no equivalen a replicaciones independientes.

**Decisión científica:** conservar C0.11 continuous/final y C0.12 ep09/10/11 como candidatos de investigación, **sin promocionar C0.12 automáticamente ni fusionar el PR**. Priorizar evaluación verdaderamente nueva y cambios controlados de representación antes de prolongar entrenamiento.
