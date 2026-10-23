# MaleCNS-Cognitive — diagnóstico de C0.13 y protocolo experimental C0.14

## Decisión actual
**No ejecutar otra época ni promover el checkpoint C0.13.** La única época nueva C0.13 consiguió +1,88 puntos porcentuales en ejercicios nuevos de TRAIN y +1,56 puntos en 192 casos reservados, mientras A/B/C/D/E perdió 3,70 puntos. La mejora reservada equivale a **3 aciertos adicionales de 192**, insuficiente para demostrar generalización robusta. Retener tanto el checkpoint de C0.12 epoch11 como el de C0.13 epoch01.

## Auditoría C0.13 pendiente de resultados
Notebook de **solo lectura**, no entrenará, no modificará conexiones:
https://www.kaggle.com/code/esthefano/malecns-cognitive-c013-retention-audit

Debe recuperar `MaleCNS_C013_OneNewEpoch_7403/report.json` y los checkpoints padre y final. Medirá:
1. Retención A/B/C/D/E antes/después, con aciertos perdidos y recuperados por cohorte.
2. Cambios por categoría **símbolos / matemáticas / lógica** en casos antiguos, los 744 ejercicios nuevos y los 192 tests nuevos.
3. Conteo y distribución de variaciones reales de `theta` (cantidad, fracción, mediana, p90/p99, L1/L2); validar los 87.331 parámetros modificados.
4. Acuerdo histórico, diferenciando coincidencia en predicciones anteriores de exactitud real.
5. Límites: ningún delta sináptico puede atribuirse por sí solo a una cohorte; eso exige medir traza de activación de neuronas/sinapsis y crédito. Una semilla entrenada **no** son tres réplicas.

No consultar repetidamente el estado de Kaggle. No ejecutar trabajo nuevo después de la auditoría, salvo autorización expresa.

## Evidencia contrastada C0.9–C0.13
- C0.10: las 372 repeticiones extra por época deterioraron A/B/C y D en la semilla 7403; las variantes sin esas repeticiones funcionaron mejor sobre conocimientos ya enseñados.
- C0.11: ocho microciclos sin selección y ocho épocas continuas produjeron el **mismo** resultado, contadores y cambios sinápticos. Guardar/cargar no crea aprendizaje.
- C0.12: las tres épocas continuas tuvieron oscilación; el efecto neto fue +0,22 pp en TRAIN macro y -1 acierto de 64 en J.
- C0.13: solo 744 instancias nuevas en una época produjeron un aprendizaje TRAIN débil con pérdida en contenidos anteriores. Nuevas **instancias** de igualdad/suma/transitividad no significan capacidades cognitivas nuevas.

## Propuesta de C0.14, NO EJECUTADA
Comparar **tres brazos de una sola pasada** desde **el mismo checkpoint de C0.12 epoch11**. Todas las variantes con 744 experiencias diferentes, dos vistas por experiencia, mismo orden intercalado estratificado y 1.488 presentaciones de feedback; sin adaptador, pseudoetiquetas, repeticiones adicionales ni aprendizaje múltiple.

| Brazo | Experiencias nuevas, una pasada | Anclas antiguas con etiqueta verdadera, una pasada | Propósito |
|---|---:|---:|---|
| 100/0 | 744 | 0 | Referencia equivalente a C0.13 |
| 75/25 | 558 | 186 | Aprender nuevo protegiendo memoria |
| 50/50 | 372 | 372 | Ver si más anclas producen menor olvido a costa de plasticidad |

Selección de anclas antiguas por cohortes A/B/C/D/E y por símbolos/matemáticas/lógica, con cuotas proporcionales al inventario:
- 75/25: A18, B20, C24, D62, E62 (total 186).
- 50/50: A36, B40, C48, D124, E124 (total 372).
- Nuevo 75/25: 186 experiencias por categoría, con 93 positivas y 93 negativas.
- Nuevo 50/50: 124 experiencias por categoría, con 62 positivas y 62 negativas.
- 100/0: 248 por categoría, con 124 positivas y 124 negativas.

En los brazos con anclas antiguas **sí existe una nueva exposición a experiencias viejas**, así que son controles futuros de repaso real, no cumplen la restricción de cero reentrenamiento de C0.13. No ponerlos en marcha sin aprobación. El total de actualizaciones se mantiene constante; por diseño, la exposición a experiencias nuevas disminuye conforme aumenta la de anclas, lo que permite observar explícitamente la frontera estabilidad/plasticidad.

## Pruebas y decisiones
- Evaluar cohortes antiguas por separado y su promedio macro; contar predicciones antiguas correctas que pasan a incorrectas y viceversa.
- Evaluar aciertos en el mismo banco completo de 744 nuevas instancias para todos los brazos, separando «vistas» y «nunca entrenadas» según el brazo.
- Preparar una nueva batería **K sellada** con al menos 192 ítems totalmente excluidos de entrenamiento y selección; no reutilizar T/J como si fueran inéditos.
- Como metas experimentales preliminares, aspirar a pérdida histórica macro <= 1,5 pp y ninguna cohorte >3 pp, mejora en nuevas tareas, y ventaja reproducible en test sellado frente al padre. **Las metas no son resultados científicos** ni criterio estadístico suficiente por sí solas.
- Evaluar sensibilidad por al menos tres **checkpoints-semilla independientes realmente entrenados**, no solo semillas del generador.
- Separar aprendizaje del mismo tipo de operación con operandos nuevos de verdadero razonamiento composicional o transferencia de regla.
- Un nuevo test de 192 preguntas binarias equilibradas tiene precisión cercana al azar alrededor de 50% y cambios de unos pocos aciertos son inciertos. Reportar incertidumbre y pruebas pareadas (por ítem) entre modelos.
- Seleccionar por validación no expuesta previamente; mantener test final sellado para una consulta única después de fijar la selección.

## Bloqueo de ingeniería: registro de estímulos
C0.13 tiene **2.608 señales originales + 1.488 señales (dos vistas × 744 experiencias nuevas) = 4.096**, el máximo de `BudgetSystem.cues`. Un test adicional se colocó solo en registros diagnósticos aislados. Antes de agregar nuevas experiencias a un checkpoint C0.13, se requiere diseñar y verificar paginación/bancos de estímulos o aumento seguro del límite de registro, sin perder el conectoma ni índices anteriores. El número de estímulos registrados **no** equivale al número de neuronas.

## Política
Trabajo exclusivo de MaleCNS-Cognitive. Preservar checkpoints y trazabilidad, no tocar Principal/MAND, no añadir adaptador de plasticidad. **No iniciar C0.14 ni crear otro entrenamiento sin aprobación del usuario.**
