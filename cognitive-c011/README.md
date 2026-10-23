# MaleCNS-Cognitive C0.11 — ocho microciclos, sin adaptador

## Objetivo
Determinar si ocho ciclos cortos de una época, con persistencia completa de pesos, optimizador, sinapsis y memorias entre ciclos, ayudan a mantener A/B/C al aprender D/E, comparados con ocho épocas continuas con el mismo número y orden de presentaciones.

## Alcance y datos
Solo MaleCNS-Cognitive. **No fusionar con Principal, MAND ni main**. Emplea el checkpoint `C0.10 / MaleCNS_C010_Long_7403 / no_extra_no_pseudo / ep12`, de **8.192 neuronas**, entrada de solo lectura. Los programas verifican el SHA256 del checkpoint antes de entrenar.

Entrenamiento nativo sin adaptador: 744 experiencias conocidas con respuestas reales, cohortes A72, B80, C96, D248, E248. **8 ciclos con una época cada uno**, con permutaciones deterministas distintas por ciclo; ambas vistas KC por experiencia. Sin 372 repasos extra ni pseudoetiquetas. Sin modificar márgenes o amortiguaciones del núcleo.

## Comparación experimental
- `micro_select`: entrenar una época desde el checkpoint seleccionado, guardar cada candidato, evaluar y aceptar solo si mejora la macroprecisión de entrenamiento y cumple límites predefinidos de pérdida de cohortes, memoria histórica y validación F.
- `micro_always`: 8 ciclos cortos independientes en ejecución pero encadenados mediante guardado/carga de un checkpoint **en cada ciclo**, aceptando siempre. No reiniciar pesos.
- `continuous`: ocho épocas consecutivas, con mismos órdenes y exposición, **sin recargar el checkpoint** entre épocas y sin evaluación intermedia. Su propósito es aislar el efecto de guardar/reanudar.
- Frozen de referencia se obtiene del padre sin cambios.

El selector tiene un papel metodológico y no es un adaptador de plasticidad: acepta o rechaza checkpoints completos entre ciclos; **no modifica las reglas de aprendizaje ni sus coeficientes**. Si el selector rechaza todos los ciclos, el resultado final seleccionado será el checkpoint base.

## Evaluación y separación
- Validación **F**: 64 preguntas nuevas etiquetadas, sin realimentación. Se puede consultar tras cada microciclo para seleccionar entre checkpoints; por tanto es *validación*, no prueba independiente.
- Test sellado **G**: otras 64 preguntas nuevas, sin realimentación y sin usarse en la selección. Consultado por primera vez al final de cada trayectoria.
- DEV y H de versiones anteriores: se muestran solo como evaluaciones exploratorias, **no para seleccionar checkpoints**.
- Cohortes A/B/C/D/E conocidas: precisión de retención/aprendizaje. La estabilidad de las 128 predicciones históricas mide coincidencia, no exactitud garantizada.
- Las pruebas F/G son sintéticas, de tipo binario, y no demuestran comprensión abierta de lenguaje.

## Seguridad y reproducibilidad
- Mismo punto de partida y órdenes de exposición para las tres trayectorias.
- Sin GPU, sin adaptador y sin crecimiento de conectoma.
- F/G nunca reciben feedback y se preservan los checkpoints de ciclos aceptados y rechazados.
- Tres semillas comprobadas para la integridad del plan de presentaciones; **solo la semilla 7403 dispone de pesos C0.10 previos**. Para afirmar robustez harán falta repeticiones reales con otros puntos de partida.

Código: `c011_microcycles.py` y `c011_long_launcher.py`.
Kaggle: preflight https://www.kaggle.com/code/esthefano/malecns-cognitive-c011-preflight , smoke https://www.kaggle.com/code/esthefano/malecns-cognitive-c011-micro-smoke , largo https://www.kaggle.com/code/esthefano/malecns-cognitive-c011-micro-long .

**Estado inicial**: código en rama independiente, no promocionar ni fusionar hasta verificaciones científicas.
