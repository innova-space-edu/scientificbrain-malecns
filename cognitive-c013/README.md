# MaleCNS-Cognitive C0.13 — una sola época, exclusivamente material nuevo

## Encargo
Un **único entrenamiento de una época**, sin repasar ni reentrenar experiencias conocidas, sin pseudoetiquetas, sin adaptador, sin replay y sin nuevas fases de entrenamiento adicionales. Evaluar después qué aprendió y cuánto olvidó.

## Datos y procedencia
- Fuente parental: Kaggle `esthefano/malecns-cognitive-c012-short-continuous`, checkpoint `MaleCNS_C012_Long_7403/checkpoints/epoch11`, **único y de solo lectura**.
- SHA-256 `system.json`: `a19561e365c9aff1b547e0ed1a2dfddadad8b66685b63bed60910c53fac3eacd`. Neuronas físicas: **8.192**. Cues antiguos preservados: **2.608**.
- Una época de **744 experiencias inéditas** y realmente etiquetadas: 248 símbolos, 248 matemáticas y 248 lógica; 372 positivos y 372 negativos. Dos vistas codificadas por experiencia, total 1.488 presentaciones. Identificadores simbólicos de un carácter, numéricos muy superiores a los vistos en C0.12. No hay experiencias A/B/C/D/E ni F/G/J en el entrenamiento.
- Conjunto final también nuevo: **192 ejercicios** adicionales (64 símbolos, 64 matemáticas, 64 lógica), semánticamente y por cues disjuntos del entrenamiento y los archivos anteriores. Se evalúa al final sin feedback ni selección.
- Comprobación estricta de firmas semánticas, unicidad de entradas KC y de que ningún índice antiguo ni de prueba recibe feedback. **Capacidad máxima del registro de estímulos: 4.096**. Los 2.608 estímulos antiguos y los 1.488 estímulos nuevos de entrenamiento ocupan exactamente el límite. Los 384 estímulos correspondientes a las 192 pruebas T se evalúan únicamente en copias de diagnóstico separadas de 2.992 entradas; no se añaden al modelo que entrena.
- Un checkpoint antes (copia sin entrenar) y otro después de la única época; se conservan en Kaggle sin modificar C0.12.

## Medidas de análisis
Precisión en 744 ejercicios antes/después (aprendizaje en nuevos casos); precisión en 192 casos reservados antes/después (transferencia); precisión por cohortes antiguas A/B/C/D/E y acuerdo con 128 predicciones históricas antes/después (olvido). F/DEV/H de versiones anteriores se usan solo para exploración. No son pruebas ciegas.

## Riesgos y limitaciones
El entrenamiento introduce **nuevas instancias** de operaciones conocidas (igualdad, suma, transitividad), no capacidades cognitivas completamente nuevas. Una muestra sintética binaria no valida comprensión abierta del lenguaje. Las modificaciones de `theta` indican plasticidad, no neuronas activas distintas. Solo se entrena realmente semilla 7403. El código fuente de C0.11 y el checkpoint parental se verifican por hashes.

## Ejecución
[Notebook Kaggle](https://www.kaggle.com/code/esthefano/malecns-cognitive-c013-new-only)

**Política:** no fusionar con MaleCNS Principal/MAND/main ni promover checkpoint automáticamente. El experimento se declara completo solo cuando `report.json` documenta una única época finalizada sin errores.
