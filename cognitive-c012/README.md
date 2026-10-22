# MaleCNS-Cognitive C0.12 — entrenamiento continuo corto (3 épocas)

## Objetivo y motivación
Volver a una única ejecución continua y más corta que C0.9 y C0.11 para reforzar conocimiento sin repeticiones intensivas. **Sin adaptador**. Exclusivo de MaleCNS-Cognitive, no fusionar con Principal/MAND/main.

## Fuente preservada
- Kaggle `esthefano/malecns-cognitive-c011-micro-long`, checkpoint **`MaleCNS_C011_Long_7403/checkpoints/continuous/final`**.
- SHA256 inmutable `system.json`: `888872be4c5b12b59414c7a98f908fad65b542c00f9a5cf5c8df34a53199e11e`.
- 8.192 neuronas, 2.480 señales KC existentes (32 posiciones cada una).
- Origen de solo lectura. Los nuevos checkpoints se guardan con el nombre C0.12 en Kaggle, sin sobrescribir C0.10/C0.11.

## Currículo
- Tres épocas consecutivas `cycle=9,10,11`, orden fijo aleatorizado por ciclo, 744 experiencias A/B/C/D/E por época, **dos vistas por experiencia**.
- Un único modelo LIVE y estado completo durante las 3 épocas, sin recargas, reinicios, ni selección de pesos entre épocas.
- Ninguna repetición extra de 372 experiencias, ninguna de las 416 pseudoetiquetas históricas, ningún adaptador y ninguna alteración automática de margin/damping.
- Guardar un checkpoint al término de cada época como **registro**, no para decidir cambios en ejecución.

## Evaluación
- A/B/C/D/E TRAIN, coincidencia con 128 predicciones históricas y DEV/H anteriores (exploratorios).
- F/G de C0.11: **exploratorios** (ya examinados). No seleccionar por ellos.
- Nuevo **J**: 64 preguntas binarias equilibradas de símbolos, matemática y lógica. 128 claves KC para ambas vistas, aisladas y no entrenadas. Evaluar **solo después de acabar** las tres épocas, entonces comparar el padre y la versión final sobre J.
- J, F, G, H y DEV no reciben feedback.
- Evaluar J una vez no demuestra generalización robusta; una muestra de 64 tiene incertidumbre considerable.

## Validación y estado
- `c012_short_continuous.py`: pruebas estáticas de integridad para tres semillas, pero solo la semilla 7403 posee un checkpoint entrenado. Gate interno de prueba corta seguido de las tres épocas.
- [Kaggle C0.12](https://www.kaggle.com/code/esthefano/malecns-cognitive-c012-short-continuous)
- Ejecutado desde el notebook privado, estado inicial RUNNING. Ningún resultado reportado como definitivo antes del cierre y verificación de `report.json`.
- No hay control negativo shuffled en esta versión de entrenamiento corto; se conserva el control congelado del checkpoint padre y los controles científicos C0.10/C0.11.

**Política:** no promover el modelo ni fusionar resultados automáticamente; comparar retención y transferencia antes de continuar con C0.13.
