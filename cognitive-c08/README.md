# MaleCNS-Cognitive C0.8 — entrenamiento nativo en tres fases

**Estado: entrenamiento completo (DEV_COMPLETE), archivo de código y resultados guardado en GitHub. Solo MaleCNS-Cognitive; NO fusionar con main/Principal/MAND.**

## Código y reproducibilidad

- [c08_smoke_source.py](c08_smoke_source.py): fuente exacta verificada del experimento Kaggle de integración, SHA256 562675750ee8087b1e83795bb3e858c149fd8ee407b8d956f9d428168067366c.
- [c08_native_runner.py](c08_native_runner.py): reglas de generación del currículo, entrenamiento, replay y evaluación C0.8.
- [c08_long_launcher.py](c08_long_launcher.py): lanzador Kaggle con control de integridad SHA y prueba corta previa, que ejecutó las tres fases.
- [c07_pinned_source.py](c07_pinned_source.py): código de C0.7 utilizado como dependencia de ejecución, SHA256 f4c3bb0237bb5b3aaecc28402de4b9c946c0178b86b613ed2ab78c166c2c9062.
- [protocol.json](protocol.json): distribución de experiencias, controles y semillas.
- [C08_RESULTADO_COMPLETO_7403.json](C08_RESULTADO_COMPLETO_7403.json): informe íntegro de los cuatro controles y métricas por etapa, SHA256 8381635d80d3d43ee12c4fa04d777f2eaabb06441b1d0a7ab9bec3cae815ecab.
- [RESULTADOS_C08_7403.md](RESULTADOS_C08_7403.md): informe humano con análisis y limitaciones.
- [archive_sha256_manifest.json](archive_sha256_manifest.json): comprobación de procedencia de las fuentes y resultados.

**No hay claves API ni pesos en el repositorio.** Checkpoint original V62 y punto de partida C0.7 native_balanced ep24 permanecen privados en Kaggle; el código referencia esos recursos y falla si no los encuentra. Para reproducir los experimentos remotos se necesitan acceso a los notebooks y los datos privados propios, además del entorno Kaggle.

## Protocolo efectivamente ejecutado

- Punto de partida: C0.7 native_balanced época 24, 8.192 neuronas.
- 10 épocas de ampliación: 248 tareas C0.7 + 248 tareas nuevas D = 496.
- 6 épocas de repaso de las 496, con una pasada sobre 416 asociaciones anteriores usando pseudoetiquetas heredadas.
- Fase final de 248 experiencias inéditas E, una sola presentación y feedback por pregunta (sin replay).
- Evaluación exclusiva de 64 nuevas H; también 32 DEV, 32 STRESS y 32 transfer anteriores, exploratorios.
- Cuatro controles: intercalado nativo, uniforme nativo, Frozen y Shuffled.
- **Sin adaptador, sin crecimiento de neuronas, CPU solamente.**

## Ejecución y resultados

- [Comprobación del checkpoint C0.7](https://www.kaggle.com/code/esthefano/malecns-cognitive-c08-checkpoint-preflight).
- [Smoke C0.8 terminado](https://www.kaggle.com/code/esthefano/malecns-cognitive-c08-threephases-smoke).
- [Experimento C0.8 completo](https://www.kaggle.com/code/esthefano/malecns-cognitive-c08-native-double-retrain-novel): estado DEV_COMPLETE, 7437,2 s de fase larga, tras smoke interno.

Precisión nativo intercalado final: TRAIN antiguo 72,58%, D 59,27%, E presentación única 55,24%, DEV 46,88%, H inédito 50,00%. La estabilidad de predicciones históricas fue 96,88% en 128 sondas. El desempeño no demuestra una mejora de generalización sobre los controles.

**No se ha promovido C0.8 a un modelo estable**. Solo hay una semilla larga (7403), no hubo replicación completa 7411/7429 y las pseudoetiquetas antiguas no equivalen a exactitud de recuerdos.
