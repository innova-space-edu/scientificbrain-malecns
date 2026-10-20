# MaleCNS-Cognitive C0.5 — StableCredit DEV

**Solo Cognitive.** No se usa ni incorpora código de MAND o MaleCNS Principal. Modelo V62 balanced: dependencia de checkpoint privado, lectura solamente, 8.192 neuronas.

### Diagnóstico C0.4
En seis épocas y semillas 7403/7411/7429, Adaptive DEV: 46.875%, 40.625%, 53.125% (promedio 46.875%). Fixed: 53.125%, 37.5%, 59.375% (promedio 50%). Frozen: 43.75%, 37.5%, 56.25%. Shuffled: 53.125%, 37.5%, 56.25%. No se ha validado C0.4; en seed 7403 empeora DEV al final pese a mejorar TRAIN.

### Nuevo C0.5
- Fuente Python c05_credit.py: plasticidad conservadora por dominio, damping entre 0.54 y 0.63 y margin entre 0.000135 y 0.000165.
- Omite feedback cuando ya hay respuesta correcta con margen elevado en todos los modos entrenables (Adaptive, Fixed, Shuffled). Usa model.discard() nativo para cerrar retroalimentación pendiente. La primera ejecución detectó este requisito y fue corregida antes de los ensayos de resultados.
- Repaso TRAIN de 12 ejemplos/época con rotación reproducible y cuotas iguales por materia y clase, sin selección por resultados DEV. 32 repasos de asociaciones antiguas/época mediante etiquetas de autodestilación, no verdad histórica verificada.
- 6 épocas, 32 a 64 slots KC efectivos, 72 TRAIN, 32 DEV, 4 controles. No se modifican las neuronas físicas y todos los cambios de pesos pasan por la proyección sináptica nativa.
- Paquete completo c05_credit.py, c05_experiment.py, test_c05.py y dependencias en MaleCNS_Cognitive_C05_StableCredit_DEV.zip. Pruebas locales 3 semillas + ciclo de descarte.

### Kaggle
- https://www.kaggle.com/code/esthefano/malecns-cognitive-c05-stable-fixed
- https://www.kaggle.com/code/esthefano/malecns-cognitive-c05-seed-7411
- https://www.kaggle.com/code/esthefano/malecns-cognitive-c05-seed-7429

Los notebooks remotos son un piloto abreviado de modificaciones C0.5 sobre el script C0.4 anclado a SHA, no una ejecución idéntica del corredor completo del ZIP.

NO FUSIONAR ni promover hasta comprobar resultados reales, estabilidad antigua, controles emparejados y pruebas nativas completas; sin heldout final.
