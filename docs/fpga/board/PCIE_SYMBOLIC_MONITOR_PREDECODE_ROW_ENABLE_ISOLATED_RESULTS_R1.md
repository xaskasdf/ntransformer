# E/D: habilitaciones por fila, banco aislado

Fecha: 2026-10-11 UTC. Tanda `20261011T230322Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-row-enable-r1`.
Revisión `PASS_ACTUAL_ED_ROW_ENABLE_R1`, respaldada en D: y NFS.

## Cobertura efectiva

Se comparó E/D real contra las referencias E/D intactas usando únicamente
entradas públicas de C. La prueba ejecutó **40.958 ciclos nuevos**, con
comparaciones antes/después del flanco y estabilidad antes del flanco:

- 14.336 bits de tabla y 51 bits de contexto retenido.
- 2.498 bits de salida D y 4.554 bits de salida E original.
- 512 bits de enables contrastados contra el oráculo de E.
- Todas las filas y controles binarios, seis puertas con cuatro estados,
  X/Z individual en cada bit de C y 1024 registros aleatorios.
- Seis mutaciones reales de fuentes detectadas y 39 corrupciones de
  evidencia rechazadas.

El cambio genera máscaras de habilitación coherentes con el registro E
existente y las consume en D; no añade un ciclo de latencia. El PASS
autoriza avanzar al banco integrado propio, que ya sigue en la cola.
Todavía no acredita equivalencia del monitor completo, síntesis, replay
mapeado, timing ni promoción de arquitectura.

## Procedencia

- Revisión: `3fbedc4102c8d8e93fd7b2977340c9d0d1c98216742032d5d74e4a3b5fc7caa8`.
- Archivo: `decedc4e4f21fe031e2c4984c4e1fef90f4af29b257c0fdb4a4617f191609bdb`.
- Snapshot: `1e6bd34e07aa779136b640ba6cb9a92a1889de3413023702592e60c1a024610c`.
- Configuración: `404e76e8ef0d0185c8fb9cdc3a132e4b596a1f5bc1de2ecb39e86009b3bef249`.
- Evidencia: `6d57ab3d70504e0d0b0656ac5383d658a3ece8635fb70dfbf75c046ee61c69d7`.
