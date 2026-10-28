# Integración completa de sumas E — resultado R1

La tanda 990, `20261011T191854Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-effect-sum-full-r1`,
aprobó la revisión independiente y quedó respaldada en D: y NFS.
Vivado terminó en **777,077 s**. La cola R3 cerró la revisión y pasó
a la síntesis propia sin repetir la simulación.

## Cambio y cobertura

Partiendo del monitor RQ 983, se sustituyó únicamente E por la suma
equilibrada calificada en 986. Control RQ, D, A, B, selección winner,
interfaz y latencia permanecen iguales a esa referencia.

- **183.058 filas nuevas**, 16 bancos y 447 bits públicos comparados.
- **183.138 muestras antes y después** del flanco, por fase.
- 31 perturbaciones de quietud, 62 muestras de perturbación/restauración.
- Seis controles negativos detectados en los testigos originales.
- 34 corrupciones de evidencia rechazadas por el revisor.
- Seis bancos drenados sin cancelar tráfico original; 26 fuentes
  efectivamente compiladas por caso.

Se verificaron por separado contabilidad, referencias originales,
fuentes compiladas, controles de entrada y advertencias. Las pruebas
incluyen X/Z; no se limitaron al subconjunto binario del replay mapeado.

## Evidencia

El archivo de resultados contiene 1.575 miembros y 50.137.862 bytes.
SHA-256 del archivo:
`431a44425aa3121aa749d60dc1adf34da2dd008b8230fdb4418df382f45b80e9`.

Snapshot `source-20261011T191731Z.zip`:
`bf7e421a5453d2563587acfb2e9bcce9523a50431efc6c6d589df6e56972bb2c`.
Configuración congelada:
`8d6e16040cf7c2712ea5ef357cba3315b6f3da2c77705ae4f0a95a211e6e00f4`.
El certificado y los archivos están bajo `artifacts/ntrpu-board-vivado/`
en la carpeta de la tanda; revisión y evidencia independiente están
en `build/cloud-setup/20261002-board/`, con el mismo prefijo.

## Alcance

Este resultado habilita la [síntesis propia](PCIE_SYMBOLIC_MONITOR_PREDECODE_EFFECT_SUM_SYNTH_R1.md).
La mejora de recursos y timing, el replay de su netlist y el cierre físico
siguen siendo comprobaciones independientes. No califica inferencia,
DDR de 512 GiB ni una placa fabricada.
