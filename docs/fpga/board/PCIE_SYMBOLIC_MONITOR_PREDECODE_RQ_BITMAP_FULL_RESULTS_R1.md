# Bitmap RQ: integración completa revisada

La tanda 983 `20261011T174801Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-rq-bitmap-full-r1`
terminó con código 0 en **779,723 s**. La revisión independiente aprobó
**183.058 filas RTL nuevas**, sin reutilizar muestras de otro candidato.
El coordinador cerró revisión y respaldo el 2026-10-11 a las 18:04 UTC.

## Cobertura

Se compararon los 447 bits públicos, con 183.138 muestras antes y después
del flanco, 31 pruebas de quietud y 62 muestras de perturbación/restauración.
Los seis bancos drenados conservaron el tráfico original sin cancelación
RAW. Los seis controles negativos de entradas divergieron donde se esperaba;
el revisor rechazó 34 corrupciones de evidencia.

Los 26 archivos realmente compilados conservan las referencias y oráculos
originales. El candidato mantiene la prioridad RC ya validada y anticipa
la comparación de tags de los cuatro lanes RQ; la admisión y la actualización
final del bitmap mantienen su orden. El banco C aislado previo es la
[tanda 982](PCIE_SYMBOLIC_MONITOR_PREDECODE_RQ_BITMAP_FOCUS_RESULTS_R1.md).

## Evidencia y respaldo

Resultados verificados en D: y NFS: **50.042.495 bytes, 1.575 archivos**,
SHA-256 `a8a6be0c15de267da9bae81961078afa9bd759c1da406a94964bb832a2d22267`.
Snapshot `source-20261011T174641Z.zip`, SHA-256
`82b7d487586f5ea2b3646ac164e61c3c54b0626bdddf393107793fd1f2278d2a`.

La revisión `-independent-symbolic-monitor-predecode-rq-bitmap-full-review-r1.json`
tiene SHA-256 `439b03df53ad2d930f9b553e9f6364f92f06135ba048eeff1d0d1ed2c882f22b`.
Los registros se conservan en `build/cloud-setup/20261002-board/`;
el certificado del archivo está bajo `artifacts/ntrpu-board-vivado/<tanda>/`.

## Siguiente criterio

El PASS habilita [síntesis propia comparable con 979](PCIE_SYMBOLIC_MONITOR_PREDECODE_RQ_BITMAP_SYNTH_R1.md),
seguida del [replay de su netlist](PCIE_SYMBOLIC_MONITOR_PREDECODE_RQ_BITMAP_MAPPED_R1.md).
No demuestra mejora de timing ni califica PCIe nativo o placa física.
La [prueba independiente de suma en E](PCIE_SYMBOLIC_MONITOR_PREDECODE_EFFECT_SUM_RTL_R1.md)
queda encolada después del cierre de esa cadena, sin modificar esta variante.
