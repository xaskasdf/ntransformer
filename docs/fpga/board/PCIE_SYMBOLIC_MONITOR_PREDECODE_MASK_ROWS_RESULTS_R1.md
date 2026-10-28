# Máscaras fijas RC: resultado aislado R1

Revisado y respaldado el 2026-10-12 a las 00:29 UTC. El selector con
índices fijos pasó el banco RTL aislado, conservando las referencias
y vectores originales.

## Cobertura observada

| Comprobación | Resultado |
| --- | ---: |
| Propiedad y ancestros, 840 bits | 1.186.064 |
| Incluidas las comprobaciones con X/Z | 1.026.544 |
| Vectores adicionales de selección, 800 bits | 89.632 |
| Mutantes de fuente rechazados | 3 de 3 |
| Corrupciones de evidencia rechazadas | 13 de 13 |

Los mutantes omiten el origen cero, invierten los bits de origen u
omiten la cabecera tres. Cada uno produjo el testigo esperado en el
comparador original; se verificaron entradas, resultados y que la
diferencia permaneciera dentro de la máscara. Un fallo de compilación
no se acepta como detección de mutante.

## Procedencia

Tanda: `20261012T002536Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-commit-priority-mask-rows-r1`.

Revisión SHA-256: `fdfc7677fb3b0635078e96ba5945f0ce31151ada78bfc41c2064727e98e3aa9a`.

Archivo SHA-256: `0277c171f52c49bc6395ff61baf1b2e9548a1154905cca5372ee7e705077ccc6`.

Fuentes SHA-256: `e3229485855c3dc85d5409b4e2c2816aaef8024deff7356f9a939b8090776822`.

La revisión `…-independent-commit-priority-mask-rows-review-r1.json`
está en `build/cloud-setup/20261002-board/`; vincula configuración,
archivos compilados, logs y respaldo verificado local/NFS.

## Siguiente paso

Habilita la [integración completa en RQ](PCIE_SYMBOLIC_MONITOR_PREDECODE_MASK_ROWS_FULL_R1.md),
ya en espera detrás de la cola existente. Su posterior
[síntesis comparable R2](PCIE_SYMBOLIC_MONITOR_PREDECODE_MASK_ROWS_SYNTH_R2.md)
también está en espera y requiere ese PASS completo.
El resultado aislado no prueba ahorro de área, mejora temporal ni
integración PCIe/DDR. No se promueve la variante todavía.
