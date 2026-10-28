# Selector RC por prioridad: resultado aislado R2

La tanda 977 `20261011T162925Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-commit-priority-r2`
terminó con `PASS_ISOLATED_COMMIT_PRIORITY_SELECTOR_R1`, revisión independiente
y respaldo en D: y NFS. R2 corrige exclusivamente el argumento Tcl del
lanzador fallido 976; el RTL candidato mantiene su identidad R1.

## Evidencia funcional

- 1.186.064 comparaciones de ownership y ancestry sobre 840 bits; incluyen
  1.026.544 casos X/Z.
- 89.632 vectores adicionales de selección sobre 800 bits.
- Tres mutaciones reales de fuente detectadas: forzar slot cero, omitir
  slot tres e invertir prioridad. Fallaron en las filas previstas 0, 0 y 4.
- Doce corrupciones de evidencia rechazadas por el revisor.

Se verificaron logs, snapshots, fuentes realmente compiladas, comparadores
originales y las cinco advertencias XSIM esperadas. El ensayo permite pasar
a integración; no demuestra equivalencia universal, síntesis ni timing.

## Trazabilidad

Archivo de resultados SHA-256:
`9be50b020ab8ac662faa46326e397d165d20527b9430c88286e96baa67b1817c`.
Snapshot `source-20261011T162815Z.zip`:
`cd9186d25d29a4e0c08c4186434d8c42f4fe75191410e64c7461d3e2400b9413`.
Configuración congelada:
`f704a8168c05f955a462ae577315e43a1566817a37df5b67470cb7da40118bfa`.

La revisión `-independent-commit-priority-review-r2.json` está en
`build/cloud-setup/20261002-board/`. La integración 978
`20261011T163442Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-commit-priority-full-r1`
se lanzó después del cierre y respaldo aislados. Su resultado debe revisarse
por separado antes de ejecutar la [síntesis propia](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMMIT_PRIORITY_SYNTH_R1.md).
