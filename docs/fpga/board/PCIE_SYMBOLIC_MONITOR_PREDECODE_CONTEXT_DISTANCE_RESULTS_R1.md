# Ensayo aislado de distancia al SOP

La tanda 973, `20261011T152938Z-V4-ffn-supervised-pcie-symbolic-late-fault-leaves-distance-r1`,
terminó con salida **0** en **147,796 s**. La revisión independiente aprobó
el banco directo y su respaldo está verificado en D: y NFS.

## Cobertura y resultado

Pasaron **47.734 muestras**: 10.240 casos de frontera, 17.352 entradas
walking X/Z, 20.000 registros aleatorios y los casos dirigidos. Se compararon
los 32 vectores RC/RQ por lane, de dieciséis bits, antes del OR, además del
agregado original. El padre por bandera permaneció como referencia.

Los seis mutantes RTL produjeron los primeros testigos previstos. El revisor
rechazó quince corrupciones de evidencia y verificó los ocho archivos SV
más `glbl` de cada uno de los siete proyectos. Los WDB se conservaron como
artefactos; sus contenidos no se interpretaron.

La [distancia seleccionada dentro del prefijo](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_DISTANCE_R1.md)
queda calificada en este banco combinacional, sin registros o ciclos nuevos.
Todavía no tiene integración completa, síntesis ni replay mapeado propios.
La [comparación `opt_design`](PCIE_SYMBOLIC_MONITOR_PREDECODE_PAIRED_OPT_PLAN_R1.md)
precede la decisión de integrar: este PASS no demuestra mejora temporal.

## Evidencia

- Paquete: **3.211.382 bytes**, 356 archivos; SHA-256
  `463a218399b50293ac505d3966fb94392c3149ccd039406e7364b674ea03118f`.
- Revisión: SHA-256 `0f310d71b743a34ecc008eeb0023d810c6860280e197b5ba7c8414f40e15a334`.
- Fuentes: `source-20261011T152832Z.zip`, SHA-256
  `37505ecbd4d3a15b2eca0898344b3b7d6831f5bfe0f33b8067f3405c1e210277`.

Scripts y resultados están en `build/cloud-setup/20261002-board/`. No hay
calificación adicional de PCIe nativo, DDR, placa o inferencia.
