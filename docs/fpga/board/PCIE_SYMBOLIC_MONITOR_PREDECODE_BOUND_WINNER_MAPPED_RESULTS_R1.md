# Límites B con winner: replay propio aprobado, tanda 996

## Resultado

`20261011T210603Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-bound-winner-binary-r1`
terminó en **1.343,466 s**, salida 0. La revisión independiente aprobó
**174.436 filas binarias nuevas en 16 bancos**, comparando los 447 bits
públicos. Pasaron diez perturbaciones quiet y sus restauraciones: veinte
muestras adicionales. Los seis controles de entrada fallaron en sus
testigos previstos y las 34 alteraciones de evidencia fueron rechazadas.

La netlist consumida corresponde exactamente a la síntesis propia B995,
no a la referencia RQ. Las referencias RTL independientes y la contabilidad
original permanecen intactas. La revisión comprobó también los cuerpos
duplicados de `glbl`, las fuentes compiladas y los hashes de la netlist.

## Recuperación y siguiente etapa

El primer lanzamiento R4 falló antes de Vivado por longitud de comando
en Windows. La [recuperación R5](PCIE_SYMBOLIC_MONITOR_PREDECODE_PIPELINE_RECOVERY_R5.md)
conservó la configuración congelada y reanudó únicamente el owner previo
al lanzamiento mediante SSH por stdin. No hubo una simulación parcial
anterior que se reutilizara para este resultado.

El coordinador R5 cerró esta etapa revisada y respaldada a las 21:31 UTC
y comenzó la preparación del conteo C aislado. La comparación temporal
B→R sigue en la cola después del replay C. Este PASS demuestra equivalencia
binaria del monitor pasivo en el banco cubierto; no usa SDF, no aprueba
PCIe nativo ni cierre temporal físico y no mide inferencia.

## Evidencia

Archivo: 83.448.817 bytes, 294 miembros, verificado a las 21:31:22 UTC
en D: y NFS. SHA-256:
`b8ff60f96601798b6db251bd76af7895ea4c744e3c8eff0a326676e803112919`.

Revisión independiente:
`39b351ab856d3d99d0de860961f6db5ddce7e20968621ba7750cded87de422ab`.
Netlist funcional B995:
`7dd15963670a55a1dec2eee537794b0fe2252a327045194ddc44b62d8751457c`.
Fuentes `source-20261011T210029Z.zip`:
`70e864b3597f4fefde23c4c4492944fb195a8eec89401fd08356cc2b0ba106b6`.
Configuración:
`82bdef88123e9b2a5e7b3d959ef7e33d1313a7dfa26398a2c1e5651d6db30f54`.

Certificado en `artifacts/ntrpu-board-vivado/`, carpeta de la tanda;
revisión, recibos de recuperación y coordinadores en
`build/cloud-setup/20261002-board/`. La copia local conserva el archivo
completo sin extracción duplicada; cada miembro fue verificado.
