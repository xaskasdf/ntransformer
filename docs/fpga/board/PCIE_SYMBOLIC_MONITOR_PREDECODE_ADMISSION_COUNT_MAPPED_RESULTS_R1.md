# Contador C: replay propio aprobado, tanda 1000

## Resultado

`20261011T220721Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-admission-count-binary-r1`
terminó en **1.344,574 s**, salida 0. La revisión independiente aprobó
**174.436 filas binarias nuevas en 16 bancos**, comparando 447 bits
públicos. Pasaron diez perturbaciones quiet y sus restauraciones: veinte
muestras adicionales. Los seis controles de entrada fallaron en sus
testigos previstos y las 34 alteraciones de evidencia fueron rechazadas.

La netlist consumida corresponde a la síntesis propia C999. Se verificaron
los 22 casos compilados, las referencias independientes, la contabilidad,
los cuerpos duplicados de `glbl` y el hash del archivo funcional exacto.
Se conservaron las filas y los bancos drenados originales; no se reutilizó
un PASS de RQ, E o B para acreditar este candidato.

## Decisión

El PASS confirma equivalencia binaria del monitor pasivo en el banco
cubierto. No usa SDF, no califica PCIe nativo ni timing físico y no mide
inferencia. La [síntesis C999](PCIE_SYMBOLIC_MONITOR_PREDECODE_ADMISSION_COUNT_SYNTHESIS_RESULTS_R1.md)
sigue mostrando WNS estimado −3,857 ns y 112 LUT adicionales frente a RQ.
Por tanto, **RQ permanece como referencia**.

La cola R5 continúa con las comparaciones temporales E, B y C frente a
RQ 984, después con E/D. Las rutas de entrada se leen junto con la matriz
entre registros; no se omite la nueva ruta crítica desde `rc_user[78]`.
El experimento posterior de máscaras RC conserva su dependencia de E/D.

## Evidencia y respaldo

Archivo: **82.958.741 bytes, 294 miembros**, verificado el 2026-10-11 a
las 22:32:46 UTC en D: y NFS. SHA-256:
`531b5b502e1c41adb3b598f9580b6f2ad3ae53a447fb489a1f12376c9b0eb071`.

Revisión independiente:
`d8e25d5ff574b832e4a7b4ffb6f1300cca076ecbd2285d7e8a7c76e274513287`.
Netlist funcional C999:
`43f13f10c468647f0acd0b7667d67e6a128f21c330be748c1236fcdc0899e564`.
Fuentes `source-20261011T220551Z.zip`:
`8ddfbe0c91899f4c983d5401157063d1ef988b1d4bee369b17cfa0666acab819`.
Configuración:
`98aa24a062fec8cb8a31f850071928e560aa816a0afa38e511de7a0382e30a6b`.

Certificado en `artifacts/ntrpu-board-vivado/<tanda>/`; revisión y
coordinadores en `build/cloud-setup/20261002-board/`. La copia local
conserva el archivo completo sin extracción duplicada, con cada miembro
verificado.
