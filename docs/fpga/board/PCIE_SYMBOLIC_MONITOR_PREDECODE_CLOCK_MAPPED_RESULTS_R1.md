# Predecode R1: replay de la netlist optimizada

## Resultado

La tanda `20261011T090650Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-clock-binary-r1`
aprobó la comparación binaria de su **propia netlist optimizada con BUFGCE**
contra el RTL original independiente. Vivado terminó con código 0 en
1.381,996 segundos. El archivo, sus 300 miembros y las fuentes quedaron
verificados en D: y NFS; la revisión independiente habilita colocación/ruteo.

| Comprobación | Resultado |
|---|---:|
| Bancos positivos | 16 |
| Filas binarias | 174.436 |
| Bits públicos comparados | 447 |
| Muestras antes / después del flanco | 174.516 / 174.516 |
| Sondas quiet / restauraciones | 10 / 10 |
| Mutantes de entrada rechazados | 6 |
| Corrupciones de evidencia rechazadas | 34 |

Los seis bancos drenados conservan cero cancelaciones de **tráfico original**
en C/E/D, A, B y R. Los contadores generales de registros cancelados pueden
incluir registros vacíos o de drenaje y no se confunden con esa propiedad.
Los mutantes detectados omiten RQ, RC, frío o perfil, o alteran secuencia y
requester. Se verificaron las fuentes compiladas en los 22 casos ejecutados.

La biblioteca fuente BUFGCE instalada conservó su hash antes y después;
se simuló con `unisims_ver`. Esto no demuestra la identidad interna de la
biblioteca precompilada. Se revisaron 24 avisos conocidos: una declaración
adelantada de `rq_held`, una redefinición idéntica de `glbl` y 22 avisos del
entorno de compilación de XSIM. No hubo advertencias críticas.

## Identidad de la evidencia

- Netlist propia: `6dccda64c5f8dc4aa8df559c56d1a9e045ba54b725d965a260e52802ffcc0499`.
- Archivo: `f80d73876502d22a1312329ae566667acb782195567a45f0a2642fcf95899166`.
- Fuentes: `source-20261011T085408Z.zip`, hash `bdb5fc44d4258b5395fa6affde5b5cbdea580a39d94180687568f5ca0dae53b1`.
- Revisión independiente: `e2fc1f410e852ea40e174ca6b0ced00d44e13d379033fe38fb4cbc157c61a49b`.

Estado del revisor: `PASS_SYMBOLIC_MONITOR_PREDECODE_CLOCK_OWN_MAPPED_BINARY`.

## Alcance y siguiente paso

Este replay usa los mismos bancos binarios previamente congelados. La
regresión RTL X/Z tiene evidencia separada; no se extiende su alcance a la
netlist. No se aplicó SDF ni se aprobó timing, PCIe nativo, CDC, reset, DDR
o placa. La [cola física](PCIE_SYMBOLIC_MONITOR_PREDECODE_PHYSICAL_QUEUE_R1.md)
puede continuar con colocación/ruteo del DCP propio, reapertura, revisión de
los diez grupos temporales y replay de la netlist ruteada resultante.
