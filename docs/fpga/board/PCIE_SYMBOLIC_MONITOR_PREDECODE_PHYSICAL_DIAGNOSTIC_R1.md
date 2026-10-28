# Predecode R1: primer diagnóstico físico

## Resultado archivado, revisión pendiente

La tanda `20261011T093541Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-placement-r1`
terminó con código 0 en 1.210,493 segundos. Usó el DCP optimizado propio,
`place_design` Default y `route_design -directive Quick`, dos threads,
período de 4 ns y E/S de 0 ns. El archivo y sus 97 miembros están
verificados en D: y NFS. La inspección autoriza **reapertura diagnóstica**;
no autoriza todavía el replay ruteado ni aprueba timing.

## Medidas reportadas

| Medida | Referencia R1, Quick | Predecode R1, Quick |
|---|---:|---:|
| LUT | 103.747 | 104.500 |
| FF | 26.540 | 27.106 |
| CARRY8 | 2.058 | 2.058 |
| WNS global | −11,835 ns | −13,498 ns |
| WHS global | −3,904 ns | −2,972 ns |
| Peor hold registro→registro | −0,104 ns | −0,084 ns |
| Niveles de la peor ruta E→D | 3 | 1 |
| Retardo de datos de esa ruta | 15,613 ns | 17,314 ns |
| Retardo de interconexión de esa ruta | 15,236 ns | 17,145 ns |

Con este flujo no hay mejora del peor setup: empeora 1,663 ns y aumenta
753 LUT y 566 FF respecto a la referencia física. El hold global mejora,
pero sigue fallando. El candidato reporta 32.507 endpoints con fallo de
setup y 30.474 con fallo de hold, sobre 70.455 endpoints temporales.

La ruta crítica va de `e_stage/out_admission_matches_reg[114]/C` a
`d_stage/symbolic_rows_reg[114][34]/CE`. Tiene una LUT5 y 99,024 % de retardo
de interconexión: 9,826 ns en una red con fanout 54 y 7,319 ns en otra con
fanout 51. El reporte dedicado E→D confirma el mismo peor setup y mide
hold −0,007 ns. No hay un reporte E→D dedicado de referencia para comparar
ese hold de forma equivalente.

Las banderas indican colocación y ruteo completos, sin errores de rutas.
Se conservan tres avisos de DRC OOC incompleto, cien avisos emitidos de
puertos sin ubicación de partición y uno de reloj sin `HD.CLK_SRC`.
Las fronteras OOC y los límites de emisión siguen impidiendo una aprobación
física de placa; ausencia de errores de ruteo no significa timing cerrado.

## Decisión siguiente

Completar primero reapertura, auditoría EDIF, identidad de los diez grupos
temporales, revisión independiente y replay ruteado propios. El cambio
de predecodificación no se presenta como una mejora física demostrada.

La siguiente comparación prioritaria será ruteo guiado por timing desde
los DCP **ya colocados**, preservando restricciones y documentando cualquier
transformación del router. Comparar referencia y candidato permite separar
el efecto de Quick del cambio de RTL. Su preparación/ejecución todavía
está pendiente; no se modifica ni reinicia esta tanda.

El [ensayo de fallos por lane](PCIE_SYMBOLIC_LATE_FAULT_LEAVES_R1.md) permanece
como comprobación independiente de B→R. No resuelve por sí solo la ruta
E→D que domina este resultado. La integración y la selección del siguiente
monitor deben responder a las medidas físicas, sin añadir ciclos ni reducir
capacidad o comprobaciones.

## Identidad

- Archivo: `6bea5dfdb9c648a2f2a41754c6d3282839b44a3b28028d596194580285fc20ff`.
- Fuentes: `source-20261011T093408Z.zip`, hash `3049daa98ab5edb3c5f7b7b0066938de673320dc3e3c0729ef66ecbcb079e716`.
- DCP colocado: `c5153f0903068ad1e6027c5bad9da2a1d1e1feeb1bb234071c7fbf7b2cce38b2`.
- DCP ruteado: `7db656909fea3814c69e40891152baa13a18ed33d6dd61e5b063e80b8e2e1d6b`.
- Netlist ruteada: `c70b29f56fc816820bf6e63e370f1f2cbc056d42d44285574a97fbe188b9e201`.
- Inspección local: `13a75c68ff4326bb17a6f5ff010ee33d216821312e4517fc26ef4456e98e4c3d`.
