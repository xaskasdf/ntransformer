# Monitor PCIe R1: primera caracterización física

Se completaron colocación y ruteo OOC sobre `xcvu5p-flva2104-2-i`,
manteniendo **4 ns**, I/O cero, un BUFGCE y las restricciones originales.
La tanda usó dos hilos y `route_design -directive Quick`; terminó en
**1276,875 s**. La reapertura independiente terminó en **255,694 s**.
Ambas tandas están respaldadas y verificadas en D: y NFS.

## Resultado medido

Vivado informa `ROUTED_FULLY=1`, sin gaps, conflictos ni errores de rutas.
Se utilizan **103747 CLB LUT**, **26540 FF**, **2058 CARRY8** y un BUFGCE;
no hay BRAM, URAM ni DSP. La colocación añadió diecinueve réplicas FDRE:
ocho de `c_stage/observing` y once de `e_stage/.../record/valid`.
La identidad del grafo reabierto no sustituye su prueba funcional propia.

La temporización **no cierra**: WNS global **−11,835 ns** y WHS global
**−3,904 ns**; fallan 37213 endpoints de setup y 30344 de hold, de 69458.
No aparecen fallos de ancho de pulso. Las holguras por clase son:

| Camino | Setup (ns) | Hold (ns) |
| --- | ---: | ---: |
| Registro–registro | -11.835 | -0.104 |
| Entrada–registro | -6.338 | -3.904 |
| Registro–salida | -3.541 | 1.979 |
| Entrada–salida | 3.025 | 0.022 |

Se conservaron veinte caminos por clase, excepto entrada–salida, con dos.
El peor camino interno parte de `e_stage/out_e_reg[origin][ops][admissions][0][tag][1]/C`
y termina en `d_stage/symbolic_rows_reg[234][20]/CE`: tres niveles,
**15,613 ns**, de los cuales **15,236 ns (97,585 %)** son interconexión.
Su última red tiene 51 cargas y 8,405 ns de retardo. Es un problema de
distribución física medido, distinto del peor camino estimado de 31 niveles.

## Verificación y límites

El DCP reabierto conserva el EDIF byte por byte salvo la fecha de exportación,
los 1831 puertos, primitivas, conectividad, constraints y ocho categorías
de temporización. La auditoría jerárquica rechaza ciclos y referencias
inválidas, con ocho controles de estructura y trece de reloj.
El BUFGCE alimenta los 26540 pines de reloj; su entrada OOC sigue siendo
`HIERPORT` sin ruta física y los puertos carecen de ubicaciones de partición.

DRC informa cero violaciones, pero advierte cobertura OOC incompleta.
Se conservan los límites de emisión de avisos y los reportes de metodología:
1000 `TIMING-15`, 1000 `TIMING-16` y 1830 `XDCH-2`.
`Quick` no busca cierre de timing. Estos resultados no acreditan el
endpoint real, CDC, reset, DDR de 512 GiB ni la placa.

## Siguiente dependencia

Reproducir **la netlist ruteada propia** sobre los dieciséis bancos,
174436 filas binarias, 447 bits públicos, dos fases, sondas de quiet y seis
mutantes reales. Mantener la regresión RTL de 183058 filas con X/Z.
Después, comparar una variante con selección de filas decodificada en E
y distribución local en D, sin añadir latencia ni reducir dos RQ/cuatro RC
por ciclo o los 256 tags. Es una hipótesis pendiente de pruebas y medida;
preservar prioridad RC antes de RQ, contexto por valor y cancelación por frío.

- Tanda física: `20261011T062034Z-V4-ffn-supervised-pcie-symbolic-monitor-placement-r2`; SHA-256 `9398867a7cc8bc1ef60393e5200a7e6dd41c9d372bec635412c9c846641d8ec4`.
- Reapertura: `20261011T064601Z-V4-ffn-supervised-pcie-symbolic-monitor-route-audit-r1`; SHA-256 `35c8d58324c78dde88a36372bb067ae43ee2da33f5d112e897b88bd352ee3b46`.
- Revisión: `20261011T062034Z-V4-ffn-supervised-pcie-symbolic-monitor-placement-r2-independent-monitor-physical-review-r1.json`; SHA-256 `d98ad2867d6d817f21e29c39f99895946ebc93517af9cbbdd919bc5cfb752e04`.
- Fuente compartida: SHA-256 `d52990ab151f601701eb8cbe756cb2b3409177016ba3f576e07b89cd25394370`.
