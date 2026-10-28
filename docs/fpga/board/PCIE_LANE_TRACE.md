# Orden de lanes PCIe: netlist sintetizada

Revisión del 2026-10-02. **PASS de conectividad** en
`20261002T160051Z-V4-pcie-lane-synth-r2`: la netlist del ejemplo serial confirma
los 64 puertos escalares, 16 transceptores y 64 bolas únicas. La comprobación
inicial de fuentes PIPE se contrastó con síntesis real del ejemplo serial.
No publica un pinout aprobado para PCB: faltan colocación, ruteo y convivencia
con MIG/PERST. [Mapa estructurado](pcie_lane_map.json).

## Hallazgo

El archivo público `ip_0/source/ntrpu_xdma_pcie4_ip_gtwizard_top.v` conecta
`GT_RXP/N` a `gtyrxp/n_in` mediante una concatenación ascendente de índices
0–15, asignada a un vector descendente. Hace la inversión equivalente en
TXP/N. Por tanto, **GT lane 0 se conecta al índice empaquetado 15**, no al 0.
Se observa en las líneas 852–855 de esta revisión; no es una propuesta de
activar lane reversal en el protocolo PCIe.

`ip_0/synth/ntrpu_xdma_pcie4_ip_gt.v` habilita índices internos 96–111.
Las funciones `f_lb_ch`/`f_ub_ch` empaquetan los canales habilitados en orden
ascendente. El XDC fija los contenedores 24–27, subcanales 0–3, en
`GTYE4_CHANNEL_X1Y0`–`X1Y15`. Combinados, estos fragmentos sugieren:

| Lane lógica | Sitios candidatos en orden | Quad |
| --- | --- | --- |
| 0–3 | X1Y15, X1Y14, X1Y13, X1Y12 | 227 |
| 4–7 | X1Y11, X1Y10, X1Y9, X1Y8 | 226 |
| 8–11 | X1Y7, X1Y6, X1Y5, X1Y4 | 225 |
| 12–15 | X1Y3, X1Y2, X1Y1, X1Y0 | 224 |

La netlist confirma la tabla a través de toda la jerarquía del endpoint y
del wrapper de primitivas. El reporte de E/S y las funciones de bolas dan:

| Lane | RXP | RXN | TXP | TXN |
| ---: | --- | --- | --- | --- |
| 0 | AA4 | AA3 | Y7 | Y6 |
| 1 | AB2 | AB1 | AB7 | AB6 |
| 2 | AC4 | AC3 | AD7 | AD6 |
| 3 | AD2 | AD1 | AF7 | AF6 |
| 4 | AE4 | AE3 | AH7 | AH6 |
| 5 | AF2 | AF1 | AK7 | AK6 |
| 6 | AG4 | AG3 | AM7 | AM6 |
| 7 | AH2 | AH1 | AN5 | AN4 |
| 8 | AJ4 | AJ3 | AP7 | AP6 |
| 9 | AK2 | AK1 | AR5 | AR4 |
| 10 | AM2 | AM1 | AT7 | AT6 |
| 11 | AP2 | AP1 | AU5 | AU4 |
| 12 | AT2 | AT1 | AW5 | AW4 |
| 13 | AV2 | AV1 | BA5 | BA4 |
| 14 | AY2 | AY1 | BC5 | BC4 |
| 15 | BB2 | BB1 | BE5 | BE4 |

La referencia AL9/AL8 en quad225 y PERST en AM17 conservan el alcance de las
comprobaciones anteriores. El ejemplo serial conserva LVCMOS18 en PERST;
esta síntesis no resuelve su conflicto con VCCO de DDR a 1,2 V.

## Evidencia y siguiente comprobación

Rutas relativas a `ntrpu_xdma_ex.gen/sources_1/ip/ntrpu_xdma/ip_0/`;
SHA-256 de los bytes originales:

- `source/ntrpu_xdma_pcie4_ip_gtwizard_top.v`:
  `0e0fb2ec4be1bf5baf78dae12a5bedea52fb2600d074b9c108c651ece31d25e1`.
- `ip_0/synth/ntrpu_xdma_pcie4_ip_gt.v`:
  `463a122ffa10b5e0eeb80da41036c6ea6c1c52204a34044018ee32a4e12cb370`.
- `ip_0/synth/ntrpu_xdma_pcie4_ip_gt_gtwizard_gtye4.v`:
  `0ea985c503e22d737914bdfe69fe98f63154946eba61092d42cf0f50c7906f61`.
- `ip_0/synth/ntrpu_xdma_pcie4_ip_gt.xdc`:
  `2b2a09fd342b8459f16749021ca8b0a9a21e73edc77479c88b3de8b9798f74bd`.

La relación comprobada es `pci_exp_rxp/rxn/txp/txn[i] → pin de primitiva → LOC
GT → bola`. `check_ntrpu_pcie_lanes.py` coteja el TSV de conectividad con
`report_io`, las cuatro bolas candidatas del sitio y su función RX/TX/P/N.
Exige 64 bolas únicas; rechazó seis alteraciones del resultado real: truncado,
duplicado, sitio, banco, polaridad y bola duplicada. Integrar PERST/MIG después
según las dependencias de [la cola](VIVADO_QUEUE.md).

La ejecución `20261002T155658Z-V4-pcie-lane-synth` intentó este paso en una
copia nueva del ejemplo serial `20261002T091826Z-V4-pcie-r2`. Falló en
48,439 s por `Synth 8-439`: faltaban los targets globales de `blk_mem_gen_0`.
No produjo checkpoint ni traza válida. El script preparaba solo XDMA;
ahora genera targets de síntesis para todas las IP del proyecto.

`20261002T160051Z-V4-pcie-lane-synth-r2` terminó en **481,806 s**, salida 0,
con esa corrección. Usa `tools/ntrpu_board_pcie_lane_trace.tcl`, cuatro hilos
y límite de 1800 s. Conserva `pcie_synth.dcp` para repetir consultas sin
sintetizar. Fuentes `source-20261002T160015Z`, SHA-256
`3ea18e5e1361592849d081a3e99901bf9cdddef1bc138503b320614bbc3a3ce9`.
Respaldo local verificado: 135.188.749 bytes, 475 archivos, SHA-256
`326f3cf47dd3366c92a306cb4621ebf997227948b8a87eeb2ad3e71a0e4077e6`.
El primer intento fallido también está respaldado: 42.062.571 bytes,
465 archivos, SHA-256
`8a91122679634255885e4bdfc3954bdd2f6afa6cc53cb8215fdc9d2bc5efc744`.

## Recursos y avisos

El endpoint serial y sus memorias de ejemplo usan **61.778 LUT, 58.653
registros, 205 tiles BRAM, cero DSP, 16 GTYE4_CHANNEL y un PCIE40E4**.
Siete entradas de IP están `Up-to-date` y con licencia `Included`. Los números
incluyen las memorias/aplicación del ejemplo; no son recursos del motor NTRPU
ni del conjunto de cuatro canales DDR. No hay timing posrutado de este endpoint.

La síntesis informa cero errores/críticos y **76.086 avisos**; Vivado limita
las instancias impresas de mensajes repetidos. El respaldo conserva todos los
logs y el conteo impreso por ID en `diagnostics_summary.json`; no se añadieron
supresiones. Predominan puertos/registros sin uso, propagación de constantes,
atributos de reloj y mapeo de RAM. Se conservan para seguimiento `Synth 8-7137`
(prioridad set/reset del contador CPLL AMD, posible diferencia de simulación)
y `Vivado_Tcl 4-2057` (waiver heredado REQP-1839 posiblemente sin objeto útil).
La traza correcta no declara resueltos esos avisos ni cierra DRC de placa.
