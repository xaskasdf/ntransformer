# V7: filtro AXI de capacidad DDR

## Problema y ubicación

El [banco frío original](DDR_FFN_COLD.md) comprueba rangos en SystemVerilog antes de
conectar las direcciones de 64 bits a MIG de 37 bits. Ese comparador no
forma parte del hardware. Un acceso fuera del canal podría convertirse en
un acceso válido a una dirección inferior si sólo se cortan bits altos.

`fpga/rtl/ntrpu_axi_capacity_guard.sv` implementa ahora ese filtro entre el
árbitro FFN/host y MIG, en el dominio UI. Su SHA-256 es
`a2a7e2452e1d41a710efa9cf09126eec8da38182c118bf7adfef6ded923f7af5`.
El RTL pasó banco aislado, síntesis, replay funcional de netlist e integración
fría MIG en dos perfiles de reloj. Las revisiones FFN y los buffers externos
conservan sus propias validaciones.

## Contrato implementado para el prototipo

- Entrada AXI de 64 bits de dirección, datos de 512 bits e ID de 4 bits;
  salida de dirección de 37 bits. Capacidad del canal: `1 << 37` bytes.
- Admitir el subconjunto usado por el camino actual: ráfagas INCR, tamaño
  de 64 bytes, dirección alineada a 64 bytes y sin cruce de página de 4 KiB.
  Rechazar explícitamente otros formatos; no reinterpretarlos.
- Calcular extremo exclusivo y suma con extensión de acarreo. Comprobar
  dirección inicial, longitud completa y capacidad **antes** de truncar.
  El último beat válido empieza en `0x1fffffffc0`.
- Capturar la decisión y los atributos al aceptar AR/AW. Mantener VALID,
  dirección, ID, atributos y datos estables mientras READY permanezca bajo.
  Preservar todos los strobes de escrituras admitidas.
- Permitir una lectura y una escritura pendientes, con estados separados.
  Conservar la independencia AW/W: W puede presentarse antes de AW aunque
  su aceptación espere a que se conozca la dirección.
- Una lectura rechazada devuelve el número de beats indicado, RID original,
  RRESP DECERR, datos definidos y RLAST correcto, sin emitir AR hacia MIG.
  Una escritura rechazada consume la ráfaga y devuelve un único DECERR con
  BID original, sin emitir AW ni W hacia MIG.
- Cerrar nuevas admisiones sin cancelar transacciones ya expuestas. `idle`
  exige vaciado completo, incluidas respuestas de error retenidas. No dar
  por drenada una escritura a la que todavía le faltan beats.
- El reset es frío o coordinado entre participantes. Este módulo por sí
  solo no resuelve la pérdida de reloj ni el fallo de reinicio del modelo DDR.

## Cobertura de validación

El banco aislado usa un destino AXI observable y una referencia
independiente. Incluye cero, direcciones superiores a 4 GiB, último beat,
primera dirección inválida, bits altos que producirían alias, suma con
acarreo y cruce de página. Prueba AW antes de W, W antes de AW, canales
simultáneos, respuestas retenidas y cierre de admisión con trabajo pendiente.

Para rechazos, exige cero handshakes aguas abajo y respuestas completas al
origen. Para accesos válidos, reconstruye direcciones, IDs, LAST y strobes
en ambos lados, incluyendo máscaras vacías y parciales.
Los productores AXI deben cumplir sus obligaciones de longitud y estabilidad;
tramas malformadas necesitan un contrato de contención adicional.

## Resultados del 2026-10-04

Los IDs siguientes tienen prefijo `20261004T` y sufijo indicado en los
certificados bajo `artifacts/ntrpu-board-vivado/`. Cada tanda conserva fuente,
trazas, revisión independiente y archivo íntegro local/NFS con SHA-256.

| Tanda | Resultado revisado |
| --- | --- |
| `090429Z-V7-ffn-supervised-capacity-guard-orders` | Tres semillas, 60 casos simultáneos R/W, 3.911 ciclos retenidos; 84 trazas alteradas rechazadas y tres mutantes RTL detectados. |
| `091025Z-V7-ffn-supervised-capacity-guard-synth` | 1.380 LUT: 740 de lógica y 640 LUTRAM; 164 FF, cero BRAM/DSP/latches/blackboxes. |
| `092421Z-V7-ffn-supervised-capacity-guard-netlist-r1` | Replay del DCP anterior: 60 casos, 3.984 ciclos retenidos y 84 trazas alteradas rechazadas; compila netlist, banco y `glbl`, sin RTL original. |
| `091348Z-V7-ffn-supervised-ddr-guard-fast` | Host 4 ns: tres FFN D32/H32, 96 FP32 exactos y 54 transacciones; cuatro ranks activos. |
| `091358Z-V7-ffn-supervised-ddr-guard-slow` | Host 6,2 ns: misma cobertura numérica y de transferencias. |

Las dos tandas MIG suman seis FFN, 19.968 B cargados, 4.224 B calculados,
768 B recuperados y 192 FP32 exactos. Cada revisor rechaza 36 alteraciones
de trazas y cuatro de conexiones; verifica las 35 señales AXI y que sólo
las salidas del filtro se estrechan a 37 bits. Estos bancos integrados
contienen **accesos válidos**: los rechazos se califican en el banco aislado.

| Host | Ciclos de cómputo sin filtro | Con filtro | Diferencia por FFN |
| --- | --- | --- | --- |
| 4 ns | 2.224 / 2.165 / 2.154 | 2.214 / 2.117 / 2.136 | −10 / −48 / −18 |
| 6,2 ns | 2.132 / 2.254 / 2.045 | 2.144 / 2.302 / 2.285 | +12 / +48 / +240 |

Cada ciclo UI equivale a 5 ns simulados. Los cambios incluyen interacción
con pausas y sondeo MMIO deterministas; no miden una latencia aislada ni
demuestran aceleración física. MIG usa `Simulation_Mode=BFM` y geometría
genérica de dos slots/cuatro ranks; la calibración acelerada no mide PHY real.

El DCP conserva SHA-256
`6eb5926ca6ff84fc0d1501257da39a984203ad089ed0367ae4928f513a38ac8c`.
Su estimación interna WNS +3,309/WHS +0,067 ns **no cierra timing**:
1.292 entradas y 1.237 salidas carecen de retardos, incluido el inicio del
comparador de direcciones; falta ubicación del reloj padre y ruteo integrado.
No se suprimen advertencias. La simulación de netlist prolonga sólo el reset
frío para superar GSR de UNISIM; no añade retardos SDF.

`085823Z` y `090053Z` conservan cobertura preliminar menor, sin calificarse
como el banco completo. `091704Z` conserva el error Tcl de `set_property`
antes de simular; se corrigió con `-dict` en un lanzamiento nuevo.

## Reproducción y próximos criterios

```bash
vivado -mode batch -source tools/ntrpu_board_axi_capacity_guard.tcl \
  -tclargs sim /ruta/nueva/unit
vivado -mode batch -source tools/ntrpu_board_axi_capacity_guard.tcl \
  -tclargs synth /ruta/nueva/synth
vivado -mode batch -source tools/ntrpu_board_capacity_guard_netlist.tcl \
  -tclargs /ruta/nueva/synth /ruta/nueva/netlist
python tools/check_ntrpu_axi_capacity_guard.py /ruta/seed-1.log --seed 1
```

Limitar cada lanzamiento a cuatro hilos. La integración usa
`tools/ntrpu_board_ffn_ddr_guard.tcl` con los argumentos HLS/modelo y modos
del [banco frío](DDR_FFN_COLD.md); usar siempre directorios nuevos.
Los revisores de procedencia y respaldo viajan en el checkpoint local.

La [tanda posterior de rechazos desde el host](DDR_AXI_GUARD_ERRORS.md) ya
comprueba DECERR completo, ausencia de tráfico DDR, conservación de tres
regiones y tres FFN posteriores en ambos relojes. Incluye un mutante RTL
detectado como control negativo. El siguiente paso es conectar el filtro en
un top sintetizable y revisar timing con interfaces y relojes completos.

Este filtro verifica capacidad, no autorización de buffers. La propiedad de
pesos, scratch y resultados durante cómputo/retirada sigue siendo otro
contrato; el host no adquiere permiso de escritura por estar dentro de 128 GiB.
También permanecen abiertos el DIMM exacto, reset DDR con tráfico, cuatro
canales, integración PCIe/DMA completa, restricciones de placa e inferencia.
