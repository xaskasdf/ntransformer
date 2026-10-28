# Coordinador sintetizable con propietario retenido

## Diseño y contrato

`fpga/rtl/ntrpu_owned_drain_coordinator.sv` instancia el propietario
certificado y el coordinador de siete contratos. La prueba de ownership
se conecta internamente al slot 0: ningún puerto permite suministrar un
ACK externo para ese slot. Las peticiones usan la generación/tag registrados
por el coordinador; el proveedor comprueba la época real del propietario.

Los seis slots externos, en orden, son cierre del host, inventario protegido
XDMA, sección AXI-Lite, sección AXI4, respuestas locales AXI-Lite y respuestas
locales AXI4. Sus identidades, ACK, fallas y readiness deben ser coherentes
en `mgmt_clk`; el módulo no sincroniza entradas externas arbitrarias.
Ambos resets son exclusivamente de configuración inicial. No se produce
reset operativo ni reapertura de propiedad.

## Simulación nativa y revisión independiente

Vivado/XSim 2026.1 ejecutó ocho perfiles sobre `xcvu5p-flva2104-2-i`,
con periodos de fuente/dispositivo de 5 y 37 ns. El banco conecta las
secciones reales con IP AMD y conserva los estímulos de la
[composición previa](OWNED_DRAIN_COMPOSITION.md): propietario vivo,
invalidación con relojes detenidos, CMD/DATA/resultados retenidos,
identidades obsoletas, solicitudes tardías y revocación.

`tools/check_ntrpu_owned_coordinator.py` reutiliza las comprobaciones
de transacciones y ownership sin modificar sus aserciones. Además,
compara todas las señales públicas capturadas en la referencia y verifica
los puertos del módulo nuevo. Solo normaliza el nombre del módulo raíz;
no transforma tiempos ni datos.

- Ocho perfiles aprobados; entre **702 y 750 señales** por perfil.
- **553.884 estados con tiempos idénticos**, sin diferencias públicas.
- Seis perfiles con host o XDMA ausente mantienen `complete_ack=0`.
- Los dos controles con fixtures producen dos certificados frescos cada
  uno, seguidos de revocación. Esos fixtures no califican proveedores reales.

Esta comparación es una reproducción acotada de trazas RTL, no una prueba
formal ni una comparación funcional de la netlist sintetizada.

## Recursos, timing y CDC OOC

| Jerarquía | LUT | FF |
| --- | ---: | ---: |
| Coordinador completo | 2.164 | 3.410 |
| Join de siete certificados | 645 | 153 |
| Propietario con proveedor | 1.519 | 3.257 |
| Proveedor de ownership, incluido en la fila anterior | 172 | 776 |

No hay BRAM, URAM, DSP, latches ni cajas negras. Los valores corresponden
a esta síntesis conjunta; no se suman sin más a los de bloques aislados,
porque Vivado optimiza la lógica en función de las conexiones.

Con relojes de 5 y 10 ns: WNS **1,941 ns**, WHS **0,066 ns** y WPWS
**2,225 ns**; cero endpoints internos sin restricción, sin reloj o en bucles.
Se conservan tres cruces CDC-3 informativos, con tres etapas de sincronización.
Solo se aplican las excepciones específicas de XPM; no se añadieron grupos
de reloj ni falsos caminos generales.

Hay **2.180 entradas y 2.592 salidas sin retardos externos**. Los ocho
avisos corresponden a lógica XPM retirada/no utilizada, falta de
`HD.CLK_SRC` en OOC y análisis entre relojes. No hubo avisos críticos.
Esto no acredita timing físico, ruteo ni resolución del CDC-15 del FIFO
AMD del endpoint nativo, que no forma parte de esta síntesis.

## Reproducción y respaldo

Fuentes nativas: `source-20261006T095847Z.zip`, SHA-256
`7ab509b79f4ba2e84952e86608d3a4c9ac70c4ae768f77a57ddcf2ef16050e84`.
Scripts: `tools/ntrpu_board_owned_coordinator.tcl`,
`tools/ntrpu_board_owned_coordinator_missing.tcl` y
`tools/ntrpu_board_owned_coordinator_synth.tcl`.

| Tanda | SHA-256 del tar.gz local/NFS |
| --- | --- |
| `20261006T095925Z-V4-ffn-supervised-owned-coordinator-frozen` | `5591f31ccd8f5eb96bf3b1b78c255102125cdfb0e48d7b4586357bb535fc8686` |
| `20261006T100616Z-V4-ffn-supervised-owned-coordinator-missing-frozen` | `a8a5b7a61b5063e175d9d765b3206b63da109e3a03dddf898561d632c48dece1` |
| `20261006T101300Z-V4-ffn-supervised-owned-coordinator-synth-frozen` | `b60d8fd4c36b031c7b49495771f6ec113840fe6849b123fbfa3ec811ae5073be` |

Los comprimidos y ZIP de fuentes están bajo
`artifacts/ntrpu-board-vivado/<run>/`, sin nueva extracción duplicada.
Las revisiones son addenda `*-independent-coordinator-review.json` y
`*-independent-coordinator-synth-review.json` en el directorio de
orquestación; el checkpoint preserva también esos revisores.

## Siguiente integración

La cola R33 debe incorporar el módulo en una nueva revisión del camino
`ntrpu_pcie_fragment_dual_managed`: control NTRD real, propietario y
certificado compartiendo el mismo ARM y la misma reserva. Luego conectar
las secciones retenidas a los mismos flujos de control/datos y al receptor
real, preservando todas las obligaciones AW/W/B/AR/R y CMD/DATA/CPL.
La certificación debe observar esos flujos, sin una segunda instancia de
propietario ni estímulos independientes que aparenten integración.

En estos ocho perfiles los estímulos de propietario y secciones todavía
son independientes. Host y colas protegidas XDMA, reset nativo completo,
DIMM exacto, cuatro canales y cierre físico conservan sus requisitos en
[VIVADO_QUEUE.md](VIVADO_QUEUE.md). No se autoriza drenaje global ni se
declara aprobado el datapath host–PCIe–DDR o la placa de 512 GiB.
