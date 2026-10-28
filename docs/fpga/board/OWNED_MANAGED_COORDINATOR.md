# Control NTRD y certificado del mismo propietario

Validado el 2026-10-06 con Vivado 2026.1, build 6511674, para
`xcvu5p-flva2104-2-i`. Este paso continúa
[el coordinador sintetizable](OWNED_DRAIN_COORDINATOR.md).

## Integración

`ntrpu_pcie_fragment_dual_coordinated.sv` conecta el control NTRD real
con `ntrpu_owned_drain_coordinator`. Las escrituras AXI-Lite producen ARM
y ACK sobre el mismo propietario que recibe los frames NTRPUF01 y genera
el certificado. No se duplica el propietario. Se conservan las interfaces
de CMD/DATA/CPL, resultados, épocas y diagnóstico.

El certificado de propiedad no certifica el drenaje de MMIO ni autoriza
reset. Los seis contratos externos permanecen explícitos y ausentes en
este banco; `complete_ack` permanece bajo. Los resets usados son de
inicialización en frío, una vez por caso.

## Simulación y revisión independiente

Ocho casos combinan relojes de fuente de 5 y 37 ns con pérdidas de sesión
mientras se retiene CMD, DATA o resultado; otros dos casos presentan una
época incorrecta. El reloj de gestión tiene un período de 10 ns.

La revisión reconstruye desde los puertos públicos del VCD:

- **176 escrituras y 86 lecturas MMIO**, incluidos descriptores completos,
  rechazos y respuestas retenidas; ocho ARM y ningún retiro aceptado.
- Dos frames por caso, una línea de 64 bytes, un CMD y un DATA; comparación
  de payload, strobes, IDs, direcciones, épocas y tags de 64 bits.
- Conservación de las dos leases y del cookie `0xffffffff`, seguido de
  `0x100000000`, durante la pérdida de sesión.
- Sustitución de solicitud con reloj detenido, un único flanco para
  registrar la pérdida sin certificar todavía, rechazo de respuestas
  obsoletas y renovación con identidad nueva.
- Certificado válido mientras sigue retenida una B de MMIO: confirma que
  cerrar ownership y drenar el bus son obligaciones distintas.
- Rechazo de reutilización de ARM/ACK/descriptores aunque vuelva admission;
  ningún certificado válido en los dos casos de época incorrecta.

Se verifican 108 eventos contra el VCD, además de cada transferencia y
estabilidad bajo backpressure. Siete alteraciones de evidencia fueron
rechazadas: drenaje global o retiro falsificado, payload, época, tag,
cookie y respuesta MMIO. Son controles del revisor, no defectos RTL
inyectados. Las fuentes compiladas se cotejan con el ZIP congelado.

| Reloj de fuente | Primer certificado desde reanudación | Renovación desde reanudación |
| --- | --- | --- |
| 5 ns | 234,998 ns | 84,998 ns |
| 37 ns | 854,998–864,998 ns | 414,998–424,998 ns |

Son latencias del escenario RTL observado; no representan throughput de
inferencia, tiempo de reset ni medidas de placa.

La regresión del banco anterior se ejecuta con semillas **1, 7 y 42**,
tanto sobre el módulo original como sobre el nuevo. El revisor anterior
aprueba ambas variantes y sus trazas `OWN_*`/`CTL_*` coinciden exactamente:
7.359, 7.391 y 6.669 líneas por semilla. Conserva las inicializaciones en
frío del banco anterior y mantiene inactivo `close_request`. Es una
comparación acotada de RTL, no equivalencia formal o de netlist.

## Síntesis y límites físicos

| Medida OOC | Resultado |
| --- | --- |
| Top integrado | 3.054 LUT / 4.334 FF |
| Control NTRD | 915 LUT / 924 FF |
| Coordinador con propietario | 2.203 LUT / 3.410 FF |
| BRAM / URAM / DSP / latches / cajas negras | 0 |
| WNS / WHS / WPWS | 1,941 / 0,066 / 2,225 ns |
| CDC reportados | 3 informativos, sincronizadores de tres etapas |

Las LUT de jerarquías no se suman directamente por el empaquetado de la
síntesis. Se conservan las excepciones puntuales de XPM: un máximo de
datapath de 5 ns y tres false paths a la primera etapa; sin agrupación
general de relojes.

Hay **1.897 entradas y 3.023 salidas sin retardos de E/S**. No se reportan
relojes internos ausentes, endpoints internos sin restricciones ni loops.
Los ocho avisos incluyen lógica XPM no usada, `src_clk` sin carga en
sincronizadores, `HD.CLK_SRC` ausente en OOC y la advertencia entre relojes.
No se ha colocado/ruteado este top ni aprobado timing físico.

## Evidencia y reproducción

Cada ID se encuentra en `artifacts/ntrpu-board-vivado/<ID>/`, con tar,
ZIP de fuentes y `VERIFIED.json`, cotejados localmente y en NFS:

| ID | Resultado |
| --- | --- |
| `20261006T103725Z-V4-ffn-supervised-owned-managed-frozen` | Ocho casos, revisión independiente aprobada |
| `20261006T103844Z-V4-ffn-supervised-owned-managed-legacy-frozen` | Diagnóstico de preparación; no simuló |
| `20261006T110052Z-V4-ffn-supervised-owned-managed-synth-frozen` | Síntesis y auditoría OOC aprobadas |
| `20261006T110438Z-V4-ffn-supervised-owned-managed-legacy-r2-frozen` | Regresión original/coordinada, tres semillas |

El diagnóstico se conserva: `set_property` interpretó la cadena inicial
`-testplusarg` como opción. La revisión R2 usa `set_property -dict` sin
cambiar RTL ni estímulos. El original permanece intacto.

Fuentes del banco y síntesis: `source-20261006T103435Z.zip`, SHA-256
`25db95e0d402d400889a3600a0e3325cc99a95c2ddee1f43f798104728c7e6a2`.
Regresión R2: `source-20261006T110230Z.zip`, SHA-256
`207db17679ec3e3fe955c91416fb17bee5a24bfd2571052cf4fb3dce4d018138`.

Los Tcl reproducibles son `tools/ntrpu_board_owned_managed.tcl`,
`tools/ntrpu_board_owned_managed_synth.tcl` y
`tools/ntrpu_board_owned_managed_legacy_r2.tcl`; requieren carpeta nueva.
El revisor público es `tools/check_ntrpu_owned_managed.py`. Los revisores,
resultados y hashes de procedencia se conservan en el checkpoint bajo
`build/cloud-setup/20261002-board/`, con prefijos de los IDs anteriores.

## Próximo tramo y criterios abiertos

La cola R34 conecta secciones AXI-Lite/AXI4 retenidas a **estos mismos
flujos** y al receptor de memoria real. Debe conservar las obligaciones
AW/W/B/AR/R y CMD/DATA/CPL, observar el recorrido del payload y revisar
CDC y restricciones del conjunto. Las secciones del banco anterior
todavía generaban tráfico independiente del propietario.

Este banco usa un productor de completions de prueba. Su sonda de CPL
tardía se retira tras observar que no es aceptada; no prueba un productor
PCIe/DDR completo. El cierre real del host, inventario protegido XDMA,
secuenciador/reset operativo, timeout PCIe nativo de 50 ms, CDC-15 del
endpoint, DIMM exacto, cuatro canales, inferencia completa y entrega física
siguen pendientes según [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
