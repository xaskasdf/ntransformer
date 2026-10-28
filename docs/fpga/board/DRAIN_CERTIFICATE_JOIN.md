# Composición de certificados de drenaje

`ntrpu_drain_certificate_join.sv` es un bloque de gestión sintetizable.
Combina siete contratos; todavía no está conectado al endpoint y a todas
las secciones de transporte. No acciona reset ni reabre admisión.

## Contrato e interfaces

| Bit | Certificado requerido |
| --- | --- |
| 0 | Propiedad anterior invalidada |
| 1 | Productor host cerrado y obligaciones externas identificadas |
| 2 | Inventario protegido XDMA completo |
| 3 | Sección AXI-Lite vaciada |
| 4 | Sección AXI4 vaciada |
| 5 | Respuestas locales AXI-Lite terminadas |
| 6 | Respuestas locales AXI4 terminadas |

Todas las entradas pertenecen a `mgmt_clk`. Un nivel asíncrono o un bit
escrito por software no sirve como certificado. Cada proveedor entrega ACK,
generación y tag coherentes y revocables. `contract_present` debe permanecer
bajo para un proveedor cuyo contrato no esté implementado, particularmente
el inventario protegido XDMA actual. Las pruebas locales 5/6 solo se solicitan
después de aceptar las cinco anteriores.

Cada ronda exige generación/tag no nulos y una identidad lexicográficamente
mayor que la anterior. Cancelar una solicitud no permite reutilizar su tag.
Cada proveedor debe mostrar ACK bajo después de recibir su nueva solicitud.
El coordinador conserva los fallos hasta inicialización de configuración.
La pérdida de un certificado previamente aceptado o de su readiness invalida
la ronda; restaurar ese nivel requiere una identidad nueva.

`complete_ack` exige simultáneamente las siete pruebas frescas. Sus campos
de identidad son válidos solo mientras ACK esté alto. Los controles X/Z
se rechazan en simulación; esto no representa un detector físico de X.
`cold_rst_n` es inicialización de configuración, nunca reset operativo.

## Evidencia funcional

- `20261006T055550Z-V4-ffn-supervised-drain-join`: 39 escenarios,
  238 comprobaciones y 47 observaciones positivas. Revisión continua de
  entradas/salidas públicas y 559 flancos de gestión. El auditor rechazó
  287 alteraciones de evidencia.
- `20261006T060016Z-V4-ffn-supervised-drain-join-mailboxes`: siete mailboxes
  XPM reales, 12 observaciones y 245 flancos de gestión. Se conservó una
  solicitud ocupada durante cancelación/reemplazo y se descartó una
  finalización de la identidad anterior. El auditor rechazó 17 alteraciones.

La segunda tanda usa proveedores de prueba. El VCD registra habilitaciones
de reloj y las esperas públicas, no cada red de reloj de los proveedores.
Ambas tandas conservan el aviso XSIM 43-3431 sobre el entorno del compilador;
la elaboración y las simulaciones terminaron. El registro no identifica
una variable concreta, por lo que no se atribuye una causa específica.

Los mailboxes pueden conservar `busy` durante la retirada de su handshake
después de devolver ACK. El coordinador no los resetea. Una futura secuencia
de reset debe preservar su dominio de retención y sus transferencias.

## Defectos deliberados y recursos

`20261006T060626Z-V4-ffn-supervised-drain-join-characterize` detectó cuatro
defectos en copias privadas: omitir respuestas locales, aceptar ACK alto sin
renovación, olvidar un fallo y no comprobar el tag. El RTL original y el banco
anterior permanecieron intactos. Para el primer defecto, el prefijo Tcl
`label=1` también coincidía con `label=11`: la revisión independiente exige
la línea Fatal completa y documenta que la detección ocurrió antes. Las
futuras revisiones del ejecutor deben usar coincidencias exactas.

La síntesis OOC del coordinador solo, para `xcvu5p-flva2104-2-i`, midió:

| Recurso/condición | Resultado |
| --- | ---: |
| LUT ajustadas por combinación | 564 |
| Primitivas LUT antes del ajuste | 630 |
| FF | 153 |
| CARRY8 | 68 |
| RAM / DSP / latches | 0 / 0 / 0 |
| Reloj de gestión restringido | 10 ns |
| WNS / WHS OOC | +8,304 / +0,077 ns |

Los recuentos de primitivas y utilización se cotejaron. La síntesis terminó
sin errores, avisos críticos ni avisos de síntesis. Quedan **1054 entradas
y 287 salidas sin retardos de E/S**; no hay colocación/ruteo ni equivalencia
funcional del netlist comprobada. No se añadieron excepciones temporales.
El informe CDC del bloque tiene un único reloj: no califica los adaptadores
ni los cruces de las secciones. Estos recursos excluyen mailboxes, datapaths,
XDMA y DDR.

## Respaldo y alcance

| Tanda | SHA-256 del archivo local/NFS |
| --- | --- |
| Composición | `f93af834f3f1b8a75362bd1c5fee80ed4624dbf2a4718d23bdc3325d45ce95f8` |
| Mailboxes | `9bad00ee7685b08ca8242178c484e1c13239817f065dac43011f65be529615ca` |
| Defectos y síntesis | `dd4789bdabcf204249631f7dd12222f12aa304640775c702bb7b290b6f2b26e5` |

Las adendas de revisión se conservan por separado. La configuración de
Vivado y las fuentes compiladas se cotejaron con los ZIP de fuentes.
Estos resultados no califican los proveedores externos, el drenaje nativo
completo, el driver, el reset global ni la placa. El siguiente paso es
la [integración de las secciones reales](DRAIN_SECTION_INTEGRATION_PLAN.md).
