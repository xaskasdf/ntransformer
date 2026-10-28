# Certificado de invalidación del propietario NTRPU

Validación del 2026-10-06, Vivado 2026.1 build 6511674,
`xcvu5p-flva2104-2-i`.

## Contrato implementado

`ntrpu_ownership_certificate.sv` observa el propietario en su reloj de origen.
Exige `session_lost` registrado, sesión inválida, épocas no nulas e iguales
en collector/bridge y bloqueo de ARM, retiro, LINE y completions. La generación
solicitada debe ser **la época real del propietario**; el tag identifica una
nueva solicitud de prueba. No basta renombrar una generación de gestión.

Un mailbox XPM transporta la identidad coherente y exige ACK bajo para cada
solicitud. La confirmación requiere tres observaciones consecutivas y un
flanco posterior de respuesta. Los cambios posteriores de propiedad o de
leases revocan el certificado mediante una cadena XPM de tres etapas; el
fallo queda retenido hasta inicialización de configuración.

`ntrpu_pcie_fragment_certified_owner.sv` conecta el proveedor directamente al
collector y bridge reales. No acciona reset. Invalidar ownership conserva
CMD/DATA/resultados ya expuestos, cookies, épocas y leases. Los handshakes de
CMD/DATA pendientes pueden terminar una vez sin convertir el certificado
en una prueba de drenaje.

El caso de arranque sin propietario previo (`epoch=0`) permanece sin
certificado. Una solicitud de otra época queda pendiente sin ACK; cancelar
la petición de gestión no borra una obligación ya entregada al peer.

## Simulaciones y controles

Dos bancos positivos de 12 escenarios comprobaron relojes de origen de
5/37 ns y gestión de 10 ns, pérdida con CMD/DATA/resultado retenido,
reloj detenido antes y después del registro de pérdida, sustitución de tag,
renovación y rechazo de otra época. En el segundo banco, seis escenarios
usan el módulo integrado; los otros seis conservan una perturbación
explícita del puerto de observación para comprobar revocación, sin forzar
estado interno del propietario.

La revisión pública del VCD reconstruyó, por banco, 24 tramas NTRPUF01,
12 transferencias de 64 bytes en la interfaz CMD/DATA del pool (sin
controlador DDR en este banco), cuatro resultados anteriores a
la pérdida y **cero retiros posteriores**. Comparó todos los bytes, direcciones,
cookies, handshakes y 138 observaciones. La revocación registrada alcanzó
gestión en **24,5–29,5 ns de simulación**, incluso con origen detenido.

Se detectaron tres defectos RTL privados: aceptar otra época, certificar
un propietario válido y omitir revocación. La revisión independiente
rechazó 13 alteraciones de logs/VCD y cotejó las fuentes compiladas.

## Recursos y límites físicos

| Síntesis OOC | Resultado |
| --- | ---: |
| Propietario + proveedor | 1.604 LUT / 3.257 FF |
| Jerarquía del proveedor, incluido mailbox | 260 LUT / 776 FF |
| RAM / DSP / latches | 0 |
| WNS / WHS | +1,941 / +0,066 ns |
| CDC | 3 CDC-3 informativos, profundidad 3 |
| Entradas / salidas sin retardos | 1.386 / 2.305 |

Hay ocho avisos: un registro XPM sin uso eliminado, cuatro puertos `src_clk`
sin carga en sincronizadores sin registro de entrada y tres avisos de
timing OOC/múltiples relojes. Se conservan las excepciones XPM específicas
y su límite de 5 ns para datos del handshake; no se añadieron grupos de
reloj generales. No hay ruteo, equivalencia funcional del netlist ni cierre
físico. El CDC-15 de la integración nativa anterior sigue abierto.

## Tandas y respaldo

Todos los archivos se verificaron localmente y en NFS, sin duplicar las
extracciones. Los IDs llevan prefijo `20261006T` y sufijo `-frozen`:

| ID abreviado | Resultado | SHA-256 del comprimido |
| --- | --- | --- |
| `085613Z-V4-ffn-supervised-ownership-certificate` | Diagnóstico: faltaba incluir el decoder; sin simulación | `c08f65b748045b6e6686f9093cd752bd2bbb3f889c9c3fe6cd5ce8e2a8f57477` |
| `090016Z-V4-ffn-supervised-ownership-certificate-r2` | 12 escenarios revisados | `3f7d530f444e0954ab5c16c98fd6eaca1616af3a2f4fedae27ac5b098b26f471` |
| `090608Z-V4-ffn-supervised-ownership-negatives` | Banco integrado y tres defectos detectados | `a40858f360846f5bdbdd43f7d254e469ce02c3a511536a42771828572d67ec41` |
| `090944Z-V4-ffn-supervised-ownership-synth` | Recursos/CDC OOC revisados | `ef012cb85b53866592af959931ba8759256d1b381b90966638f8026b10442af8` |

Las revisiones independientes son adendas bajo
`build/cloud-setup/20261002-board/`, incluidas en el checkpoint posterior.

## Próxima integración

Conectar este propietario al slot 0 del coordinador de siete certificados
y repetir tráfico con las secciones AXI-Lite/AXI4 y proveedores locales R2.
Mantener ausentes los contratos host/XDMA todavía no calificados y probar
que impiden `complete_ack`. No se ha cerrado el driver, el inventario
protegido XDMA, el reset global, el DIMM exacto ni la implementación de placa.
