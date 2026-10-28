# Seguimiento de drenaje AXI-Lite

Implementación V4/V7: `fpga/rtl/ntrpu_axil_drain_monitor.sv`.
El banco `tb_ntrpu_axil_drain_monitor.sv` y el Tcl
`tools/ntrpu_board_axil_drain_monitor.tcl` pasan la validación aislada
en `20261005T155453Z-V4-ffn-supervised-axil-drain`.
No están conectados al endpoint AMD ni a sus CDC actuales.

## Qué demuestra el contador

El monitor observa continuamente los handshakes de **una interfaz AXI4-Lite**
desde el reset de configuración. Cuenta por separado direcciones AW,
datos W y direcciones AR aceptados. Una respuesta B aceptada retira
ambas mitades de una escritura; R retira una lectura. Que los pines
VALID estén bajos no borra las transacciones pendientes.

Solo `cold_rst_n` borra el inventario. No conectarlo a PERST ni a reset
de transporte. Una B sin ambas mitades previas, una R sin AR previa o
un contador desbordado fijan `protocol_fault` hasta reconfiguración.
Los contadores saturan para evitar que un desbordamiento parezca vacío.
El valor por defecto, cuatro bits, permite quince operaciones pendientes
por contador; debe dimensionarse para la interfaz concreta.

## Certificado de drenaje

`req_valid` y `req_tag` pertenecen al mismo dominio de reloj del monitor.
Un identificador nuevo se captura antes de contar dos observaciones
consecutivas de inventario vacío, sin ningún VALID activo y con
`source_fenced` confirmado. El ACK conserva ese identificador completo.
Cambiar el identificador o retirar la solicitud/fence invalida el ACK;
no se copia una etiqueta nueva sobre un ACK antiguo.

`source_fenced` es un contrato externo independiente: debe probar que
no llegarán nuevas transacciones desde el emisor, colas y cruces de reloj.
El monitor **no impone ese bloqueo**. Los participantes deben mantenerlo
hasta completar la coordinación. Un bus multibit de solicitud/ACK no
puede cruzar relojes conectándose directamente a este módulo.

## Banco y dependencias

Los trece casos cubren identidad de 64 bits, AW antes de W y viceversa,
B/R bloqueadas, VALID aún no aceptado, retirada del fence, varias
operaciones, aceptación/retiro simultáneos, respuestas inesperadas y
desbordamiento de cada contador. Usan dos bits para alcanzar los límites.
La síntesis aislada usa el valor predeterminado de cuatro bits.

Un caso conecta el monitor al secuenciador aprobado en **un mismo reloj**:
AW sola y B bloqueada deben mantener QUIESCE sin solicitar resets;
solo el drenaje completo con ACK fresco permite reset local. El fence
y los ACK de las otras acciones siguen siendo estímulos del banco.

Esto no certifica ráfagas AXI, inventario interno de XDMA, DDR, ni el
drenaje conjunto de host/source. Faltan monitores a ambos extremos,
transferencias atómicas de solicitud/ACK, aislamiento verificable y
coordinación con el guard, propietario y MIG reales. Una transacción
perdida ante PERST puede dejar el drenaje bloqueado; no se fuerza un ACK
para recuperar la sesión.

## Resultado y respaldo

Los trece casos pasan en XSIM y el verificador comprueba 21 eventos.
La tanda termina con código 0 en 82,358 s, incluida la síntesis OOC:
77 LUT y 80 FF, sin BRAM, URAM ni DSP. Los dos avisos corresponden al
entorno de compilación de XSIM y a la fuente de reloj OOC sin ubicación;
no hay avisos de síntesis ni de ejecución del banco.

Setup +8,685 ns y hold +0,087 ns son estimaciones previas a colocación.
Faltan retardos de 77 entradas y 78 salidas; no aprueban timing físico.
Respaldo verificado local/NFS: 65 archivos, 349.798 bytes, SHA-256
`359ceb6f6f1b2cc2e95ba203f3d40f0adf00def484e88f1e2b360a44e4e866c7`.
La adenda de revisión conserva el resultado original y clasifica avisos
y límites: `3b2ff1c535b18bcb62f2b6fb7028e17785b598273cdd663ac94acd6577d95a49`.

La tanda posterior [TAGGED_RESET_MAILBOX](TAGGED_RESET_MAILBOX.md) conecta
este monitor al secuenciador mediante un XPM real entre dos relojes y
repite AW incompleta/B bloqueada. Todavía no observa ambos extremos de
los CDC AXI existentes ni impone el aislamiento del emisor real.

## Avance: dos interfaces y reset local

El [banco con el convertidor AMD real](AXIL_PAIR_DRAIN.md) pasa ambos perfiles.
Distingue certificado externo de estados internos de handshake; la secuencia
de reset todav?a la conduce el banco y no autoriza recuperaci?n del endpoint.
