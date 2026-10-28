# Revocación de certificados AXI-Lite ante fallos

Actualización posterior: [composición real de admisión y certificado R4](AXIL_FAULT_COMPOSITION.md).
Resuelve el CDC de admisión; el nuevo banco previo al certificado descubre
una respuesta huérfana que debe contenerse antes del manager. Los resultados
R2/R3 siguientes se conservan como historial de estas revisiones.

## Problema y alcance

El controlador original conserva un certificado de sección después de que
aparece un fallo local. La tanda `20261005T190117Z-V4-ffn-supervised-shutdown-fault-probe-r2`
lo demuestra en admisión, H y D, con relojes de dispositivo de 5 y 37 ns.
Cambiar generación/tag impide reutilizar ese certificado, pero dejar la
petición original activa mantiene el ACK obsoleto.

R2 introduce la revocación. La revisión R3 de esta tanda es
`fpga/rtl/ntrpu_axil_fence_certificate_r3.sv`, con el retorno agregado
a host registrado antes del CDC.
Conserva el controlador original y no modifica la admisión, los inventarios,
el actuador ni los buzones sellados. Esto certifica una sección AXI-Lite;
las colas locales de rechazo y el productor/colas protegidas de XDMA siguen
fuera de ella. No autoriza reset global, reapertura ni reutilización NTRD.

## Implementación

Tres XPM CDC de tres etapas transportan los fallos retenidos de admisión,
H y D a gestión. No añaden registro en origen: su reloj puede detenerse
justo después de registrar el fallo. `fault_mgmt[2:0]` corresponde a
`{device_fault, host_fault, admission_fault}`.

Gestión retiene cada bit durante la vida de configuración y enmascara todos
los certificados expuestos. Bloquea solicitudes nuevas sin cortar el
handshake interno de un buzón pendiente. La retirada de una petición,
un cambio de identidad o el reset del datapath no borra esta alarma.
`cold_rst_n` corresponde exclusivamente a inicialización de configuración.
Un fallo no acciona el reset de transporte.

La revisión también conecta el fallo a cierre permanente de admisión:
fallos host directamente y fallo agregado mediante CDC de vuelta al host.
La calificación del cierre por fallo **antes** de solicitar una prueba
queda pendiente; los bancos de esta tanda inyectan después de certificar.

## Simulación con origen detenido

`20261005T190354Z-V4-ffn-supervised-shutdown-fault-revoke` completa tres
escenarios por perfil con Shutdown Manager y Clock Converter AMD reales.
Antes de inyectar fallos conserva la regresión completa de saturación,
datos, strobes, respuestas, cierre, identidad fresca y reset local.

| Origen del fallo | Dispositivo 5 ns | Dispositivo 37 ns |
| --- | ---: | ---: |
| Admisión | 24,8 ns | 28,8 ns |
| Inventario H | 20,8 ns | 25,8 ns |
| Inventario D | 21,4 ns | 28,4 ns |

Son latencias simuladas desde la bandera registrada hasta revocar el ACK,
con gestión a 10 ns y host a 7 ns; no cotas físicas de metastabilidad.
La prueba detiene el reloj que detectó el fallo, cambia generación/tag,
reanuda ambos relojes y retira artificialmente las banderas de origen.
Gestión conserva el error y las salidas permanecen bloqueadas, sin reset.

En admisión se fuerza una respuesta inesperada después del cierre. En H/D
se fuerza un handshake B huérfano sobre la interfaz observada; los
inventarios RTL reales detectan el underflow. Ambos VALID y READY se fuerzan
porque la IP inactiva puede negarse a aceptar una respuesta inesperada.
Es inyección adversa de protocolo, no tráfico legal producido por la IP.

## Diagnósticos conservados

La tanda `20261005T191031Z-V4-ffn-supervised-shutdown-fault-running`
repite los tres fallos en ambos perfiles manteniendo activos los relojes
de origen. Pasa con exactamente las mismas seis latencias. Solo este banco
nuevo adelanta la declaración de `fault_mgmt`, eliminando el aviso de orden;
el banco sellado anterior se conserva. Reproducir con
`tools/ntrpu_board_axil_fault_running.tcl` y revisar con
`tools/check_ntrpu_axil_fault_running.py`.

`20261005T190740Z-V4-ffn-supervised-shutdown-fault-mutants` ejecuta dos
copias privadas incorrectas contra el banco positivo sin modificar:

| Mutación RTL | Rechazo observado |
| --- | --- |
| Ignorar el fallo de admisión al cruzar a gestión | ACK sin revocar, 17.650 ns |
| Excluir los bits retenidos de `fault_mgmt` | Alarma perdida al retirar las banderas, 15.280 ns |

Ambas fallan en la aserción prevista. La copia de producción mantiene su
hash. XSIM expresa esos instantes enteros en ns; los campos opcionales de
tiempo del revisor original, que buscaba ps, se conservan vacíos y una
observación adicional registra la unidad real. Reproducir con
`tools/ntrpu_board_axil_fault_mutants.tcl`.

El primer banco `185759Z` confirma el problema de admisión, pero falla al
esperar que D acepte una respuesta no solicitada; se conserva como intento
fallido. R2 cambia el estímulo H/D descrito arriba.

La regresión corregida pasa funcionalmente, pero el recolector inicial
rechaza un quinto aviso: una conexión precede a la declaración explícita
de tres bits de `fault_mgmt` en el banco. La adenda verifica los bits 1/2/4,
ambos perfiles, todas las fuentes y bibliotecas; no hay truncamiento ni
aviso de ancho. Los otros cuatro avisos son los conocidos de entorno y
parámetro aplicado a `glbl`. No se repite simulación para corregir el revisor.

## Reproducción

Tras cargar Vivado 2026.1, ejecutar en una carpeta nueva:

```bash
vivado -mode batch -source tools/ntrpu_board_axil_fault_probe_r2.tcl \
  -tclargs /ruta/nueva/diagnostico
vivado -mode batch -source tools/ntrpu_board_axil_fault_revoke.tcl \
  -tclargs /ruta/nueva/revocacion
```

El revisor `tools/check_ntrpu_axil_fault_probe.py` separa la regresión
anterior al fallo de su traza adversa. Un marcador final o salida cero de
Vivado por sí solos no acreditan éxito. Fuentes, resultados y revisiones
viajan con cada respaldo local/NFS; se conservan también los intentos fallidos.


## Revisión R3 y síntesis

`20261005T191739Z-V4-ffn-supervised-shutdown-fault-r3` pasa doce casos en cuatro combinaciones
de reloj y conserva las latencias de la tabla. El revisor rechaza 36
alteraciones de evidencia. Una adenda revisa los ocho avisos reales: el
recolector esperaba avisos adicionales para el segundo genérico desconocido
de `glbl`. Los modos efectivos se verifican en las trazas; no se repite simulación.

R2 sintetiza 544 LUT/1.604 FF, pero presenta CDC-10 por lógica combinacional
antes del sincronizador de retorno y CDC-26 sobre la entrada de admisión.
R3 registra el agregado en gestión y consume **544 LUT y 1.605 FF**, sin
RAM, DSP, latches ni cajas negras. El inventario lógico contiene 570 primitivas
LUT; se conserva por separado del cómputo del informe de utilización.
Los dos controles negativos se repiten contra R3 en `20261005T192546Z-V4-ffn-supervised-shutdown-fault-mutants-r3`.

R3 elimina el CDC-10 y reconoce nueve CDC-3 de tres etapas. **CDC-26 permanece
abierto en `admission_fault`**, aunque el modelo OOC atribuye su origen a
`host_clk`. No hay LUTRAM en el inventario; hay que analizar este cruce con
el productor real antes de calificar la composición. No se suprime el aviso.

Se conservan diez false paths de XPM a primera etapa y tres max-delay de
10/7/7 ns, sin grupos de reloj generales ni waivers propios. La restricción
`set_input_delay -clock host_clk 0.000 [get_ports admission_fault]` representa
el dominio lógico de esa interfaz OOC; **no es un retardo medido de placa ni Tco**.

WNS 5.231 ns, WHS 0.003 ns y WPWS 3.225 ns
son estimaciones OOC. Faltan retardos en 216 entradas/85 salidas, colocación
de relojes, ruteo y equivalencia funcional del netlist. No se aprueba timing físico.
Los avisos XPM y OOC de síntesis se conservan y revisan.

Reproducir R3 con `tools/ntrpu_board_axil_fault_r3.tcl`,
`tools/ntrpu_board_axil_fault_synth_r3.tcl` y
`tools/ntrpu_board_axil_fault_mutants_r3.tcl`, siempre en directorios nuevos.

## Respaldos de la tanda

| Tanda | ID | SHA-256 del archivo |
| --- | --- | --- |
| fault-probe | `20261005T185759Z-V4-ffn-supervised-shutdown-fault-probe` | `93bfac93f5a43cc1e8484b316be7fcfab94fa78d9af15397209287c411023866` |
| fault-probe-r2 | `20261005T190117Z-V4-ffn-supervised-shutdown-fault-probe-r2` | `2f32e07936c19b08cf743556270e121b720d2e889d7a59abff743007cb106746` |
| fault-revoke | `20261005T190354Z-V4-ffn-supervised-shutdown-fault-revoke` | `60a847118c7aca5a0e16546b4d8c2482ed210e44398ac62f6115e83afdc70629` |
| fault-mutants | `20261005T190740Z-V4-ffn-supervised-shutdown-fault-mutants` | `f106b5c1318e652e00df3f9ebd0d6ed094f0421d8626c099ad3fdb9074fa7ae9` |
| fault-running | `20261005T191031Z-V4-ffn-supervised-shutdown-fault-running` | `a1d534bc031fb50dfab1bb39219227218f8c14888b81c3dd75d195811c5ae2c6` |
| fault-synth | `20261005T191301Z-V4-ffn-supervised-shutdown-fault-synth` | `178a3eb4151b1753e0fe68dc863ba492ae800d7e2fef27ed9382c4dfce88bb3a` |
| fault-r3 | `20261005T191739Z-V4-ffn-supervised-shutdown-fault-r3` | `2723364b10adb30a8e39b0d299e0b3caf0500d94d7ea9a8c41832be614770cc8` |
| fault-synth-r3 | `20261005T192200Z-V4-ffn-supervised-shutdown-fault-synth-r3` | `1749fdab1ba60685048e357026cd6166ffa039b88ea4b2a6d9856a8b41ada084` |
| fault-mutants-r3 | `20261005T192546Z-V4-ffn-supervised-shutdown-fault-mutants-r3` | `09c80fdc18150769a1048ac4fa15d89e327222c183412f6a50564841d14b2aef` |

Adenda funcional R3: `42e8aec037109cdf71e6c1682b221abf3a3a2b7a40270e3a5831e0075f89d429`.
Adenda comparativa de síntesis: `da61560a4a2c0486dfe23aca936184aabbf08ed0cc8265dbf037b8b1ed96e2f0`.
Todos tienen copias locales/NFS verificadas; los intentos fallidos se conservan.

## Siguiente tanda

Prioridad: composición admisión/controlador R3 y fallos antes de la prueba,
durante un buzón pendiente y con respuestas locales retenidas. Después
continúan ráfagas AXI4/IDs/WLAST/RLAST y colas protegidas de XDMA. Los
criterios V2–V9 de DIMM, placa, inferencia y ensayo físico permanecen abiertos.
