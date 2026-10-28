# Certificados de reset entre dominios de reloj

Prototipo V4/V7: `fpga/rtl/ntrpu_tagged_reset_mailbox.sv`.
Tanda `20261005T161235Z-V4-ffn-supervised-tagged-cdc`, aprobada para
funcionamiento aislado y respaldada localmente y en NFS.
Conecta un dominio de gestión con un participante dentro de la FPGA.
No constituye transporte entre tarjetas ni coordinación completa de XDMA.

## Contrato e identidad

Cada solicitud contiene generación de configuración de 64 bits,
identificador de 64 bits y fase de tres bits. El llamador debe asignar
una generación adecuada y no reutilizar identificadores dentro de ella.
El secuenciador existente conserva esa unicidad de tags hasta agotamiento;
la generación se incorpora en su adaptador. El buzón no genera identidades
ni mantiene una lista de todos los mensajes históricos.

Solo existe una transferencia pendiente. Sus campos permanecen estables
durante el handshake. El receptor presenta la solicitud en su reloj y
solo acepta un ACK con los tres campos coincidentes, después de observar
ACK bajo para esa petición. Un ACK que ya estaba alto no la aprueba.
El retorno confirma la identidad capturada originalmente, sin copiarle
la etiqueta de una solicitud posterior.

La implementación usa `XPM_CDC_HANDSHAKE`, con tres etapas por dirección,
131 bits y comprobaciones de simulación activadas. Se mantiene el ciclo
completo de aceptación y retirada antes de otro envío, conforme al
[contrato de AMD](https://docs.amd.com/r/en-US/pg382-xpm-cdc-generator/XPM_CDC_HANDSHAKE).
Las restricciones XPM requieren revisión junto con las de integración;
la simulación digital no mide metastabilidad ni acredita CDC físico.

## Cancelación, relojes y vida de configuración

Retirar o reemplazar una solicitud invalida su certificado de gestión,
pero **no retracta la petición ya entregada**. Su handshake debe terminar;
después se envía la solicitud vigente. Varias sustituciones se condensan
en la última: este puerto no es una cola de comandos.

Un participante sin reloj o sin ACK conserva la transferencia pendiente.
No hay timeout que borre inventario ni puerto de reset unilateral.
Los registros y XPM se inicializan con la configuración completa de FPGA.
No usar el buzón para ejecutar órdenes de reset cancelables: transporta
peticiones de comprobación de acciones controladas por el coordinador.

Una identidad completada no se vuelve a emitir mientras sea la última;
retirar y volver a presentar esa misma identidad tampoco restituye su
ACK. La prohibición de reutilizar otras identidades sigue siendo contrato
del emisor, no un filtro histórico implementado por el buzón.

## Banco y alcance

El banco usa relojes de 10 ns y 7 ns, detiene ambos por separado y prueba
doce escenarios: retrasos, cada campo erróneo, ACK previamente alto,
cancelación, reemplazos, cambios de generación y repetición de identidad.
Los eventos registran entrega al participante, aceptación del certificado
y observación de su retorno en gestión.

El último caso conecta el [monitor AXI-Lite](AXIL_DRAIN_MONITOR.md), este
buzón XPM y el [secuenciador](ENDPOINT_RESET_SEQUENCER.md). AW incompleta
y B bloqueada deben impedir reset local hasta recibir el certificado
de drenaje a través del reloj distinto. El aislamiento del emisor y la
invalidación del propietario siguen siendo estímulos controlados.
No se aprueban aún drenaje de todos los CDC/ráfagas/DMA, retorno completo
del endpoint, reutilización NTRD ni restricciones físicas de placa.

## Resultado, restricciones y respaldo

Los doce casos pasan en 99,279 s, incluida síntesis y reportes. Se
entregan trece peticiones y se aceptan trece certificados en el peer;
diez llegan a gestión como respuestas vigentes y tres se descartan por
cancelación/reemplazo. El retorno desde la aceptación en el peer hasta
el ACK de gestión mide 31,2–38,7 ns simulados con relojes activos;
el caso con gestión detenida mide 244,7 ns. No son medidas físicas ni
latencia completa del drenaje.

La síntesis OOC reporta 212 LUT y 608 FF, sin BRAM, URAM ni DSP. Los
siete avisos se revisan: entorno de XSIM, fusión de dos registros XPM
idénticos, dos puertos de reloj sin carga por `SRC_INPUT_REG=0`, fuentes
OOC de ambos relojes y advertencia sobre relaciones entre relojes.

`report_cdc` muestra dos sincronizadores de tres etapas con `ASYNC_REG`.
El bus de 131 bits está cubierto por la **exención CDC-15 de AMD**, aplicada
solo dentro de `transfer`; su ausencia del informe no significa que
se haya comprobado sin exenciones. La restricción efectiva es
`set_max_delay -datapath_only 7 ns`; las otras dos excepciones alcanzan
solo la primera etapa de cada sincronizador de control. No se añadieron
grupos de reloj asíncronos que sustituyan ese límite.

Los márgenes de síntesis son +8,374 ns en gestión, +5,472 ns en el peer
y +6,749 ns bajo el límite del bus. Faltan retardos de 264 entradas y
329 salidas, ubicación de relojes y ruteo. Estos informes no cierran
implementación física. La biblioteca XPM instalada, su configuración de
simulación y los Tcl de restricciones quedan identificados por hashes.

Respaldo: 72 archivos, 572.408 bytes, SHA-256
`260bf16cca3f846cd61f57c472cf026f0b975c34e5fd6e4f75336ee881d334c6`.
Adenda de revisión independiente:
`98ea11bcae07583f5749af50969857c777cedfeef3211f27f95e5ca1603b55f9`.
