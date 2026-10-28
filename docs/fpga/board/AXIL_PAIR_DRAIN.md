# Drenaje externo AXI-Lite y reset coordinado

## Resultado y alcance

El 2026-10-05, `20261005T164228Z-V4-ffn-supervised-axil-pair-r2`
pas? seis casos en cada uno de dos perfiles, en 63,885 s. Instancia el
convertidor AMD AXI4-Lite real, dos inventarios retenidos y dos buzones XPM.
Certifica las interfaces externas bajo un contrato de aislamiento; no acredita
el drenaje completo de XDMA ni recuperaci?n de sesi?n, DDR o placa.

## Contrato del componente

`fpga/rtl/ntrpu_axil_pair_drain.sv` pide primero un certificado al lado host.
Solo despu?s solicita uno nuevo al dispositivo. Cada respuesta conserva generaci?n
y etiqueta de 64 bits. AW, W y AR se contabilizan separadamente frente a B y R;
una mitad incompleta o una respuesta retenida impide el certificado.

El productor de `host_source_fenced` debe mantener el aislamiento hasta terminar
el mantenimiento. El componente no bloquea VALID/READY ni crea ese aislamiento.
Se debe retirar la petici?n antes de permitir tr?fico nuevo. Sus contadores se
conservan durante resets de transporte; solo se inicializan al configurar FPGA.
Los fallos locales a?n requieren integraci?n con el coordinador.

## Evidencia de simulaci?n

Reloj de gesti?n: 10 ns; host: 7 ns; dispositivo: 5 ns y 37 ns. El perfil lento
es estr?s de CDC, no una frecuencia aprobada para la placa.

Cada perfil compara cuatro escrituras y cuatro lecturas en ambos extremos:
AW antes/despu?s de W, escritura con m?scaras, B/R bloqueadas, AR pendiente,
ausencia de aislamiento, cambio de etiqueta y tr?fico posterior al reset.
Conserva exactamente datos, m?scaras y los cuatro valores de RRESP.

Con el host detenido despu?s de emitir su prueba, el perfil lento obtiene
inventarios externos vac?os mientras el estado interno AMD es `0x0084a70000`:
los canales W y B a?n completan su handshake interno. Esta observaci?n no es
un defecto de AMD ni una certificaci?n de vaciado interno. El banco solicita
entonces reset solapado y usa contadores RTL de 64 flancos y sincronizadores
reales; no cuenta tiempo transcurrido como flancos del reloj detenido.

| Perfil dispositivo | Flancos host/dispositivo al confirmar hold | Estado interno antes de liberar |
| --- | --- | --- |
| 5 ns | 71 / 716 | cero |
| 37 ns | 70 / 96 | cero |

En esta tanda hist?rica la secuencia la conduce el banco. El actuador RTL
se valida posteriormente en la revisi?n enlazada al final.
El criterio de solapamiento y sincronizaci?n sigue [AMD PG059](https://docs.amd.com/r/en-US/pg059-axi-interconnect/Resets).
No hubo s?ntesis en esta tanda.

## Revisi?n y respaldo

El intento `20261005T164021Z-V4-ffn-supervised-axil-pair` fall? por muestrear
el ACK combinacional en el mismo delta que retiraba la petici?n. R2 a?ade
1 ps ?nicamente al banco; conserva RTL y verificador. Ambos intentos quedan
respaldados. El revisor terminal R2 acepta la abreviatura que XSIM imprime al
compilar, contrastada con biblioteca, wrapper y fuente instalada; conserva
el revisor original fallido.

Los cuatro avisos son dos de entorno LIBRARY_PATH y dos por el par?metro de
perfil aplicado tambi?n a `glbl`. Se verifican los flancos de ambos perfiles;
no hay avisos de ejecuci?n. El verificador rechaza diez alteraciones de evidencia.

- R1: 101 archivos; SHA-256 `8365a7affd0f068654c183dff561ed4c955d2291493d2a49aefca943cd9565cd`.
- R2: 102 archivos; SHA-256 `83e3027fef07cc91ac4e349bb7348dd86225117d946c2028fda080038ef25ee2`.
- Adenda revisada: `517a34d90778785d6902b56873d31c29e20125f358a5abc9e6f5c9606b87b402`.

Copias verificadas en `artifacts/ntrpu-board-vivado/` y NFS. Para reproducir:
`vivado -mode batch -source tools/ntrpu_board_axil_pair_drain_r2.tcl -tclargs NUEVA_CARPETA`.
Consultar [buz?n](TAGGED_RESET_MAILBOX.md), [inventario](AXIL_DRAIN_MONITOR.md)
y [coordinaci?n pendiente](ENDPOINT_RESET_COORDINATION.md).

## Avance: actuador local en RTL

El [actuador de reset](LOCAL_RESET_ACTUATOR.md) pasa sus ocho casos y dos
perfiles con el convertidor AMD real. Implementa conteo y liberaci?n autom?tica;
la uni?n completa con el secuenciador y el drenaje del endpoint sigue pendiente.

## Avance: aislamiento con IP AMD

El [Shutdown Manager real](AXIL_SHUTDOWN_MANAGER.md) sustituye la entrada
de aislamiento ideal en una nueva tanda. Dos perfiles pasan, conservando
un inventario adicional arriba: cierre hacia memoria no equivale a vaciado
del host o XDMA. Saturaci?n y coordinaci?n completa siguen pendientes.
