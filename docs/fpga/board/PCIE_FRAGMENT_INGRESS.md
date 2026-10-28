# Captura y ensamblado de fragmentos NTRPUF01

Estado del 2026-10-05 UTC: el bloque `ntrpu_pcie_fragment_ingress.sv` aprueba RTL,
netlist sintetizada e integración nativa XDMA. El
[formato y su mapeo de 64 bytes](PCIE_INGRESS_FRAGMENTS.md) tienen evidencia
separada; el [ingreso DWORD](PCIE_INGRESS_PLAN.md) conserva su RTL y contrato.

## Contrato de transporte y reserva

Una reserva autoriza una línea de 64 bytes, con ventana, destino, época y
etiqueta de 64 bits. Dos paquetes completos aportan sus mitades de 32 bytes.
AW y W se capturan independientemente, con una escritura pendiente. Mientras
B espera READY no se admite otra pareja. Una ráfaga ajena al perfil se consume
hasta su WLAST real y recibe un único B; no publica bytes. La ausencia de
WLAST bloquea la reutilización, incluso tras más de 256 beats.

El ensamblador publica únicamente un bitmap completo y espera que se drenen
los B de ingreso. Mantiene estables descriptor y datos mientras LINE espera
READY. Un B de ingreso confirma staging; el resultado del trabajo requiere
además DONE del destino con identidad correcta y la barrera externa de
aislamiento correspondiente a esa época. Un DONE ajeno no completa la reserva.
STOP conserva las escrituras y líneas ya admitidas hasta drenarlas.

ARM exige transporte vacío, ausencia de llegadas, retiro del resultado
anterior, direcciones alineadas, una época estrictamente mayor y barrera
válida. El reset sigue siendo **frío y coordinado**. Este bloque no demuestra
el vaciado de PCIe ni permite eliminar la barrera externa.

## Errores y repetición

- Formato, magic, rango, ventana, época o etiqueta incorrectos rechazan el
  paquete sin alterar datos, bitmap ni resultado de la reserva vigente.
- Una repetición idéntica, anterior a la publicación, es idempotente. Una
  repetición conflictiva con formato e identidad válidos invalida el trabajo.
- Un paquete válido de esa reserva después de publicar la línea invalida un
  trabajo aún no reportado. No retracta LINE ni revierte un prefijo escrito.
  Después de reportar, se rechaza sin cambiar el resultado ya publicado.
- Un error del destino o STOP invalida el trabajo; la finalización espera
  igualmente las obligaciones de transporte y destino pendientes.

`rejected_frames` y `rejected_identity` son contadores globales de 32 bits,
con saturación implementada; ARM no los borra. El segundo cuenta rechazos
exclusivamente de época/etiqueta sobre un formato y ventana válidos.
`last_reject_code` distingue STOP=1, formato=2, rango/reserva=3, conflicto=4,
destino=5, escritura tardía=6 e identidad=7. `last_reject_faults` conserva los
bits del decodificador. En una ráfaga rechazada se retiene el primer motivo.
El límite de saturación de los contadores todavía no se ejercitó.

## Evidencia del bloque aislado

Tanda `20261004T233833Z-V7-ffn-supervised-fragment-ingress`: 20 escenarios con
semillas 1, 7 y 42, tanto en RTL como en netlist. La reconstrucción independiente
por byte y transacción coincide entre ambas versiones. Cada ejecución tiene
58 AW/B, 320 W, 23 tramas rechazadas —cinco por identidad—, 20 resultados y
16 líneas publicadas, equivalentes a 1.024 bytes. Incluye drenaje de 261 beats,
rearme sobre la misma ventana, identidades con bits altos, duplicados, huecos,
STOP, respuestas ajenas y tráfico posterior al retiro.

Con semilla 1 hay 10 ciclos de stall AW, 10 W, 237 B, 88 LINE y 80 RESULT;
termina en 8,975 µs simulados. Diez mutaciones RTL fallan en sus aserciones
previstas y 52 alteraciones de evidencia son rechazadas por el revisor.
La tanda completa consume 292,873 s de herramienta.

Síntesis `xcvu5p-flva2104-2-i`: 897 LUT, 1.600 FF, sin BRAM, URAM, DSP,
latches ni cajas negras. Con reloj de 5 ns, WNS estimado 1,960 ns y WHS
0,066 ns. **No es cierre temporal físico:** hay 1.119 entradas y 1.056 salidas
sin retardos presupuestados; falta `HD.CLK_SRC` para estimar skew. El DRC OOC
no informa comprobaciones fallidas, pero limita las de conectividad. Las doce
elaboraciones conservan el aviso de entorno `LIBRARY_PATH`; los positivos no
tienen avisos de compilación ni simulación.

Respaldo local/NFS verificado a las 23:45:44 UTC: 423 archivos, 5.026.260 bytes,
SHA-256 `bd75e487cfd5754ab88519e6077774ba681ed57b6b0b8f6c5e835a714b3aad6e`.

## Integración nativa

`20261004T234908Z-V7-ffn-supervised-fragment-ingress-link` aprueba el XDMA
nativo conectado al mismo RTL, con cuatro reservas y 16 escrituras: nueve
aceptadas y siete rechazadas, tres exclusivamente por identidad. Hay cuatro
líneas, cuatro confirmaciones de destino, cuatro resultados y 12 lecturas
correctas de las tres líneas finales. La cuarta reserva reutiliza la ventana
de la segunda; los paquetes antiguos no cambian su bitmap ni su resultado.

La revisión reconstruye 95 señales y coteja todos los paquetes y eventos de
ARM, publicación, DONE y retiro. Comprueba contadores por ciclo, estabilidad
bajo READY bajo, drenaje de B antes de publicar y DONE antes del resultado.
Las 173 alteraciones de evidencia son rechazadas. Se cotejan 76 inputs
compilados y 134 archivos AMD intactos. Los 139 avisos de herramienta y uno
de `unique case` en tiempo cero mantienen el conjunto de la tanda nativa
anterior: únicamente cambian los ordinales de temporales generados por XSIM,
con líneas, índices seleccionados y multiplicidades conservados.

El banco dura 452,954 s de herramienta y termina en 81,374 µs simulados.
Se observan 12 ciclos de stall AR, 19 LINE y 16 RESULT; ninguno en
RQ/RC/AW/W/B/R, ARM ni DONE. Desde el último beat RQ hasta la decisión pasan
196–200 ns; decisión a B, 4 ns; último B aceptado a publicación, 20–32 ns.
El destino BFM confirma 44 ns después de LINE. Estas cifras describen esta
simulación PIPE y su banco, no latencias físicas ni caudal de inferencia.

La captura y el ensamblado son RTL de producción. **ARM, aislamiento, destino
y lectura de memoria siguen siendo BFMs**; no hay control por BAR0, CDC ni
MIG en esta tanda. Respaldo local/NFS verificado el 2026-10-05 a las 00:00:05
UTC: 845 archivos, 107.499.829 bytes, SHA-256
`362eeb54e9950b425c52718598c2e4d1d61439cbbfaf49895d3388f6bd83e9ec`.
El certificado está bajo el ID de tanda en `artifacts/ntrpu-board-vivado/`.

## Puertas restantes

Después: componer el controlador de reservas, ambos dominios de reloj y el
receptor DDR; probar STOP y errores con datos en tránsito. Siguen pendientes
el transporte real desde host, múltiples productores/reservas, reset con
tráfico, DIMM exacto, cuatro canales, implementación integrada y placa.

El control DWORD conserva `NTRI`, versión `0x00010000` y su capacidad de
barrera externa. La [variante de fragmentos NTRF](PCIE_FRAGMENT_CONTROL.md)
ya aprueba el banco de control aislado, con perfil explícito y captura
conjunta de telemetría de rechazo. Su composición con el receptor y el
guardián de capacidad se califica por separado, conservando la autorización
de destino y el retiro y rearme coordinados por época/etiqueta.
