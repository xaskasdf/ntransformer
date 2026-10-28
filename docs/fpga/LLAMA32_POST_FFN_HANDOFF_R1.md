# Propiedad post-O → FFN y residual

Actualizado el 2026-10-16, 01:23 UTC. **Control aprobado en XSim y síntesis
OOC**, después de la [composición FFN/residual real](LLAMA32_FFN_RESIDUAL_REAL_R1.md).
El banco usa productores de respuestas controlados; no acredita todavía
la conexión aritmética con atención viva.

## Resultado Vivado

Tanda `20261016T011916Z-V7-llama32-post-ffn-handoff-vivado-r2`:
**1.566 escenarios** en tres semillas y **309 controles de traza** rechazados.
Síntesis: 2.480 LUT, 3.106 registros, sin BRAM ni DSP. A 5 ns, WNS
+1,571 ns y WHS +0,077 ns; es síntesis OOC, no cierre temporal ruteado.
Sin cajas negras, latches ni bucles combinacionales detectados.

Archivo respaldado D:/NFS:
`826769e336fb5a731619c152ff84ced905d1494efbdd63c946485e9d912806c9`.
Restauración completa verificada: 67 archivos de resultado y 27 fuentes.
Revisión independiente:
`a9355fbf1a18f2611f2c88eaabbf40586a31a63391a951d4bddbda445e58b5bf`.
R2 del coordinador conserva el paquete fuente R1 y sus estímulos; sólo
actualiza la dependencia hacia el resultado FFN revisado. La composición
aritmética atención/FFN inició su propia tanda a las 01:23 UTC, sin
desplazar la captura DDR posterior a BRAM.

## Contrato de integración

[`ntrpu_post_ffn_handoff_r2.sv`](../../fpga/rtl/ntrpu_post_ffn_handoff_r2.sv)
retiene el par post-O: ABI1 de 4.096 bytes y residual FP32 de 8.192 bytes.
La entrada ABI1 pasa directamente al hijo FFN/residual, sin recuantización.
El resultado final sólo se publica con identidad completa de 64 bits,
metadatos coherentes y estados correctos. Ambos productores conservan
sus respuestas hasta el mismo ACK público.

El padre debe capturar ocho asignaciones reales, en este orden: caché KV,
pesos O, activación O, salida O, entrada post-O, pesos de normalización,
residual y ABI1. El controlador comprueba cada uno de los cinco destinos
contra esas regiones completas: gate/up, 65.536 bytes cada uno; ABI1
intermedio, 16.384; salida FFN y residual final, 8.192 cada uno. Comprueba
alineación, extensión y desbordamiento con sumas de 65 bits. El hijo
valida las relaciones internas de sus matrices y espacios de trabajo.

Las regiones contiguas son válidas. Cambios de época, cancelaciones o
mutaciones del productor invalidan el resultado de forma persistente.
Una petición ya ofrecida conserva sus campos bajo contrapresión; una
respuesta con identidad ajena nunca recibe ACK.

## Evidencia nativa

Banco [`tb_ntrpu_post_ffn_handoff_r4.sv`](../../fpga/tb/tb_ntrpu_post_ffn_handoff_r4.sv),
semillas 1/7/42, RTL R2 sin modificaciones entre las tandas finales:

| Medida | Resultado |
| --- | --- |
| Trabajos / completions | 1.566 / 1.554 |
| Solapamientos prohibidos / contigüidades válidas | 240 / 120 |
| Descriptores ofrecidos / aceptados sin revocación | 975 / 912 |
| ACK al productor / al hijo | 1.545 / 1.071 |
| Alteraciones de trazas rechazadas | 309 |
| Mutantes compilados detectados por aserción directa | 41 |
| Ciclos del banco por semilla | 26.458 / 26.389 / 26.451 |

El [revisor independiente](../../tools/check_ntrpu_post_ffn_handoff_r1.py)
comprueba las asignaciones capturadas, metadatos, orden y liberación.
Los ciclos anteriores son del banco de control, no latencia de inferencia.

R1 no compiló en Icarus por declaraciones ANSI de arrays agrupadas; R2
explicita cada puerto. Banco R3 detectaba una falsa denegación mediante
espera acotada; R4 la detecta inmediatamente. Se conservan todos los
intentos. Ningún timeout cuenta como mutante detectado.

| Evidencia | SHA-256 |
| --- | --- |
| RTL R2 | `dc1f8525d7277095178040e5023cca7a70bf0f661b1a4c06148a4dc9c86d8ce5` |
| Banco R4 | `1b7d51b42185a70f15eaa263474e1a14f8ed87deab848ffc8ba07715feb894c2` |
| Nativo `20261015T185438Z` | `a856b8ec79efcd65dd37f27185e87efcd48daded4a6f1e2e41f1e61b3404277f` |
| Mutantes `20261015T185448Z` | `fd8a80dcb8340095126c287a930761b46f6bc0c48c60700b13c4ed5ce0cacab8` |
| Revisión nativa | `5b3a363f515c8ce36dde3609f7ea0870f5f8186c9d1c8b12b940646c81fbcc35` |
| Paquete fuente Vivado | `0bcd47c6c8842237ae235cad48e4483b6c9608e2ec3f02deceafbc018528a59f` |
| Configuración Vivado | `25921281c37e0c93a41b52454a9cf84eba4473b877f513be64839fa18f26a350` |

## Cola y trabajo pendiente

El paquete tiene 27 miembros y dos fuentes compiladas. El coordinador
`queue_llama32_post_ffn_handoff_vivado_r1.py` exige revisión y respaldo
verificado de FFN/residual real antes de ocupar el único slot Vivado.
Ejecutará síntesis OOC a 5 ns y tres semillas XSim, dos CPU y límite de
1.800 segundos. Los respaldos nativos y el paquete están verificados en
D: y NFS; el resultado Vivado permanece pendiente.

Siguiente integración: envolver `ntrpu_attention_post_path_r2` y
`ntrpu_llama32_ffn_residual_r3`, capturar las ocho asignaciones del
productor, conservar la exclusión de mantenimiento KV hasta el ACK
final y arbitrar AXI sin liberar propietarios prematuramente. Después,
probar la cadena con pesos originales y datos realmente producidos.
También faltan cierre físico, DIMM exacto, DDR4/PCIe, inferencia completa
y varias tarjetas. Las asignaciones suministradas son un contrato del
padre, no una autenticación de memoria.

Avance 19:27 UTC: el [wrapper con atenci?n y FFN reales](LLAMA32_ATTENTION_FFN_CONTROL_R1.md)
ya est? conectado y pas? el banco nativo de control y elaboraci?n de
interfaces. Su validaci?n aritm?tica integrada sigue en preparaci?n.
