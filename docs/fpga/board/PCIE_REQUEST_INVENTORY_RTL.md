# Inventario RTL de solicitudes PCIe

R6 tiene [síntesis y netlist funcional revisados](PCIE_REQUEST_INVENTORY_SYNTHESIS.md),
pero todavía falla timing a 4 ns: sus métricas son idénticas a R5. Su netlist
pasó 67 escenarios binarios y coincide con los estados emitidos por RTL.
R6 y R7 pasaron los 68 escenarios RTL con 22.339 estados idénticos.
R7 pasó además los 67 escenarios de netlist, pero no mejora el peor
camino: WNS −11,267 ns, con 46.617 LUT y 9.531 FF.
Los resultados históricos siguientes
se conservan; el informe enlazado distingue cada revisión y su alcance.

## Revisión de arquitectura del 2026-10-07

La primera síntesis, `20261006T234952Z`, agotó 900 s durante optimización:
**no generó netlist, utilización mapeada ni timing**. El log intermedio
registró 28.976 multiplexores de 13 bits y 28.944 de 12 bits; son estadísticas
RTL, no LUT físicas. Además, las restricciones estaban después de
`synth_design` y no llegaron a aplicarse. El diagnóstico y respaldo se
conservan, con SHA-256 de archivo
`b689ecc709916f5db6c335502ac59ad00e646add1c9eae4749d23ceeb1003a0f`.

`fpga/rtl/ntrpu_pcie_request_inventory_r2.sv` sustituye las mutaciones
sucesivas de toda la tabla por **cuatro registros de actualización RC y dos
RQ**. Los accesos RC reciben la actualización anterior más reciente del
mismo tag, y las asignaciones RQ se aplican al final. Se mantiene el perfil
completo y la liberación del tag en el descriptor terminal. El SHA-256 es
`7084fc24824322ea4a3ff4c97abb8b73bbdfcf759231b8c3ce9f1cbe435821c3`.

| Tanda | Revisión funcional | Respaldo SHA-256 |
| --- | --- | --- |
| `20261007T131318Z`, R5 | 21.903 ciclos/39 escenarios originales; 35,543 s | `c8b20849d74fabd0d45db75be02c4c89419eb91fab92a35c2b8ed4a455fe23ca` |
| `20261007T131706Z`, R6 | 21.942 ciclos/43 escenarios; 35,617 s | `8158ef9b81526d32f88ce8dd083b56e84d06955183411d4e4d9ba78f36aa08a0` |

Ambas pasaron revisión por fila contra la referencia regenerada, incluidos
los 22 fallos, 3.778 ciclos nativos reproducidos y máximos 256/257/1.052.672.
Cada revisión rechazó nueve corrupciones de evidencia. R6 agrega cuatro
casos: cuatro splits del mismo tag, persistencia de su última actualización,
cuatro actualizaciones distintas simultáneas con RQ2, y continuación más
tres splits con reutilización y otra solicitud RQ en el mismo ciclo.
La revisión independiente de R6 tiene SHA-256
`26ce79175db6187338db37eb4a14c528753319c5fe6c167c18c1f287422fde20`.

La síntesis `20261007T132006Z` terminó en 633,921 s: 63.050 LUT y
9.524 registros, sin latches. El XDC de entrada fue rechazado parcialmente;
una revisión posterior sobre copia del DCP cubrió todos los I/O.
El WNS resultante es −17,501 ns a 4 ns: el cierre temporal sigue pendiente.

Se preparó el banco binario `pcie_request_inventory_binary_r2`: 21.933
ciclos/42 escenarios. Conserva 21 fallos binarios y excluye explícitamente
solo la detección X/Z, imposible como entrada digital física. El nuevo
banco de netlist espera 200 ns con reset frío y tráfico inactivo para el
arranque UNISIM, sin forzar señales protegidas. La tanda `20261007T134048Z`
pasó revisión funcional independiente de todos sus ciclos. No lleva SDF
y no acredita cumplimiento temporal ni funcionamiento de placa.

## Resultado funcional

`fpga/rtl/ntrpu_pcie_request_inventory.sv` pasó XSIM y revisión
independiente el 2026-10-06. Su SHA-256 permanece idéntico en las cuatro
tandas iniciales:
`04938380945a7b668db9182da191f656f92ca1b18e26c5b40f39372831ac9665`.

| Tanda | Resultado | Respaldo SHA-256 |
| --- | --- | --- |
| `20261006T233659Z`, R1 | Diagnóstico Tcl anterior a compilación | `42ab59d6e74d24f669ed391c5b2ab84ecbaaa97b6b6411ff0601d5515cfb9434` |
| `20261006T233949Z`, R2 | 4.314 filas correctas; banco falla al leer después de EOF | `efe3a868ec1e77dc81dd351ffa4e6e3dce96d2d91f7702a08aa391919382f193` |
| `20261006T234402Z`, R3 | 5.065 ciclos, 38 escenarios; revisión aprobada | `50b05f4ab858c89c3bd5a778e8fe35008ecaf8410eb1ddb8b61e6e25f9e0e10a` |
| `20261006T234656Z`, R4 | 21.903 ciclos, 39 escenarios; revisión aprobada | `c4098ec10fd5903e64b82838768708798fa01136e337c0d0e50c6ef0c5bc06b4` |

Todos los paquetes están verificados en D:/NFS. R1 se corrigió mediante
una nueva revisión Tcl que usa `set_property -dict`. R2 conserva el fallo;
el banco nuevo termina solo cuando `$fscanf` devuelve -1 y `$feof` está
activo, y sigue rechazando filas incompletas. No se alteraron retrospectivamente
fuentes, registros ni resultados de esos intentos.

El informe independiente de R4 tiene SHA-256
`c634f816240930a48ad2de8860876e8a1cb41f11795577ebfa6f9af806cc7bf6`.
Regenera los vectores, comprueba cada fila y los 256 bits del inventario,
y rechaza nueve corrupciones del resultado. XSIM tardó 33,794 s, incluida
la compilación; son medidas de ejecución del banco, no de una placa.

## Arquitectura y alcance

El monitor observa un PF del perfil generado: cliente asigna tags de ocho
bits, ARI deshabilitado, datos de 512 bits, alineación DWORD, RQ2/RC4.
Procesa Memory Read/Write; los tipos no soportados producen fallo.
`profile_valid` debe acreditar el Requester ID y la máscara de atributos
efectivos, con una frontera pública inicialmente quieta. Los cambios
posteriores de perfil, reset o enlace dejan fallo retenido.

La tabla contiene 256 tags con dirección siguiente, bytes restantes,
TC y atributos. RC retira antes de admitir RQ del mismo ciclo. Un
descriptor terminal libera el tag, mientras el contexto del payload
permanece hasta EOP. Por eso pueden coexistir **257 lecturas** y hasta
**1.052.672 bytes** pendientes. Solo `cold_rst_n` borra la observación;
un reset operativo no borra obligaciones. El módulo no tiene salida ACK.

`inventory_empty` exige perfil válido, enlace activo, ausencia de errores,
tags, payloads parciales y beats ofrecidos. No acredita recepción host de
escrituras posted, colas internas del core ni cierre del productor.

## Cobertura ejecutada

- 256 tags simultáneos con direcciones, TC y atributos distintos;
  respuestas en orden inverso, dos RQ y cuatro RC por beat.
- Reutilización de tag en el ciclo del descriptor terminal, conservando
  el payload previo; 257 lecturas de 4 KiB pendientes simultáneamente.
- Splits del mismo tag en un beat, continuación más tres completions,
  solicitudes read/write compartiendo beat y transferencias de 4 KiB.
- Byte enables parciales, huecos, longitud cero y dirección de 64 bits alta.
- Backpressure, discontinuación tardía, timeout y 46 escenarios de fallo
  persistente. Las comprobaciones de X/Z son de simulación; no equivalen
  a detectar estados analógicos desconocidos en hardware.
- Reproducción de **3.778 ciclos** de la captura nativa, sin volver a
  ejecutar el endpoint AMD dentro de este banco RTL.

Vectores actuales: `fpga/tb/data/pcie_request_inventory_r7.{txt,json}`.
Generador: `tools/generate_ntrpu_pcie_inventory_vectors_r7.py`.
Banco: `fpga/tb/tb_ntrpu_pcie_request_inventory_r2.sv`.
Entrada Vivado para RTL R7: `tools/ntrpu_board_pcie_request_inventory_r15.tcl`.
R6 y R7 pasaron 22.339 ciclos/68 escenarios, incluidos cuatro retiros
con dos reasignaciones bajo saturación, duplicados en el mismo beat
y siete casos de longitud inválida, además de 15 pares de punteros SOP
inválidos. Todos sus estados emitidos coinciden.

## Dependencias pendientes

R4 usa 58.225 LUT y reduce WNS a −11,964 ns, pero sigue fallando
setup y hold a 4 ns. Su netlist pasó 45 escenarios binarios.
R5 limita los anchos temporales: 46.359 LUT y 9.524 FF; el timing reportado
sigue fallando. Su EDIF final es acíclico, pese a dos avisos intermedios
de optimización. El replay funcional pasó y el DCP reabierto confirma
cero excepciones temporales. R6 conserva esos recursos y timing; su
netlist pasó 67 escenarios. R7 pasó los bancos RTL/68 y netlist/67;
el retiming de la guarda de rango no mejoró el peor camino. Sigue
pendiente una arquitectura por etapas con inventario de eventos en vuelo.
Los retardos I/O de 0 ns son de caracterización y no restricciones
físicas de la tarjeta. Falta cerrar timing y validar la siguiente arquitectura,
distinguiendo estímulos binarios de las comprobaciones
X/Z exclusivas de simulación. Después faltan la integración nativa del
RTL y una conexión sintetizable soportada dentro
del wrapper XDMA. Mantener el join coherente de siete contratos y la
recepción host independientes de este inventario.

[Captura nativa y perfil](XDMA_PCIE_INVENTORY_PLAN.md) ·
[Cola completa, DDR/512 GiB e inferencia](VIVADO_QUEUE.md).
