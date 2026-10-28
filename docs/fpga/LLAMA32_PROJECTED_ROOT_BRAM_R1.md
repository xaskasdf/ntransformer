# Lectura privada de BRAM: equivalencia y experimento físico R1

Respaldo cerrado 2026-10-16 12:24 UTC: **447 archivos de resultados y
165 fuentes restaurados y verificados**. Archivo de 335.081.432 bytes,
SHA-256 `af89519c5abd8258bcdbf289d41eac727194767ad12b035eac69c31f2a1164b6`.
Recibos `VERIFIED.json` y `RESTORATION_R1.json` en
`artifacts/ntrpu-board-vivado/20261016T101133Z-V7-projected-root-bram-vivado-r3/`;
copia NFS verificada. DDR R1 inició su turno después del cierre de la tanda.

Actualización 2026-10-16 12:16 UTC: **aritmética y medición física aprobadas;
respaldo completo en transferencia**. Vivado terminó con código cero a las
12:14:59 UTC. La revisión independiente aprobó las ocho condiciones físicas,
69 controles aritméticos, 14 controles de medición y cinco de buffers.
DDR conserva el siguiente turno tras completar el respaldo.

| Medición del bloque aislado | Resultado |
| --- | --- |
| Reloj / setup / hold | 5 ns / +0,056 ns / +0,010 ns |
| Redes ruteables completas / errores | 87.158 de 87.158 / 0 |
| Filas del inventario, sin `gap` explícito | 576.845 |
| LUT / registros / BRAM / DSP | 48.256 / 45.208 / 84,5 / 134 |
| Banco aritmético | 3.823.015 ciclos, dos proyecciones y cancelación/recuperación |
| Salida numérica | 4.096 valores, 16.384 bytes FP32 exactos |

Revisión `20261016T101133Z-V7-projected-root-bram-vivado-r3-independent-projected-root-bram-review-r1.json`,
SHA-256 `271924264c4a78e2059fc4ac9ab939b9e060b29c5b163809296355049b4a39b0`.
Checkpoint ruteado: `5d8ac5bd0aa95f038b2078aac71203b71dddd926441506722b6c3738029b93a1`.
Esto aprueba timing interno OOC a 200 MHz; quedan pendientes presupuestos
de puertos externos, reloj y colocación del diseño padre, simulación
funcional posterior al ruteo y la placa DDR/PCIe integrada.

La exportación original terminó dentro de su límite. El
[helper por lotes](PROJECTED_ROOT_ROUTE_EXPORT_R1.md) queda preparado para
futuras tandas, probado con comandos simulados; no se usó en este resultado.

Actualización 2026-10-16 10:31 UTC: **síntesis realizada; XSim en ejecución**
en `20261016T101133Z-V7-projected-root-bram-vivado-r3`, coordinador PID 15928.
La observación del proceso XSim 883625 llegó al ciclo 541.774, sin fallo en
los últimos 64 KiB inspeccionados; no es una revisión del log completo ni
un resultado aprobado. Aritmética y ruteo siguen pendientes de cierre.
El ZIP fuente conserva el SHA-256 indicado abajo. DDR mantiene el turno
siguiente. El observador PID 23628 restaurará resultados y fuentes cuando
el coordinador termine su respaldo, sin modificar la cola.

Estado al 2026-10-15 20:26 UTC: **control nativo aprobado; aritmética y
ruteo nuevos encolados**. La FFN R4 continúa ejecutándose. Esta tanda
sigue a atención/FFN/residual real R1; conserva sus dependencias y exige
revisión y respaldo antes de ocupar el único slot Vivado.

## Problema y cambio

El [ruteo anterior](LLAMA32_PROJECTED_ROOT_PHYSOPT_R1.md) terminó con
setup −0,034 ns y hold +0,003 ns a 5 ns. El peor camino iba desde
`completion_epoch_reg[25]` hasta `output_mem_reg/ENARDEN`: 4,589 ns,
18 niveles lógicos y comprobaciones de identidad, cancelación y consumidor.

El [propietario R3](../../fpga/rtl/ntrpu_attention_quantize_owner_r3.sv)
lee su RAM privada durante el estado registrado `RESPONSE`. Conserva
íntegros `result_pending`, `result_valid`, `result_q_valid`, la salida
cero cuando es inválida y todas las condiciones de revocación y liberación.
La lectura adicional no solicita memoria externa. Puede aumentar actividad
interna mientras se retiene una respuesta; falta medir su coste energético.

Los padres `quantize_store_r2`, `output_path_r2` y `projected_root_r2`
seleccionan esa versión. Los archivos ya ejecutados o encolados permanecen
intactos.

## Evidencia disponible

- Tres semillas, 219 trabajos y 59.904 palabras verificadas.
- Las 17 salidas públicas coinciden con R2 en 1.191.819 muestras; incluye
  pausas, reset, identidad de 64 bits, cancelación y revocación tardía.
- 99 alteraciones de trazas rechazadas y seis mutantes RTL detectados por
  aserciones directas, sin depender de timeouts.
- Elaboración Icarus de 38 fuentes: 34 RTL, tres firmas aritméticas y TB.
  Esta prueba no ejecuta aritmética ni acredita síntesis.
- Revisor físico probado con evidencia antigua explícitamente renombrada:
  14 controles de medición y cinco de buffers. No representa ruteo nuevo.

## Tanda pendiente

Paquete `20261015T202505Z-projected-root-bram-vivado-r1`: 88 fuentes
originales compilables y 14 IP; espera 15 archivos de simulación generados.
Usa los mismos fixtures y pesos Q8 de Llama 3.2 1B Instruct ya verificados.
Primero ejecuta síntesis y XSim con dos proyecciones completas y una
cancelación. La revisión numérica independiente exige 69 controles antes
de permitir colocación, ruteo y `phys_opt_design -directive Explore`.

Dos CPU; límites de 7.200/600/3.600 segundos para simulación, revisión y
ruteo. Mantiene el reloj OOC de 5 ns y BUFGCE provisional `X0Y60`, sin
excepciones nuevas. Timing negativo sigue siendo un resultado fallido de
timing aunque la medición termine correctamente. DDR/PCIe físicos,
simulación posterior al ruteo y la inferencia completa continúan pendientes.

## Respaldo

Fuente sellada SHA-256:
`6c91557983bdc73610f97b97cdf9c0a3115456839a45a07671a6e3c28744ed07`.
Configuración:
`89a84cc20d0981d17520de5d85ce6c4412b8410ccccb05441c5e04b4c23298aa`.
Coordinador local PID 10204. Paquete y tres tandas nativas están verificados
en `D:/AI/ntransformer/artifacts/` y en NFS; no se ha ejecutado el experimento
Vivado nuevo. El archivo original `AGENTS.md` sigue intacto.
