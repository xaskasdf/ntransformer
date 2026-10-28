# Motor de atención: XSim y síntesis OOC revisados

Estado 20261013T200013Z: **simulación, síntesis y procedencia verificadas;
respaldo completo confirmado en D:/NFS**. El timing a 5 ns sigue sin cerrar.
Tanda `20261013T191941Z-V7-llama32-attention-engine-vivado-r1`, coordinador R7; parte `xcvu5p-flva2104-2-i`.

## Resultado funcional

El RTL de atención generado por Vitis se conectó a contexto, estado privado
y camino K/V con AXI BFM. Pasaron 155 consultas y 9.600 resultados públicos;
155 tiles comparados contra la referencia de alta precisión, con **0 ULP**
de error máximo de salida y error escalado de estado máximo
3.599e-16. Se rechazaron 42 alteraciones
de trazas. El banco incluye cuatro resets y recuperación tras errores.

Las cuatro continuaciones del estado ejercitaron la corrección del índice
especulativo 104. Los diagnósticos de las tres tandas anteriores permanecen
archivados. Se verificaron los hashes de los archivos realmente compilados,
los 33 archivos generados congelados y los padres HLS/memoria respaldados.

El banco completo consumió 2,489,451 ciclos simulados,
con 3,096,576 bytes escritos y 561,152
leídos por AXI BFM. Estas cifras incluyen fallas, pausas y reset; no representan
tiempo por token ni rendimiento medido en una tarjeta.

## Recursos y timing estimados

| Medida del top integrado | Resultado |
| --- | --- |
| LUT | 19410 |
| Registros | 16513 |
| Tiles BRAM | 33.5 |
| DSP | 77 |
| Setup WNS a 5 ns | -0.456 ns |
| Hold WHS | +0.030 ns |

Sin black boxes, latches, bucles combinacionales ni endpoints internos sin
restricción. Hay puertos externos sin retardos definidos, coherente con
síntesis OOC; todavía no se califica timing de placa ni ruteado.

La peor ruta estimada va de `memory_path/kv_path/ram/received_reg[0]/C`
a `memory_path/kv_path/FSM_onehot_state_reg[0]/CE`: 18 niveles lógicos y
5,352 ns de camino de datos, con 76,7 % de retardo de interconexión estimado.
El siguiente estudio físico debe comprobar colocación/ruteo antes de atribuir
todo el déficit al RTL. No se acredita operación a 200 MHz. Estos recursos
incluyen submódulos ya medidos; no deben sumarse a sus cifras individuales.

## Respaldo y siguientes dependencias

Archivo local: `artifacts/ntrpu-board-vivado/20261013T191941Z-V7-llama32-attention-engine-vivado-r1/20261013T191941Z-V7-llama32-attention-engine-vivado-r1.tar.gz`.

- SHA-256 del archivo: `1a59031097823e87939bd2c9ea1103389fd97cc6516aeaa4cd61e9f15c1e67ed`.
- Revisión independiente: `48c740c1713c40121ebc5fc670ac517679068bc485a1cc06c1aecabe500c2448`.
- Evidencia: `be3f6f57329fc0135b93aa60ab83481f22b48466bbe6741b9640953d5ea500ab`.
- Fuentes de la tanda: `0ff884611cca2f544c9132a5f88ff0168ec3690941b8c4182d9500e89a3dfb4f`.

El respaldo contiene fuentes, proyectos, trazas, checkpoint, reportes y
certificado de integridad. El coordinador libera el slot solo después de
verificarlo. Sigue el secuenciador de 32 cabezas R4, guard raíz R3, wrapper
completo R2, cuantizador HLS R1 y controlador de cuantización R1, en ese orden.
La inferencia completa, la unión a proyección O/FFN, DDR física y el cierre
de timing integrado continúan pendientes.
