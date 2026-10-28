# Publicación AXI del par post-O — R1

**Estado 2026-10-15:** síntesis y XSim R3 aprobados y respaldados en D:/NFS.
Tres semillas: 330 trabajos, 180 pares, 2.211.840 bytes y 126 controles.
Síntesis OOC a 5 ns: 1.525 LUT, 1.084 FF, 0 BRAM y 0 DSP;
WNS +1,816 ns, WHS +0,066 ns, sin cajas negras, latches ni lazos.
Son estimaciones de síntesis; faltan ruteo y restricciones externas.
Ejecución `20261015T082949Z-V7-llama32-post-attention-store-vivado-r3`;
archivo `b14f50cdf0bb7f643989a4d37eac5c052f5faee35b2966321ac06149521de4e8`.

`fpga/rtl/ntrpu_post_attention_store_r1.sv` publica conjuntamente un vector
residual FP32 de 8.192 bytes y la entrada FFN ABI1 de 4.096 bytes.
El banco carga resultados de referencia reales en un productor RAM;
todavía no une el cálculo HLS con este transporte.

## Contrato de transferencia y visibilidad

El comando conserva secuencia y época completas de 64 bits y captura dos
regiones de destino. Antes de leer RAM comprueba alineación de 64 bytes
de los destinos, límites con sumas de 65 bits y ausencia de solapamiento
entre los dos intervalos escritos. Regiones adyacentes son válidas.

Cada lectura RAM tiene CE y respuesta válida de un ciclo: 2.048 palabras
FP32 residuales y 512 palabras ABI1 de 64 bits. Se comprueban finitud y
formato Q8. El escritor empaqueta beats de 512 bits y mantiene una sola
escritura de 64 bytes pendiente: 128 beats residuales y luego 64 ABI1.
AW y W se aceptan independientemente; dirección, datos y valid permanecen
estables hasta sus respectivos handshakes, incluso tras cancelar.

Los contadores reflejan escrituras confirmadas por B correcto e ID cero.
El descriptor conjunto aparece únicamente después del último B correcto,
sin fallos ni revocación y tras liberar la respuesta propia del productor.
Un cambio tardío de época o cancelación invalida ambas salidas de forma
persistente. La publicación conjunta del descriptor **no hace atómicas las
escrituras físicas**: el llamador debe mantener privados ambos destinos
hasta recibir el resultado válido. Un error no revierte bytes ya escritos.

## Fallos, drenaje y reset

Después de detectar un fallo no se inicia otro registro, pero se drena
cualquier AW/W ofrecido y su B. El watchdog predeterminado de 65.536 ciclos
cuenta falta de progreso AXI; su expiración no permite abandonar el bus.
Un productor con identidad ajena no recibe ACK. Si la identidad nunca se
restaura o el bus no responde, se requiere reset coordinado con ambos pares.
El banco prueba umbral 64, cancelaciones en cada canal, reset con escrituras
parciales, identidad en bits altos, límites de dirección y recuperación.

## Evidencia nativa y controles

Tres semillas (1, 7 y 42) ejecutaron 110 trabajos cada una: **330 trabajos,
312 respuestas, 309 ACK de origen y 21 resets**. Hubo 180 publicaciones
correctas: **2.211.840 bytes**, comparados con cuatro vectores post-O de
Llama 3.2 1B Instruct Q8_0. El modelo de memoria materializa datos solamente
tras B correcto, usando AW y W aceptados. El verificador reconstruye ese
registro de escrituras y compara cada byte publicado.

Los bancos completos consumieron 413.528, 413.406 y 413.347 ciclos. Incluyen
fallos, pausas y recuperación; no son medidas de rendimiento de inferencia.
Los 20 mutantes de RTL compilaron y fallaron con diagnóstico explícito.
Los 126 controles de corrupción de traza fueron rechazados.

El verificador R2 conserva todos los criterios positivos de R1. Solo cambia
un control negativo: alterar ACK con XOR 1 podía producir un retraso legal
en la semilla 42. R2 fuerza ACK en ciclo cero para comprobar liberación
prematura. Se conserva el diagnóstico R1 y no se modificó RTL, banco ni
datos para obtener el resultado aprobado.

| Evidencia | Identificador / SHA-256 |
| --- | --- |
| Banco nativo | `20261015T072828Z-V7-post-attention-store-native-r1` |
| Archivo nativo | `c930308492b19e341cf3b70e0f559cc1fc78dc75adf80e961d4658fc1bd8e5b5` |
| Mutantes | `20261015T073348Z-V7-post-attention-store-mutants-r1` |
| Archivo de mutantes | `21685e8c99061118b07978240787f20708e01848abdd84bf89ab53987395ec5f` |
| Revisión independiente R2 | `3f764b0be14f57856b57042150ddf5235f7a03f8133ace5cfaaef67aa5aacf73` |
| RTL probado | `7b16c351b830158eed66616776b90e69c2924d786c3a538e692ab9e00caf0d8a` |

## Tanda Vivado y siguiente integración

Paquete `20261015T074530Z-post-attention-store-vivado-r1`, verificado en
D: y NFS; 38 archivos y dos fuentes compiladas. Su coordinador espera la
revisión y respaldo del propietario R1 antes de tomar el turno principal.
Exige síntesis sin lazos, latches ni cajas negras; tres semillas XSim;
revisión independiente de bytes, controles y fuentes compiladas; y respaldo
verificado antes de liberar el turno. Objetivo VU5P a 5 ns, CPU0/1,
límite de herramienta 1.800 segundos. Timing se informa por separado.

- Fuente sellada: `6d9b3da54b8734ed809bbcd43985ae4132b5f3ae4e527ce8d16ef1068f2baa6b`.
- Paquete: `c5d33764d26afd179e564b794d5cb85d3aa4eec0cda5747faa821bb6f3f45eb9`.

Después del [HLS post-O](LLAMA32_POST_ATTENTION_R1.md) real, inspeccionar
sus puertos generados y unir cargador, propietario, cálculo y escritor.
La prueba integrada debe consumir O confirmado en memoria, verificar los
intervalos reales sin actividad HLS contra el watchdog, conservar el residual
para la segunda suma y alimentar FFN solo desde ABI1 publicado. La unión,
el cierre temporal integrado, DDR física y la inferencia completa siguen
pendientes. Orden vigente: [recuperación y cola R3](LLAMA32_PROJECTED_ROOT_TAG_RECOVERY_R3.md).
