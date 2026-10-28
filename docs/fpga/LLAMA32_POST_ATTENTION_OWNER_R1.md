# Propiedad de RAM post-O y publicación del par — R1

**Estado 2026-10-15:** síntesis y XSim R2 aprobados y respaldados en D:/NFS.
Tres semillas: 255 trabajos, 345.600 palabras RAM y 75 controles de traza.
Síntesis OOC a 5 ns: 665 LUT, 219 FF, 3 BRAM y 0 DSP;
WNS +2,252 ns, WHS +0,066 ns, sin cajas negras, latches ni lazos.
Son estimaciones de síntesis; faltan ruteo y restricciones externas.
Ejecución `20261015T082404Z-V7-llama32-post-attention-owner-vivado-r2`;
archivo `849dd80e0a29ccd0ff216c61798612191257af0c2a324fd90ea358b4a03b2305`.

El nuevo `fpga/rtl/ntrpu_post_attention_owner_r1.sv` mantiene la propiedad
de las entradas del cargador durante HLS y controla dos RAM de salida.
Los pares de prueba que representan al cargador y a HLS son sintéticos;
esta tanda no acredita aritmética post-O ni escrituras AXI de resultados.

## Contrato de memoria y vida del resultado

La respuesta retenida del cargador identifica secuencia y época completas
de 64 bits. Un comando ajeno produce error sin consumir esa respuesta.
Una respuesta propia fallida puede drenarse. Cambiar la identidad del
productor de forma permanente exige reset coordinado; el propietario no
debe reconocer una respuesta ajena para desbloquearse.

Los puertos 0, 1 y 2 son entrada residual, salida O y peso RMSNorm.
Cada RAM tiene dirección de 11 bits, datos FP32 y respuesta síncrona de
un ciclo, con CE/validez independientes. El propietario fija longitud
2.048 y epsilon `0x3727c5ac`. Rechaza respuestas ausentes o no finitas.

HLS escribe dos RAM privadas distintas: 2.048 palabras residuales FP32 y
512 palabras ABI1 de 64 bits. Las escrituras de cada RAM deben recorrer
sus direcciones en orden sin faltar ni repetirse. Comprueba finitud de
FP32 y formato ABI1: INT8 distintos de −128, escala válida y relleno cero.
La visibilidad del par exige ambos conteos completos, `ap_done`,
`ap_ready`, retorno cero y liberación de la entrada propia. Los lectores
de salida tienen CE y latencia de un ciclo independientes.

Un fallo invalida el par completo. Aunque HLS haya escrito la RAM residual
antes de fallar en normalización, ninguna salida queda visible. Cancelación
o cambio de época durante una respuesta retenida revocan sus datos de
inmediato; restaurar la época no revive el resultado. La respuesta conserva
su estado hasta ACK y bloquea la reutilización de ambas RAM.

## Cancelación, watchdog y reset

Los fallos se retienen hasta completar la operación. El propietario espera
`ap_done` antes de reconocer la entrada, incluso tras cancelación o timeout.
Un núcleo que nunca termina requiere reset coordinado con cargador y HLS.
No se convierte la expiración del watchdog en permiso para reutilizar RAM.

El watchdog cuenta ciclos sin accesos de entrada, escrituras ni finalización;
su valor predeterminado es 262.144. El banco reduce el umbral a 64 para
probarlo. **El umbral de integración todavía debe contrastarse con los
intervalos sin actividad externa del HLS real**, que mantiene resultados
internos antes de copiar las salidas. La ruta que detecta progreso utiliza
el fallo registrado para evitar realimentación combinacional del watchdog.

## Evidencia reproducible

El banco `tb_ntrpu_post_attention_owner_r1.sv` ejecutó 85 trabajos por
semilla (1, 7 y 42): 255 trabajos, 240 respuestas y 18 resets en total.
Comprobó **276.480 palabras residuales y 69.120 palabras ABI1**, con
permutaciones distintas que cubren todas las direcciones de ambas RAM.
Los ciclos completos fueron 474.632, 474.675 y 474.661; incluyen inyección
de fallos, esperas y lectura pública, no rendimiento del cálculo post-O.

Cubre identidades con diferencias en bits altos, productor revocado,
respuesta ausente en cada entrada, escrituras faltantes/duplicadas/de más,
formato inválido, error HLS temprano/tardío, cancelación antes y durante
cómputo, entre los dos buffers, timeout, respuesta retenida y recuperación
tras reset. La revisión independiente reconstruye el plan y compara cada
palabra pública; el forwarding de entradas se comprueba mediante aserciones
del banco congelado y mutantes compilados, no con una traza por beat.

Los 75 controles de corrupción de traza se rechazaron. Los 17 mutantes
compilaron y fallaron explícitamente en el banco: identidad truncada,
entrada intercambiada, datos alterados, publicación/liberación/reutilización
prematuras, conteos incompletos, respuesta RAM ignorada, cancelación,
timeout, revocación olvidada, epsilon incorrecto y validación de datos/HLS.

| Evidencia | Identificador / SHA-256 del archivo |
| --- | --- |
| Banco nativo | `20261015T070350Z-V7-post-attention-owner-native-r1` |
| Archivo nativo | `fe37c6a08b66caa964345ef15c79f1fdc47d0cc7006f40f6328b890dc3c24adc` |
| Mutantes | `20261015T070752Z-V7-post-attention-owner-mutants-r1` |
| Archivo de mutantes | `775434dfd3561e9203730f3798c52510ba2a7d2f2c571a8ec6bff6201ade9d9d` |
| Revisión independiente | `cd8226dd087c8e9f95638d4ec326aae707f2ca62bce0d16cd0e30e3192900aa5` |

Fuentes y resultados están verificados en D: y NFS. El RTL aprobado tiene
SHA-256 `965dd31e2f62ce8c74896fec16827e671cacfbc706ec00951e100e2c0cdec98f`.

## Vivado y criterios pendientes

`llama32-post-attention-owner-vivado-r1-coordinator.json` espera revisión
y respaldo del cargador Vivado R2. La tanda exige síntesis sin cajas negras,
latches ni lazos, tres semillas XSim, comparación de las fuentes realmente
compiladas y respaldo antes de liberar el turno. Usa VU5P, 5 ns, CPU0/1
y límite de 1.800 segundos; sigue siendo una prueba de protocolo.

- Fuente sellada: `208e8743bf267e7fca5cf5385bd5ad4ec2db78fd81ab77775dad8d140817c905`.
- Paquete: `8a8710fa708beb43d03b8a51dda8188116862955eceb6088d19a5850751acd1b`.

Después de aprobar el HLS real, inspeccionar sus puertos generados y unir
el [cargador AXI](LLAMA32_POST_ATTENTION_LOADER_R1.md), este propietario y
el [cálculo post-O](LLAMA32_POST_ATTENTION_R1.md). La integración debe leer
O confirmado en memoria, conservar la suma residual durante FFN y publicar
el descriptor residual/ABI1 solo tras todos los B de sus escrituras. La
publicación atómica del descriptor no significa escrituras físicas atómicas;
los buffers parciales permanecen privados. El [escritor AXI del par](LLAMA32_POST_ATTENTION_STORE_R1.md)
ya pasó su banco nativo y espera esta tanda en Vivado. Su integración con
HLS, la segunda suma residual, timing integrado y DDR física siguen pendientes.
