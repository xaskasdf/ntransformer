# Post-O real con transporte AXI — R1

Estado 2026-10-15 09:00 UTC: **síntesis/XSim y revisión independiente
aprobados; respaldo completo verificado en D: y NFS**.
Tanda `20261015T084117Z-V7-llama32-post-attention-real-vivado-r1`;
VU5P `xcvu5p-flva2104-2-i`, reloj OOC de 5 ns, Vivado 2026.1.

## Composición y datos

`ntrpu_post_attention_real_r2.sv` une el cargador AXI, propietario de RAM,
escritor del par y controlador de transporte con el RTL HLS aprobado de
residual → RMSNorm → Q8/ABI1. El paquete contiene 44 fuentes compilables:
siete módulos propios, un banco y 36 Verilog generados; incluye los siete
Tcl que recrean sus IP aritméticas. No compila el sustituto de aritmética.

Las tres entradas son el residual conservado, O y los pesos RMSNorm de FFN,
2.048 FP32 cada una. Proceden de cuatro vectores de referencia de la capa 0
de Llama 3.2 1B Instruct Q8_0. Los resultados de referencia se usan solo
para comparar: no alimentan al cálculo HLS. Este banco no genera O desde
la raíz de atención ni ejecuta FFN o inferencia completa.

El modelo conserva SHA-256
`3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1`.
El [contrato de propiedad y drenaje](LLAMA32_POST_ATTENTION_PATH_R1.md)
sigue vigente; el llamador mantiene las fuentes y destinos privados hasta
la respuesta final. Reset debe coordinar todos los participantes.

## Criterios de aprobación

- Cuatro trabajos normales, cancelación durante HLS y recuperación,
  cancelación durante escritura y recuperación: ocho trabajos en total.
- Seis pares completos: 73.728 bytes exactos, materializados desde AW/W
  aceptados y confirmados por B; ningún resultado parcial publicado.
- Direcciones superiores a 4 GiB, división de ráfagas en páginas de 4 KiB,
  backpressure, captura de descriptores y retención hasta ACK.
- Medir latencia y máximo intervalo HLS sin progreso por trabajo;
  contrastarlos con el watchdog de 262.144 ciclos.
- Verificación independiente de bytes, estados y orden, catorce controles
  negativos, hashes de las 44 fuentes realmente compiladas y del checkpoint.
- Síntesis sin cajas negras, latches ni lazos; informar recursos y timing
  como estimaciones OOC, con ruteo e interfaces externas aún pendientes.
- Respaldar resultados y diagnósticos en D:/NFS y verificar cada miembro
  antes de liberar el turno de Vivado.

## Resultado medido

Los ocho trabajos pasaron: **seis pares publicados, 73.728 bytes exactos**,
72 AR, 3.072 beats R, 1.280 respuestas B y ocho inicios/finales HLS.
El banco completo consumió 1.430.804 ciclos. La revisión rechazó los
catorce controles y comprobó hashes de las 44 fuentes realmente compiladas.

Cada trabajo con cálculo íntegro tuvo 173.353 ciclos HLS y un máximo de
165.137 ciclos sin acceso externo. El watchdog de 262.144 deja un margen
observado de 97.007 ciclos para estos vectores. El trabajo cancelado
durante HLS drenó en 126.249 ciclos, devolvió estado 7 y no publicó ni
escribió salidas AXI. Cancelar la escritura después de 128 respuestas B
dejó 8.192 bytes residuales confirmados y cero ABI1, sin publicar el par.
Ambas recuperaciones posteriores entregaron resultados completos exactos.

Síntesis OOC: **13.131 LUT, 11.389 FF, 26,5 BRAM y 11 DSP**;
WNS +0,953 ns y WHS +0,039 ns a 5 ns. Cero cajas negras, latches, lazos
y endpoints internos sin restricción. Hay 1.815 entradas y 949 salidas
sin presupuesto externo: estos datos no cierran timing de placa.
No se ejecutó colocación/ruteo ni simulación posterior a implementación
de este bloque. Los ciclos medidos no son tokens/s ni rendimiento DDR.

Revisión `20261015T084117Z-V7-llama32-post-attention-real-vivado-r1-independent-post-attention-real-review-r1.json`;
SHA-256 `478c4e959f3a47d4060670749ef9b811e958c22d2ac3cbc83d94d3d34546f963`.
Archivo de resultados de 56.130.215 bytes y 245 miembros verificados:
`2fc7909a175e77e61f4548b95ec58ac1c284f47b629fa40946810594788be14c`.
El [ruteo de la raíz → O](LLAMA32_PROJECTED_ROOT_ROUTE_R1.md) tomó el
turno después de ese respaldo; no implementa físicamente este post-O.

## Dependencias y evidencia de preparación

HLS R3, cargador R3, propietario R2 y escritor R3 terminaron aprobados y
respaldados antes de esta ejecución. El controlador pasó 144 trabajos
nativos, 39 controles de traza y doce mutantes compilados. El adaptador
HLS pasó 79 controles de contrato. La compilación nativa R2 acredita
sintaxis/conexiones con un módulo vacío de firma exacta; no aritmética.

| Registro | SHA-256 |
| --- | --- |
| Fuente sellada D:/NFS | `648028177fea4214e1484fba7eec2975eb4d204e78b6e4dfa3b2dbadfd412e7a` |
| Paquete R1 | `a672636c54d7d668fb75699eb8cf6669d9237bb062367c3337cc34e125edb232` |
| Archivo HLS aprobado | `ac6e55a53a11e7fd4cb6867483e788fe03abf9e1fb01b4969949ea1cf455b979` |
| Compilación nativa R2 | `db92da744a8d808cf130ec03a02b1c202be340f1abd61c6bcb2400d0b715c544` |

La revisión R2 del top conserva públicamente `completion_hls_status`;
R1 lo había clasificado como señal interna durante la generación. Ambos
intentos de compilación se conservaron; los módulos hijos no cambiaron.
El coordinador R1 y la auditoría R27 preservan las validaciones externas
y limitan esta cola a una sola tanda principal de Vivado.
