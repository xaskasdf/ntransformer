# Escritor AXI de la segunda suma residual

Estado 2026-10-15 17:04 UTC: **nativo aprobado y respaldado; Vivado en cola**.
`fpga/rtl/ntrpu_residual_store_r1.sv` transporta una salida FP32 de 2.048
elementos desde RAM hasta un destino AXI privado de 8.192 bytes.

## Propiedad y publicación

Captura secuencia, época y descriptor de 64 bits. Verifica identidad,
alineación de 64 bytes, tamaño, límites con aritmética de 65 bits y finitud
de cada palabra RAM. Empaqueta 16 palabras por beat de 512 bits; emite
128 escrituras de 64 bytes, con una sola operación pendiente.

AW y W se aceptan independientemente y permanecen estables durante las
pausas. Cada contador avanza únicamente por B correcto e ID cero. El
descriptor público requiere las 128 confirmaciones, ausencia de fallo o
revocación y liberación del productor. Una cancelación o cambio tardío de
época revoca permanentemente el resultado retenido.

Si falla una operación ofrecida, mantiene AW/W hasta sus handshakes y
drena B antes de liberar el origen. El watchdog no autoriza abandonar
tráfico pendiente. Un bus detenido o una identidad que nunca se restaura
requieren reset coordinado. La publicación del descriptor no hace atómicas
las escrituras: el destino debe permanecer privado y los bytes ya escritos
no se revierten ante error.

## Pruebas con bytes de Llama

El fixture contiene las cuatro salidas finales de la referencia
[post-O → FFN → segunda suma](LLAMA32_POST_FFN_R1.md), capa 0, tokens
0, 42, 128000 y 128255 del GGUF Q8_0 verificado. Esas salidas se cargan
en la RAM del productor; este banco no ejecuta la aritmética HLS.

Tres semillas aprobaron 82 trabajos cada una: **246 trabajos, 132 vectores
publicados y 1.081.344 bytes exactos**. El revisor reconstruye la memoria
confirmada usando AW/W aceptados y B exitosos; compara cada byte publicado.
Cada semilla registró 108.839 lecturas RAM, 6.798 AW, 6.798 W y 6.796 B.
La diferencia corresponde a resets con transacciones pendientes.

Incluye direcciones altas, overflow, destino corto o desalineado, RAM
sin respuesta, Inf, B con error/ID ajeno, cancelación en cada canal,
revocación tardía, identidad ajena, watchdog y reset/recuperación.
Los **117 controles de revisión y 21 mutantes RTL** fueron rechazados;
no se aceptan fallos de compilación ni timeouts como detección de mutantes.

El banco R1 comparaba el contador retenido de la operación anterior contra
el contador del siguiente trabajo mientras estaba inactivo. R2 acota esa
aserción a `busy`; mantiene la comprobación durante toda transferencia y
respuesta retenida. Se conservó el diagnóstico y no se modificó el RTL.

## Medidas y evidencia

Las transferencias correctas consumieron **4.441–4.513 ciclos**, contando
desde la primera lectura RAM hasta el último B, ambos incluidos, con las
pausas del banco. No incluye cálculo, DDR física ni tráfico host; no mide
inferencia. Los bancos completos consumieron 238.997, 238.748 y 238.707 ciclos.

| Evidencia D:/NFS | SHA-256 |
| --- | --- |
| Fixture | `f19bba97868ed043cc928916bb313b011960b62adfdeb7ffb914282414e0e2b3` |
| Nativo `20261015T165608Z`, R2 | `3f8bffdebced9d3bac64e0709de048629d8ed3bc5b095c78fd1211f29b8beecc` |
| Mutantes `20261015T165805Z`, R1 | `489014e5efe3b0a7e74b349eb0810f2e863efe837a1b021a9ea45c59f212a6c6` |
| Revisión independiente | `052a5c6b812dcb146ae70adcf0dd98b9c9cb2fbb2ae02bcdbc5a2a8517b5324e` |
| Fuente sellada Vivado | `9361a07bd64bf61c168f66f48bb876eeb8e661688ea019eeaf3d6c1c734142fd` |

`queue_llama32_residual_store_vivado_r1.py` espera la revisión y respaldo
del [propietario residual](LLAMA32_RESIDUAL_OWNER_R1.md). Ejecutará síntesis
OOC a 5 ns y tres semillas XSim, CPU0/1, dos hilos y límite de 1.800 segundos.

## Integración siguiente

Se inspeccionaron los nueve Verilog originales del
[HLS residual aprobado](LLAMA32_RESIDUAL_R1.md). El adaptador
`ntrpu_residual_hls_ports_r1.sv` conecta 18 puertos sin añadir registros;
55 cambios de dirección, ancho, presencia o puertos adicionales fueron
rechazados. SHA del adaptador:
`642c54806b0e78d96b31cfc28e03e0921d229ecb9a7876a7e14893abc0a07767`.
La [composición residual real](LLAMA32_RESIDUAL_REAL_R1.md) posteriormente
aprobó la compilación de los nueve módulos originales y 144 trabajos
nativos con operandos AXI, 147 controles y 12 mutantes. Su validación
Vivado espera al escritor R1. Sigue conectar la salida FFN viva;
permanecen pendientes timing integrado, DDR4/PCIe e inferencia completa.
