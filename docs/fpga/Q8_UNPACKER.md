# Camino AXI a bloques Q8_0

Fecha: 2026-10-02. `fpga/rtl/ntrpu_q8_unpacker.sv` convierte beats de 512 bits en bloques Q8_0; `ntrpu_q8_weight_stream.sv` lo conecta al lector y los buffers dobles. El conjunto está probado y sintetizado. Todavía no multiplica pesos ni ejecuta GEMV.

## Formato y señales

Se conserva el formato del [GEMV Q8_0 existente](../../fpga/hls/gemv_q8_0/ntrpu_gemv_q8_0.h): dos bytes de escala FP16 little-endian y 32 pesos INT8 con signo. El byte de menor dirección ocupa los bits bajos del beat. Un bloque de 34 bytes puede cruzar dos beats de 64 bytes; el desempaquetador conserva los bytes sobrantes en un registro de 128 bytes.

| Salida | Contrato |
| --- | --- |
| `block_scale[15:0]` | Bits FP16 originales, sin conversión aritmética |
| `block_weights[255:0]` | Lane `i` en `[8*i +: 8]`; el consumidor debe interpretarla como INT8 con signo |
| `block_index[9:0]` | Índice lineal 0–1.023 dentro del paquete |
| `block_tag[31:0]` | Etiqueta del primer beat, constante durante el paquete |
| `block_last` | Último bloque del paquete completo |
| `block_valid/block_ready` | Handshake; datos y metadatos estables durante pausas |

La entrada no tiene máscara de bytes: todos los beats contienen 64 bytes útiles. Por eso los paquetes válidos deben medir **17, 34, …, 544 beats**, equivalentes a **32, 64, …, 1.024 bloques Q8_0**. Un tile máximo ocupa 34.816 bytes. No hay relleno implícito. El índice no codifica fila ni posición K: esa interpretación corresponde al descriptor del futuro motor.

Se conservan exactamente ceros con signo, escalas negativas y subnormales FP16. Se rechazan exponentes FP16 de cinco unos, correspondientes a infinitos y NaN. No se dequantiza ni se cambia el redondeo de la referencia HLS.

## Transporte y validación son finalizaciones distintas

El top integrado expone dos canales:

1. `load_done_*`: resultado de copiar memoria al banco privado. Su contrato es el del [buffer doble](WEIGHT_TILE_BUFFER.md). Si falla, el banco no se publica y **no habrá finalización de decodificación** para esa carga.
2. `decode_done_*`: resultado de interpretar un paquete publicado. Estado 0 significa correcto; 1, longitud/etiqueta/framing inválido; 2, escala no finita. `decode_done_blocks` cuenta bloques aceptados por el consumidor y `decode_done_tag` identifica el paquete.

Una lectura AXI correcta puede contener una escala inválida. Por eso los bloques entregados al cómputo son **provisionales hasta `decode_done_status=0`**: mantener los acumuladores/resultados privados y descartarlos ante error. La atomicidad del banco protege contra fallos de transporte, no valida numéricamente sus bytes.

Al detectar un error, el desempaquetador deja de producir bloques nuevos y drena la entrada hasta `in_last`. Un bloque correcto ya preparado conserva `VALID` y sus datos hasta que el consumidor lo acepte. El prefijo entregado antes de un error de framing puede variar con las pausas; el contador final registra lo efectivamente recibido. Si se detectan simultáneamente errores de formato y escala, el formato tiene prioridad; después se conserva el estado.

La finalización espera a que se consuma la salida pendiente y permanece estable hasta aceptación. No empieza otro paquete mientras esa notificación esté detenida. Los bancos pueden seguir precargando según su capacidad. No hay timeout para una fuente que nunca envíe `last`; el reset debe coordinarse con el resto del camino y descartar el trabajo parcial.

## Pruebas y síntesis

```bash
source /ruta/AMDDesignTools/2026.1/Vivado/settings64.sh
python3 scripts/run_ntrpu_axi_reader.py --design q8_unpacker
python3 scripts/run_ntrpu_axi_reader.py --design q8_stream --synth
```

Resultados con XSIM/Vivado 2026.1 y `xczu7ev-ffvc1156-2-e`:

- Desempaquetador, `20261002T082824Z-git2zj0b`: **105 casos correctos** (35 × semillas 1/7/42). Comprueba los 32 offsets posibles de bloque dentro del beat, todos los lanes, ceros/subnormales/signos, infinitos/NaN, errores al inicio y final, longitudes inválidas, etiquetas cambiadas, backpressure y reset en reposo.
- Integración, `20261002T083147Z-sbzlh2_z`: **30 casos correctos** (10 × 3 semillas), 12.573 bloques comparados. Incluye ambos bancos ocupados, un tercer comando bloqueado, carga/decodificación simultáneas, error AXI final sin paquete publicado, escala final inválida, paquete de 64 bytes rechazado por formato y recuperación posterior.
- Síntesis del conjunto lector + buffers + desempaquetador: **5.037 LUT, 2.395 FF, 28 RAMB36 + 2 RAMB18, 0 DSP**.
- Reloj solicitado de 5 ns; estimación WNS +2,244 ns y WHS +0,047 ns. No hay colocación/ruteo: faltan retrasos de 654 entradas y 498 salidas y HD.CLK_SRC. No equivale a cierre temporal ni a ancho de banda físico medido.

Los datos son sintéticos y se construyen byte a byte independientemente del registro de ensamblado. Estas pruebas no comparan logits ni resultados GEMV, y no usan pesos de Llama. Fuentes, informes, checkpoint y hashes quedan en `artifacts/ntrpu-q8-stream-20261002/` y en el volumen persistente.

## Capacidad de servicio y siguiente integración

El diseño tiene una sola salida de bloque por ciclo. A un reloj hipotético de 200 MHz, su techo sería `34 × 200 MHz = 6,8 GB/s` de Q8 empaquetado, inferior al techo de 12,8 GB/s del lector de 512 bits. Las pausas y transiciones reducen el promedio. Para el presupuesto Llama 1B, eso da como máximo ideal unos 5,18 tokens/s limitados solo por esta salida; **no es rendimiento observado** y omite cómputo, KV, DDR y planificación.

El [núcleo MAC por bloque](Q8_BLOCK_MAC.md) ya implementa producto INT8/INT32,
escalado y realimentación FP64, con pruebas CPU y 317 operaciones C/RTL PASS
en VU5P; la versión optimizada tarda 21 ciclos por bloque válido.
El [adaptador GEMV](Q8_GEMV_STREAM.md) ya asocia activaciones, conserva
acumuladores entre tiles K y publica después de ambas validaciones: 72 casos
RTL con memoria sintética, errores tardíos y reset coordinado. Falta conectar
la FFN y completar la implementación física.
Replicar salidas o cambiar el empaquetado deberá justificarse con medidas de
ese camino completo.
