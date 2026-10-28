# Bloque de activación para la FFN en memoria

Estado al 2026-10-03 UTC: co-simulación C/RTL ampliada **27/27 PASS** en VU5P;
el banco inicial de 26 casos conserva también su referencia CPU.
`fpga/hls/swiglu_q8_block/ntrpu_swiglu_q8_block.cpp` conecta las operaciones
numéricas existentes sin modificar sus fuentes. Su interfaz es local;
el [adaptador de memoria](SWIGLU_MEMORY.md) la conecta a AXI y tiene
validación separada. La cadena [FFN](FFN_MEMORY.md) y su integración
[PCIe/XDMA](PCIE_FFN_INTEGRATION.md) ya pasan bancos propios con el operador
original. El candidato con pipeline descrito abajo se califica por separado.

## Operación y contrato

Procesa 1–32 parejas gate/up procedentes de salidas GEMV binary64:

1. Valida todas las entradas utilizadas y convierte a FP32 con redondeo
   al más cercano, empates al par y subnormales conservados.
2. Ejecuta `SiLU(gate) × up` mediante el SwiGLU existente: intermediarios
   FP64, exponencial AMD y resultado FP32 comprobado.
3. Completa con ceros las posiciones restantes y cuantiza un bloque de
   32 componentes con el cuantizador Q8 existente.
4. Emite ocho palabras de 64 bits que forman un registro de
   [activación ABI1](ACTIVATION_MEMORY.md): 32 bytes INT8, escala FP64
   little-endian y 24 bytes reservados cero.

Las entradas son arrays de 32 palabras de 64 bits; se ignoran las posiciones
posteriores a `elements`, aunque contengan NaN. El destino tiene ocho
palabras. Los buffers deben ser válidos y no solaparse. Ninguna palabra
del destino se modifica si falla la operación.

| Estado local | Significado |
| --- | --- |
| 0 | Registro completo correcto |
| 1 | `elements` fuera de 1–32 |
| 2 | NaN/infinito en una entrada utilizada |
| 3 | Entrada finita fuera del rango FP32 |
| 4 | Overflow de la activación |
| 5 | Fallo del cuantizador; reservado para propagar su contrato |

La validación de NaN/infinito tiene prioridad sobre el overflow de conversión,
independientemente de su posición en el bloque. El controlador de memoria
deberá traducir estos estados al protocolo del job y mantener su identidad.

## Interfaces generadas

El top HLS utiliza `ap_ctrl_hs`, reloj `ap_clk` y reset `ap_rst` activo alto.
Conservar `elements` hasta `ap_done`. Los puertos locales observados en RTL:

- `gate_address0[4:0]`, `gate_ce0`, `gate_q0[63:0]`.
- `up_address0[4:0]`, `up_ce0`, `up_q0[63:0]`.
- `record_r_address0[2:0]`, `record_r_ce0`, `record_r_we0`, `record_r_d0[63:0]`.
- `ap_start`, `ap_done`, `ap_ready`, `ap_idle`, `ap_return[31:0]`.

Son interfaces `ap_memory`, no masters AXI ni punteros del host. El adaptador
debe respetar la latencia de lectura del modelo de memoria HLS y capturar
las ocho escrituras privadas. Solo `ap_done` con estado cero autoriza
escribir el registro completo en memoria del dispositivo.

## Evidencia numérica y temporal

Tanda `20261003T065730Z-V7-swiglu-block-cosim`, Vivado/Vitis 2026.1,
`xcvu5p-flva2104-2-i`, reloj solicitado de 5 ns; duración 243,270 s.
Contiene 26 transacciones C/RTL consecutivas y 13 registros exitosos.

El banco compara contra conversiones IEEE nativas y `std::exp`, sin usar
los helpers numéricos del DUT en esa referencia: los 32 INT8 deben coincidir
exactamente y la escala admite error relativo de hasta 2×10⁻⁷. C simulation
AMD y co-simulación RTL deben reproducir los mismos registros. Se revisan
padding, reservados, guardas, NaN/infinito tardíos, overflow, prioridad de
errores, entradas subnormales, empates de conversión y recuperación.

Latencia RTL medida: **2934–3794 ciclos** para bloques válidos; 1 ciclo para
forma inválida, 192–195 para fallos de entrada y 2182 para overflow de
activación. El banco completo reporta 49.464 ciclos. No incluye transporte
AXI ni demuestra una FFN completa.

Estimación HLS: 13.047 LUT, 4779 FF, 37 DSP, 7 BRAM18 y periodo 3,647 ns.
Estos datos proceden del XML de síntesis; no representan recursos mapeados
ni timing posruteo; los resultados Vivado se detallan más abajo.

Se conservan avisos de flujo batch obsoleto, nombres de funciones y entorno
XSIM. Tres `HLS 214-450` indican dirección ignorada en puertos de registro
gate/up/record. La inspección del RTL confirma los puertos de memoria y sus
anchos enumerados arriba; pasan datos no uniformes y colas parciales.
Esto no califica aún su direccionamiento dentro del sistema AXI.

La tanda ampliada `20261003T072154Z-V7-swiglu-block-raw-payload` pasa
**27/27 casos**, con 14 registros correctos y 13 errores sin modificar
la salida; total 53.259 ciclos y duración 233,599 s. Añade gate=257 y
up=−1, salvo up[0]=127. Produce las palabras `ffffffffffffff7f`,
tres veces `ffffffffffffffff`, escala `4070100000000000` y tres ceros.
Los bytes INT8 son válidos aunque esas palabras, interpretadas erróneamente
como doubles, parezcan NaN. Los 30 archivos RTL generados son idénticos
a los del banco inicial y del operador sintetizado/ruteado; sus hashes
están en `swiglu-block-rtl-identity.json` del índice de entrega.

## Implementación fuera de contexto

Esta tabla corresponde al operador original, sin el experimento de pipeline.

| Fase | LUT | FF | DSP | BRAM | WNS / WHS |
| --- | --- | --- | --- | --- | --- |
| Síntesis Vivado | 9372 | 6532 | 37 | 2 RAMB36 + 1 RAMB18 | +1,747 / +0,039 ns |
| Ruteo OOC | 9020 | 6428 | 37 | 2 RAMB36 + 1 RAMB18 | +1,038 / +0,011 ns |

Tandas `20261003T070338Z-V7-swiglu-block-synth`,
`20261003T071037Z-V7-swiglu-block-route` y
`20261003T071520Z-V7-swiglu-block-route-audit`. Se conservan checkpoints,
reportes, avisos DSP, revisión y fuentes con respaldos locales verificados.

Vivado cuenta 14.953 nets ruteables completas y cero errores de ruteo.
Queda un gap explícito del reloj: 615 aliases comparten su ruta `PARTIAL`.
La ubicación `BUFGCE_X0Y60` es una suposición OOC; 162 entradas y
87 salidas carecen de retardos externos. No hay endpoints internos sin
restricción ni nuevas excepciones. Estos márgenes internos no cierran
el timing del sistema ni califican una placa.

## Integración con memoria

El [adaptador AXI](SWIGLU_MEMORY.md) implementa los pasos 1–4 del contrato
siguiente y pasa 105 casos con gate variable, además de 8192 elementos
completos. El controlador FFN raíz también implementa el paso 5; conservar
estas reglas al sustituir el operador:

1. Capturar el job completo: número de elementos, secuencia/época y
   regiones gate/up/salida. Validar rangos, overflow y solapamientos antes
   del primer AR. Cada vector de entrada reserva hasta el siguiente beat
   de 64 bytes; el último bloque puede leer padding pero no calcularlo.
2. Reutilizar el lector AXI para cargar hasta 256 bytes de gate y 256 de
   up en buffers privados. No arrancar HLS hasta completar ambas lecturas
   sin error. Adaptar explícitamente sus puertos locales de 64 bits.
3. Capturar direcciones y las ocho palabras emitidas; rechazar duplicados,
   palabras faltantes o finalización fallida. Usar un escritor de un
   beat AXI con AW/W independientes y completion después de B.
4. Repetir hasta 256 bloques para 8192 componentes. Los registros escritos
   permanecen privados hasta completar el job entero. Fallo, cancelación
   o reset invalidan todo el vector; las transacciones emitidas se drenan
   o se recuperan con reset coordinado.
5. Arrancar down únicamente tras esa completion correcta. La FFN raíz
   publica su resultado final cuando también termina down sin errores.

El escritor binary64 existente **no acepta sin cambios** registros ABI1:
sus bytes INT8 podrían parecer un NaN y activar su comprobación de finitud.
El nuevo [escritor de activaciones](ACTIVATION_WRITER.md) valida el formato
ABI1 y conserva los bits; pasa 159 casos RTL y 159 sobre netlist con errores
B/BID, backpressure, watchdog y reset. Su conexión al operador mediante
`ntrpu_swiglu_memory.sv` tiene una tanda de integración independiente.

La integración cubre 1/31/32/33/720/8192 elementos, cruces de página,
direcciones altas, errores tardíos de lectura/HLS/B, descriptores alterados,
cancelación, guardas y recuperación tras reset. La inyección específica de
palabras HLS duplicadas/faltantes sigue pendiente; los bancos actuales
comprueban los datos emitidos por el operador real.

## Reproducción

Con el entorno Vitis cargado:

```bash
python3 scripts/run_ntrpu_hls.py --kernel swiglu_q8_block \
  --part xcvu5p-flva2104-2-i --clock-ns 5 --stage cosim \
  --output-root build/NEW_SWIGLU_BLOCK
```

El runner conserva todas las dependencias y sus hashes en una carpeta nueva.
Las pruebas de staging pasan 7/7; no se consideran evidencia RTL.
El lanzador de la cola es `tools/ntrpu_board_swiglu_block_cosim.tcl`.
Los respaldos incluyen fuentes, logs, reportes, comandos y revisión local;
consultar [VIVADO_STATUS](board/VIVADO_STATUS.md) para el estado de cada tanda.

## Candidato con pipeline en dos bucles

`20261003T233132Z-V7-swiglu-pipeline-both` aprueba 27 transacciones C/RTL,
incluidos 14 registros válidos idénticos bit a bit a la tanda ampliada.
Conserva estados, escala FP64, palabras INT8 y los 18 puertos locales.
Solo cambia dos directivas en copias de las fuentes: `PIPELINE off` pasa
a `PIPELINE II=1` en el bucle aritmético SwiGLU y en división/redondeo Q8.
No cambia exponencial, división, redondeo ni las fuentes HLS originales.

Los bloques válidos bajan de 2.934–3.794 a **598–722 ciclos**;
la mejora por caso es **4,24–6,14×**. Ambos bucles alcanzan II=1.
La reducción de máximo del cuantizador continúa secuencial. HLS estima
13.570 LUT, 8.388 FF, 45 DSP y 7 BRAM18: ocho DSP más que la referencia.
La estimación de período sigue siendo 3,647 ns; no es timing posruteo.
Ocho alteraciones de evidencia son rechazadas por el revisor independiente.

Reproducir en un directorio nuevo:

```bash
python3 tools/run_ntrpu_swiglu_pipeline.py --variant both \
  --tool /ruta/Vitis/bin/vitis-run --output build/NEW_SWIGLU_PIPELINE
```

El runner conserva originales, fuentes copiadas y hashes de cada cambio.
Respaldo verificado local/NFS: 57.201.148 bytes, SHA-256
`f06ef8cda2da0e856c3e5fdbed0da02132f3e20393b3f578ba67a9c9936b7e9c`.
Fuentes de ejecución:
`46c0b65e1e85234c8d7bc40a61dc556978235955801b946afdde69e7ebd42ed9`.
La cola separada ya aprueba y respalda 105 casos de memoria y 63 FFN
pequeñas (`20261003T234033Z-V7-swiglu-pipeline-memory` y
`20261003T234033Z-V7-ffn-quad-pipeline-errors`). El vector de 8192 elementos
`20261003T234644Z-V7-swiglu-pipeline-memory-large` aprueba sus 256 registros,
con nueve negativos y respaldo verificado. Brandon y síntesis también
pasan y están respaldados: la [comparación FFN](FFN_PERFORMANCE.md#ffn-quadfifo-con-swiglu-optimizado)
mide 34,02–34,09% menos ciclos con ocho DSP adicionales. El ruteo nativo
pasa timing interno a 5 ns (WNS +0,046/WHS +0,009 ns), con contexto OOC
pendiente. El banco corto PCIe optimizado también pasa y está respaldado;
Brandon por PCIe tiene una nueva tanda en curso. El factor del operador
aislado no es el de la FFN.
