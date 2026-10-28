# Núcleo GEMV INT8 y ejecución distribuida

Este hito añade cálculo numérico al simulador: `y = W × x`, con pesos y entrada INT8 firmados y salida INT32. Es una implementación nueva en CPU, comprobada contra una referencia INT64 independiente. No son fuentes recuperadas, síntesis HLS ni inferencia de un modelo GGUF.

## Ejecutar la demostración

Usar la [compilación sin CUDA del simulador](SIMULATOR.md). Después, en PowerShell con Visual Studio:

```powershell
.\build\ntrpu-sim\src\ntrpu\Release\ntrpu_gemv_demo.exe
```

En Linux, con un generador de configuración única, la ruta es `./build/ntrpu-sim/src/ntrpu/ntrpu_gemv_demo`. La validación de este hito se realizó en Windows x64, MSVC 19.51, Release; Linux, CUDA y herramientas HLS no se ejecutaron.

La demo divide las siete filas de una matriz de 7 × 65 entre dos unidades, en grupos de tres y cuatro. Replica la entrada, ejecuta cada grupo junto a sus pesos y reúne la salida. Comprueba estos resultados:

```text
Output: 56577 -33200 -35937 89838 -58051 66444 -43301
Logical capacity: 1024 GiB
Local weight bytes: 455
Peer bytes: 81 (65 input + 16 output)
Materialized page backing: 28672 bytes
```

Son 455 multiplicaciones-acumulaciones (MAC). La copia local de las primeras tres salidas agrega 12 bytes al contador DMA total, pero no al tráfico entre unidades. La carga inicial desde el host no está incluida en esos contadores. La capacidad es virtual; las cifras no miden rendimiento ni demuestran hardware de 1 TiB.

## Contrato numérico

| Propiedad | Regla |
| --- | --- |
| Pesos | `int8_row_major`, filas contiguas, sin padding ni escalas |
| Entrada | Vector INT8 de `columns` elementos, rango −128…127 |
| Salida | Un INT32 por fila, complemento a dos, little endian |
| Dimensiones | Filas positivas; `1 <= columns <= 131071` |
| Trabajo por comando | Como máximo 1 MiB de pesos; dividir matrices mayores por filas |
| Aritmética | Exacta, sin bias, saturación, redondeo ni wraparound |

El mayor valor absoluto de un producto es `(-128) × (-128) = 16384`. Con `K <= floor(INT32_MAX / 16384) = 131071`, todas las sumas parciales caben en INT32. Se rechazan columnas mayores incluso si una entrada concreta no desbordaría. La función CPU procesa grupos de 32 elementos y su resto; esto no acredita 32 lanes físicos ni un intervalo de iniciación HLS.

`gemv_int8()` recibe spans de pesos y entrada y devuelve un vector INT32. Comprueba que el tamaño coincida con la matriz sin multiplicaciones desbordadas. `pack_int32_le()` y `unpack_int32_le()` definen explícitamente la representación de salida.

## Comandos y reparto

Al publicar un tensor con `format = "int8_row_major"`, indicar `rows` y `columns`. La publicación exige que `size_bytes == rows × columns`. `submit_gemv()` acepta únicamente ese formato; Q8_0 tiene [su propio contrato y comando](Q8_DISTRIBUTED.md). Los formatos desconocidos permanecen como metadatos opacos para transporte.

```cpp
fabric.publish_model(1, {{42, rows * columns,
    {{0, {w0}, split * columns},
     {split * columns, {w1}, (rows - split) * columns}},
    "int8_row_major", rows, columns}});

// Tras replicar la entrada a x0 y x1:
fabric.submit_gemv({2, 1, 0, 42, 0, split, {x0}, {y0}});
fabric.submit_gemv({3, 1, 1, 42, split, rows - split, {x1}, {y1}});
fabric.execute(3);
fabric.execute(2);
```

Cada comando especifica secuencia, época, unidad, tensor de pesos, primera fila, cantidad de filas, entrada y salida. Todos sus operandos deben pertenecer a la unidad ejecutora. Fragmentos de una fila pueden ocupar varios buffers de esa misma unidad; una fila que cruce unidades se rechaza con `NonLocal`. Habilitar una ruta P2P no elimina esta restricción. El ejemplo reúne filas completas y no requiere reducir sumas parciales entre tarjetas.

GEMV comparte cola, reserva de buffers, completions y cancelación con DMA. El consumidor debe ejecutar la copia de entrada antes del cálculo y el cálculo antes de reunir resultados; no hay scheduler automático. Se vuelven a comprobar los dispositivos antes de ejecutar. Una cancelación, pérdida de unidad o falta de páginas no cambia el payload de salida ni contabiliza MAC exitosos. Se leen todos los operandos antes de escribir, por lo que entrada y salida pueden solaparse en un buffer mutable.

Las completions GEMV tienen `bytes_transferred = 0`, `output_bytes = 4 × row_count` y `macs = row_count × columns` cuando terminan correctamente. `bytes_copied` y `peer_bytes` siguen midiendo únicamente DMA simulado. Los contadores de cómputo son `gemv_commands`, `macs`, `local_weight_bytes` y `compute_output_bytes`; representan operaciones lógicas exitosas, no transacciones de un controlador DDR.

## Relación con GGUF Q8_0

El formato local [`BlockQ8_0`](../../src/core/types.h) ocupa **34 bytes**: una escala FP16 de dos bytes y 32 pesos INT8. El kernel actual [`gemv_q8_0_kernel`](../../src/cuda/gemm.cu) combina esos pesos con activaciones FP32 y aplica una escala por bloque:

```text
y[r] = sum_b dW[r,b] * sum_j qW[r,b,j] * x[32*b+j]
```

El nuevo núcleo calcula únicamente productos de enteros. No se puede eliminar la escala del bloque y tratar un GGUF como una matriz INT8 plana. Si se decide cuantizar también la entrada por bloques, la expresión propuesta sería:

```text
x[32*b+j] ≈ dX[b] * qX[b,j]
y[r] ≈ sum_b dW[r,b] * dX[b] * dot_int8(qW[r,b], qX[b])
```

Cada dot de 32 elementos cabe en INT32, pero las escalas deben aplicarse **antes de sumar bloques con escalas diferentes**. La [adaptación experimental Q8_0](HLS.md) implementa en CPU la lectura FP16, cuantización por bloques, tratamiento de ceros y rechazo de valores no finitos. Llama al top HLS como función C++ y aplica escalas en double. Está [conectada al runtime distribuido](Q8_DISTRIBUTED.md), con resultados FP32 y pruebas sintéticas. Faltan implementación hardware de escalas y evaluación contra logits de un modelo.

## Pruebas y siguiente paso

`tests/test_ntrpu_gemv.cpp` añade siete casos a CTest, más la demo: tamaños irregulares y límites de 32 elementos; valores extremos y límite de acumulación; bytes de salida; seis particiones reproducibles entre unidades; formatos y comandos inválidos; cancelación y pérdida de unidad; presupuesto de páginas y solapamiento. Las referencias usan bucles escalares INT64 independientes del núcleo. Las 24 pruebas del componente completo pasaron en Release el 2026-09-30, con fallos que conservan código de salida no cero aunque esté definido `NDEBUG`.

El [top HLS de 32 × 1024 y su flujo reproducible](HLS.md) ya están escritos y verificados como C++14. Las pruebas completas ahora son 40, incluida la adaptación Q8_0 distribuida y la preparación del flujo. Falta ejecutar Vitis C simulation, síntesis y co-simulación. El informe recuperado sirve como evidencia histórica; las nuevas estimaciones deberán conservar su propia procedencia. FFN y el backend de inferencia completo siguen pendientes.
