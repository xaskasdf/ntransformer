# Q8_0 distribuido y operaciones encadenadas

El simulador ya ejecuta `submit_q8_0_gemv()` sobre filas de pesos Q8_0 residentes en cada NTRPU. Recibe y produce FP32, por lo que un resultado puede alimentar otra operación mediante los comandos DMA existentes. El cálculo usa el top HLS como función C++ y aplica cuantización y escalas en CPU; todavía no es ejecución FPGA ni un backend de inferencia completo.

## Ejecutar la demo

Después de [compilar sin CUDA](SIMULATOR.md):

```powershell
.\build\ntrpu-sim\src\ntrpu\Release\ntrpu_q8_demo.exe
```

En Linux con generador de configuración única, el ejecutable corresponde a `./build/ntrpu-sim/src/ntrpu/ntrpu_q8_demo`. Esta versión se validó en Windows x64, MSVC 19.51, Release.

La demo publica dos matrices y ejecuta ocho comandos:

1. Replica la entrada de 96 valores FP32 de la unidad 0 a la 1.
2. Calcula una matriz de 64 × 96, dividida en 33 y 31 filas entre ambas unidades.
3. Verifica las dos completions y reúne las 64 salidas FP32 en la unidad 0.
4. Copia esos bytes a la unidad 1, donde una matriz local de 32 × 64 produce la salida final.
5. Devuelve los 32 resultados a la unidad 0 y los compara con la misma ruta cuantizada sin repartir.

Los buffers de resultados alimentan la segunda operación sin llamadas intermedias a `Fabric::read()` ni `Fabric::write()`. Esta demo es una cadena de dos operaciones lineales; la [prueba de Brandon](BRANDON_REFERENCE.md) añade una FFN con SwiGLU.

| Resultado comprobado | Valor |
| --- | --- |
| Igualdad distribuido frente a modelo cuantizado monolítico | Byte por byte |
| Comandos GEMV Q8_0 correctos | 3 |
| MAC correctos | 8.192 |
| Pesos locales leídos, incluidas escalas | 8.704 bytes |
| Tráfico entre unidades | 892 bytes |
| DMA total, incluida la copia local de 33 salidas | 1.024 bytes |
| Respaldo de páginas materializado | 45.056 bytes |
| Capacidad lógica virtual | 2 × 512 GiB |

Los 892 bytes corresponden a `384 entrada + 124 reunión remota + 256 entrada siguiente + 128 resultado final`. La carga inicial de pesos desde el host queda fuera de esos contadores. No son medidas de rendimiento ni evidencia de memoria física de 1 TiB.

## Contrato de publicación y ejecución

Publicar los pesos con `format = "q8_0_row_major"`, `rows` y `columns`. Cada fila contiene bloques de 34 bytes: escala FP16 little endian y 32 valores INT8. Debe cumplirse `size_bytes = rows × (columns / 32) × 34`, sin padding. La publicación comprueba dimensiones, overflow y cobertura; los valores de escalas y activaciones se comprueban al ejecutar.

```cpp
fabric.publish_model(epoch, {{tensor_id, weight_bytes,
    {{0, {weights_a}, first_bytes},
     {first_bytes, {weights_b}, remaining_bytes}},
    "q8_0_row_major", rows, columns}});

fabric.submit_q8_0_gemv({sequence, epoch, unit, tensor_id,
                        first_row, row_count, {local_input}, {local_output}});
```

| Propiedad | Regla |
| --- | --- |
| Filas | Intervalo positivo dentro del tensor publicado |
| Columnas por comando | Múltiplo de 32, entre 32 y 131.040 |
| Pesos por comando | Hasta 1 MiB, contando escalas |
| Entrada | `columns × 4` bytes FP32 little endian |
| Salida | `row_count × 4` bytes FP32 little endian |
| Localidad | Pesos seleccionados, entrada y salida pertenecen a `unit` |
| Colas y permisos | Épocas, secuencias, pins, completions y cancelación compartidos con DMA |

Los fragmentos pueden dividir una escala entre buffers de la misma unidad: se reúnen antes del cálculo. No se acepta una fila repartida entre unidades ni un operando remoto, aunque exista una ruta P2P. El llamador replica las entradas y reúne las salidas mediante DMA.

`submit_gemv()` conserva su contrato INT8/INT32 y rechaza este formato. La etiqueta opaca `Q8_0` no equivale al formato tipado `q8_0_row_major`; se exige este último para evitar interpretar bytes sin dimensiones verificadas.

## Precisión y fallos

La ruta aplica la [cuantización por bloques documentada en HLS.md](HLS.md): INT8 para activaciones, dot INT32 y escalas/acumulación double en CPU. Al finalizar convierte a FP32. La demo también compara con dos GEMV de referencia usando las activaciones originales de cada etapa, con una conversión FP32 entre ellas: obtuvo error L2 relativo `0.00670101`, aproximadamente **0,670 %**, sobre estos datos sintéticos. Esto no mide calidad de un modelo ni equivale a una comparación bit a bit con CUDA.

NaN/Inf en activaciones o escalas, o un resultado double fuera del rango finito FP32, producen `NumericError`. La conversión final puede redondear valores pequeños a subnormales o cero. No se escriben resultados parciales de un comando: se calcula y valida toda su salida antes de modificar el destino. Una falta de páginas, cancelación o pérdida de unidad tampoco publica su salida.

La atomicidad es por comando. No existe rollback de toda la cadena ni scheduler de dependencias: el consumidor debe comprobar todas las completions de una etapa antes de emitir trabajo dependiente, y gestionar cualquier trabajo pendiente tras un fallo. La demo cumple ese orden explícitamente.

`macs`, `local_weight_bytes` y `compute_output_bytes` cuentan únicamente comandos correctos. `local_weight_bytes` incluye los dos bytes de escala de cada bloque. `q8_0_commands` es un subconjunto de `gemv_commands`; `bytes_copied` y `peer_bytes` siguen contando solo DMA simulado. Una completion numérica correcta informa `bytes_transferred = 0`, los bytes de salida y los MAC; una fallida informa cero en esos tres campos.

## Validación y trabajo pendiente

Pasaron **40/40 pruebas** en Release el 2026-09-30. Los seis casos nuevos cubren codificación FP32, cuatro matrices de 67 filas repartidas como 34+33, tamaños que requieren varios tiles HLS, offsets mayores de 4 GiB, escalas cortadas entre spans locales, límites de formato/tamaño, errores numéricos sin escritura parcial, pérdida de unidad, cancelación, cola, presupuesto y buffers solapados. La demo encadenada es el séptimo caso añadido.

La etapa posterior ya incorpora una [FFN SwiGLU con pesos de Brandon](BRANDON_REFERENCE.md), leídos desde una revisión fija safetensors y comparados con CPU/GPU. Siguen pendientes la adaptación completa del modelo, la carga GGUF, síntesis/co-simulación con Vitis, precisión hardware de escalas y conexión AXI/DDR/PCIe. El top HLS permanece sin métricas RTL nuevas.
