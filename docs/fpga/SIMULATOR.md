# Simulador funcional de NTRPU

El simulador implementa memoria, transferencias, GEMV INT8/INT32, GEMV Q8_0 y SwiGLU FP32 entre NTRPU en C++20, sin CUDA ni bibliotecas externas. Las demostraciones configuran dos unidades de **512 GiB** y verifican tensores y cálculo distribuido entre ambas. La capacidad es virtual: solo se materializan las páginas escritas. No es todavía un motor de inferencia, un emulador de PCIe ni un diseño RTL.

## Compilación y ejecución

Desde la raíz del repositorio, con CMake 3.24+ y un compilador C++20:

```bash
cmake -S . -B build/ntrpu-sim -DNTRANSFORMER_ENABLE_CUDA=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/ntrpu-sim --config Release --parallel
ctest --test-dir build/ntrpu-sim -C Release --output-on-failure
```

En Linux con un generador de configuración única:

```bash
./build/ntrpu-sim/src/ntrpu/ntrpu_sim
./build/ntrpu-sim/src/ntrpu/ntrpu_gemv_demo
./build/ntrpu-sim/src/ntrpu/ntrpu_q8_demo
```

En PowerShell con Visual Studio:

```powershell
.\build\ntrpu-sim\src\ntrpu\Release\ntrpu_sim.exe
.\build\ntrpu-sim\src\ntrpu\Release\ntrpu_gemv_demo.exe
.\build\ntrpu-sim\src\ntrpu\Release\ntrpu_q8_demo.exe
```

La demostración `ntrpu_sim` imprime completions en orden `2, 1`, verifica los bytes recibidos y muestra capacidad lógica, memoria materializada y tráfico entre unidades. Mueve deliberadamente solo 96 bytes de prueba; no representa una carga de pesos real ni mide ancho de banda. `ntrpu_gemv_demo` calcula una matriz de 7 × 65 dividida por filas entre unidades y comprueba el resultado contra INT64; ver [contrato numérico y resultados](GEMV_INT8.md).

`ntrpu_q8_demo` ejecuta [dos operaciones Q8_0 encadenadas](Q8_DISTRIBUTED.md), con pesos repartidos y transferencia directa de resultados FP32 hacia la siguiente operación.

`ntrpu_ffn_fixture` ejecuta una [FFN con SwiGLU y pesos reales de Brandon](BRANDON_REFERENCE.md); una herramienta Python prepara entradas y referencias numéricas. CUDA se utiliza solo si se solicita la referencia PyTorch GPU.

`NTRANSFORMER_ENABLE_CUDA` permanece activado por defecto para conservar la compilación habitual del motor. Al desactivarlo, CMake no habilita el lenguaje CUDA ni busca el toolkit. `NTRANSFORMER_BUILD_NTRPU_SIM=OFF` omite este componente. `BUILD_TESTING=OFF` omite los tests.

## Componentes

| Archivo | Responsabilidad |
| --- | --- |
| `src/ntrpu/simulator.h` | API de buffers, tensores, comandos y completions |
| `src/ntrpu/simulator.cpp` | Memoria dispersa, validación, colas y ejecución funcional |
| `src/ntrpu/demo.cpp` | Demostración de dos unidades y pesos distribuidos |
| `src/ntrpu/gemv.h`, `gemv.cpp` | GEMV INT8/INT32 y codificación de salida |
| `src/ntrpu/gemv_demo.cpp` | Cálculo con filas repartidas y referencia INT64 |
| `src/ntrpu/q8_0.cpp` | Referencia Q8_0 y adaptación experimental al modelo C++ del top HLS |
| `src/ntrpu/numeric_io.cpp`, `q8_demo.cpp` | Codificación FP32 y demo Q8_0 encadenada |
| `fpga/hls/gemv_int8/` | Fuente HLS acotada y flujo Tcl para Vitis |
| `tests/test_ntrpu_sim.cpp` | Pruebas funcionales y de errores |
| `tests/test_ntrpu_gemv.cpp` | Pruebas numéricas, reparto y fallos de cómputo |

El namespace es `nt::ntrpu`. La API C++ es experimental y todavía no es un ABI binario para hardware.

## Contrato de memoria

Cada unidad tiene capacidad de 64 bits y páginas de respaldo de 4 KiB. Las regiones nunca escritas se leen como cero. Por defecto se permite materializar hasta 4.096 páginas por unidad, equivalentes a 16 MiB, además de los metadatos y buffers temporales del proceso. El contador `resident_bytes()` mide exclusivamente el respaldo de páginas, no el consumo total de RAM.

`register_buffer(unit, offset, size)` reserva un intervalo físico simulado sin asignar todo su respaldo. Rechaza intervalos fuera de capacidad y solapamientos con otras reservas de esa unidad. Los identificadores no se reutilizan y permiten invalidar referencias tras `unregister_buffer`. Liberar una reserva no borra ni devuelve las páginas materializadas en esta versión.

`publish_model(epoch, tensors)` valida la cobertura exacta de todos los tensores y publica el directorio de forma conjunta. Admite fragmentos en diferentes unidades y offsets superiores a 4 GB. Para `int8_row_major`, exige dimensiones positivas y tamaño `rows × columns`. Para `q8_0_row_major`, exige columnas múltiplos de 32 y tamaño `rows × (columns / 32) × 34`, sin overflow. Las escalas se comprueban al ejecutar Q8_0. Los demás formatos siguen siendo metadatos opacos para transporte; no hay un cargador de archivos GGUF en este componente.

Las épocas son positivas y estrictamente crecientes. Los buffers completos referenciados por los tensores publicados quedan protegidos contra escritura y liberación. Una nueva publicación reemplaza el directorio y libera las protecciones anteriores que ya no correspondan. Publicar un directorio vacío retira el modelo. Se exige consumir todas las completions antes de cambiar de época.

## Contrato de transferencias

`submit()` copia entre buffers y `submit_tensor_copy()` reúne fragmentos de un tensor en un destino registrado. Cada comando lleva época, secuencia, offsets y longitud. Por defecto la transferencia máxima es 1 MiB; el consumidor debe dividir operaciones mayores. Las rutas entre unidades se habilitan explícitamente y son direccionales.

La cola limita el total de trabajos pendientes más completions sin consumir. Una secuencia debe ser distinta de las que estén pendientes o sin consumir; puede reutilizarse después de `poll()`. La aplicación real deberá añadir las reglas de sesiones y protección contra replays que requiera el transporte.

Un comando aceptado retiene sus buffers. Las lecturas/escrituras host y la liberación se bloquean mientras tengan trabajo pendiente. Otros comandos pueden referenciarlos; el orden explícito de `execute(sequence)` determina sus dependencias y resultados. No se infiere un grafo de dependencias ni se ejecutan trabajos automáticamente.

`execute()` vuelve a verificar disponibilidad y rutas, reúne primero los bytes fuente y después escribe el destino. Las copias solapadas tienen semántica de `memmove`. La completion se hace visible después de actualizar los datos. Errores de validación o de presupuesto de páginas no modifican el contenido del destino y reportan cero bytes transferidos. Un fallo de asignación del proceso puede dejar páginas cero materializadas, sin cambiar los bytes del payload.

`cancel()` completa un trabajo pendiente con estado cancelado, sin transferir datos. `set_online()` y `set_peer_access()` inyectan fallos antes de ejecutar; restablecer disponibilidad conserva la memoria simulada, no representa un reset físico. El simulador es determinista y de un solo hilo: no reproduce escrituras parciales, tiempos de espera, caches, IOMMU ni tráfico real.

## Validación

Hay **45 pruebas** registradas en CTest: 15 casos de memoria/transporte, siete de GEMV INT8, cuatro del top HLS en CPU, cuatro de Q8_0, seis de Q8_0 distribuido, tres demos, cuatro casos de FFN/SwiGLU y dos pruebas Python (lanzador HLS y archivos FFN) si Python está disponible. Cubren límites y overflow, permisos, épocas, colas, completions fuera de orden, fallos de unidades/enlaces, cancelación, presupuesto, solapamientos, aritmética, formatos y errores numéricos. La [guía HLS](HLS.md) distingue las pruebas host de las verificaciones pendientes en Vitis.

Una prueba publica un tensor virtual de **800 GiB** repartido en fragmentos de 512 y 288 GiB. Lee 16 bytes que cruzan la frontera entre unidades usando solo tres páginas materializadas. Otra contrasta 200 transferencias reproducibles con un vector de referencia, incluyendo offsets en origen y destino y escrituras que cruzan páginas.

Las comprobaciones fallan con un código de salida distinto de cero también en Release; no dependen de `assert` ni desaparecen con `NDEBUG`. Las 45 pruebas pasaron en Windows x64/MSVC 19.51 y Linux/WSL/GCC 13.3, Release, el 2026-10-01. La compilación del motor CUDA original no se validó en estos hitos; la comparación PyTorch GPU se documenta por separado.

## Siguiente hito

Los contratos [INT8/INT32](GEMV_INT8.md), [Q8_0/FP32](Q8_DISTRIBUTED.md) y la [FFN SwiGLU](BRANDON_REFERENCE.md) ya están implementados y verificados con pesos distribuidos. Siguen un bloque completo, adaptación del modelo y carga GGUF, además de validar el [top HLS](HLS.md) con Vitis. Las ventanas BAR, los descriptores binarios y los drivers deberán desarrollarse y validarse posteriormente contra hardware real.
