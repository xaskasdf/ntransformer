# Primer núcleo HLS de NTRPU

> Estado actualizado (2026-10-01): licencia instalada; INT8 pasa co-simulación C/RTL. Q8 y cuantizador siguen en validación; el cuantizador no cumple 5 ns. Véase [evidencia HLS](validation/hls-20261001.md). Las menciones anteriores a licencia pendiente describen el bloqueo histórico.

La fuente nueva `fpga/hls/gemv_int8/ntrpu_gemv_int8.cpp` implementa un GEMV INT8/INT32 con memoria acotada y pragmas HLS. Comparte banco de pruebas entre compilación C++14 normal y el flujo preparado para Vitis. No reemplaza las fuentes perdidas ni reutiliza las métricas del informe recuperado.

## Estado comprobado

Actualización 2026-10-01: Vivado/Vitis **2026.1**, build 6511674 (HLS 6493734), quedó instalado en la VM. La primera ejecución real `csim`, `20261001T062609Z-csim-zrxml0ox`, terminó con código 1 en `set_part`: Vivado informó que falta una licencia válida. No llegó a `csim_design`; tampoco hay síntesis ni RTL validado. El informe fallido y su manifiesto se conservaron en el volumen persistente. El estado histórico siguiente describe las comprobaciones previas a esta instalación.

En el hito HLS inicial del 2026-09-30 pasaron **33 pruebas** en Windows x64, MSVC 19.51, Release: las 24 anteriores, cuatro del top HLS ejecutado en CPU, cuatro de Q8_0 y una del lanzador Python. La prueba Python contiene tres verificaciones de preparación, parámetros e instalación ausente. En ese hito había 32 pruebas C++ sin Python.

La integración posterior de [Q8_0 al simulador distribuido](Q8_DISTRIBUTED.md) amplió la suite a **40 pruebas**, todas correctas en el mismo entorno (39 si Python no está disponible). El estado RTL descrito a continuación permanece pendiente.

Se verificó también la sintaxis Tcl y el orden de las cuatro etapas con comandos AMD sustituidos por stubs. El `dry-run` copió y registró las fuentes. **No se ejecutaron Vitis C simulation, síntesis, co-simulación RTL ni exportación de IP**: no se encontró `vitis-run` en el PATH ni una instalación en las ubicaciones revisadas. No hay cifras nuevas de frecuencia, DSP, LUT, latencia o ancho de banda.

## Contrato del top

```cpp
uint32_t ntrpu_gemv_int8(const int8_t* weights, const int8_t* input,
                        int32_t* output, uint32_t rows, uint32_t columns);
```

| Elemento | Contrato |
| --- | --- |
| Dimensiones | `1..32` filas y `1..1024` columnas |
| Pesos | INT8 firmado, filas contiguas, exactamente `rows * columns` valores |
| Entrada | `columns` valores INT8 firmados |
| Salida | `rows` valores INT32; se sobrescriben, no se acumulan sobre el destino |
| Retorno | `0`: correcto; `1`: dimensiones inválidas, sin acceder a memoria |
| Aritmética | Sin escalas ni bias; suma exacta, sin saturación |
| Memoria | Direcciones válidas y regiones no solapadas, garantizadas por el llamador |

El máximo absoluto por fila es `1024 * 16384 = 16777216`, dentro de INT32. La función host admite más columnas; este top es un tile. Para INT8 plano se pueden dividir columnas y sumar resultados parciales respetando el límite global. Para Q8_0 se deben aplicar las escalas de cada bloque antes de sumar.

La implementación carga la entrada en un buffer local y procesa una fila a la vez: carga pesos, acumula en 32 lanes y reduce a una salida. Las memorias locales se particionan cíclicamente en 32 bancos; los acumuladores se particionan completamente. `PIPELINE II=1` y `UNROLL` son solicitudes al sintetizador, no resultados medidos. Esta primera versión separa carga y cálculo, sin doble buffer ni optimización de ancho de banda DDR.

## Interfaces propuestas

Hay tres bundles AXI4 master: `weights_mem`, `input_mem` y `output_mem`. El bundle AXI4-Lite `control` recibe direcciones base y dimensiones y expone control/finalización y retorno. `offset=slave` lleva las direcciones base al plano de control; `config_interface -m_axi_addr64=true` solicita direccionamiento AXI de 64 bits.

Estas opciones siguen [las interfaces HLS](https://docs.amd.com/r/2025.1-English/ug1399-vitis-hls/pragma-HLS-interface) y [la configuración de interfaces de AMD](https://docs.amd.com/r/2025.2-English/ug1399-vitis-hls/config_interface). Tener tres bundles no equivale a disponer de tres canales DDR físicos. Vivado debe conectarlos a la memoria y arbitraje elegidos.

El mapa de registros definitivo se tomará de los headers del IP generado; todavía no es un ABI NTRPU. Faltan PCIe/BAR, DDR/MIG, colas, driver, resets y gestión de errores AXI. El control de dimensiones no valida registros de memoria ni permisos DMA; esa comprobación pertenece al runtime y al sistema de hardware.

## Compilar y probar en CPU

Usar los [comandos habituales del simulador](SIMULATOR.md). Para seleccionar las pruebas nuevas:

```powershell
ctest --test-dir build/ntrpu-sim -C Release -R 'ntrpu_(hls|q8_0)' --output-on-failure
```

`tests/test_ntrpu_hls.cpp` contiene 39 llamadas válidas y seis inválidas: dimensiones pequeñas y máximas, restos de 32, signos extremos, ceros y ejecuciones repetidas con distintas longitudes. Usa una referencia INT64 y códigos de fallo activos también en Release. El compilador host ignora los pragmas HLS; estas pruebas comprueban la función C++, no los adaptadores AXI ni la temporización RTL.

## Preparar y ejecutar Vitis

Para instalar y ejecutar las herramientas en la instancia cloud propuesta, consultar [Vivado/Vitis en una estación remota](CLOUD_HLS.md).

El lanzador usa el flujo Vivado IP mediante Tcl y la [CLI `vitis-run` documentada por AMD para 2025.2](https://docs.amd.com/r/2025.2-English/ug1399-vitis-hls/vitis-v-and-vitis-run-Commands). Requiere Python 3.10+ y, para ejecutar herramientas, el entorno AMD cargado y soporte para la pieza seleccionada.

Ejemplo de preparación, usando únicamente como referencia la pieza que aparece en el informe recuperado:

```powershell
python scripts/run_ntrpu_hls.py --part xczu7ev-ffvc1156-2-e --clock-ns 5 --stage cosim --dry-run
```

Esa pieza no confirma la placa disponible ni el diseño final de 512 GB. Los 5 ns son un objetivo inicial de síntesis, no una frecuencia alcanzada. Para el dispositivo elegido, ejecutar desde un entorno con Vitis:

```powershell
python scripts/run_ntrpu_hls.py --part <pieza-exacta> --clock-ns 5 --stage csim
python scripts/run_ntrpu_hls.py --part <pieza-exacta> --clock-ns 5 --stage cosim
```

Reemplazar `<pieza-exacta>` antes de ejecutar. `--tool` permite indicar una ruta a `vitis-run` si el resto del entorno ya está configurado. Las etapas son acumulativas:

| Etapa | Operaciones |
| --- | --- |
| `csim` | Compilación y ejecución del banco C++ en Vitis |
| `synth` | C simulation y síntesis HLS |
| `cosim` | Las anteriores y co-simulación C/RTL Verilog |
| `export` | Las anteriores y exportación como IP catalog |

Cada invocación crea un directorio nuevo `build/ntrpu-hls/<fecha-UTC>-<etapa>-<id>/`. Contiene copia de las fuentes y testbench, hashes SHA256, pieza, reloj, comando y estado en `manifest.json`. Al ejecutar herramientas agrega `tool-version.log`, `tool.log` y los resultados bajo `project/solution1/`. No borra ni reinicia proyectos anteriores. `dry_run` significa únicamente preparación; `tool_succeeded` registra salida cero del proceso y requiere revisar sus informes, no certifica timing ni una placa funcional.

Conservar y revisar el informe de síntesis, warnings, interfaz generada y reporte de co-simulación antes de integrar el IP. La exportación no realiza place-and-route de una tarjeta ni genera su bitstream.

## Adaptación experimental Q8_0

Actualización: el [nuevo top Q8_0 con escalas](HLS_Q8_0.md) implementa la decodificación FP16 y acumulación escalada dentro de su fuente HLS. El adaptador lo ejecuta como C++ para hasta 1024 columnas; para anchos mayores conserva el camino por bloques descrito abajo. La cuantización de activaciones continúa en el host y ninguna de las dos rutas ha pasado co-simulación RTL.

`src/ntrpu/q8_0.cpp` recibe bloques Q8_0 ya extraídos, no un archivo GGUF completo. Lee escalas FP16 little endian, incluidos subnormales y escalas negativas finitas, y rechaza escalas o activaciones NaN/Inf.

`q8_0_gemv_reference()` descuantiza los pesos y usa las activaciones FP32 originales con acumulación double. En el camino conservado para anchos mayores de 1024, `q8_0_gemv_hls_model()` cuantiza cada bloque de 32 activaciones con `dX = maxabs / 127`, rango −127…127 y empates alejándose de cero; para un bloque cero usa escala y enteros cero. Invoca la misma fuente del top como función C++ para hasta 32 filas por bloque y acumula `dW * dX * dot_int8` en CPU/double. **La cuantización, escalas y acumulación double aún no son hardware HLS**. Ahora se invocan desde `Fabric::submit_q8_0_gemv()`, con conversión final a FP32 y [pruebas distribuidas](Q8_DISTRIBUTED.md).

La prueba sintética de 35 × 96, semilla `20260930`, obtuvo error absoluto máximo `44.7415` y error L2 relativo `0.00434318` (aproximadamente 0,434 %). Cada salida respeta la cota `sum(abs(weight) * dX / 2)`, más tolerancia de redondeo. Es un resultado sobre datos sintéticos; no mide calidad de un modelo, logits o perplexity. La referencia double tampoco reproduce exactamente el orden de sumas FP32 del backend CUDA.

El siguiente paso verificable es ejecutar este flujo en Vitis, corregir lo que revele la co-simulación y seleccionar la precisión y el lugar de aplicación de escalas. Luego integrar memoria y una operación completa antes de añadir FFN y el backend de inferencia.
