# Matrices completas sobre memoria AXI

Actualizado el 2026-10-03 UTC. V7 sigue abierto: este camino utiliza memoria
AXI simulada, el MAC HLS real y el [GEMV con salida en memoria](RESULT_MEMORY.md).
No integra todavía DDR, DMA PCIe ni una FFN encadenada.

## Control y formato

`fpga/rtl/ntrpu_matrix_sequencer.sv` acepta una matriz y emite grupos de
hasta 32 filas hacia el motor existente. `ntrpu_q8_matrix_memory.sv` conecta
ambos módulos; expone AR/R y AW/W/B de 512 bits, direcciones de 64 bits y
un único reloj. No hace falta enviar cada tile desde el host.

| Campo | Contrato |
| --- | --- |
| `job_layout` | 2: bloques Q8 de 34 bytes, escala FP16 little-endian |
| `job_rows` | 1–8192 filas |
| `job_k_blocks` | 1–256 bloques de 32 componentes |
| `job_tile_k` | 1–32 bloques por tile |
| Descriptores | Dirección, base y longitud registradas; activaciones/salida con época |
| Completion | Estado, identidad raíz, filas privadas completadas y bytes W con strobe |

Orden físico: `[grupo de filas][tile K][bloque K][fila][34 bytes]`.
Los tiles son contiguos y su tamaño debe ser múltiplo de 64 bytes. Se
comprueba toda la geometría de la última fila/grupo antes del primer acceso;
el módulo no añade padding. El host debe preparar ese padding explícitamente
cuando la forma lo requiera. La salida es un vector de binary64 little-endian.

Se validan alineación de 64 bytes, rangos completos, overflow de 64 bits,
épocas y ausencia de solapamiento de la salida con ambas regiones de entrada.
Las activaciones se recargan para cada grupo. El secuenciador conserva el
orden de acumulación del MAC y suma las direcciones sin truncarlas.

Solo una completion raíz con estado cero permite publicar **toda** la matriz.
Un fallo o cancelación tardíos pueden dejar grupos correctos escritos en
el destino privado: deben descartarse. `done_rows` y `done_bytes` sirven para
diagnóstico, no autorizan publicación parcial. El reset debe coordinarse con
el motor y sus peers AXI; un trabajo interrumpido no emite completion.

## Validación y resultados

El secuenciador aislado pasa 135/135 casos con un motor de protocolo simulado:
límites, 65.536 tiles, wrap de secuencia, respuestas incorrectas, cancelación,
reset y backpressure. Esto comprueba descriptores, **no** 65.536 cálculos MAC.
La primera síntesis falló setup, WNS −1,310 ns; registrar conteos, productos
y direcciones corrigió esa ruta sin excepciones de timing. La revisión pasa
135/135 y alcanza WNS estimado +1,770 ns.

El conjunto con MAC real pasa 30/30 casos, semillas 1/7/42. Cada semilla
calcula 387 valores privados y escribe 363; solo 195 pertenecen a matrices
completadas correctamente. Incluye fallos RRESP/BRESP del segundo grupo,
cancelación durante su carga, reset durante W, rechazo previo de rangos y
recuperación. Un modelo conserva todos los bytes del destino entre grupos
y verifica guardas, direcciones altas, cruces de página y respuestas B.
Python recalcula independientemente cada valor escrito; 13 alteraciones
de evidencia son rechazadas por el revisor.

Las tres matrices reales de Brandon pasan 9/9 ejecuciones y **5088 valores
escritos exactos**, con entradas sintéticas independientes. Revisión del modelo:
`ca6329de991a1cfc17b2a7a1aaae2cc38c08577a`. Los pesos y referencias proceden
del fixture ya revisado, sin nueva cuantización; el decodificador CPU revisa
cada bloque físico y recalcula todos los productos Q8 ordenados.

| Matriz | Forma lógica/almacenada | Ciclos por semilla |
| --- | --- | --- |
| Gate | 720 × 256 | 73.095 / 72.926 / 72.941 |
| Up | 720 × 256 | 73.071 / 72.978 / 72.988 |
| Down | 256 × 720; K almacenado 736 | 73.228 / 73.101 / 73.144 |
| Total | 1696 valores, 54 grupos | 219.394 / 219.005 / 219.073 |

Por semilla: 35.328 bytes de activaciones, 591.872 de pesos y 13.568 de
salida, en 59 ráfagas de escritura. Los 2,921–2,926 bytes de bus por ciclo
incluyen recargas y pausas del banco; no son caudal DDR ni tokens/s.
La separación gate/up/down usa entradas independientes, sin SwiGLU entre ellas.

El banco anterior enviaba 54 jobs y 116 tiles externamente; ahora envía
3 comandos de matriz y el secuenciador genera esos tiles. En las mismas
tres semillas, los ciclos Brandon anteriores fueron
219.853 / 219.453 / 219.552. El cambio reduce la intervención externa;
conserva los dos MAC. Las pausas dependen del flujo de cada banco, por lo
que esta diferencia pequeña no mide aisladamente una aceleración ni el
coste de comandos PCIe del host.

La síntesis OOC del conjunto usa **19.369 LUT**, 15.191 FF, 54 DSP,
37 RAMB36 + 2 RAMB18 y 320 LUTRAM. Con reloj de 5 ns obtiene WNS +1,450 ns
y WHS +0,039 ns, sin endpoints internos sin restricción. Faltan retardos de
1269 entradas y 825 salidas; quedan advertencias DSP y de puertos aritméticos
generados. Es síntesis, no cierre temporal de placa.

El ruteo OOC posterior usa **19.212 LUT**, 15.095 FF, 54 DSP y
37 RAMB36 + 2 RAMB18. Obtiene WNS +0,614 ns y WHS +0,010 ns con las
restricciones internas de 5 ns. Vivado cuenta 32.842 nets ruteables completas
y cero errores de ruteo; permanece un gap explícito del reloj padre.
La auditoría identifica 450 segmentos jerárquicos con la misma ruta
`PARTIAL`. `HD.CLK_SRC=BUFGCE_X0Y60` es provisional; las 1269 entradas
y 825 salidas siguen sin retardos externos. No se añadieron excepciones
ni dispensas: falta el reloj y las interfaces del sistema para cerrar timing.

## Formas completas de Llama 1B

Dos matrices **sintéticas** pasan en RTL con el MAC real, semilla 1:

| Forma | Valores escritos exactos | Ciclos del job | Bytes de salida | Respuestas B |
| --- | --- | --- | --- | --- |
| Up, 8192 × 2048 | 8192 | 6.463.892 | 65.536 | 272 |
| Down, 2048 × 8192 | 2048 | 6.451.524 | 16.384 | 68 |

Cada job lee 1.048.576 bytes de activaciones, incluidas recargas, y
17.825.792 bytes de pesos: 18 MiB en total. Procesa 512 tiles; up usa
256 grupos de filas y down 64. La memoria final conserva y compara
independientemente los **10.240 valores**, con guardas y todas las B drenadas.
Las duraciones fueron 2603,995 s y 2582,270 s de ejecución de herramientas.

Con el reloj simulado de 5 ns, los jobs representan 32,319460 ms y
32,257620 ms; el tráfico medido equivale a 2,9301 y 2,9281 bytes de bus
por ciclo. Son medidas de estos bancos con pausas, sin DDR/PCIe físicos,
pesos Meta ni una FFN encadenada. No representan inferencia ni tokens/s.

Ambas imágenes pasaron decodificación CPU y referencia Q8 independiente;
el revisor también rechaza 13 alteraciones de evidencia. Su primera revisión
se detuvo por esperar cuatro avisos de tamaño de arrays cuando el banco
grande emite tres. Se corrigió esa expectativa, conservando los logs y
sin cambiar el DUT, los datos ni repetir las simulaciones aprobadas.

## Reproducción y evidencia

Preparación y revisión CPU:

```powershell
python -m unittest discover -s tests -p test_ntrpu_matrix_layout.py -v
python tools/prepare_ntrpu_matrix_fixture.py --profile brandon `
  --memory-fixture build/ntrpu-gemv-memory-20261003 `
  --output build/NEW_MATRIX_FIXTURE
```

`tools/ntrpu_matrix_layout.py` empaqueta Q8 denso sin modificar un directorio
existente. `tools/check_ntrpu_matrix_fixture.py` decodifica sin utilizar el
empaquetador. `tools/check_ntrpu_matrix_memory.py` comprueba logs, identidades,
grupos, tráfico y bits. En Linux, con Vivado 2026.1 cargado:

```bash
vivado -mode batch -source tools/ntrpu_board_matrix_memory.tcl \
  -tclargs "$MAC_COSIM_RUN" sim "$NEW_OUTPUT" "$MATRIX_FIXTURE"
```

Omitir el último argumento ejecuta los diez casos de fallos/recuperación;
`synth` genera un DCP separado. Un quinto argumento `1` o `2` selecciona una
matriz grande, habilita un máximo de imagen de 20 MiB y ejecuta solo semilla 1.
No modifica los RTL ni las fuentes de ejecuciones anteriores.

Respaldos verificados en `artifacts/ntrpu-board-vivado/<run>/`, con fuentes,
logs, revisión, DCP cuando corresponde, `FILES.json` y `VERIFIED.json`:

- `20261003T060606Z-V7-matrix-sequencer-unit` y `...-synth`: referencia inicial y fallo temporal conservado.
- `20261003T061228Z-V7-matrix-sequencer-unit-r2` y `...-synth-r2`: corrección validada.
- `20261003T062214Z-V7-matrix-memory-unit` y `...-synth`: conjunto integrado.
- `20261003T063235Z-V7-matrix-memory-brandon`: matrices reales completas.
- `20261003T064137Z-V7-matrix-memory-route` y `20261003T064736Z-V7-matrix-memory-route-audit`: implementación y límites de reloj.
- `20261003T064207Z-V7-matrix-llama-up` y `...-down`: formas completas sintéticas, memoria final y tráfico revisados.

## Siguiente tanda

La [FFN por memoria](FFN_MEMORY.md) ya encadena gate/up,
[SwiGLU/Q8](SWIGLU_MEMORY.md) y down con completion raíz y pasa 63 casos
sintéticos de cálculo/error/reset y tres cadenas con pesos reales de Brandon:
45.120 bytes exactos frente a referencia independiente en binary64. La
[salida FP32 integrada](FFN_FP32.md) pasa sus propios bancos de escritura
y cadena; falta ampliar dimensiones numéricas de la FFN. Mantener los destinos
privados hasta completar todas las fases y descartar los de jobs fallidos.
Continúan pendientes runtime de buffers, DMA y DDR con sus dependencias
de placa, además de la conversión de pesos Meta autorizados.

El [bloque local SwiGLU/Q8](SWIGLU_Q8_BLOCK.md) ya pasa 27/27
transacciones C/RTL: convierte las salidas FP64 y produce registros ABI1.
Su adaptador AXI ya pasa 105 casos y un vector de 8192 elementos;
la secuenciación raíz pasa además 498 casos de protocolo, incluidas formas
Llama, sin atribuirles validación numérica de la FFN completa.
