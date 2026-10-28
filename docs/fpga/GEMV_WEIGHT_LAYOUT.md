# Empaquetado de pesos para GEMV

Para cargar también las activaciones por AXI, consultar el
[ABI y controlador de memoria](ACTIVATION_MEMORY.md). Los descriptores
de pesos de este documento se mantienen sin cambios.

`tools/ntrpu_gemv_layout.py` convierte bloques Q8 densos en una imagen de
memoria y descriptores para el [GEMV escalar/dual](Q8_GEMV_PARALLEL.md).
El archivo de entrada recorre filas y luego bloques K: cada bloque contiene
una escala FP16 little-endian y 32 bytes INT8. No cuantiza ni rellena filas
o columnas silenciosamente; el llamador debe preparar ese padding.

## Contrato de bytes y direcciones

| Layout declarado | Orden dentro del tile | Consumidor |
| --- | --- | --- |
| 1 | `[fila][bloque K][34 bytes]` | GEMV escalar |
| 2 | `[bloque K][fila][34 bytes]` | GEMV dual, `job_layout=2` |

Se admiten 1–32 filas, 1–256 bloques K y 1–32 bloques K por tile. El
producto filas × bloques del tile, incluido el último, debe ser múltiplo
de 32. Cada payload queda así alineado a beats de 64 bytes. Se rechazan
escalas FP16 no finitas, tamaño de entrada incorrecto, direcciones sin
alinear, solapamientos, exceso del rango registrado y overflow de 64 bits.
La API exige `bytes` inmutables y devuelve otra imagen independiente.

La imagen comienza en `region_base`. Por defecto, el primer payload está
en offset `0xfc0` y los siguientes a intervalos de `0x10000`; esos valores
ejercitan cruces de 4 KiB y conservan direcciones distintas entre tiles.
Los huecos contienen `0xa5`. Son decisiones del fixture inicial, no una
asignación física de memoria. El runtime todavía debe reservar/registrar
el buffer, transferirlo y mantenerlo inmutable hasta completar el job.

Cada descriptor registra `k_base`, `k_count`, dirección, offset, cantidad
de bytes y SHA-256 del payload. El manifiesto `NTRTILE02` añade layout,
dimensiones, rango, tamaño/hash de imagen y hash de entrada. Los hashes
identifican los artefactos; el controlador RTL no verifica SHA-256.

```bash
python tools/ntrpu_gemv_layout.py matriz.q8 --output NUEVA_CARPETA \
  --rows 32 --k-blocks 256 --tile-k 32 --layout 2 \
  --region-base 0x100000000 --region-bytes 0x200000
```

Produce `weights-image.bin`, su representación byte a byte
`weights-image.hex` y `tiles.json`. Rechaza una carpeta de salida existente.
Las direcciones son del espacio AXI del dispositivo propuesto; no son
punteros de CPU ni permiten acceder automáticamente a un BAR.

## Banco que consume el archivo del host

`prepare_ntrpu_gemv_packed_fixture.py` fija el manifiesto Brandon anterior
por hash y crea `NTRGEMV02`: 54 proyecciones reales y cuatro formas sintéticas
32×2048, 32×8192, 1×8192 y 31×2048. Conserva las referencias Q8 de Brandon
y calcula las sintéticas independientemente en Python FP64.

El banco `NTRPU_PACKED` carga la imagen en un array y contesta AXI leyendo
esos bytes, sin reconstruir pesos según fila/K durante la transacción.
Carga además los descriptores del host: cada línea HEX tiene 128 bits
`{dirección[63:0], K_base[31:0], K_count[31:0]}`. Las activaciones siguen
entrando por el puerto de carga directo, no por DMA.

```bash
python tools/prepare_ntrpu_gemv_packed_fixture.py \
  --brandon-fixture FIXTURE_BRANDON_VALIDADO --output NUEVO_FIXTURE
vivado -mode batch -source tools/ntrpu_board_gemv_stream_sim.tcl \
  -tclargs RTL_HLS_MAC NUEVA_SALIDA NUEVO_FIXTURE dual packed
python tools/check_ntrpu_gemv_fixture.py LOG --fixture NUEVO_FIXTURE --seed 1
python -m unittest discover -s tests -p test_ntrpu_gemv_layout.py -v
```

El revisor vuelve a decodificar cada byte de las imágenes, comprueba su
posición contra la matriz densa, descriptores, huecos, hashes y orden K;
después compara cada salida RTL con su referencia. Las pruebas CPU incluyen
formatos conocidos, tail, filas impares, tamaño máximo y rechazos de rangos.
La CLI produjo la misma imagen máxima de 497.600 bytes y rechazó sobrescribirla.
Las tandas y su evidencia se registran en [VIVADO_STATUS](board/VIVADO_STATUS.md).

## Evidencia RTL

`20261003T034056Z-V7-gemv-host-packed` terminó en 544,588 s reales:
**174/174 jobs PASS**, tres semillas y 5376 salidas comparadas con diferencia
numérica cero. De ellas, 5088 corresponden a Brandon y 288 a formas sintéticas.
Cada semilla procesa 29.888 bloques, 496 ráfagas y 1.016.192 bytes útiles;
375877/375715/375749 ciclos de jobs, más 174 ciclos entre operaciones.
El caudal de este banco ronda 2,704 bytes/ciclo, unos 0,541 GB/s con reloj
simulado de 200 MHz. No es caudal medido en DDR.

Fixture SHA-256:
`e32298e2e12864a823310633b24747066744738f31758ccead4205ed677b72c8`.
Se verifican 407 archivos de entrada, la copia del MAC HLS y los contadores
por fase; ocho alteraciones de métricas y ocho de resultados son rechazadas.
Se conservan tres avisos XSIM 43-3431 y seis Wavedata 42-489: los arrays de
pesos/imagen superan el límite de visualización de ondas, no de simulación.

La prueba negativa `20261003T034748Z-V7-gemv-host-packed-negative` altera
solo el byte de imagen 4034, de 225 a 224, conservando matriz fuente,
activaciones y referencia. Actualiza los hashes para que sigan siendo
consistentes. El decodificador independiente rechaza el byte discordante
con la fuente y el banco RTL falla en la primera salida, ciclo 3171:
0,446020 frente a 0,445988, valores redondeados del log. Exit1 en 61,303 s
es el fallo esperado de esta prueba negativa, no un PASS numérico.
Demuestra que la imagen se consume realmente y que funciona el comparador;
el hardware no detecta por sí mismo la alteración de un INT8 válido.

`20261003T035216Z-V7-gemv-packed-driver-regression` repite el banco dual
normal después de ampliar el lanzador: **87/87 PASS**, 129,490 s,
sin cambios en las métricas de la regresión previa. Los RTL sintetizables
siguen coincidiendo por hash con la síntesis BRAM y el ruteo documentados;
esta iteración modifica empaquetado, banco y revisión, no ese circuito.

Esto valida el contrato host→imagen→AXI simulado→MAC. No sustituye registro
de buffers, carga DMA, DDR, cadena FFN ni inferencia con pesos de Llama.
