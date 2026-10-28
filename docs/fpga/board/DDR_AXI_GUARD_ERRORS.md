# V7: rechazos de capacidad desde el host sobre MIG

## Resultado y alcance

El 2026-10-04, ambos perfiles host pasaron la revisión independiente del
camino `host AXI → CDC AMD → árbitro → filtro de capacidad → MIG`. Cada
perfil completa 36 transacciones de protección y después tres FFN D32/H32.
Los accesos fuera de capacidad reciben DECERR sin emitir VALID hacia MIG;
los datos de control se conservan y las FFN devuelven 96 FP32 exactos.

Se usa el RTL del [filtro ya calificado](DDR_AXI_GUARD.md), sin modificarlo:
SHA-256 `a2a7e2452e1d41a710efa9cf09126eec8da38182c118bf7adfef6ded923f7af5`.
MIG utiliza geometría genérica de 128 GiB, ECC, dos slots y cuatro ranks,
con `Simulation_Mode=BFM`. Son resultados funcionales de simulación.

## Secuencia y comprobaciones

El host escribe y lee tres regiones de 256 bytes: `0`, `0x10000` y
`0x1fffffff00`. Después ejecuta una escritura y una lectura por dirección:

| Dirección inválida | Beats de 64 B | Dirección que resultaría de truncar |
| --- | --- | --- |
| `0x0000002000000000` | 1 | `0` |
| `0x0000002000010000` | 4 | `0x10000` |
| `0x0001000000010000` | 4 | `0x10000` |
| `0x8000001fffffff00` | 4 | `0x1fffffff00` |
| `0xffffffffffffff00` | 4 | `0x1fffffff00` |
| `0x0000010000010000` | 3 | `0x10000` |

Las ráfagas mantienen alineación, INCR, tamaño, longitud y límite de 4 KiB
legales para atravesar CDC/árbitro. Los formatos no admitidos y cruces de
página mantienen la cobertura aislada anterior.

Después de cada par inválido, el host relee **las tres regiones**. En cada
perfil hay 12 descriptores rechazados, 768 bytes de control y 21 lecturas
completas de esas regiones. Se observan los cinco canales en tres fronteras:
host, entrada del filtro y salida hacia MIG. La referencia reconstruye
direcciones de 64 bits, atributos, datos, strobes, IDs, RESP, LAST, orden,
retenciones y causalidad entre relojes. Las lecturas MIG se comparan con
escrituras aceptadas previamente; no se inicializa memoria por jerarquía.

La fase FFN original conserva sus 54 transacciones, 9.984 B de entrada,
2.112 B calculados y 384 B devueltos. El revisor original sigue comprobando
bytes, clocks, ranks y ABI2. Los registros `NTRPU_GP_*` se revisan con su
propio esquema; todos los diagnósticos del simulador/modelo se conservan.

## Evidencia por tanda

IDs bajo `artifacts/ntrpu-board-vivado/`, con prefijo `20261004T`:

| Sufijo del ID | Resultado | Tiempo de herramienta |
| --- | --- | --- |
| `093814Z-V7-ffn-supervised-ddr-guard-errors-fast` | Host 4 ns: 499 ciclos retenidos; fase de protección 11,820 µs simulados; FFN en 2.122 / 2.121 / 2.132 ciclos UI. | 395,449 s |
| `093824Z-V7-ffn-supervised-ddr-guard-errors-slow` | Host 6,2 ns: 497 ciclos retenidos; protección 13,265 µs simulados; FFN en 2.102 / 2.149 / 2.292 ciclos UI. | 412,733 s |
| `094552Z-V7-ffn-supervised-ddr-guard-errors-negative` | Control negativo: al quitar únicamente los dos comparadores de capacidad, falla en la primera escritura inválida, a 29,775998 µs. | 223,318 s |

Las tres tandas tienen respaldo local/NFS verificado, con fuentes, revisión,
trazas, manifiesto y SHA-256. Los perfiles normales suman **180 transacciones host, 24 rechazos, seis FFN
y 192 FP32 exactos**. Cada revisión rechaza 62 alteraciones de evidencia y
tres de conexiones. Los periodos y las pausas son deterministas: estos
tiempos no son mediciones de ancho de banda ni frecuencia física.

El control negativo compila una copia aislada del mutante, verificada contra
el original. Las primeras seis operaciones válidas completan; la séptima
conserva `0x2000000000` hasta el filtro y provoca `invalid probe exposed to
MIG`. Su salida 1 es el fallo esperado, no una FFN aprobada.

Se revisan todas las advertencias. La adicional `VRFC 10-3380` corresponde
a `probing`, un `bit` de ámbito de módulo explícitamente inicializado y
declarado después de su primer uso; no es una red implícita. Permanecen las
limitaciones conocidas del modelo AMD y de los FIFO de simulación.

## Reproducción y siguiente paso

Usar `tools/ntrpu_board_ffn_ddr_guard_errors.tcl` con los cinco argumentos,
rutas HLS/modelo y modos del [banco frío](DDR_FFN_COLD.md). Cada lanzamiento
requiere una carpeta nueva, cuatro hilos y límite de 1.800 s.

```bash
python tools/check_ntrpu_ffn_ddr_guard_errors.py /ruta/seed-1.log \
  --host-period-ps 4000
```

Para el otro perfil, usar `6200`. El lanzador terminado en `_negative.tcl`
debe fallar exactamente como se describe; no sirve para calificar el diseño
normal. Los revisores de procedencia y respaldo viajan con el checkpoint.

El [top sintetizable FFN y filtro](FFN_GUARDED_TOP.md) ya pasa este banco
en ambos perfiles con trazas idénticas. Sigue la revisión de implementación
con sus relojes e interfaces. También faltan propiedad de
buffers, adaptación de calibración/reset, DIMM exacto, integración PCIe/DMA
completa, cuatro canales, constraints de placa y ensayos físicos. Los accesos
inválidos ocurren antes del cómputo; esta tanda no prueba errores concurrentes
con FFN, reset DDR con tráfico ni inferencia completa.
