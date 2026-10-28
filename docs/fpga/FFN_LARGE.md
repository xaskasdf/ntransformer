# FFN completa a dimensiones Llama: banco sintético

Actualizado el 2026-10-03 UTC. La prueba encadena gate → up → SwiGLU/Q8 →
down mediante la raíz [FFN FP32](FFN_FP32.md), con `D=2048`, `H=8192` y
`tile_k=32`. **La tanda completa pasó la revisión numérica independiente.**
Los pesos son sintéticos y dispersos. No son pesos Meta ni inferencia Llama.

## Contrato del banco

El RTL del dispositivo es idéntico al validado y ruteado anteriormente:
18 archivos, watchdog 65.536. Solo se amplía el banco. Cada matriz contiene
17 MiB de Q8_0 y se calcula íntegra, incluidos los bloques de pesos cero;
el modelo de memoria genera sus bytes al leerlos para evitar grandes
arrays del simulador. Mantiene imágenes inmutables, direcciones sobre 4 GiB,
inicio a offset 4032, cruces de página y pausas AR/AW/W/completion.

Gate/up se almacenan y vuelven a leer como binary64; SwiGLU escribe registros
ABI1 y down almacena FP32. El banco exige lectura de intermediarios ya
escritos, comparación de cada byte de salida, guardas intactas, drenaje de B,
completion raíz correcta y conservación del descriptor tras sobrescribir
sus entradas. Los casos de errores/reset permanecen en su banco separado.

La referencia Python construye matrices densas Q8, codifica las escalas con
`struct.pack('<e')`, las empaqueta mediante la API del host y calcula la FFN
escalar con acumulación ordenada binary64 y conversiones FP32. Comprueba
además el resultado contra la fórmula cerrada del patrón sintético. Las
escalas gate 257…767 cruzan el cambio de espaciado FP16 en 512.

El verificador coteja todo el plan AR, sus fases/direcciones/longitudes y
un checksum por buffer de los datos R aceptados. Este checksum aplica
`hash=(hash XOR palabra_LE64)*0x100000001b3 mod 2^64`, empezando en
`0xcbf29ce484222325`; es un diagnóstico de transporte, no integridad
criptográfica. Los SHA-256 de imágenes CPU quedan registrados aparte.

## Resultados medidos

| Medida | Prueba corta, 2048/96 | Prueba completa, 2048/8192 |
| --- | ---: | ---: |
| Ciclos de la cadena | 243.755 | 20.332.664 |
| Bytes R | 665.088 | 56.754.176 |
| AR / R | 376 / 10.392 | 16.480 / 886.784 |
| Bytes W | 9920 | 155.648 |
| AW / W / B | 77 / 155 / 77 | 866 / 2432 / 866 |
| FP32 down | 2048 valores | 2048 valores |

La prueba corta pasó en 206,900 s: 75.693/75.651/11.512/80.899 ciclos por
fase, 133/29/51/6 pausas AR/AW/W/completion. Diez alteraciones de la evidencia
fueron rechazadas, incluidos datos W, máscara, checksum R y dirección AR.
Tanda `20261003T112205Z-V7-ffn-large-smoke`, respaldo de 78.246.234 bytes,
SHA-256 `9c4e26e76ec28674116b7b343315a88d081e4c5dfb605d4f5bb1220f0f4c155d`.

La regresión `20261003T113156Z-V7-ffn-large-bank-regression` pasa los 63
casos anteriores en 373,311 s. Sus ciclos, pausas y 71.040 bytes físicos
coinciden exactamente con la revisión FP32 previa; ampliar el banco no
cambió sus resultados.

La exploración `20261003T113913Z-V7-ffn-large-fast-smoke` repite el smoke
con depuración de xelab desactivada, sin desactivar assertions. La traza
completa `NTRPU_*` es idéntica. Tardó 206,987 s frente a 206,900 s:
**no se observó aceleración**. El pico de memoria del kernel XSIM pasó de
468.136 a 324.280 KB. El nombre `fast` expresa el objetivo del experimento;
la tanda completa conserva su compilación original.

La tanda completa `20261003T113023Z-V7-ffn-full-llama-shape` empezó a las
11:30:26 UTC y terminó a las **14:36:28 UTC**, salida 0, **11.160,111 s** de
pared, dentro del presupuesto de 12.000 s. Se mantuvo separada de SSH, sin
reinicios ni cambios de fuentes mientras corrían las demás tandas. La fuente es
`source-20261003T113015Z`, SHA-256
`ecdbbff034a667e1ff0f0979513f5407bd3b1e0ed2c32ea8639a64d11d1865b5`.

| Fase | Ciclos medidos |
| --- | ---: |
| Gate | 6.454.374 |
| Up | 6.455.250 |
| SwiGLU/Q8 | 980.311 |
| Down FP32 | 6.442.729 |

Los 155.648 bytes escritos, incluidos 2.048 valores down FP32, coinciden
exactamente con la referencia escalar. Se revisan el plan completo de 16.480
AR y los checksums por buffer de 56.754.176 bytes leídos. Las pausas medidas
son 5416/295/847/6 ciclos AR/AW/W/completion. El revisor rechaza diez
corrupciones de evidencia y confirma los 18 RTL originales sin cambios y
las copias HLS idénticas. Conserva XSIM 43-3431 y dos Wavedata 42-489.

Gate/up/down ocupan el **95,18%** de los ciclos; la siguiente investigación
de prestaciones debe medir el trabajo dentro de esas etapas y el coste por
tile antes de replicar motores. El cociente de bytes R aceptados sobre ciclos
totales es **2,791 bytes/ciclo** en este banco. Incluye relecturas y periodos de
cómputo; no es caudal continuo del bus ni una medida de DDR.

El respaldo local y NFS está verificado: **483.736.666 bytes**, 287 archivos,
SHA-256 `1d486d3a9ef2ce9216c326dfb4a409d380cf8ffe3a2bfdd335269ee9c0589216`.
Se conserva en `artifacts/ntrpu-board-vivado/20261003T113023Z-V7-ffn-full-llama-shape/`.
Esta prueba usa
la raíz FFN directamente; el arbitraje y la carga desde host tienen sus bancos
independientes. No demuestra esta dimensión a través de PCIe o DDR.

## Reproducción

```bash
vivado -mode batch -source tools/ntrpu_board_ffn_memory.tcl \
  -tclargs "$MAC_HLS_RUN" "$SWIGLU_HLS_RUN" sim_fp32_large_smoke NEW_SMOKE
python tools/check_ntrpu_ffn_large.py --hidden 96 --log NEW_SMOKE/seed-1.log
# Después del smoke revisado; usar otra carpeta y presupuesto explícito.
vivado -mode batch -source tools/ntrpu_board_ffn_memory.tcl \
  -tclargs "$MAC_HLS_RUN" "$SWIGLU_HLS_RUN" sim_fp32_large NEW_FULL
python tools/check_ntrpu_ffn_large.py --hidden 8192 --log NEW_FULL/seed-1.log
```

No extrapolar los ciclos a tokens/s ni DDR física. Faltan los pesos Meta,
atención, normas, RoPE/KV, runtime/DMA e integración con la memoria de placa.
