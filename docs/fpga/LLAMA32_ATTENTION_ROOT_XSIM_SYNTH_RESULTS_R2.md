# Llama 3.2 1B: wrapper completo de atención, XSim y síntesis

Estado 20261013T210023Z: **simulación integrada y síntesis revisadas y respaldadas**.
Diseño `ntrpu_attention_root_r1`, secuenciador de 32 cabezas, protección de
metadatos, caché KV, interfaz AXI con BFM y núcleo HLS aritmético generado.
No equivale a inferencia completa ni a DDR física.

## Evidencia funcional

Tanda real `20261013T200719Z-V7-llama32-attention-root-vivado-r1`:

- 23 trabajos raíz, 19 completions, 426 consultas hijas y 424 liberaciones.
- 24.576 resultados FP32 públicos; 423 tiles aritméticos revisados frente a
  la referencia independiente. Diferencia máxima de salida **0 ULP**;
  error escalado máximo del estado **3,456 × 10⁻¹⁶**.
- 5 resets, 7 asociaciones de caché, 28 append; 476 AW/B y 1.792 beats W
  (114.688 bytes), 1.075 AR y 8.853 beats R (566.592 bytes).
- Se verifican copia/concatenación de las 32 cabezas, orden y contenido KV,
  errores de metadatos y AXI, cancelación, revocación y recuperación tras reset.
- 26 corrupciones de traza y siete alteraciones de identidad de diseño rechazadas.

## Ciclos observados

El banco completo consumió 2.757.841 ciclos simulados. Los cuatro primeros
trabajos, sin fallos, midieron lo siguiente entre eventos BEGIN y COMPLETE:

| Tokens del prefijo | Ciclos | Tiempo simulado a 5 ns (µs) |
|---|---:|---:|
| 1 | 112,455 | 562.275 |
| 2 | 171,700 | 858.500 |
| 3 | 230,971 | 1154.855 |
| 4 | 290,234 | 1451.170 |

Estos intervalos incluyen carga/lectura, cómputo y extracción del resultado
por el banco. No son ciclos por token de una inferencia completa ni throughput
medido en una placa. El reloj de 5 ns todavía no cumple timing.

## Síntesis OOC

VU5P `xcvu5p-flva2104-2-i`, periodo de 5 ns:
20.709 LUT, 19.414 FF, 43 tiles BRAM y 77 DSP.
Setup WNS **−0,443 ns**, hold WHS **+0,030 ns**. Sin cajas negras, latches,
bucles ni endpoints internos sin restricciones. Permanecen 2.845 entradas
y 1.373 salidas externas sin presupuesto temporal. No hay cierre de timing
ni calificación de ruteo del wrapper. Los recursos incluyen submódulos;
no deben sumarse a los de esos mismos módulos aislados.

## Corrección de la revisión y respaldo

El verificador original confundía el inicio del contador `root_start` con
el evento BEGIN. En el banco R2 congelado, la tarea guarda el contador en
el flanco descendente previo y luego espera el siguiente para emitir BEGIN.
Los 19 trabajos terminados tienen exactamente un ciclo de diferencia.

`tools/check_ntrpu_attention_root_r2.py` exige estrictamente ese desfase
de +1 y obtiene las métricas del intervalo entre eventos; conserva también
el contador original como `bench_elapsed`. Se añadieron controles negativos
que alteran el ciclo de BEGIN y el elapsed de COMPLETE. No se cambiaron RTL,
estímulos, logs, criterios numéricos ni constraints; **no se repitió Vivado**.

El diagnóstico original permanece archivado con SHA-256
`7486d15c8d6bbeaac4d460232b993e523434ed1d5b2eb04184dfe7b3969bfd01`.
La revisión nueva copia y verifica cada miembro de esa tanda y añade
verificador, procedencia y resultado:

- Archivo revisado `20261013T205248Z-V7-attention-root-evidence-review-r2` SHA-256 `a631ae22ddc88f29c98bbf62f0c36021704412b51c7fc50883aa0ccb6f7ff01e`.
- Revisión SHA-256 `480beb2c71e89466da7526feaa48f9e002bad44abeb4d9d756a1f266e97b181d`.
- Fuente original SHA-256 `10abd430b1adbb9c001eae84ac50290c5b447087149b6051bd7d0647f1ad54cf`.
- Coordinador vigente `llama32-attention-root-vivado-r3-coordinator.json`.

Ambos archivos se conservan en D:/NFS. La cola continúa con cuantización
HLS R2, propietario Vivado R3 (RTL R2), ruteo R3 y proyección R3.
Auditoría vigente: `audit_attention_composition_active_r9.py`.
