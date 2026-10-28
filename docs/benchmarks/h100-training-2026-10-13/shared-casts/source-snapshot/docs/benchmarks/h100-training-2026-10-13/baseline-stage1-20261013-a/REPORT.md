# Referencia de entrenamiento: Brandon-Mini stage1-v1

Captura parcial al **13 de octubre de 2026, 20:54:40 UTC (17:54:40 en Santiago)**.
El preentrenamiento estaba activo en el paso 25.400/32.768 (77,51%). Esta captura
es inmutable; el resultado final del entrenamiento se debe guardar por separado.

## Arquitectura y ejecución

| Ajuste | Valor |
|---|---|
| Parámetros | 107.604.288 |
| Arquitectura | 30 capas; dimensión 576; FFN 1472; SwiGLU; RMSNorm; RoPE |
| Atención | 9 heads Q, 3 KV; head dimension 64; PyTorch SDPA causal |
| Vocabulario / contexto | SentencePiece 8192 / 1024 tokens |
| Embeddings | Compartidos con salida; dropout 0 |
| Precisión | Autocast BF16; parámetros FP32; matmul precision `high` |
| Paralelismo | DDP/NCCL, 2 H100; `no_sync` en el primer microbatch |
| Batch | 16 secuencias/GPU × 2 GPU × 2 acumulaciones = 64 secuencias/paso |
| Tokens/paso | 65.536 posiciones supervisadas globales |
| Optimizador | AdamW fused; betas (0,9; 0,95); weight decay 0,1 |
| LR | Pico 3e-4; warmup 400 pasos; decay coseno en el 20% final; mínimo 3e-5 |
| Clipping / semilla | Norma global 1,0 / 20261013 |
| Inicialización | Mejor checkpoint del piloto de 400 pasos; optimizador nuevo |
| Software | Python 3.12.15; PyTorch 2.14.1; CUDA build 13.0; NCCL 2.30.7; driver 580.178.04 |
| Host | Xeon Platinum 8480+, 52 CPU lógicas; GPUs conectadas por NVLink (`NV18`) |

La ejecución observada no usa `torch.compile` ni activation checkpointing.
SDPA selecciona su backend; no se perfiló qué kernel concreto ejecutó.

## Rendimiento medido

| Métrica | Resultado |
|---|---:|
| Tokens procesados en esta etapa | 1.664.614.400 |
| Presupuesto de la etapa | 2.147.483.648 |
| Throughput, últimos 1.000 pasos | 307.985 tokens/s globales |
| Throughput, todas las ventanas | 306.374 tokens/s globales |
| Throughput efectivo desde lanzamiento | 298.102 tokens/s globales |
| Tiempo registrado en ventanas de training | 5.433,28 s (90,55 min) |
| Tiempo desde lanzamiento hasta última métrica | 5.584,04 s (93,07 min) |
| Diferencia no desglosada | 150,75 s; 2,70% del tiempo total |
| Mediana del tiempo medio/paso por ventana | 212,83 ms |
| P95 del tiempo medio/paso por ventana | 214,67 ms |
| Mediana / P95 recientes (últimos 1.000 pasos) | 212,78 / 212,90 ms |
| Variación relativa del throughput reciente (CV) | 0,0317% |
| Ventana más lenta | Paso 8.700: 219.348 tokens/s |

Cada registro cubre 100 pasos. Los percentiles anteriores son de **medias de
ventanas**, no percentiles de latencia de pasos individuales. Throughput agregado
= suma de tokens / suma de tiempos, no media aritmética de tasas. Los tiempos de
ventana se reconstruyeron de los tokens y tokens/s registrados.

El contador del trainer incluye preparación del batch, transferencias, forward,
backward, comunicación DDP, clipping, optimizer y sincronización. Se reinicia tras
validar y guardar checkpoints. La diferencia con el reloj total agrupa arranque,
validación inicial/periódica, guardados y logging; **no permite separar sus costes**.
El reloj total excluye preparación del corpus y el piloto anterior. Se registraron
ventanas más lentas alrededor de los pasos 8.500–8.800; su causa no está medida.

## Pérdidas y estabilidad

| Paso | FineWeb-Edu | Cosmopedia | Media de validación |
|---:|---:|---:|---:|
| 0 | 4,3079 | 3,5625 | 3,9352 |
| 5.000 | 3,1021 | 2,2009 | 2,6515 |
| 10.000 | 2,9442 | 2,0145 | 2,4794 |
| 15.000 | 2,8678 | 1,9226 | 2,3952 |
| 20.000 | 2,8183 | 1,8714 | 2,3448 |
| 25.000 | 2,7889 | 1,8306 | 2,3097 |

Última pérdida de entrenamiento: **2,4284**, media de los pasos 25.301–25.400.
La media de validación pondera ambas fuentes por igual; el entrenamiento usa
aproximadamente 60/40 según ventanas disponibles, por lo que ambas pérdidas no
son directamente equivalentes. `exp(media de validación)` = 10,0718 es la media
geométrica de las perplejidades por fuente, no una perplejidad global ponderada.

Validación cada 1.000 pasos, con 8 batches por fuente y GPU: 262.144 posiciones
por fuente y evaluación. Son muestras fijas con reemplazo. Los archivos de
validación contienen 1M tokens por fuente; no se evalúan íntegramente cada vez.
El corpus de entrenamiento contiene ~1.200M tokens FineWeb-Edu y ~800M Cosmopedia;
los 1.665M tokens procesados no equivalen a cobertura única del corpus.

Norma de gradiente registrada: mínimo 0,3059; máximo 1,9868 (paso 7.500,
antes del clipping). Última: 0,3460. Sólo se conserva el valor del último paso de
cada ventana: no se puede inferir la frecuencia real de clipping. No hay valores
no finitos en las métricas capturadas. Aún no hay resultados de SFT en esta captura.

## GPU: diez observaciones recientes

| Métrica | GPU 0 | GPU 1 |
|---|---:|---:|
| Utilización media | 97,9% | 97,8% |
| Memoria utilizada | 21.321 MiB | 21.321 MiB |
| Potencia media | 547,6 W | 566,6 W |
| Límite de potencia | 700 W | 700 W |
| Clock SM / memoria | 1980 / 2619 MHz | 1980 / 2619 MHz |
| Temperatura media | 51,0 °C | 64,9 °C |

Muestreo de `nvidia-smi` a intervalos de aproximadamente un segundo. Memoria y
potencia son observaciones del dispositivo, no picos históricos del allocator
PyTorch ni energía total del entrenamiento. Utilización GPU no equivale a MFU.

## Comparación con la aceleración experimental

1. Para aislar cambios de implementación, conservar arquitectura, pesos iniciales,
   tokenizer, hashes del corpus, muestreo/semilla, máscara/shift de targets, batch
   efectivo, contexto, optimizador, LR, precisión y topología. La identidad del
   checkpoint inicial y las fuentes exactas están en `identity.json` y `baseline.json`.
2. Comparar throughput de training con throughput de training, y tiempo total con
   tiempo total. Mantener igual frecuencia de validación/guardado. Si se cambia
   precisión, batch o algoritmo, etiquetar esa corrida como una variante distinta.
3. Comparar pérdida por fuente frente a tokens procesados, pérdida al mismo
   presupuesto de tiempo y tiempo hasta alcanzar la misma calidad. Conservar el
   diagnóstico independiente y sus reglas de generación/scoring para evaluar SFT.
4. Instrumentar desde el inicio de la próxima corrida: timestamps, tiempo acumulado
   de training/validación/checkpoints, picos de memoria, y ventanas de medición de
   latencia. Hacer una prueba de equivalencia de pérdidas/gradientes antes de medir
   velocidad. Repetir ventanas comparables para no usar únicamente el mejor resultado.
5. Si ntransformer acelera sólo generación del teacher, medir prefill, decode,
   latencia, respuestas válidas/s y coste por ejemplo aceptado en un experimento
   separado: estos tokens/s de entrenamiento no son tokens/s autoregresivos.

`quality-milestones.json` conserva el primer punto evaluado que cruza cada umbral,
con tokens y tiempo acumulado de ventanas de training. No contiene tiempos
históricos totales hasta esos umbrales: el trainer no los registró.

## Archivos reproducibles

- `baseline.json`: métricas, hardware/software, configuración, identidades y hashes.
- `training-windows.csv` y `validation.csv`: series completas para comparación.
- `pretrain-metrics.jsonl`: registros originales capturados sin modificación.
- `gpu-samples.csv`: diez observaciones por GPU con timestamp.
- `source/`: código y configuraciones congeladas de la corrida.
- `training-curves.png` / `.svg`: curvas exportables.

Copia persistente: `/lambda/nfs/btiny/brandon-mini/baselines/stage1-20261013-a`.
Recrear otra captura con `scripts/snapshot_mini_training.py`, usando un directorio
nuevo. No mezclar esta captura parcial con un resultado final.
