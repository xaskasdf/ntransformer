# Atención, cuantización y O: resultado integrado y revisión R4

Actualización 2026-10-15. **Regresión funcional aprobada y respaldada**:
la raíz de atención lee K/V confirmado, produce FP32 mediante HLS y lo
cuantiza. La proyección O consume exclusivamente ABI1 escrito en memoria
y confirma sus resultados por AXI. El banco incluye éxito con un token,
cancelación tras ocho beats Q y recuperación con cuatro tokens.
No constituye inferencia completa ni ensayo sobre DDR física.

## Métricas verificadas

Ejecución real: `20261015T064819Z-V7-llama32-attention-projected-root-real-vivado-r3`.
Terminó en **3.823.015 ciclos**, con dos proyecciones completas:
4.096 resultados FP64 y 16.384 bytes FP32 exactos. También verificó
64 cabezas, 4.096 valores de atención, 1.024 palabras ABI1, doce incorporaciones
K/V y lecturas posteriores a sus confirmaciones B. Rechazó 69 controles.

| Medida | Un token | Recuperación, cuatro tokens |
| --- | ---: | ---: |
| Raíz, comando a publicación | 109.902 ciclos | 286.935 ciclos |
| Matriz O | 1.608.074 ciclos | 1.608.056 ciclos |
| Trabajo con preparación y retención del banco | 1.822.384 ciclos | 1.999.385 ciclos |
| Mayor intervalo sin actividad AXI durante O | 11.820 ciclos | 11.820 ciclos |

Síntesis VU5P: 48.599 LUT, 45.117 registros, 84,5 BRAM y 134 DSP;
cero cajas negras, latches y lazos. A 5 ns, **WNS −0,203 ns** y WHS
+0,030 ns: falta cierre temporal. Son ciclos RTL con memoria simulada,
no tokens/s de una tarjeta; las E/S externas tampoco tienen presupuesto
temporal completo.

## Corrección del verificador sin repetir simulación

El diagnóstico original quedó conservado. Exigía que `APPEND_DONE` tuviera
un ciclo estrictamente mayor que el último B. El RTL registra `RESPONSE`
al aceptar ese B; el banco registra B en el flanco positivo y observa la
respuesta en el siguiente negativo, con el mismo contador. Esto sucede
en los doce cierres K/V del banco.

Los verificadores `check_ntrpu_attention_projected_root_real_r2.py` y
`check_ntrpu_attention_projected_root_materialization_r2.py` admiten esa
igualdad. Además exigen que **todos los eventos B precedan a APPEND_DONE**
en la traza. Dos controles nuevos rechazan publicación prematura por ciclo
y por orden dentro del mismo ciclo. Conservan los 67 controles anteriores
y las comparaciones numéricas. No cambió RTL, banco, pesos ni checkpoint.

R4 revisó fuentes realmente compiladas, resultados, recursos, restricciones,
checkpoint y respaldo original. La revisión nueva vive en un suplemento;
el archivo de la ejecución original permanece intacto.

| Evidencia | SHA-256 |
| --- | --- |
| Archivo original, D:/NFS | `9c9d1bbb8945781b04322a18f8b31034c2d906681aa66d8883e651c264268a57` |
| Revisión corregida | `45f3d08f324ee2e30bd34c88847efcf80e872317785aaea3f5c663cda4ec7e65` |
| Suplemento R4, D:/NFS | `6b8f09725216aa5a7c656230df952a6877491bcd725324d1b41d27c023ff2f59` |

## Continuación vigente

Revisión terminal: `llama32-attention-projected-root-real-review-r4-coordinator.json`.
Cola nueva, con revisión y respaldo entre etapas:

1. Post-O HLS R3, ejecución `20261015T080756Z`: C/RTL aprobado y respaldado;
   [26 transacciones exactas](LLAMA32_POST_ATTENTION_R1.md), máximo 173.353 ciclos.
2. Cargador AXI Vivado R3, ejecución `20261015T081949Z`: aprobado y respaldado.
3. Propietario de RAM Vivado R2, ejecución `20261015T082404Z`: aprobado y respaldado.
4. Escritor AXI del par Vivado R3, ejecución `20261015T082949Z`: aprobado y respaldado.
5. [Post-O con HLS y AXI reales](LLAMA32_POST_ATTENTION_REAL_R1.md),
   ejecución `20261015T084117Z`: ocho trabajos aprobados, 73.728 bytes
   y respaldo completo verificado en D:/NFS.
6. [Colocación y ruteo OOC de esta raíz](LLAMA32_PROJECTED_ROOT_ROUTE_R1.md):
   ejecución `20261015T090048Z` terminada y respaldada. Rutas completas;
   WNS −0,079 ns, WHS +0,010 ns a 5 ns. La revisión R2 conserva ese fallo.
7. [Unión real raíz → post-O](LLAMA32_ATTENTION_POST_REAL_R1.md):
   165 trabajos nativos y 16 mutantes detectados; composición aritmética
   en validación Vivado R2 tras corregir la lista de fuentes de R1.

Los sucesores antiguos detenidos se conservan. El escritor R2 no lanzó
Vivado: intentó leer el registro antecesor antes de su creación. R3 espera
ese archivo; las fuentes RTL son idénticas. Los
sucesores comparten una sola tanda principal. Se preservan trabajos externos.

La [composición de transporte](LLAMA32_POST_ATTENTION_PATH_R1.md) pasó su
banco nativo y la unión HLS real post-O pasó su propio banco. Falta calificar
la nueva unión que consume O desde B confirmado, integrar FFN y la segunda suma
residual, cerrar timing y ejecutar inferencia completa.
