# Raíz completa: recuperación de etiquetas y cola R3

**Resultado actualizado:** R3 terminó y pasó la
[revisión suplementaria R4](LLAMA32_PROJECTED_ROOT_REVIEW_R4.md): 4.096
resultados FP64, 16.384 bytes FP32 y 69 controles. La cola vigente está
en ese documento. El relato siguiente conserva el estado anterior.

Actualización 2026-10-15. La regresión corregida está ejecutándose como
`20261015T064819Z-V7-llama32-attention-projected-root-real-vivado-r3`.
Todavía no se ha aprobado la composición completa.

## Resultados que habilitaron esta tanda

Las dos etapas anteriores terminaron y tienen revisión independiente y
respaldo verificado en D: y NFS:

| Etapa | Resultado numérico XSim | Síntesis VU5P, 5 ns |
| --- | --- | --- |
| O R3, `20261014T041103Z` | 7 trabajos; 5 proyecciones, 10.240 FP64 y 40.960 bytes FP32 exactos; 32 controles | 19.981 LUT, 17.996 FF, 38 BRAM, 54 DSP; WNS +1,019 ns |
| Cuantización → memoria ABI1 → O R3, `20261014T051924Z` | 4 trabajos; 3 proyecciones, 6.144 FP64 y 24.576 bytes FP32 exactos; 38 controles | 27.480 LUT, 25.286 FF, 41,5 BRAM, 57 DSP; WNS −0,787 ns |

Ambas síntesis tienen cero cajas negras y cero lazos. La segunda **no cumple
el objetivo temporal OOC de 5 ns**. Sus pruebas funcionales aprobaron;
ninguna de estas dos tandas acredita timing ruteado ni DDR física.
O requiere entre 1.606.198 y 1.607.222 ciclos de matriz en los cinco casos
correctos. Son ciclos RTL con memoria simulada, no tokens/s de una placa.

Archivos terminales SHA-256, respectivamente:

- `021a1fb6e4ed1bae31e2cb45f36f7c01845a25df1a9511ddd02e39687e36f824`.
- `543396a3bdb07a73638d33f19184f7e4adc5c536040f02ee5b1bf628fca5bc8a`.

## Diagnóstico conservado

La raíz R2 (`20261014T061323Z`) sintetizó y llegó a la primera cuantización
en XSim. Se detuvo en el ciclo 212.732 con `Quant publication before commit`.
El banco inicializaba `INITIAL_QUANT_TAG` en `0xffffffff`; el almacén exige
`next_tag[5:0] == 0`, porque reserva 64 etiquetas por trabajo. Por inspección
del RTL, esa configuración provoca el rechazo de admisión antes de escribir
ABI1. La traza fallida no imprime el estado de esa respuesta y no se usa
para afirmar una proyección completa.

El diagnóstico, fuentes y checkpoint permanecen intactos. Archivo terminal:
`e1822e1bacd78445c4ea2164b0210290fd0aa1d8affe0591263e6e172b8eca52`.
Las dos etapas posteriores R1 se detuvieron sin lanzar herramientas.

## Cambio y aceptación

`fpga/tb/tb_ntrpu_attention_projected_root_real_r2.sv` inicia cuantización
en `0xffffffc0`, igual que la proyección. Añade una comprobación inicial
de alineación; conserva todas las aserciones anteriores y el cruce de
32 bits en el segundo trabajo exitoso. No cambia RTL, pesos ni aritmética.
La compilación de conectividad R3 pasó y está respaldada; usa interfaces
HLS vacías y no sustituye la regresión aritmética real.

La tanda nueva vuelve a sintetizar y simular las 88 fuentes reales:
34 RTL propios, banco y 53 módulos Verilog generados. Mantiene dos salidas
O completas, cancelación/recuperación, K/V y ABI1 materializados solo tras B,
67 controles de traza y comprobación de las fuentes realmente compiladas.
Límite: 7.200 segundos, CPU0/1, parte `xcvu5p-flva2104-2-i`.

- Paquete R3: `6df293d90f8bc840e2fabef44209f91d6f0d104be64ecb6a90b96bb55c541a28`.
- Fuentes R3: `0053a907e3c37e14e38daad89a01e54f0c6560d389c0fdd400c4ac4dabc029d9`.
- Banco R2: `550da07cd13f5dfaa9567066b875b71117f13f3c66b7b561149f53054b3d03cf`.
- Compilación previa: `a40d0d6a3c6ee11305335aaadc7dce2244792ff00f6140c8be288eed18937be0`.

## Continuación automática

Los coordinadores actuales son, en orden:

1. `llama32-attention-projected-root-real-vivado-r3-coordinator.json`.
2. `llama32-post-attention-hls-r2-coordinator.json`.
3. `llama32-post-attention-loader-vivado-r2-coordinator.json`.
4. `llama32-post-attention-owner-vivado-r1-coordinator.json`:
   [propietario de RAM aprobado en protocolo nativo](LLAMA32_POST_ATTENTION_OWNER_R1.md).
5. `llama32-post-attention-store-vivado-r1-coordinator.json`:
   [publicación AXI del par con bytes reales](LLAMA32_POST_ATTENTION_STORE_R1.md).

Cada sucesor exige revisión satisfactoria, hashes de su dependencia,
respaldo completo verificado y turno libre. Una observación SSH fallida
no autoriza reiniciar herramientas. Se preservan los trabajos externos
en `ntrpu-side`. La auditoría vigente es R24.

El GGUF proporcionado por el usuario ya existe en
`D:/AI/models/ntrpu/unsloth-llama32-1b-instruct-b69aef11/`:
1.321.082.528 bytes; SHA-256
`3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1`.
Se recalculó el hash local y coincide con el publicado en
[Hugging Face](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-Q8_0.gguf).

Véanse el [banco completo](LLAMA32_ATTENTION_PROJECTED_ROOT_REAL_QUEUE_R2.md),
el [bloque post-O](LLAMA32_POST_ATTENTION_R1.md) y el
[cargador AXI](LLAMA32_POST_ATTENTION_LOADER_R1.md).
