# Proyección O real: composición de memoria y tanda Vivado (R1)

Estado 20261013T220812Z: **preflight de compilación aprobado; aritmética integrada
pendiente de XSim**. Coordinador: `waiting_for_attention_quantize_store_review_and_backup`.

## Fuentes y contrato

`ntrpu_attention_projection_memory_r1.sv` conecta el controlador O con el
GEMV matricial existente y su escritor FP32. Conserva identidad pública de
64 bits, reserva 64 identidades internas por trabajo y espera el drenaje AXI
antes de publicar. Las regiones de pesos y activaciones deben permanecer
inmutables hasta consumir la completion; el reset debe coordinarse con el
receptor AXI externo.

La geometría es **2.048 × 2.048**: 64 grupos de 32 filas, dos tiles de 32
bloques Q8 por grupo. Se usa el MAC dual HLS generado que participó en la
FFN aprobada; no se cambia el RTL matricial ni se activa la variante cuádruple.
La actividad del watchdog procede de handshakes AXI reales; se medirá el
mayor intervalo inactivo con su límite de producción de 65.536 ciclos.

El [Q8_0 de Unsloth](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/blob/main/Llama-3.2-1B-Instruct-Q8_0.gguf)
ya está en D:, fuera de Git: **1.321.082.528 bytes**, SHA-256
`3f87a880027e7b9ea8e0da9e4009584336f352af444a0e6e5c20721ac4c7ffd1`. El hash local coincide con el publicado; no se repite la descarga.
Se reutilizan cuatro activaciones de atención y referencias del
[oráculo CPU revisado](LLAMA32_ATTENTION_OUTPUT_NATIVE_RESULTS_R1.md).

## Prueba programada y criterios

`tb_ntrpu_attention_projection_memory_r2.sv` ejecuta siete trabajos:
cuatro vectores completos; región de salida un byte insuficiente; cancelación
durante lectura de pesos; recuperación con un vector completo.

Por proyección válida se exigen 2.048 resultados FP64 exactos y 8.192 bytes
FP32 exactos; 1.344 AR, 73.728 R, 66 AW, 128 W y 66 B. Direcciones iniciales
a 4.032 bytes de página fuerzan particiones de 4 KiB. El banco introduce
backpressure y admite W antes de AW. La completion queda retenida siete ciclos.
La región insuficiente debe fallar antes de AXI; la cancelación debe drenar
las lecturas aceptadas y dejar cero resultados publicados.

El verificador independiente compara cada lectura con los bytes congelados,
reconstruye bursts/grupos, comprueba FP64/FP32, identidades y publicación, y
recalcula los intervalos sin actividad. Tiene 31 controles negativos preparados;
su ejecución contra la traza real aún está pendiente.

## Compilación, cola y respaldo

El preflight R1 falló porque una función del testbench llamaba a una tarea.
Se preservó su archivo de diagnóstico: SHA-256
`93f6f6df9a042cbb4d2bccd51c84ea3aee0f9cfa21c62d3af1746997d947df23`.
R2 reemplaza esa llamada por `$fatal` dentro de la función, conserva el RTL
y **compila con salida 0**. Archivo R2:
`b6c21713a5c2e446983935f029fb1894b2fa0822cbc7adcfcf2de06cf0361a48`.
Ambos están verificados en D: y NFS. Icarus emite advertencias sobre enums
y `always_*` heredados; este preflight usa solo la interfaz del MAC y **no
califica simulación ni aritmética**. El paquete Vivado excluye ese stub.

Paquete `20261013T220604Z-llama32-attention-projection-memory-vivado-r1`, 67 miembros, fuente ZIP
`4e5c525c1c118edde8f012e20619d60f5d43f805a0d687b156f8fd389476606b`; configuración
`cc4371698e8c6c5fcac30dc518a9443fec79b4cf0e820b4ea662ed3a4dc5eb40`. Respaldo local:
`D:\AI\ntransformer\artifacts\ntrpu-model-validation\20261013T220604Z-llama32-attention-projection-memory-vivado-r1`; réplica NFS verificada por miembro.

La etapa 23 espera a cuantizador → propietario → AXI R2, su revisión y su
respaldo. Verifica además las evidencias terminales del controlador O y FFN.
Usa VU5P `xcvu5p-flva2104-2-i`, reloj OOC de 5 ns, dos CPU y límite de 7.200 s.
El observador guarda estados atómicamente, revisa las fuentes realmente
compiladas y respalda incluso diagnósticos antes de liberar el turno.
Auditoría vigente: `audit_attention_composition_active_r13.py`.

Quedan pendientes enlazar raíz → cuantizador → O, residual/norma/FFN,
inferencia completa, integración DDR y timing de tarjeta. Los ciclos que
produzca este banco corresponden a una proyección contra memoria simulada.
