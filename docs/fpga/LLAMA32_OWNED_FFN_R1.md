# FFN real con identidad pública y residual retenido

Estado 2026-10-15 16:50 UTC: **Vivado R4 en ejecución**, todavía
sin resultado aritmético de esta composición en XSim.
Tanda `20261015T164923Z-V7-llama32-owned-ffn-vivado-r4`.

## Diseño y contrato

`fpga/rtl/ntrpu_llama32_owned_ffn_r1.sv` conecta el
[propietario FFN R3](LLAMA32_FFN_OWNER_R1.md) con `ntrpu_ffn_memory` y sus
16 módulos originales, sin modificar el núcleo calificado. Conserva
secuencia/época públicas de 64 bits; reserva cuatro etiquetas privadas
para gate, up, SwiGLU y down. `OUTPUT_FP32=1`, D=2048, H=8192.
El residual retenido queda protegido frente a los destinos FFN.

El banco `fpga/tb/tb_ntrpu_llama32_owned_ffn_r1.sv` ejecutará un caso
completo con 53.477.376 bytes de matrices Q8_0 del GGUF verificado.
La entrada es el ABI1 exacto de la referencia post-O, muestra 1/token 0;
no es todavía una conexión al productor post-O vivo. Mantiene pausas
aleatorias AXI y destruye los pines de comando tras capturarlos.

La memoria del banco confirma las escrituras reales W sólo al aceptar B
exitoso. Los intermedios deben estar confirmados antes de leerlos; ninguna
referencia numérica alimenta la memoria de salida. Se esperan 155.648 bytes
escritos entre etapas y 2.048 resultados FP32 finales, 8.192 bytes.

## Preflight y límites de la evidencia

- Icarus no aceptó puertos ANSI de arrays del núcleo original. Se conservó
  el diagnóstico; no se cambió el núcleo para acomodar ese compilador.
- `xvlog` analizó 18 RTL, el banco y dos firmas HLS; `xelab` elaboró sólo
  el top. Las firmas estaban vacías: demuestra compatibilidad de puertos,
  sin ejecución aritmética ni elaboración completa del banco.
- El autoensayo del revisor usa una traza artificial y rechaza 34 controles
  de identidad/causalidad y 14 de métricas. Sus ciclos no son medidas RTL.
- El paquete completo excluye esas firmas y compila 51 fuentes propias/HLS
  y once archivos IP de simulación, heredados del banco FFN calificado.

## Cola, evidencia y recuperación

`queue_llama32_owned_ffn_vivado_r4.py` verificó la revisión y respaldo del
propietario Vivado R4, posterior a la [optimización física R5](LLAMA32_PROJECTED_ROOT_PHYSOPT_R1.md). Usa CPU0/1,
dos hilos, síntesis OOC a 5 ns y hasta ocho horas para un caso XSim completo.
El límite es de seguridad; no es una estimación de duración ni latencia.

| Evidencia D:/NFS | SHA-256 |
| --- | --- |
| Manifiesto del fixture | `bf36c6afbe46df663aa9823769f6c6c74e1608826fce7182e21ffe42646e0194` |
| Compilación vendor R2 | `85ac5cab519c736fc12637fda5b4dc8f6c42859d9b470e933f5fe69ee254c52e` |
| Revisión de compilación R3 | `c26a2cc21ab4f84feb9053554a1982e0a342f6e9e14cb03aff7c0f977faf191b` |
| Fuente completa Vivado R4 | `cb1a49ed317ba1871b59d645f91c8fb201023172693958d2a16ec7109bdc7fbd` |
| Configuración Vivado R4 | `4dec12254966409a29a4055066d644b4f5b73a83fbe8201b84aab7b3398116d6` |

R1 quedó detenido antes de lanzar herramientas por un fallo del observador
de respaldo precedente. R2 conserva todos sus miembros y añade el sucesor
del coordinador y las dependencias reparadas, con verificación por hash.
R2 también se detuvo antes de ejecutar herramientas cuando el preflight
físico rechazó una propiedad ausente del checkpoint. R3 actualiza esa
dependencia; los 51 archivos compilados y el fixture permanecen idénticos.
R3 también se detuvo antes de ejecutar por la revisión del encabezado
físico. R4 conserva de nuevo los mismos archivos compilados y depende
de la revisión corregida y respaldada; no repite una FFN ya ejecutada.

Después de aprobar: unir al productor post-O, transportar la segunda suma
residual y calificar errores/cancelaciones de la cadena completa. Siguen
pendientes inferencia completa, timing físico e integración DDR4/PCIe.
