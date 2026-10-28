# Recuperación de portabilidad Vivado: KV y atención

Fecha: 2026-10-13. Se preservaron las fuentes, resultados y fallos originales.
Las revisiones nuevas mantienen dependencias y revisores; ninguna calificación
anterior se sustituye por una prueba con mocks.

## Diagnósticos y cambios

`20261013T173053Z-V7-llama32-kv-loader-vivado-r1` falló antes de simular:
Vivado interpretó `-testplusarg SEED=1` como opción de `set_property`.
Se cambió exclusivamente la asignación a
`set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SEED=$seed"] ...`.
La prueba instalada verificó lectura posterior para semillas 1, 7 y 42,
y para la combinación de semilla y ruta del fixture.

`20261013T173720Z-V7-llama32-kv-loader-vivado-r1` pasó las cuatro suites
XSim, 85 casos cada una, pero falló en síntesis con `Synth 8-6735`:
puertos `input logic` bajo `default_nettype none` sin tipo net explícito.

La segunda recuperación añade copias en `fpga/rtl/vivado_explicit_nets_r1/`.
Sólo convierte declaraciones de entrada en las cabeceras a `input wire logic`;
los cuerpos, nombres de módulos, bancos, fixtures y núcleo HLS generado
se conservan. Cada copia tiene hash original y nuevo, y número de declaraciones
modificadas en `VIVADO_EXPLICITNETS_RECOVERY_R1.json`. Los Tcl y ejecutores
revisados seleccionan esas copias; los revisores comprueban sus hashes reales
en los proyectos compilados. La elaboración de `ntrpu_attention_memory_r1`
con Vivado 2026.1 pasó antes de liberar la cola.

## Revisión de identidad del diseño

La ejecución `20261013T174606Z-V7-llama32-kv-loader-vivado-r1` terminó
simulación y síntesis con código 0. El revisor confundía `NAME=design_1`
con el módulo superior. Se revisó la evidencia ya respaldada mediante
`check_ntrpu_vivado_design_identity_r1.py`, que exige el nombre exacto en
tres informes y en el comando real de síntesis, y rechaza siete corrupciones.
La adopción `20261013T175407Z-V7-kv-loader-evidence-review-r2` conserva
el diagnóstico original y demuestra PASS sin repetir herramientas.
[Métricas del cargador](../LLAMA32_KV_LOADER_XSIM_SYNTH_RESULTS_R1.md).

## Orden reanudado

Cargador KV R8, adoptado → append R9 → caché R7 → RAM de atención R6 → camino KV R6
→ contexto R5 → memoria R4 → motor generado R4. Estos números identifican
coordinadores; el contrato RTL conserva sus nombres originales.
Atención HLS R5 ya terminó y permanece como padre calificado.

Cada etapa exige PASS revisado y archivo local/NFS verificado de su padre.
La integración final exige además que los 33 archivos HLS generados coincidan
con el archivo terminal de atención. Un diagnóstico detiene los descendientes.

## Evidencia

- Fallo inicial: `658df11694b81173b92783051d5ba5faa1a0a4cab0be093f4d4e751e36af1bfc`.
- XSim PASS / síntesis fallida: `5ea7b22001da82413fe2740d1fd3b7b8874682bac171b6f1bdfc2b9950b1768d`.
- Cápsula de sintaxis Tcl: `5215c87cdc239444111bb076ac5a360d0cc4be1aff9c40e8f57b096a7aeabafa`.
- Prueba de elaboración: `13bb24faacd5596659749c0bf40436a84255686b40bb2444cbe5dd93da18e6f4`.
- Cápsula de puertos explícitos: `7d8ddfb3826a50728426885edd5b4a06ebb29737ac07ed3b4303f5a109ebab0b`.
- Archivo de adopción: `bec4f0992c194fd25fbde7d9659dd94d066c3080b781e5f8fec1109934e531a9`.
- Cápsula de revisión de identidad: `fc64279a1e4d491a584436d768ed9f2cd1c434c8c4e6a5c855b38dfeacc29f91`.

Los respaldos están en `D:/AI/ntransformer/artifacts/ntrpu-queue-recovery/`
y el directorio NFS de la sesión. El estado vivo corresponde a
`vivado-design-identity-recovery-r1-launch.json` y sus siete coordinadores.
La elaboración previa no reemplaza las simulaciones ni la síntesis OOC completa.
