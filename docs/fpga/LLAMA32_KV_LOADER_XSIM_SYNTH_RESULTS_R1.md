# Cargador KV: XSim y síntesis OOC revisados

Estado al 2026-10-13: **PASS_KV_LOADER_R1_XSIM_SYNTH**.
Ejecución real `20261013T174606Z-V7-llama32-kv-loader-vivado-r1`;
revisión de evidencia `20261013T175407Z-V7-kv-loader-evidence-review-r2`.
La segunda identificación corresponde a una revisión y un archivo nuevo,
**sin repetir herramientas**. El diagnóstico original permanece respaldado.

## Simulación

Cuatro suites XSim: semillas 1, 7 y 42 con watchdog 32, y semilla 1 con el
watchdog de producción 65.536. Cada suite verifica 85 casos y cinco resets,
392 bursts AXI y 18.827 beats recibidos. De ellos, 18.120 beats / 1.159.680 bytes
pertenecen a operaciones publicables; 707 beats de operaciones fallidas o
interrumpidas permanecen privados. Se incluyen stalls en AR, salida y finalización.

Los cuatro revisores rechazaron 12 corrupciones de evidencia cada uno:
omisiones/duplicaciones, direcciones, datos, identidad, resets, estados,
errores AXI y valores no finitos. Los hashes de las tres fuentes realmente
compiladas coinciden con el paquete, incluidas las copias de puertos explícitos.

## Síntesis

| Métrica | Resultado |
| --- | ---: |
| Dispositivo | `xcvu5p-flva2104-2-i` |
| Reloj solicitado | 5 ns |
| LUT / registros / DSP / BRAM | 1.300 / 2.013 / 2 / 0 |
| WNS / WHS | +2,572 ns / +0,066 ns |
| Cajas negras / latches / loops | 0 / 0 / 0 |

No hay clocks faltantes ni endpoints internos sin restringir.
Las E/S externas carecen de delays porque es síntesis OOC: quedan 1.292
entradas y 880 salidas por restringir en integración. Estos márgenes no
califican colocación/ruteo ni el enlace físico DDR.

## Corrección del revisor y trazabilidad

Vivado informó `NAME=design_1` para el objeto de diseño. El revisor anterior
lo confundía con el módulo superior y falló después de comprobar las suites.
La corrección exige el módulo exacto en utilización, jerarquía y timing,
el estado `Synthesized` y el comando real `synth_design -top`.
Siete controles de identidad alterada fueron rechazados. No se admitió el
nombre genérico como prueba de identidad ni se omitieron otras validaciones.

El archivo de adopción conserva todos los archivos originales con sus bytes
intactos, el diagnóstico anterior, la nueva revisión y `ADOPTION.json`, que
identifica el run real. Cambia el prefijo del contenedor para distinguir la
revisión; los comandos y fechas internos originales permanecen intactos.

- Archivo original: `48239fd2719549a198013441607212f77730692ba1a801383ecedd7958341728`.
- Archivo de adopción D:/NFS: `bec4f0992c194fd25fbde7d9659dd94d066c3080b781e5f8fec1109934e531a9`.
- Revisión nueva: `af34a64798bea9df11b90232047cef7d3554ec2d6f143fa4d673545fa8f1d01e`.
- DCP original: `55710334497bdd38ecb4b8c619839b309ce2442cd300b0a4972fd1f74889383a`.

El cargador todavía recibe la autoridad de lease/prefijo desde fuera.
Append, caché, atención integrada y DDR física tienen validaciones separadas.
