# Síntesis propia del selector RC por prioridad — R1

## Dependencia y propósito

La síntesis queda condicionada al PASS revisado y respaldado de la
integración `commit-priority-full-r1`. El candidato debe aprobar 183.058
filas RTL nuevas, 447 bits públicos, seis controles negativos, 34 controles
de evidencia y los seis bancos drenados sin cancelar el tráfico original.
El PASS aislado de la tanda 977 no sustituye esta condición.

Se comparará con la síntesis 970
`20261011T143058Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-fault-leaves-winner-synth-r1`:
105.518 LUT, 27.534 FF, 2.104 CARRY8 y WNS estimado −2,768 ns.
La finalidad es medir el cambio de selección RC manteniendo las demás
etapas y los cinco ciclos de latencia.

## Condiciones de ejecución

- VU5P `xcvu5p-flva2104-2-i`, reloj de 4 ns y retardos E/S de cero.
- `synth_design -directive RuntimeOptimized`, OOC, jerarquía `rebuilt`.
- Dos hilos; límite total de 1.800 s, incluido el preflight de 529 filas.
- Preflight con 26 unidades de simulación: conserva los selectores y
  módulos C originales en las referencias y los nombres propios del candidato.
- Síntesis de un monitor y 14 unidades fuente: incorpora únicamente C y
  selector nuevos, conservando E/D/A/B/R de la referencia 970.
- Sin excepciones de timing, cambios de reloj, colocación ni ruteo.

## Revisión y respaldo

Antes de lanzar se congela un snapshot y se respalda en D: y NFS. Se
archiva todo resultado terminal, incluidos fallos y timeouts. La revisión
comprueba archivos efectivamente compilados, preflight, advertencias,
restricciones, recursos, timing y ausencia de ciclos combinacionales en el
DCP. Los controles de corrupción del auditor estructural se conservan.

La comparación valida iguales restricciones, estrategia y builds de
Vivado antes de calcular diferencias. LUT y timing de síntesis son medidas
del flujo OOC, no frecuencia física ni rendimiento de inferencia.
Después se requiere replay binario de la netlist propia; la validación RTL
de X/Z y el cierre físico siguen siendo condiciones distintas.

Orquestación: `continue_symbolic_monitor_predecode_commit_priority_synth_r1.py`,
en `build/cloud-setup/20261002-board/`. La preparación y todas sus
transformaciones están registradas en el manifiesto `-synth-r1-preparation.json`.
