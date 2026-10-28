# Síntesis comparable de máscaras fijas: cola R2

Preparación 2026-10-12 UTC. Exige el PASS revisado y respaldado de la
[integración completa](PCIE_SYMBOLIC_MONITOR_PREDECODE_MASK_ROWS_FULL_R1.md).
Todavía no aporta métricas del candidato.

## Ensayo conservado

Preflight RTL de 529 filas y 31 sondas quietas; 26 archivos compilados
en simulación y 14 en síntesis. Parte `xcvu5p-flva2104-2-i`, reloj de
4 ns, retardos de E/S de 0 ns, `RuntimeOptimized`, jerarquía `rebuilt`,
dos hilos y límite de 1.800 s. La referencia es RQ984, síntesis
`20261011T180537Z`, con las mismas condiciones y versión de Vivado.

Se conservan la auditoría EDIF de ciclos, doce controles de cobertura,
revisión de advertencias por archivo/línea y comparación de recursos,
WNS/TNS, hold y peor camino. No se introducen excepciones ni cambios
de latencia para obtener slack favorable.

## Corrección previa a la ejecución

R1 no fue lanzada. La comprobación estática detectó tres referencias
heredadas de RQ: el sufijo de la revisión completa, el controlador de
esa dependencia y una referencia al coordinador de síntesis. R2 corrige
esos enlaces y usa archivos propios; conserva los borradores R1.
El RTL, Tcl, runner de Vivado, auditores, revisor y comparador no cambian.

## Cola y alcance

`continue_symbolic_monitor_predecode_mask_rows_synth_r2.py` espera
el resultado completo, congela y respalda antes de ejecutar. Cualquier
resultado terminal se conserva localmente y en NFS. Un fallo detiene
esta rama para inspección.

La revisión de síntesis solo habilita preparar el replay de la netlist
propia. Siguen pendientes ese replay, implementación física y cualquier
aprobación de PCIe, DDR, placa o inferencia completa.
