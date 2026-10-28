# Integración de prioridad RC — borrador R1

Preparada mientras la simulación 975 continúa. El ensayo aislado de
[prioridad RC](PCIE_SYMBOLIC_MONITOR_PREDECODE_COMMIT_PRIORITY_PLAN_R1.md)
está encolado detrás de su cierre. La integración completa queda en cola
condicional: solo se congela y ejecuta tras el PASS aislado revisado y
respaldado. No tiene todavía resultados de simulación.

## Separación de referencias

El selector aislado conserva el nombre de módulo histórico para usar
los bancos existentes. Compilarlo con ese nombre dentro del monitor
completo también cambiaría el selector usado por algunas referencias.
Por ello se prepararon nombres independientes:

- Selector `ntrpu_pcie_rc_update_select_commit_priority_r1`, en el archivo
  `ntrpu_pcie_rc_update_select_commit_priority_named_r1.sv`: solo renombra
  el módulo respecto del candidato aislado.
- C `ntrpu_pcie_symbolic_control_predecode_commit_priority_r1`: copia de
  C R3 con nombre propio y enlace al nuevo selector.
- Monitor `ntrpu_pcie_symbolic_monitor_predecode_commit_priority_r1`:
  copia del candidato 969/970 que cambia únicamente el enlace a C y su
  nombre. E/D/A/B/R, puertos y cinco ciclos se conservan.

El testbench cambia únicamente el nombre del candidato. El Tcl añade
los dos módulos nuevos y mantiene explícitamente los originales R3
para las referencias. Se esperan 26 unidades compiladas.

## Requisitos antes de ejecutar

Exigir PASS aislado, testigos de los tres mutantes, revisión y respaldo.
El flujo propio liga esos hashes al banco integrado:
183.058 filas nuevas, 447 bits públicos, 31 probes quiet, seis bancos
drenados sin cancelación de tráfico y seis controles de entrada.
Las funciones de contabilidad y los oráculos permanecen iguales.
El revisor exige ambas copias de C y sus avisos conocidos por separado;
los únicos cambios en el banco son el nombre del candidato y las
unidades añadidas. Cada desenlace se respalda antes de revisarlo.

No reutilizar filas aprobadas de 969. Tras el PASS completo, sintetizar
la variante exacta y comparar con 970 bajo las mismas restricciones;
probar después su propia netlist. El ahorro y el timing son hipótesis.
La implementación física y los demás requisitos de la cola siguen abiertos.

Manifiesto `pcie-symbolic-monitor-predecode-commit-priority-full-r1-sources.json`,
SHA-256 `aee4bfbdb95c229aac6ac0636f005c3372c8d0f8ac943ba5894672302e5548f0`.
Flujo `pcie-symbolic-monitor-predecode-commit-priority-full-r1-flow.json`,
SHA-256 `b1f6d2026b2839eba6be34e34d64f8f06cec4ecf74dc1bb921b290f509424133`.
