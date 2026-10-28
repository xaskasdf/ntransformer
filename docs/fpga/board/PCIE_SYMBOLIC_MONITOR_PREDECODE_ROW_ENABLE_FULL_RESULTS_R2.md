# E/D por enables: integración completa y revisión R2

Fecha: 2026-10-11 UTC. Tanda
`20261011T230854Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-row-enable-full-r1`.
Vivado terminó con salida 0 a las 23:23:43 UTC; el resultado está respaldado
en D: y NFS. La revisión R2 aprueba la integración RTL.

## Evidencia y alcance

Se ejecutaron **183.058 filas nuevas** en 16 bancos, comparando los 447 bits
públicos antes y después del flanco: 183.138 muestras por fase. El guardado
interno compara 21.951 bits de E/D y alineamiento, con 62 muestras adicionales
de inserción/restauración en el banco dirigido. Se compilaron las 28 unidades
esperadas y se verificaron sus hashes contra el snapshot y archivo terminal.

Los seis controles de entrada divergieron en sus testigos previstos.
El revisor rechazó 34 corrupciones de evidencia pública y ocho de cobertura
interna. Los seis bancos drenados no cancelan tráfico original. La comparación
conserva las referencias y oráculos anteriores.

## Corrección del revisor

R1 esperaba una advertencia de un módulo de control llamado
`ntrpu_pcie_symbolic_control_predecode_row_enable_r1.sv`, que no existe.
El monitor E/D realmente instancia el control RQ previamente validado,
`ntrpu_pcie_symbolic_control_predecode_rq_bitmap_r1.sv`. Su uso de `rq_held`
en la línea 42 precede a su declaración, igual que en el control original.

R2 corrige exclusivamente ese nombre esperado; mantiene 66 advertencias:
22 de cada control y 22 del entorno XSIM. Se rechazaron seis alteraciones
de número, nombre, línea y procedencia. No se cambiaron fuentes RTL, entradas,
resultados ni criterios numéricos; no se repitió Vivado. R1 y su fallo permanecen.

- Revisión R2: `3b20228ef232c6ee234ba1c5432ad5c74d4e8a31c0fe58549818f12725476f0a`.
- Archivo terminal: `0686cfa0d70baa6a5b34ff446a5e91d25394af5e03afb5c67c2e01df3f237cbc`.
- Fuentes: `6895b268baf40ac7297eb4e5b3abe69d87d34e0ce373a8089a35753fa177012f`.
- Configuración: `10b63912b124f27f21c222767b888f6025709c89dd3456eceb0e32ac771f90b3`.

## Continuación

La cola R6 conserva los coordinadores detenidos y ejecuta solamente las cuatro
etapas nunca iniciadas: síntesis E/D → replay de su netlist → comparación
de timing E/D–RQ984 → selector `mask_rows`. Cada etapa exige revisión y respaldo
de su predecesora. Después siguen los bancos GEMV y FFN con pesos Llama reales.
El PASS RTL habilita síntesis; todavía no acredita recursos, timing, PCIe nativo
ni funcionamiento físico de esta variante.

La síntesis comenzó a las 23:36:31 UTC, tanda
`20261011T233631Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-row-enable-synth-r1`.
El controlador R6 tiene PID local 161328 y plan
`c8c34109cc6cf6c6b1a4c0535b3c417e9be805ae44ddaf644bd49d88c6cc405d`.
Estos identificadores son una observación de ejecución, no una aprobación.
