# Prioridad RC precalculada — experimento aislado R1

La comparación optimizada 974 mantiene dos fallos en C: entrada RC→C
−2,546 ns y realimentación C→C −2,407 ns. Esta última atraviesa seis
CARRY8 en tres cadenas, con nets asociados a `rc_selector/newest` y
`commit_valid`, hasta `registered_ops.commit_slot[422]`.
[Mediciones y límites](PCIE_SYMBOLIC_MONITOR_PREDECODE_PAIRED_OPT_RESULTS_R3.md).

## Cambio propuesto

Derivar `ntrpu_pcie_rc_update_select_commit_priority_r1.sv` del selector
R3 realmente compilado. Precalcular la prioridad entre los cuatro lanes
EOP antes de conocer qué eventos tienen propiedad y coinciden con cada
tag. Una tabla combinacional de 16 entradas × 2 bits cubre las máscaras
posibles. Cada tag genera su máscara mediante condiciones procedurales
conocidas y selecciona la entrada correspondiente. Empates conservan
el primer slot; la máscara vacía devuelve valid=0 y slot=0.

Solo cambia la selección final de commit. Se mantienen decodificación,
propiedad, ancestros, máscaras de prefijo, puertos y ausencia de estado
registrado. No cambia la latencia pública ni el perfil RQ2/RC4/256 tags.
No se integra todavía al monitor completo.

La alternativa de estrechar `newest` ya se probó en el monitor anterior:
[R9 redujo área, pero empeoró su WNS](PCIE_INVENTORY_RC_SELECTOR_WIDTH_RESULTS.md).
Este experimento intenta separar la prioridad de la condición tardía;
la síntesis deberá medir si lo consigue.

## Comprobación local

El modelo finito aprobó 1.336.336 comparaciones: 1.048.576 binarias y
287.760 con un operando relacional desconocido, más seis controles
negativos. SHA-256 de `pcie-symbolic-monitor-predecode-commit-priority-r1-proof.json`:
`eab1817fbc7605078758585aa781630ad4f76bc8a9e725bc0b153e85f075b9db`.
Es un modelo de selección, no ejecución de SystemVerilog ni prueba
universal del selector con X/Z.

## Cola y aceptación

Esperar el cierre revisado y respaldado del replay optimizado 975.
Después, ejecutar sin modificar los bancos heredados: 1.186.064
comparaciones de propiedad/ancestros, incluidas 1.026.544 con X/Z y
840 bits, más 89.632 vectores de selección de 800 bits.
Tres mutantes reales fuerzan slot cero, omiten slot tres e invierten la
prioridad. Exigir testigos del comparador, compilación exacta, revisión
independiente y respaldo de cualquier desenlace en D: y NFS.

Límite: dos hilos y 900 segundos. No cancelar validaciones en curso.
El PASS aislado permite evaluar una integración separada; aún harán
falta regresión completa, síntesis comparable y replay de su netlist.
