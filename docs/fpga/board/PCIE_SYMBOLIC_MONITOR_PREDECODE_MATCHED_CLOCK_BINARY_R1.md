# Replay de los dos monitores optimizados

Fecha: 2026-10-11 UTC. Cola posterior a los
[diagnósticos de reloj RQ984/E/D](PCIE_SYMBOLIC_MONITOR_PREDECODE_MATCHED_CLOCK_R1.md).
Todavía no tiene resultados; no sustituye los replays sintetizados ya aprobados.

## Entradas y criterios

Para cada miembro se usa su propia `optimized_clock_funcsim.v`, extraída del
archivo verificado de la optimización BUFGCE. Su SHA-256 debe coincidir con
la revisión estructural, la fuente remota y el archivo copiado por XSIM.
Se conservan todos los puertos externos y se evita colisión entre módulos
de la netlist y las referencias RTL independientes.

El banco de cada candidato se conserva sin modificar: 16 bancos, 174.436
filas binarias, 447 bits públicos antes y después del flanco, diez sondas
de quiet y sus restauraciones, seis mutantes de entrada y 34 corrupciones
de evidencia. Las funciones de revisión mantienen el mismo AST que el
revisor sintetizado de ese miembro.

La biblioteca fuente AMD `BUFGCE.v` se verifica antes y después del banco
y contra el archivo de optimización. Se exige `unisims_ver` en las 22
elaboraciones y exactamente un BUFGCE en la netlist. Esto conserva la
limitación anterior: no prueba identidad interna de la biblioteca precompilada.

## Ejecución

`continue_symbolic_monitor_predecode_matched_clock_binary_r1.py` espera que
los dos diagnósticos de reloj terminen revisados y respaldados. Luego ejecuta
RQ y E/D secuencialmente, con dos hilos y 3000 s por miembro. Congela entradas,
valida dependencias y respalda cada resultado en D: y NFS. No crea una segunda
tanda si el observador pierde conexión ni altera los propietarios anteriores.

Solo el PASS de ambos replays permite preparar la comparación de colocación
y ruteo Quick, todavía separada. Cada DCP ruteado necesitará reapertura,
auditoría y otro replay propio. Ningún PASS funcional sin SDF califica
setup/hold, interfaces físicas, placa o inferencia completa.
