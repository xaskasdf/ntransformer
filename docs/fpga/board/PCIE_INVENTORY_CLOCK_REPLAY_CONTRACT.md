# Replay de la netlist con reloj interno

## Entrada y puerta de ejecución

La auditoría `20261008T205958Z` habilita solamente replay funcional.
Usar `optimized_clock_funcsim.v` conservada por el preflight
`20261008T205100Z`, vinculada por SHA-256 al DCP y EDIF recargados.
No volver a optimizar ni modificar RTL, netlist o biblioteca AMD.

Conservar las fuentes y el resultado del preflight detenido. La lectura
posterior confirma nombres y conexiones iguales, pero no reproduce la
desigualdad Tcl original: su causa sigue sin determinarse.

## Banco y revisión

Ejecutar los cinco bancos binarios completos R12: principal 22.330 filas,
ownership 46.356, RQ 8.850, prefix 544 y lanes 632. Conservar sus 30
controles de evidencia, comparaciones de estado público, cancelaciones y
filas posteriores a fallos. Referencia independiente R7 con entradas crudas;
no compilar RTL del DUT. Arranque 200 ns, muestreo 1 ns, sin SDF.

Mantener el arranque XSIM sin captura automática de ondas. Registrar cinco
arranques, archivos compilados, advertencias, duración y respaldo de todos
los resultados, incluso fallos. Comparar filas completas con el replay RTL
R12 congelado; un marcador PASS por sí solo no aprueba la tanda.

## Biblioteca y alcance

La netlist exporta BUFGCE sin parámetros. La fuente instalada de Vivado
2026.1 declara SIM_DEVICE="ULTRASCALE", mientras UG974 2026.1 documenta
"ULTRASCALE_PLUS". Preservar el hash de la fuente instalada antes y después
del replay y registrar el uso de `unisims_ver` en XSIM. Esto no acredita
por sí solo identidad entre la fuente y las bibliotecas precompiladas.
No corregir silenciosamente esta diferencia.

Sólo una revisión completa aprobada habilita preparar implementación
aislada del DCP con BUFGCE interno. Esta simulación no califica tiempos,
arranque físico, endpoint nativo, DDR, 512 GiB ni inferencia Llama.
