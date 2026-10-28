# Replay del ruteo Quick con BUFGCE interno

Preparado; no ejecutado ni aprobado por este documento.

## Dependencias

Partir exclusivamente de `routed_funcsim.v` de la tanda
`20261008T224646Z-V4-ffn-supervised-pcie-inventory-clock-quick-route`.
Exigir respaldo verificado de la implementación y de su lectura posterior,
identidad del DCP y del EDIF reexportado, auditoría de ciclos y del BUFGCE,
y revisión independiente de restricciones, rutas, advertencias y timing.
Conservar los fallos de timing como diagnóstico: un replay funcional no
los corrige ni demuestra una frecuencia de operación.

## Ejecución

Usar los cinco bancos binarios R12 sin modificar vectores, referencia,
arranque de 200 ns ni muestreo de 1 ns: principal (22.330 filas), propiedad
por tag (46.356), RQ (8.850), prefijos (544) y contribuciones RC (632).
Compilar la netlist exacta como DUT y la referencia independiente R7;
no sustituir el DUT por RTL ni añadir SDF. Mantener el arranque XSIM sin
ondas automáticas y registrar los cinco inicios.

Vincular el hash de la fuente instalada `BUFGCE.v`, sus valores por defecto
y la biblioteca `unisims_ver` usada por XSIM. La identidad entre esa fuente
y la biblioteca precompilada sigue sin estar demostrada; conservar también
la discrepancia documentada de `SIM_DEVICE`.

## Cierre

Respaldar cada resultado, incluso fallos. Comparar todas las filas públicas,
canceladas y posteriores a fallos contra el RTL R12; ejecutar los 30 controles
de corrupción del revisor. Verificar archivos compilados y hash de la netlist,
diagnósticos del simulador y tamaños de ondas. Registrar duración como dato
observado, sin atribuir causalmente diferencias a la arquitectura del reloj.

El resultado sólo califica este replay binario funcional. No cierra timing,
endpoint nativo, reset de 50 ms, DDR, 512 GiB ni diseño físico de placa.

La directiva Quick no optimizó timing. Se mantienen explícitamente los fallos de setup y hold, incluyendo el hold entre registros.
