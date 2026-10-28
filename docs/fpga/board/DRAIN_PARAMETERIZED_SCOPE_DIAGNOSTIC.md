# Capturas de bancos parametrizados

La tanda `20261006T074352Z-V4-ffn-supervised-drain-sections-r3-frozen`
terminó con código 1 y quedó respaldada. El perfil por defecto produjo
4.560.201 bytes de VCD; el siguiente se detuvo antes de simular.

XSIM enumeró la raíz como ` /\tb_ntrpu_drain_sections(EXTERNAL_FIXTURE=1) `,
con identificador escapado y espacio final. El patrón anterior
`/tb_ntrpu_drain_sections*` no la encontraba. Este resultado identifica
un error de selección de jerarquía; no demuestra un fallo del RTL ni una
contaminación de estado entre simuladores.

Las revisiones Tcl R5 (hardware original), R6 (proveedor local R2) y
`ntrpu_board_drain_shared_fault_r2.tcl` conservan el objeto devuelto por
`get_scopes /*`, seleccionan la raíz por nombre normalizado y usan
`current_scope` con búsquedas relativas de señales. La normalización solo
sirve para seleccionar: nunca reemplaza el identificador escapado real.
Cada grupo debe contener señales; el script registra su cantidad, rechaza
capturas vacías y sale de XSIM al detectar errores.

La búsqueda relativa sigue la API documentada de
[get_objects, UG835 2026.1](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/get_objects).
La validación definitiva requiere ejecución nativa de todos los perfiles
y comparación independiente de transacciones, marcadores y VCD. Un mensaje
HDL PASS aislado no cierra esa validación.

La cola R2 usa otro ZIP inmutable, verifica sus archivos locales y remotos,
respeta las ejecuciones existentes y respalda cada resultado. No modifica
las tandas originales ni omite el preflight de recursos. Se mantienen
pendientes los controles negativos RTL, los contratos externos de XDMA y
host, y la calificación física de la tarjeta.
