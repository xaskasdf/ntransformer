# Revisión de timing del reset retenido

Plan del 2026-10-06, posterior a [la auditoría OOC](RETAINED_SYNTHESIS_AUDIT.md).
Entrada inmutable: DCP del supervisor
`20261006T132220Z-V4-ffn-supervised-retained-start-synth-frozen`, SHA-256
`669510702f9fa0c3339a89c9c6da8584744579181f761306add3400bc7a539d1`.

## Diagnóstico que debe resolverse

Vivado informa seis pares de clocks con `TIMING-6/7`. El inventario
`ntrpu_board_retained_start_followup.tcl timing` debe identificar sus
startpoints, endpoints, primitivas y pines. El revisor exige comprobar
la hipótesis de que terminan únicamente en CLR de las cadenas de reset;
si aparece otro destino, se debe conservar el diagnóstico y ampliar el
análisis antes de aplicar excepciones.

El RTL propio tiene tres registros FDCE por cadena, `ASYNC_REG` y
INIT=0. El valor inicial mantiene cerrado el acceso incluso antes de
recibir un reset. Una sustitución que cambie ese valor no es equivalente.

## Criterios para una nueva revisión de constraints

1. Enumerar las cadenas y verificar sus conexiones, polaridad, clock,
   INIT y etapas. Usar objetos exactos; rechazar selectores vacíos.
2. Si la estructura lo justifica, ensayar excepciones limitadas al clear
   asíncrono de esas cadenas. No excluir pares completos de clocks,
   todos los registros, ni las salidas sincronizadas hacia sus consumidores.
3. Comparar antes/después: rutas entre clocks, rutas D entre etapas,
   recovery/removal del reset ya sincronizado, metodología, CDC y
   endpoints internos sin restricción. Conservar ambos reportes.
4. Ejecutar el banco original de 13 escenarios sobre el netlist a 5 y
   37 ns. Una simulación funcional no demuestra ausencia de metastabilidad.

AMD distingue la aserción asíncrona de la liberación sincronizada y
advierte contra mezclar CLEAR/PRESET en la misma cadena o sincronizar
sin coordinación un reset varias veces en un dominio.
[UG906 2026.1: sincronizador de reset](https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/Asynchronous-Reset-Synchronizer).
El Tcl XPM instalado restringe su entrada `src_arst` dentro de su propia
instancia; esto sirve como referencia de alcance, no como autorización
para excluir los caminos de datos del diseño propio.

## Aviso independiente

`CDC-26` continúa abierto: el reporte señala la entrada externa y un PRE,
pero su descripción corresponde a una colisión de LUTRAM. La documentación
oficial describe una RAM entre clocks de lectura/escritura; nuestro
inventario no contiene RAM. Hace falta revisar el reporte nativo y, si
procede, reproducir el caso mínimo; no se añade un waiver por semejanza.
[UG906 2026.1: condición LUTRAM](https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/LUTRAM-Read/Write-Potential-Collision).

El cierre físico, los contratos host/XDMA y la autorización de reset
operativo siguen siendo requisitos separados.
