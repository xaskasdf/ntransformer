# Colocación con BUFGCE y timeout del ruteo predeterminado

Revisado el 2026-10-08. La colocación terminó; **el ruteo no terminó**
dentro de su presupuesto de una hora. Se conservaron y verificaron ambos
intentos en D: y NFS. No existe netlist final de este ruteo para reproducir.

## Intentos conservados

`20261008T213458Z` abrió correctamente el DCP, pero falló antes de colocar:
la consulta de propiedades recibió nombres de puertos en lugar de objetos
Vivado. R1 cambió únicamente ese bucle. No se modificaron restricciones,
entrada, algoritmos ni reportes; se conservaron el Tcl y el fallo originales.

`20261008T213940Z` ejecutó `place_design` en 383 s. El comando de ruteo
predeterminado avanzó por dos iteraciones globales y entró en corrección
intensiva de hold. El límite de 3.600 s terminó la herramienta con código
**124**, duración registrada **3.600,007 s**. El wrapper local salió con
código 0 después de recopilar; eso no convierte el resultado Vivado en éxito.

## Métricas disponibles

| Fase | WNS, ns | TNS, ns | WHS, ns | THS, ns |
| --- | ---: | ---: | ---: | ---: |
| Entrada optimizada | −7,160 | −70.329,727 | −2,702 | −68.385,430 |
| Colocación completa | −6,192 | −80.700,734 | −3,066 | −61.096,625 |
| Último resumen intermedio del router | −27,224 | −355.196,077 | −3,043 | −39.957,920 |

La última fila no es STA final ni prueba de ruteo completo. Los 51.828 LUT,
10.733 FF y 1.895 CARRY8 corresponden a colocación. El análisis de sus
[20 rutas de setup](PCIE_INVENTORY_CLOCK_PLACED_PATHS.md) incluye retardos
estimados; el peor camino tiene 42 niveles y 10,055 ns de datos.

El reporte inicial del router contó 17.563 pines con restricciones ajustadas
de setup/hold, pero su archivo contiene sólo 10.000. No intercambiar ese
total con estadísticas de fases posteriores. `Route 35-197` conserva la
advertencia de origen externo sin contexto; `Route 35-198` llegó al límite
de 100 mensajes. El estado OOC no acredita la frontera de placa.

## Artefactos y revisión

`placed.dcp`: 19.054.760 bytes, SHA-256
`bbc5657611d5bdb7987a1b4624cbf43d93ad56757946ef6c39d326037598c377`.
Sus 23 miembros son legibles. Sus restricciones temporales guardadas coinciden
con las del DCP de entrada, incluido el XDC de contexto: 4 ns, E/S cero,
sin nuevas excepciones ni `HD.CLK_SRC` supuesto.

No existen `route_attempt.dcp`, `routed.edf` ni `routed_funcsim.v`.
La continuación automática se detuvo en esa puerta y no lanzó auditoría
ni replay sobre artefactos ausentes. La revisión del timeout corrigió su
primera expectativa: el reloj del XDC de contexto ya existía en la entrada;
no debe confundirse con una restricción añadida por colocación.

| Intento | Archivo SHA-256 |
| --- | --- |
| `20261008T213458Z` | `c5d794d1fcac71a220f3a2c02a9a1a68110780f04c0c5bb4a33bee081dd35fe6` |
| `20261008T213940Z` | `980131e4dc07c74b998949ba3c3627bf5618fcf883b5bc1f7741b5684b56d169` |

El manifiesto R1 verifica 47 archivos; el paquete ocupa 40.148.735 bytes. La revisión
`…-independent-default-route-timeout-review-r1.json` tiene SHA-256
`460dd43b5a4b7a1d4c08a9b14a83c60a82689ebd714a9d993328b9a98eacab93`.
La revisión `…-observation-archive-review.json`, SHA-256
`fc5bc301d594783f0c36c2d688a0ebc9c607fc3d2b9f7bb444d9dd539497f5ab`,
coteja las observaciones previas con los archivos terminales exactos.
Los JSON usan el prefijo R1 completo en `build/cloud-setup/20261002-board/`.

## Siguiente medición

`20261008T224646Z` ejecuta un [diagnóstico Quick](PCIE_INVENTORY_CLOCK_QUICK_ROUTE_CONTRACT.md)
desde el checkpoint colocado, máximo 1.200 s. No repite síntesis ni colocación
y no optimiza timing. Sus artefactos requerirán revisión y replay propios.
El objetivo de cierre temporal permanece abierto; esta medición no lo sustituye.
La integración debe considerar el [BUFG_GT real de XDMA](PCIE_INVENTORY_NATIVE_CLOCK_ORIGIN.md).

El replay aprobado anteriormente pertenece a la netlist **optimizada antes
de colocar**. No demuestra funcionalidad de los cambios del placer ni cierre
de endpoint, reset/CDC, DDR/512 GiB, inferencia o diseño físico de placa.
