# Optimización física del checkpoint de raíz proyectada

Estado 2026-10-15 16:46 UTC: **R5 terminada, revisada y respaldada; timing pendiente**.
Parte `xcvu5p-flva2104-2-i`, reloj 5 ns, CPU0/1, máximo dos hilos y
3.600 segundos. No se cambian RTL, checkpoint de entrada ni restricciones.

## Punto de partida y criterios

La [raíz ruteada](LLAMA32_PROJECTED_ROOT_ROUTE_R1.md) tiene 86.884 conexiones
ruteables, todas completas y sin errores, pero WNS −0,079 ns y WHS +0,010 ns.
Se aplica `phys_opt_design -directive Explore` y se reabre el resultado
para revisar rutas, recursos, setup/hold, reloj y restricciones.
No se añaden excepciones ni se reduce la frecuencia para aprobar.

El contexto es OOC, con BUFGCE provisional en `BUFGCE_X0Y60` y E/S externas
sin presupuesto. Aun si mejora el timing interno, faltarán integración de
placa y simulación funcional posterior al ruteo.

## Diagnóstico conservado y corrección

R4 (`20261015T160507Z`) terminó con error antes de optimizar. El guard
consultaba `get_property ROUTED_FULLY`, vacío en el checkpoint reabierto.
El informe original ya conserva ese campo vacío junto con 86.884 rutas
completas. Archivo diagnóstico verificado en D:/NFS:
`1fface677fd6b2f281b93e2ff67ef55642c3acb237d819ac119a831480f27a76`.

El Tcl R2 exige ahora los tres contadores exactos del informe real mediante
[`report_route_status -return_string`](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/report_route_status),
además de identidad, reloj y ausencia de cajas negras. Guarda informe e
identidad del padre y añade cuatro controles negativos al revisor: total,
ruteo incompleto, errores e identidad equivocada. No elimina la comprobación
de ruteo; reemplaza una consulta inválida por evidencia del diseño.

## Paquete y continuación

La ejecución terminó correctamente en 1.660,27 segundos. Tras reabrir el
checkpoint: 86.884/86.884 nets ruteables completos, cero errores y cero
huecos explícitos en 575.743 entradas del inventario jerárquico. Setup
mejoró de −0,079 a **−0,034 ns**; hold pasó de +0,010 a **+0,003 ns**.
La mejora de 0,045 ns no cierra 200 MHz. Se conservan 48.175 LUT,
44.887 FF, 84,5 BRAM y 134 DSP; no hay cajas negras, latches ni rutas
internas sin reloj/restricción. Las E/S siguen sin presupuesto de placa.

El revisor anterior exigía el encabezado `Routed`; Vivado informa
`Physopt postRoute` después de esta optimización. El revisor R3 exige
ese estado exacto en los tres informes y conserva todos los criterios
de rutas, reloj, netlist y restricciones. Rechazó 18 controles de evidencia
y cinco del netlist. Se revisaron los mismos resultados sin repetir Vivado.

| Evidencia D:/NFS | SHA-256 |
| --- | --- |
| Archivo de ejecución R5 | `e6025abb78830424d28d2d6ffc4bc295b717adbff6d2de3d5706e5337f593219` |
| Revisión R3 | `38e53de0cea5ae9d6e406d0b46392c01415bb9c700032e33b93c823b81bb95a4` |
| Suplemento de revisión | `a16017193a5659aec2aa3cf22839b35cdf0b10be584f3a65bab5e0093a22629b` |
| Checkpoint resultante | `349422ac3cdaeae5fb2bb10050044241224649e907aae188355e530f841a6d0f` |

- Run R5: `20261015T161045Z-V7-llama32-attention-projected-root-physopt-vivado-r5`.
- Checkpoint original: `05c0cd78f012aa1ba811ff1f64a52bcef3fcc0ec98b7cee10f33d33a99e1da51`.
- Fuente R5: `a1e4b95930fd2ced5a4cb8903a77b0d7e51f96d690574bfb5483778519bfaef1`.
- Configuración: `3cdfd68ba7c8f8d72d9bd3ac3c155f0988396800cc5b1c28b2ccb479c4d64a2e`.

El coordinador exige revisión y archivo íntegro antes de liberar el slot.
Después siguen propietario FFN Vivado R4, FFN completa R4, cargador residual
R2 y [propietario residual R1](LLAMA32_RESIDUAL_OWNER_R1.md). Los sucesores
conservan las fuentes compiladas de sus antecesores. Un timing
negativo se registra como tal; no invalida por sí solo una medición íntegra.
