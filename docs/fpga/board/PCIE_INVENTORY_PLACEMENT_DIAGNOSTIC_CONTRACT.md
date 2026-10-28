# Contrato: implementación diagnóstica del inventario PCIe

Después de cerrar R12 y verificar R98, estudiar colocación y ruteo del
monitor funcionalmente revisado con mejor WNS OOC entre R8 y R12.
Esta tanda mide el bloque aislado: no cierra V5–V8, pinout, DDR/512 GiB
ni integración con el endpoint. La selección del DCP se registra por hash.

## Entradas y contexto

Reutilizar el DCP de síntesis respaldado y su netlist ya reproducida;
no resintetizar con otro RTL o directiva dentro de esta comparación.
Verificar top `ntrpu_pcie_inventory_join`, parte `xcvu5p-flva2104-2-i`,
un reloj `pcie_user_clk` de 4 ns, puertos, ausencia de cajas negras y
latches, y las restricciones heredadas de E/S cero. Exportar XDC y
reportes antes de cualquier modificación.

El flujo diagnóstico anterior del repositorio usa `HD.CLK_SRC` para
declarar ubicaciones de reloj padre supuestas. Probar explícitamente
`BUFGCE_X0Y60` sólo si la herramienta confirma su existencia y tipo;
no elegir silenciosamente otro sitio. No representa un buffer colocado
en la placa ni una asignación aprobada del reloj PCIe. Registrar propiedades
y comparar timing antes y después de aplicar este contexto.

AMD distingue el reloj externo supuesto de un buffer incluido en el módulo:
el primero conserva estimaciones de su distribución en OOC. Por ello hay
que identificar las rutas parciales de frontera, no declararlas ruteadas.
Ver [UG905: buffers y relojes](https://docs.amd.com/r/en-US/ug905-vivado-hierarchical-design/I/O-and-Clock-Buffers).

## Ejecución y aceptación

1. Registrar `help` de los comandos en Vivado 2026.1 y propiedades reales
   del DCP. Mantener constantes período, incertidumbre, puertos y
   restricciones; el único contexto añadido es el origen de reloj supuesto.
2. Ejecutar una tanda aislada `opt_design`, `place_design` y `route_design`
   con configuración predeterminada y cuatro hilos como máximo. Guardar
   checkpoints y timing tras contexto, optimización, colocación y ruteo.
   Vivado documenta este flujo no basado en proyectos en
   [UG835: place_design](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/place_design).
3. Conservar intentos fallidos, logs y reportes parciales. Exportar recursos,
   setup/hold, `check_timing`, rutas, DRC, metodología, relojes, excepciones,
   XDC, EDIF y netlist funcional. No introducir excepciones temporales,
   relajaciones de severidad ni cambios de capacidad/latencia para pasar.
4. Auditar el DCP resultante: separar nets completamente ruteadas, gaps
   atribuibles al reloj padre y problemas internos. Mantener visibles DRC
   como `RTSTAT-10`; un retorno cero no acredita cierre físico.
5. Si existe una netlist utilizable, exigir DAG acíclico y replay de esa
   netlist: 22.330/67 más propiedad 46.356/3.872, RQ 8.850/240, prefijos
   544/60 y carriles 632/60. Conservar todos los estados posteriores a
   fallos, controles de evidencia, arranque binario de 200 ns y muestreo
   de 1 ns. El DUT no debe compilar RTL. Sin SDF: es validación funcional.

Si se elige R8, preparar primero su replay RTL con el nuevo suplemento
binario de 632 filas; no atribuirle resultados de R12. Si se elige R12,
reutilizar únicamente su replay correspondiente ya revisado. Respaldar cada
tanda localmente y en NFS. Las mediciones internas con contexto supuesto
deben volver a validarse al integrar reloj, endpoint, DDR y restricciones
de placa reales. [Cola y criterios completos](VIVADO_QUEUE.md).
