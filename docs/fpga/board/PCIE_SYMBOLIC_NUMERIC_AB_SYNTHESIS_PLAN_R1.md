# A/B: medición y replay propios pendientes

Plan posterior a las pruebas RTL y focalizadas del 2026-10-09.
**Sin medidas de síntesis de estas etapas todavía.**

## Unidades y restricciones

Medir `ntrpu_pcie_symbolic_apply_r1`, `ntrpu_pcie_symbolic_delivery_r1`
y un contenedor que conecte ambas exactamente como el banco aprobado.
El contenedor recibe control y registro descriptivo, conserva el registro
intermedio A y publica B; no incorpora control C/E/D, validación tardía ni R.
La instancia conjunta se mide directamente, sin sumar unidades aisladas.

Usar `xcvu5p-flva2104-2-i`, síntesis OOC `RuntimeOptimized`, jerarquía
`rebuilt`, dos threads, reloj de 4 ns, I/O cero y ninguna excepción.
Exportar DCP, EDIF, XDC, netlist funcional y reportes de recursos,
temporización, metodología, DRC y `check_timing` de cada unidad.
Mantener todos los bits del registro publicado; no eliminar diagnóstico
para mejorar artificialmente el resultado de la comparación.

## Revisión y netlist

Auditar primitivas y ciclos combinacionales del EDIF, restricciones reales
y advertencias. Publicar LUT, FF, CARRY8, memoria/DSP y peores caminos
setup/hold. Un WNS favorable sin colocación/ruteo no certifica frecuencia
en placa. El C/E/D R3 previo sigue sin cerrar 4 ns.

Después comparar la netlist propia con los módulos RTL inmutables y las
observaciones R7 por origen. El replay mapeado usa entradas binarias,
GSR explícito, muestreo tras flanco y vaciado contabilizado; la prueba
de cuatro estados ya aprobada sigue siendo requisito independiente.
Derivar los puertos del circuito realmente exportado y comprobar las
fuentes compiladas. Los controles negativos deben detectar omisión de
orígenes, desalineación de datos/control y falta de reset frío.

Cada resultado, incluso diagnóstico, se archiva en D: y NFS con sus fuentes.
La síntesis aislada no cierra el monitor completo, retiro de bytes,
validación tardía, CDC/reset, DDR de 512 GiB, pesos reales ni placa.
