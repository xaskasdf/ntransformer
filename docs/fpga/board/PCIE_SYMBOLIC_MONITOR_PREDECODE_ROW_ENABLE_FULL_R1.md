# Integración de habilitaciones E/D — R1

## Cambio y referencia

La variante conecta al monitor RQ 983 las habilitaciones por fila del
[par E/D](PCIE_SYMBOLIC_MONITOR_PREDECODE_ROW_ENABLE_R1.md). Mantiene C,
B, retiro winner y las cinco etapas originales. Conserva la versión
original de suma E: este ensayo mide por separado el cambio de escritura
de tabla y no combina los candidatos E-sum, B-bound o C-count.

Se agrega al banco una pareja original E/D independiente, alimentada
únicamente por el mismo registro C que observa el candidato. Sus salidas
no realimentan el diseño. Permite comparar estado y alineación sin asumir
que campos internos de otras revisiones C son idénticos. Las referencias
de entradas PCIe y los oráculos públicos existentes permanecen intactos.

## Cobertura y criterios

La ejecución conserva **183.058 filas nuevas en 16 bancos**, seis bancos
drenados, 31 sondas quiet y seis controles negativos de entradas. Compara
447 bits públicos antes y después del flanco, con las funciones de
contabilidad calificadas. Los controles negativos mantienen sus primeros
testigos originales; las comprobaciones internas se aplican a los bancos
positivos.

En cada muestra positiva también compara **21.951 bits internos**: las
256 filas completas de descripción y atributos, contexto retenido,
salidas D, entrada E de D, selecciones de admisión y los 512 enables nuevos.
La máscara esperada se calcula sobre la pareja de referencia. Su cobertura
debe coincidir exactamente con las 183.138 muestras pre y post y las
62 perturbaciones/restauraciones quiet. El revisor comprueba cada banco
y rechaza ocho alteraciones adicionales de su resumen.

## Orden y respaldo

La cola espera el PASS del banco aislado de 40.958 ciclos, sus seis
mutantes, 39 controles de evidencia y respaldo verificado. Después
congela un snapshot nuevo y ejecuta una sola tanda con las 28 unidades
de compilación previstas. Todo resultado terminal se preserva en D: y
NFS; R17 conserva los borradores y snapshots pendientes. Un fallo detiene
la cadena sin reintento automático.

La auditoría estática verifica recetas, conexiones y 16 resúmenes válidos;
rechaza 128 alteraciones sintéticas. Esto no sustituye la ejecución RTL.
Solo un PASS integrado revisado habilitará síntesis y replay de su propia
netlist; el timing físico y la placa siguen pendientes.
