# Ruta crítica mapeada de la variante por hojas

## Comparación en la misma etapa

Se analizaron las netlists e informes archivados de las síntesis 951 y 962,
sin ejecutar otra tanda de Vivado. Ambas mantienen las condiciones verificadas
en la [comparación de síntesis](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_SYNTHESIS_RESULTS_R1.md).

| Peor ruta de setup | Referencia 951 | Por hojas 962 |
| --- | ---: | ---: |
| Bit de origen `B.rc_user` | 64 | 75 |
| WNS estimado | −3,584 ns | −3,746 ns |
| Niveles / primitivas | 31: 30 LUT + CARRY8 | 31: 30 LUT + CARRY8 |
| Mayor cadena directa de factores AND | 11 LUT | 10 LUT |
| Mayor cadena directa de factores OR | 9 LUT | 10 LUT |

Ambas terminan en `R.fault_latched[10]`. Las cadenas son factores respecto
al pin atravesado por esa ruta, no un inventario de todos los AND/OR del
diseño. Las peores rutas parten de bits distintos; estas diferencias no
atribuyen causalmente una LUT individual a la modificación del RTL.

## Función comprobada en la netlist nueva

Diez LUT6 consecutivas, `_i_2614` a `_i_2605` dentro de `delivery_stage`,
comparten `INIT=64'hEEE222E200000000`. Cada una calcula:

`salida = entrada_del_camino AND bit_seleccionado`

Los datos son los bits 2–11 de nets nombradas `query_delivered[1]`,
`context_delivered` y `query_delivered[0]`. Las dos señales selectoras son
las mismas en las diez LUT: si I1 es 0 se selecciona query 1; si I1 es 1,
I3 elige entre contexto y query 0. La LUT siguiente, `_i_2465`, calcula XOR
con el bit 12 seleccionado. Se comprobaron las 64 combinaciones binarias
de cada celda, **704 casos**, y tres alteraciones deliberadas se rechazaron.

La composición exacta es:

`salida_final = dato[12] XOR (entrada_en_bit2 AND (&dato[11:2]))`

Esto identifica una estructura de propagación de acarreo sobre datos
seleccionados. La entrada anterior al bit 2 no quedó clasificada; por tanto,
no constituye una atribución formal a una sentencia RTL concreta.

El tramo OR tiene diez LUT y acumula otros términos de fallo. Expresar un
árbol OR en RTL no impuso una estructura equilibrada en la netlist: el mapeo conservó
una cadena larga. Cada tramo de diez LUT suma 0,380 ns de arcos lógicos;
se excluyen interconexión y los otros arcos. No son ahorros pronosticados.

## Verificación y siguiente decisión

Los análisis enlazan los archivos certificados, netlist, informe y revisión.
Se verificaron conexiones directas y la interrupción que pasa realmente por
`CARRY8.DI[5] → CO[6]`. No se clasificó la función completa de ese CARRY8.
El orden de índices de LUT3 se comprobó con 2.048 casos de la tabla UDP
instalada; LUT4/5/6 usan la expresión indexada de sus bibliotecas.

El contexto por prefijo sigue en cola. Sus resultados determinarán si elimina
la dependencia larga. Si persiste, se estudiará calcular el límite de payload
antes de seleccionar el contexto, comprobando primero anchos, X/Z y el timing
de la etapa que reciba la aritmética. No se cambia ahora el banco en ejecución.

Análisis de la variante SHA-256:
`4df165d5f8dc9fbeb878cfdb30aebd5a3083064d0ed9ac2ab1222f0b53bd3b09`.
Análisis de referencia: `2935c29120525366fec6555b8e425949c5076a84afdb77a152848f4f3f7d865e`.
Prueba de recurrencia: `9e265daed81e3d5ffa56701cbbc744341d3adc222fc572fc2a30adbbd14c0b61`.
Son pruebas binarias de funciones locales, no equivalencia X/Z, sensitización
de todo el camino ni timing posterior al ruteo.
