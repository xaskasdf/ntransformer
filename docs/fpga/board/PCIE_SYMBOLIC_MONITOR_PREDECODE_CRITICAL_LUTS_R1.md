# Predecode R1: funciones de la ruta crítica estimada

## Evidencia

Se inspeccionó la netlist optimizada de `20261011T085600Z`, sin modificarla.
La ruta B→R parte de `rc_user[64]` y termina en `fault_latched[10]`:
WNS −3,723 ns, retardo de datos 7,568 ns, 31 niveles —30 LUT y un CARRY8—.
La interconexión de este informe es estimada, no ruteada.

El análisis enlaza el archivo certificado, el informe y el Verilog exportado.
Resuelve `delivery_stage` al módulo correspondiente, extrae el `INIT` y los
pines de cada LUT del camino y comprueba sus conexiones consecutivas. La
única interrupción de conexión directa corresponde al CARRY8 del informe.
La fuente instalada de LUT4/5/6 confirma que I0 es el bit menos significativo
del índice de `INIT`. Se enumeraron todos los valores **binarios** restantes
para cada pin atravesado; esto no demuestra equivalencia X/Z del RTL.

## Dos cadenas identificadas

| Cadena | Extremos dentro de `delivery_stage` | Función respecto al pin del camino |
|---|---|---|
| 11 LUT6 | `fault_latched[10]_i_2682` → `_i_2430` | `salida = entrada AND g(otros pines)` |
| 9 LUT5/6 | `fault_latched[10]_i_1715` → `_i_3` | `salida = entrada OR g(otros pines)` |

Las once LUT6 comparten `INIT=64'hA8A8A8080808A808` y el pin I0.
Los otros datos corresponden a los bits 2–12 de `query_delivered[1]`,
`context_delivered` y `query_delivered[0]`; los dos selectores son comunes.
La cadena sigue acumulando el AND de esos bits seleccionados. Sus arcos
lógicos suman 0,418 ns; los nueve arcos OR suman 0,342 ns. Estas sumas
excluyen interconexión y no son un ahorro predicho.

De las treinta LUT, 23 tienen un valor del pin del camino que fija la
salida independientemente de los otros pines. Veinticinco tienen influencia
positiva, tres negativa y dos dependen de la combinación de los otros pines.
El camino completo, por tanto, no es una simple reducción OR.

## Consecuencia para el siguiente experimento

La reducción de fallos por lane propuesta en las
[notas de validación tardía](PCIE_SYMBOLIC_MONITOR_PREDECODE_LATE_TIMING_NOTES_R1.md)
tiene un motivo concreto para medirse. Aunque redujera el tramo OR,
permanecerían la selección aritmética anterior y el CARRY8. No se asigna
cada LUT a una sentencia RTL solo por su nombre.

Una alternativa posterior es estudiar el cálculo anticipado de
`address[1:0] + delivered` en B para cada contexto, transportándolo por
valor hasta R. El máximo binario es 8.194: requiere al menos 14 bits.
Antes de implementarlo deben revisarse anchos de expresión, semántica X/Z,
valores inválidos, selección SOP/EOP y el timing A→B. No se ha implementado
ni aprobado esta alternativa, y no debe añadir latencia ni reducir capacidad.

Se mantiene como siguiente decisión la comparación física E/D en curso.
Toda modificación exige pruebas directas, regresión integrada, síntesis y
replay propios antes de sustituir el candidato actual.

## Reproducción y respaldo

Analizador: `build/cloud-setup/20261002-board/analyze_pcie_symbolic_monitor_predecode_critical_luts_r1.py`.
Resultado: `pcie-symbolic-monitor-predecode-critical-luts-r1-analysis.json`,
SHA-256 `9d8368b1b0014b014e1eaea8eccbc7fa6588b92f4ad43e29f4fa0a22d3095a92`.
El registro conserva cada INIT, conexión, cofactor y hash de las fuentes
de biblioteca. Forma parte del respaldo de trabajo predecode en D: y NFS.
