# C/E/D R1: comprobación del circuito sintetizado

## Dependencias

Esta prueba queda detrás de la síntesis congelada en
`pcie-symbolic-ced-r1-synth-frozen.json`. Antes de ejecutarla se exige:

1. Terminación y respaldo verificado de la tanda, incluidas sus fuentes.
2. Revisión de recursos, advertencias, reloj de 4 ns, cobertura de restricciones
   y ausencia de excepciones de temporización para C, E, D y el conjunto.
3. Auditoría del EDIF interno de los cuatro DCP, cotejo de primitivas y
   rechazo de controles de corrupción y ciclos combinacionales.

La simulación RTL previa conserva su alcance y evidencia en
[los resultados C/E/D](PCIE_SYMBOLIC_CED_R1_RESULTS.md).

## Adaptador y referencias

Se utilizarán los bytes exactos de `combined/pcie_symbolic_ced_combined_funcsim.v`.
El adaptador se obtiene de sus declaraciones reales: 75 puertos de la
estructura de 2.107 bits y 392 bits adicionales, **2.499 bits externos** en total.
Los módulos de referencia reciben únicamente nombres distintos para evitar
colisiones; las transformaciones y hashes quedan registrados.

Se conserva el banco RTL con sus verificaciones del control capturado,
origen, tablas de 256 entradas, contexto retenido, R7 independiente y
cancelación por reset frío. La comparación del netlist añade 200 ns de
arranque GSR y muestreo a 1 ns después del flanco. Los dos ciclos finales
de vaciado son fijos y también se comparan. No se aplica SDF.

## Cobertura y criterios

El manifiesto `fpga/tb/data/pcie_symbolic_ced_binary_r1.json` define 16 casos
positivos y **205.493 filas**, incluida una prueba inicial de 64 filas.
Contiene versiones binarias de los bancos originales, sus variantes con
vaciado previo al reset y los casos dirigidos. Solo se excluyen escenarios
completos con entradas X/Z; las pruebas RTL de cuatro estados permanecen
obligatorias. El corpus binario principal anterior y su variante con vaciado
se conservan como pruebas adicionales con procedencia propia.

Se exige igualdad de todas las salidas en cada muestra, incluido
`pipeline_pending`, y conservación de las comprobaciones internas del RTL.
Omitir RQ, RC o un reset, o alterar la secuencia, debe provocar el error de
comparación esperado. Tráfico binario malformado y posterior a un fallo
permanece en los bancos.

## Límites

El plan no constituye un resultado de simulación. Un éxito funcional tampoco
cierra temporización física, aplicación numérica, validación tardía, retiro
de bytes ni integración del monitor completo. R12 sigue siendo la referencia
de ese monitor hasta completar sus criterios de promoción.
