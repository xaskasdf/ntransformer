# Replay propio de límites B con winner — R1

## Entrada y dependencia

El replay espera la síntesis propia revisada, comparada y respaldada.
Extrae la netlist funcional de ese archivo terminal y verifica su SHA-256,
tamaño, módulo superior y puertos: 1.384 bits de entrada y 447 de salida.
Los módulos de referencia tienen nombres separados para impedir que
Vivado los sustituya por el candidato.

La referencia del banco y su contabilidad proceden del replay RQ 985,
ya calificado. Se reutilizan el banco y los vectores, pero se ejecutan
todas las filas de nuevo contra la netlist B-bound/winner. No se usan
resultados de otra variante como evidencia de equivalencia.

## Cobertura

Se ejecutan 174.436 filas binarias en 16 bancos, con los 447 bits públicos
comparados antes y después del flanco. Se conservan diez perturbaciones
entre flancos y su restauración, seis controles negativos de entrada y
34 alteraciones de evidencia. Los seis bancos drenados deben conservar
el tráfico original sin cancelarlo artificialmente.

El banco mantiene los tiempos de estabilización de la referencia
calificada. No incluye SDF ni inspección jerárquica de los 75 registros
internos: esa alineación se verifica en la integración RTL y su preflight.
Aquí se comprueba el comportamiento público de la netlist resultante.

## Evidencia y límites

Configuración, fuentes, referencias, netlist y revisores se fijan por hash.
El coordinador espera su dependencia, respalda antes de lanzar y recoge
todo resultado terminal en D: y NFS. Un fallo detiene la cadena para
inspección y no provoca una repetición automática.

Un PASS acredita equivalencia funcional binaria dentro de estos bancos.
Las pruebas RTL X/Z, el cierre físico, PCIe nativo y la placa completa
conservan sus criterios independientes.
