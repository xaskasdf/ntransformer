# A/B R1: replay de la netlist propia aprobado

Resultado del 2026-10-09: **169.601 filas binarias y 169.657 flancos
comparados**, incluyendo cuatro flancos de vaciado por banco. Se usaron
los bytes exactos de la netlist exportada por la [síntesis A/B](PCIE_SYMBOLIC_NUMERIC_AB_R1_SYNTHESIS.md).
La prueba es funcional sin SDF; no certifica temporización física.

## Comparación por origen

El adaptador conecta los 89 puertos de entrada y 109 de salida reales.
Los 2.892 bits B alimentan las comparaciones numéricas originales contra
R7, el probe y el modelo: consultas, contexto, contribuciones EOP y candidato
de bytes. Una copia RTL con nombres distintos comprueba además todo B.
A interno se observa en la referencia RTL; no se presenta como un puerto
independiente de la netlist conjunta.

Se conservan las comprobaciones de control C/E/D, tabla de 256 tags,
orígenes y filas posteriores al fallo. GSR explícito de 200 ns, muestreo
1 ns después del flanco y retiro/cancelación contados independientemente.

| Banco | Filas | B retirados | B cancelados por cold |
| --- | ---: | ---: | ---: |
| raw_main | 18,538 | 18,152 | 256 |
| raw_ownership | 46,356 | 23,128 | 15,484 |
| raw_rq | 8,850 | 7,414 | 956 |
| raw_prefix | 544 | 188 | 236 |
| raw_lanes | 632 | 276 | 236 |
| raw_context | 997 | 917 | 52 |
| drained_main | 18,794 | 18,408 | 256 |
| drained_ownership | 61,840 | 38,612 | 15,484 |
| drained_rq | 9,806 | 8,370 | 956 |
| drained_prefix | 780 | 424 | 236 |
| drained_lanes | 872 | 516 | 236 |
| drained_context | 1,049 | 969 | 52 |
| directed | 191 | 112 | 51 |
| controls | 352 | 177 | 115 |

Ningún banco con vaciado cancela tráfico original. El corpus selecciona
escenarios completos sin X/Z de los catorce bancos crudos aprobados;
no elimina filas por error o fallo. Las 178.185 filas RTL de cuatro estados
y los 266.681 pasos focalizados siguen siendo requisitos independientes.

## Controles y procedencia

Cinco alteraciones de entradas/reset del candidato produjeron
`NUMERIC_MAPPED_PORT_MISMATCH`:

| Alteración | Fila |
| --- | ---: |
| omit_control | 6 |
| omit_record | 6 |
| omit_cold | 13 |
| wrong_origin | 6 |
| ignore_enable | 6 |

Estos controles detectan cambios de la frontera y del registro transportado;
los siete mutantes aritméticos/funcionales pertenecen a las pruebas RTL
previas. Dieciocho corrupciones de evidencia fueron rechazadas. Las
dieciséis fuentes SV y la netlist propias coinciden en las compilaciones
de los diecinueve casos; los cuerpos duplicados de `glbl` son idénticos.

- Tanda: `20261009T163601Z-V4-ffn-supervised-pcie-symbolic-numeric-ab-binary-r1`.
- Archivo D:/NFS: SHA-256 `17659d10af31e0f3a9528c9228d1d35564edf183ffe0e1facb9e840cfc1c490f`.
- Netlist: SHA-256 `28d6361eefd358437863b072e4c6706314e52dbcc9bdd3ce0876cc8b1885d84a`.
- Revisión: `20261009T163601Z-V4-ffn-supervised-pcie-symbolic-numeric-ab-binary-r1-independent-symbolic-numeric-ab-binary-review.json`, SHA-256 `d12d039a6bb56ee9d2541748cc13845c3a378cb63b1ef7008a05e2d950d5dc81`.

## Próxima dependencia

Aplicar el [diseño de validación tardía](PCIE_SYMBOLIC_LATE_VALIDATION_R1_DESIGN.md):
conservar los temporales de 15 bits antes del truncamiento, integrar retiro
de 23 bits y probar fallos y estado público del mismo origen. A/B conserva
WNS OOC +1,730 ns; C/E/D R3 sigue en −2,546 ns. Monitor completo, quiet,
CDC/reset, DDR de 512 GiB, pesos reales y placa permanecen abiertos.
