# R1: netlist funcional propia comparada con RTL y R7

Resultado del 2026-10-09: **174184 filas binarias en quince bancos**,
con **463 bits públicos** comparados con RTL y los estados R7 por origen.
Se usó la netlist archivada de la síntesis R1, sin modificarla. Los 111
puertos de entrada y 18 de salida proceden de sus declaraciones reales.

## Cobertura

| Banco | Filas | Retiradas | Canceladas por frío |
| --- | ---: | ---: | ---: |
| `main` | 18538 | 18088 | 320 |
| `ownership` | 46356 | 19257 | 19355 |
| `rq` | 8850 | 7177 | 1193 |
| `prefix` | 544 | 129 | 295 |
| `lanes` | 632 | 217 | 295 |
| `context` | 997 | 904 | 65 |
| `drained_main` | 18858 | 18408 | 320 |
| `drained_ownership` | 65711 | 38612 | 19355 |
| `drained_rq` | 10045 | 8370 | 1195 |
| `drained_prefix` | 839 | 424 | 295 |
| `drained_lanes` | 932 | 517 | 295 |
| `drained_context` | 1062 | 969 | 65 |
| `directed` | 191 | 102 | 61 |
| `controls` | 352 | 150 | 142 |
| `range_return` | 277 | 268 | 5 |

La selección excluye escenarios completos con entradas X/Z; mantiene
tráfico malformado y fallos binarios. La regresión RTL de 182806 filas
conserva la cobertura de cuatro estados. Los seis bancos con vaciado
cancelan cero registros de tráfico original.

Cada caso estabiliza GSR durante 200 ns, muestrea 1 ns después del flanco
y añade cinco flancos de vaciado. Conserva las comprobaciones originales
R7/modelo/S1/C/E/D/A/B/validación tardía/retiro y todas sus tablas.
Las tres guardas internas se observan en el RTL de referencia; la netlist
se comprueba por sus 463 bits públicos y sus efectos de rango y fallos.

El caso dirigido repite pico 1056760, retorno 1048572, persistencia del
bit 13 con fallo temprano 13 a cero y borrado por frío. Sus testigos se
validan individualmente por origen y fila.

## Controles y procedencia

Seis alteraciones reales de entradas/reset del candidato fueron detectadas:

| Control | Banco | Fila de detección |
| --- | --- | ---: |
| `omit_control` | `controls` | 7 |
| `omit_record` | `controls` | 7 |
| `omit_cold` | `controls` | 13 |
| `wrong_origin` | `controls` | 7 |
| `drop_range_step` | `range_return` | 264 |
| `drop_terminal` | `range_return` | 265 |

La revisión rechazó diecisiete corrupciones de contabilidad, cobertura y
testigos. Verificó fuentes realmente compiladas, semilla, vectores,
scripts y logs dentro del respaldo. Las definiciones duplicadas de
`glbl` tienen cuerpos idénticos; las advertencias de XSIM y `rq_held`
quedan registradas. No hay una prueba de temporización SDF.

- Tanda: `20261009T184518Z-V4-ffn-supervised-pcie-symbolic-retirement-binary-r1`.
- Semilla: SHA-256 `d09195c02f2b491a7caeb8249060c1f27df670a9df3c7236501a59ca7eb9862a`.
- Archivo D:/NFS: SHA-256 `e4e511c256dae6dd0dbeaa1ab11d521dfc1b3ad12d5c65f54c97b1ea334627a5`.
- Revisión: `20261009T184518Z-V4-ffn-supervised-pcie-symbolic-retirement-binary-r1-independent-symbolic-retirement-binary-review.json`, SHA-256 `3f90012981c815fdb99eb9d1e8da7bc04fd9b454a60985bcbe7168f3e19b1ba0`.
- Fuentes: SHA-256 `bf1b67a3ff5737688a17b8f38ee0ae24fe8f90c0d5ceec3349a7457882f745a5`.

## Siguiente trabajo

El [prototipo del monitor completo](../../../fpga/rtl/ntrpu_pcie_symbolic_monitor_r1.sv)
está escrito, pero todavía no se ha compilado ni validado. Ejecutar el
[plan de quiet e integración](PCIE_SYMBOLIC_MONITOR_TEST_PLAN_R1.md),
usando el camino real C/E/D/A/B/R y un oráculo independiente antes y
después del flanco. Conservar la referencia funcional al optimizar.

El WNS OOC de R continúa en **−3,096 ns**, y C/E/D en **−2,546 ns**.
Quiet, cierre físico, monitor completo y las validaciones nativas y de
placa siguen pendientes.
