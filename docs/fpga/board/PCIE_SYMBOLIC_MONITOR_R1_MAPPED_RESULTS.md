# Monitor PCIe R1: netlist propia comprobada

La netlist de la [síntesis conjunta](PCIE_SYMBOLIC_MONITOR_R1_SYNTHESIS.md)
pasó **174436 filas binarias** en dieciséis bancos. Sus **447 bits de
salida** se compararon con una instancia RTL independiente, antes y
después de cada flanco, con cinco vaciados adicionales por banco.
Se conservan los oráculos originales R7/S1 y la historia de registros;
los chequeos internos de la referencia no se presentan como sondas
internas de la netlist.

## Cobertura y controles

| Banco | Filas | Muestras previas | Muestras posteriores |
| --- | ---: | ---: | ---: |
| `monitor` | 252 | 257 | 257 |
| `main` | 18538 | 18543 | 18543 |
| `ownership` | 46356 | 46361 | 46361 |
| `rq` | 8850 | 8855 | 8855 |
| `prefix` | 544 | 549 | 549 |
| `lanes` | 632 | 637 | 637 |
| `context` | 997 | 1002 | 1002 |
| `drained_main` | 18858 | 18863 | 18863 |
| `drained_ownership` | 65711 | 65716 | 65716 |
| `drained_rq` | 10045 | 10050 | 10050 |
| `drained_prefix` | 839 | 844 | 844 |
| `drained_lanes` | 932 | 937 | 937 |
| `drained_context` | 1062 | 1067 | 1067 |
| `directed` | 191 | 196 | 196 |
| `controls` | 352 | 357 | 357 |
| `range_return` | 277 | 282 | 282 |

Son 174516 muestras por fase. Los seis bancos con vaciado cancelan cero
filas de tráfico original en C/E/D, A, B y R. La selección excluye solo
escenarios completos que contienen X/Z en entradas: 8622 filas respecto
a las 183058 de la [regresión RTL](PCIE_SYMBOLIC_MONITOR_R1_FULL_RESULTS.md).
Mantiene tráfico malformado binario y el banco de retorno de rango.

Diez cambios binarios entre flancos comprobaron las condiciones de
inventario vacío, con otras diez muestras después de restaurar las
entradas. Se esperaron 200 ns de GSR y 1 ns después del flanco para
muestrear la netlist funcional; no se anotó SDF.

| Alteración real de entrada del candidato | Primera fila detectada | Fase |
| --- | ---: | ---: |
| `omit_rq` | 4 | 0 |
| `omit_rc` | 10 | 1 |
| `omit_cold` | 13 | 1 |
| `wrong_sequence` | 7 | 1 |
| `wrong_requester` | 12 | 1 |
| `omit_profile` | 12 | 1 |

Las fases 0 y 1 corresponden a antes y después del flanco. Las seis
alteraciones produjeron una discrepancia en puertos públicos, sin
modificar las expectativas. El revisor rechazó 34 corrupciones de
evidencia y comprobó archivo, fuentes, netlist, puertos, scripts y
diecinueve fuentes SV compiladas por caso.

## Alcance y próximo paso

Esto acredita equivalencia funcional en los bancos ejecutados. La
estimación WNS **−3,584 ns** a 4 ns sigue abierta; no hay medidas ruteadas
nuevas. Seguir el [plan físico](PCIE_SYMBOLIC_MONITOR_PHYSICAL_PLAN_R1.md):
optimizar el mismo DCP con un BUFGCE real, revisar y reproducir esa
netlist, y solo después colocar y rutear. Conservar caudal, contexto por
valor, cancelación exclusivamente por frío y publicación alineada.
PCIe nativo, CDC, reset, recepción posted, DDR de 512 GiB y placa siguen
requiriendo sus validaciones.

- Tanda: `20261011T050428Z-V4-ffn-supervised-pcie-symbolic-monitor-binary-r1`.
- Archivo D:/NFS: SHA-256 `d1e4923715396132292aeec1f90ccd3afec918e60acabb4e72e5128c573a3509`.
- Fuentes: SHA-256 `e9c9320f7a16ec8bbb8dd1da369c900f9f881867dcc4fdeaa9baae18750e9fa4`.
- Netlist propia: SHA-256 `7d7abb8bca39a296b3d589667344614e2bb63c9ef84ebae9b1d30ef701bbbc57`.
- Revisión: `20261011T050428Z-V4-ffn-supervised-pcie-symbolic-monitor-binary-r1-independent-symbolic-monitor-binary-review.json`, SHA-256 `b797d1a29d97cd36f7ab9df3a70c67856af33e7499d1d0f6584e3f1f7bc6cd00`.
