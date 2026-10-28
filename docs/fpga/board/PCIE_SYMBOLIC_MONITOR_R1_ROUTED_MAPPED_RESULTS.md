# Monitor PCIe R1: netlist ruteada comprobada

La netlist exportada del DCP físicamente colocado y ruteado pasó sus
**174436 filas binarias** en dieciséis bancos. Se compararon **447 bits
públicos** con RTL independiente antes y después de cada flanco:
174516 muestras por fase, incluidos cinco ciclos de vaciado por banco.
Se mantuvieron los oráculos R7/S1, la historia de operaciones y los
diecinueve archivos SV usados en las pruebas anteriores.

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

## Controles

Los seis bancos drenados cancelan cero filas de tráfico original.
Diez cambios entre flancos y sus diez restauraciones comprobaron quiet.
Los seis mutantes reales de entrada fueron detectados en los puertos
públicos; el revisor rechazó 34 corrupciones de evidencia. La regresión
RTL separada conserva las 183058 filas con X/Z.

Se verificaron hashes de fuente, netlist, archivos compilados, cada caso,
cuerpos `glbl` y biblioteca BUFGCE instalada antes y después. `unisims_ver`
usó la netlist funcional sin SDF. La identidad interna de su biblioteca
precompilada sigue sin estar demostrada por el hash de la fuente instalada.

## Alcance y siguiente diseño

Este resultado comprueba la equivalencia funcional binaria observada de
la netlist ruteada, incluidas diecinueve réplicas de registros añadidas
por colocación. **No cierra temporización**: las medidas físicas conservan
WNS −11,835 ns, WHS global −3,904 ns y hold interno −0,104 ns a 4 ns.
Ver [ruteo, reapertura y límites OOC](PCIE_SYMBOLIC_MONITOR_R1_PHYSICAL_RESULTS.md).

El siguiente experimento anticipará la selección de filas en E y medirá
su efecto sobre la distribución en D. Debe conservar la latencia de cinco
clocks C→R, dos RQ/cuatro RC por ciclo, 256 tags, RC antes de RQ,
contexto por valor y cancelación únicamente por frío. Exige comparación
RTL de todas las filas X/Z, controles de mutación, síntesis, netlist propia
y nueva medida física. No se afirma mejora antes de medirla.
PCIe nativo, CDC, reset, posted, DDR de 512 GiB, ECAD y placa siguen abiertos.

- Tanda: `20261011T070809Z-V4-ffn-supervised-pcie-symbolic-monitor-route-binary-r1`.
- Archivo D:/NFS: SHA-256 `379ba92acd12f838257e5efaaaca7de016b12c63f0445d0a26e1abdbefd4b5b9`.
- Fuente: SHA-256 `d52990ab151f601701eb8cbe756cb2b3409177016ba3f576e07b89cd25394370`.
- Netlist ruteada propia: SHA-256 `1b0718cf041597fb3010b83f45e2fbf4dd8b7c39663491a9f92bd8d4c94bcfae`.
- Revisión: `20261011T070809Z-V4-ffn-supervised-pcie-symbolic-monitor-route-binary-r1-independent-symbolic-monitor-route-binary-review.json`, SHA-256 `4d3c866548114385615e5453bcf5aaa313238cd09a32eec2f18209f25e4dffe4`.
