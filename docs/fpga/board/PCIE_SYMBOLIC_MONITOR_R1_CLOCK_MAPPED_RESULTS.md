# Monitor PCIe R1: netlist optimizada con BUFGCE comprobada

La netlist propia de la [optimización con reloj global](PCIE_SYMBOLIC_MONITOR_R1_CLOCK_RESULTS.md)
pasó las **174436 filas binarias** de dieciséis bancos. Se compararon sus
**447 bits de salida** con RTL independiente antes y después de cada
flanco: 174516 muestras por fase, incluidos cinco ciclos de vaciado por
banco. La referencia conserva los oráculos R7/S1 y el historial interno.

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

Los seis bancos con vaciado cancelan cero filas de tráfico original.
Diez cambios binarios entre flancos y diez restauraciones comprobaron
quiet. Las seis mutaciones reales de entrada produjeron discrepancias
en puertos públicos; el revisor rechazó 34 corrupciones de evidencia.
El banco, vectores y diecinueve fuentes SV son idénticos a los de la
[netlist sintetizada](PCIE_SYMBOLIC_MONITOR_R1_MAPPED_RESULTS.md).
La regresión RTL separada conserva las 183058 filas con pruebas X/Z.

## Biblioteca y alcance

Se usó `unisims_ver` con la netlist exportada sin parametrizar ni SDF.
La fuente instalada de BUFGCE mantuvo su hash antes y después; su
`SIM_DEVICE` predeterminado es `ULTRASCALE`. Esto registra la fuente,
sin afirmar identidad interna de la biblioteca precompilada.
Se verificaron archivos, fuentes compiladas, cuerpos `glbl`, parámetros
del simulador y hashes de la netlist por caso.

La equivalencia funcional habilita preparar colocación y ruteo del
mismo DCP. La estimación WNS **−3,723 ns** a 4 ns sigue sin cerrar;
esta simulación no aporta retardos físicos. Seguir el
[plan físico](PCIE_SYMBOLIC_MONITOR_PHYSICAL_PLAN_R1.md), conservando
caudal, contexto por valor, cancelación por frío y publicación alineada.
PCIe nativo, CDC, reset, posted, DDR de 512 GiB y placa siguen abiertos.

- Tanda: `20261011T055058Z-V4-ffn-supervised-pcie-symbolic-monitor-clock-binary-r1`.
- Archivo D:/NFS: SHA-256 `2d7d6fc4c883cca9c31b0e05af2ee2ab43464a46dd1daac8f1680510e5289f4b`.
- Fuente: SHA-256 `504c1ae2d6d421ca6c7b39ba8d305da0816a08021668bfd883420e3afbab2fc9`.
- Netlist propia: SHA-256 `9520e678925e2f8026568c9c6949084d022dce2e6b5884e8e42a451136b1ec98`.
- Revisión: `20261011T055058Z-V4-ffn-supervised-pcie-symbolic-monitor-clock-binary-r1-independent-symbolic-monitor-clock-binary-review.json`, SHA-256 `22427b46211d945b34076c0fd4f07e4f3f2f9f6bf27a6674538904a3705c14c2`.
