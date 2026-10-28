# Predecode R1: netlist propia comprobada

## Resultado

La tanda `20261011T082444Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-binary-r1`
terminó con código 0 en 1.519,539 segundos. La revisión independiente aprobó
**174.436 filas binarias** de dieciséis bancos contra la referencia RTL original,
sin modificar sus expectativas. Comparó **447 bits públicos** antes y después
del flanco: 174.516 muestras por fase, incluidos cinco vaciados por banco.

Se usó la netlist generada por la
[síntesis del candidato](PCIE_SYMBOLIC_MONITOR_PREDECODE_SYNTHESIS_R1.md).
Los chequeos internos pertenecen a las referencias RTL; no se presentan
como inspección interna de todos los registros de la netlist.

## Cobertura

Los seis bancos drenados cancelaron cero filas de tráfico original en
C/E/D, A, B y R. Los contadores generales incluyen registros sin tráfico
y no equivalen a ese conteo por origen. Se conservaron tráfico malformado
binario, fronteras, propiedad de tags y retorno de rango.

Los bancos y filas coinciden con la
[prueba binaria de referencia](PCIE_SYMBOLIC_MONITOR_R1_MAPPED_RESULTS.md).
Se excluyeron únicamente escenarios completos con X/Z en entradas: 8.622
filas respecto a la [regresión RTL completa](PCIE_SYMBOLIC_MONITOR_PREDECODE_FULL_R1_RESULTS.md).
Se comprobaron diez sondas binarias de quiet y diez restauraciones.

| Mutante de entrada | Primera fila detectada | Fase |
|---|---:|---:|
| `omit_rq` | 4 | 0 |
| `omit_rc` | 10 | 1 |
| `omit_cold` | 13 | 1 |
| `wrong_sequence` | 7 | 1 |
| `wrong_requester` | 12 | 1 |
| `omit_profile` | 12 | 1 |

Fase 0: antes del flanco; fase 1: después. El revisor rechazó 34 corrupciones
de evidencia y comprobó diecinueve fuentes SV más la netlist propia por caso.
Hubo 24 avisos conocidos: declaración adelantada de `rq_held`, duplicación
idéntica de `glbl` y entorno XSIM; ninguna advertencia crítica.

## Respaldo y siguiente dependencia

Archivo D:/NFS, 94.079.279 bytes y 305 miembros verificados:
`2ac9c3e4638e87e3757acb8d717a9f975dc68ec393e0605c6305db5886d3e0ac`.
Netlist propia:
`97033ba40e2b7110bc8fb136fb5f828b7f67a34faaeaf0890200c3fde3404232`.
Revisión:
`9ac43cf3a3c1767e15e7f1df7d4a35012884e98a33889322ced5e0256c9de162`.

La simulación esperó 200 ns de GSR y muestreó 1 ns después del flanco;
no usó SDF. El WNS sintetizado sigue en −3,584 ns. La
[cola física](PCIE_SYMBOLIC_MONITOR_PREDECODE_PHYSICAL_QUEUE_R1.md) continúa
con optimización BUFGCE, replay de esa netlist y después colocación/ruteo.
No se califican timing físico, PCIe nativo, CDC, DDR ni placa.
