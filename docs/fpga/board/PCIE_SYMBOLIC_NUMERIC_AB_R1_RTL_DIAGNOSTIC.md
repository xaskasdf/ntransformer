# A/B R1: bancos positivos y cobertura pendiente

Tanda `20261009T154102Z-V4-ffn-supervised-pcie-symbolic-numeric-ab-r1-rtl`, revisada el 2026-10-09. **177.833 filas positivas aprobadas**
con los módulos numéricos completos detrás de C/E/D R3. Son pruebas
funcionales; un control negativo no fue detectado y dos no se ejecutaron.
Todavía no hay síntesis ni replay propios de A/B.

## Cobertura y contabilidad

| Banco | Filas | Capturados | Retirados A | Retirados B | Cancelados A/B |
| --- | ---: | ---: | ---: | ---: | ---: |
| main | 22,339 | 22,203 | 22,002 | 21,935 | 201/268 |
| ownership | 46,356 | 38,612 | 26,999 | 23,128 | 11,613/15,484 |
| rq | 8,850 | 8,370 | 7,653 | 7,414 | 717/956 |
| prefix | 544 | 424 | 247 | 188 | 177/236 |
| lanes | 984 | 800 | 527 | 436 | 273/364 |
| context | 1,047 | 1,011 | 960 | 943 | 51/68 |
| directed | 217 | 185 | 140 | 126 | 45/59 |
| drained_main | 22,607 | 22,471 | 22,270 | 22,203 | 201/268 |
| drained_ownership | 61,840 | 54,096 | 42,483 | 38,612 | 11,613/15,484 |
| drained_rq | 9,806 | 9,326 | 8,609 | 8,370 | 717/956 |
| drained_prefix | 780 | 660 | 483 | 424 | 177/236 |
| drained_lanes | 1,348 | 1,164 | 891 | 800 | 273/364 |
| drained_context | 1,115 | 1,079 | 1,028 | 1,011 | 51/68 |

Se compararon los 2.723 bits de A y los 2.892 de B, alineados por origen.
Se conservaron las comprobaciones independientes de R7/probe/modelo/S1,
las tablas y el contexto C/E/D. Las consultas, contexto numérico, dieciséis
contribuciones por lane y candidato de bytes también coincidieron.

Los seis bancos con cuatro ciclos de vaciado conservaron todas las filas
anteriores y no cancelaron tráfico original en A ni B. Los bancos crudos
conservaron sus resets, X/Z, wrap, casos malformados y filas posteriores al
fallo. La revisión calculó captura/retiro/cancelación desde los vectores,
cotejó cada fuente compilada y rechazó nueve corrupciones de evidencia.

## Módulos mutados

| Mutación | Comparación que detectó el cambio | Fila |
| --- | --- | ---: |
| swap_fields | NUMERIC_A_MISMATCH | 8 |
| drop_unowned_context | NUMERIC_A_MISMATCH | 11 |
| omit_origin | NUMERIC_A_MISMATCH | 6 |
| wrong_fault | NUMERIC_A_MISMATCH | 5 |

La variante `ignore_enable` completó el banco de contexto sin discrepancia.
Vivado terminó con código 1; `omit_terminal` y `omit_cold` no llegaron a
ejecutarse. Se conserva el resultado y se preparará un banco crudo
dirigido que diferencie estos cambios, además de la prueba focalizada.

Cada variante es un módulo real modificado y compilado en su propio caso.
No se aceptó un error de compilación o watchdog como detección funcional.

## Evidencia y dependencias

- Archivo D:/NFS: SHA-256 `1826f499ade405ec93984f9bf8bd8f4e5a931684000335b73f93ddff15a6b649`.
- Fuentes: SHA-256 `6f6621e4742b06382ef0e14cb1ee38a57e3dea67bd5ac6c67cc28692fc3aea27`.
- Revisión: `20261009T154102Z-V4-ffn-supervised-pcie-symbolic-numeric-ab-r1-rtl-independent-symbolic-numeric-ab-diagnostic-review.json`, SHA-256 `05c7ffc907a51aeac4304997b1280d8e1911196aad659124bafad77bb040a92a`.

Sigue la [prueba focalizada](PCIE_SYMBOLIC_NUMERIC_AB_FOCUS_R1_PLAN.md),
y la tanda cruda dirigida de los tres controles pendientes. Solo tras
aprobar ambas se habilitan síntesis a 4 ns, auditoría y replay propios.
El cálculo de bytes aquí pertenece al banco: aún falta el estado de retiro R,
validación tardía y certificado de vacío del monitor completo. Se conservan
R12, las [notas de anchos](PCIE_SYMBOLIC_NUMERIC_RETIREMENT_NOTES_R1.md) y
todas las dependencias de [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
