# Bitmap RQ: resultado del banco C real

## Resultado

La tanda 982 `20261011T174335Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-rq-bitmap-focus-r1`
aprobó el banco enfocado en **113,867 s**, con salida 0. Su revisión
independiente y respaldo terminaron correctamente antes de habilitar
la regresión completa del monitor.

Se compararon **42.910 filas**, antes y después de cada flanco: el registro
C de 3.574 bits, `control_pending` y el bitmap combinacional de 256 bits.
El banco empleó entradas públicas y el C padre sin modificar, sin
escrituras a estado interno ni exclusión de escenarios con fallos.

Cobertura: 8.192 escenarios SOP/tag con X/Z, 256 combinaciones de tipos,
64 handshakes y 8.850 filas del suplemento RQ, incluida capacidad y
reutilización RQ después de liberaciones RC reales. Los doce ciclos
restantes corresponden a los testigos dirigidos iniciales.

## Controles y revisión

Las cuatro fuentes defectuosas se compilaron por separado y fallaron
antes del flanco en los testigos esperados:

| Alteración | Fila | Divergencia del bitmap |
| --- | ---: | --- |
| Omitir lane 3 | 5 | Falta el tag 23 |
| Rotar el origen del tag | 5 | Tag 53 en lugar de 23 |
| Aceptar igualdad desconocida | 8 | Todos los tags en lugar de ninguno |
| Omitir admisión | 0 | Tag 0 activo durante idle |

El revisor verificó fuentes compiladas, recetas exactas de cambios,
archivo de resultados y snapshot; rechazó 16 corrupciones de evidencia.
Se contabilizaron las quince advertencias esperadas de compilación/XSIM.

## Evidencia y límites

Paquete de 2.489.559 bytes y 253 archivos, verificado en D: y NFS:
`c31590a9ea461b823077fa4d97e2c44bd92ca537b93fbf4ef1df2aa93bd5a779`.
Revisión `-independent-rq-bitmap-focus-review-r1.json`:
`9f3f18892b6a991458dc36a7e0e0d6fc473302ad3f025ad2cf589167a0f2ca48`.
C candidato:
`488ad613738fcc812e9f053092ba4ddbd613553ede73a4e2916d10def18ff907`.

El PASS habilita las **183.058 filas de regresión completa**; todavía no
califica la integración del monitor, su netlist, timing o hardware.
Síntesis y replay propios permanecen condicionados a sus pruebas previas.
