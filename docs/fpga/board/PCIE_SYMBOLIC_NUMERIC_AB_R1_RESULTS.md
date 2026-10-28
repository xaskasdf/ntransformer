# A/B R1: validación funcional completa

Revisión del 2026-10-09: **178.185 filas crudas positivas y 266.681 flancos
focalizados aprobados**, con el mismo RTL A/B en todas las tandas.
Se habilita medir su síntesis; todavía no hay temporización propia ni
calificación del monitor completo.

## Pruebas por origen

Los trece bancos iniciales aportaron 177.833 filas. Conservaron todas las
comparaciones R7/probe/modelo/S1/C/E/D, los 2.723 bits de A, los 2.892 de B,
consultas, contexto retenido, dieciséis contribuciones por lane y candidato
de bytes. Los bancos con cuatro ciclos de vaciado no cancelaron tráfico
original. Los resets de los bancos crudos conservaron su contabilidad.

El banco complementario de 352 filas cerró la cobertura de enable,
terminal y cold sin repetir los bancos anteriores. Su primer caso admite
un tag con 31 bytes en dirección `0x101` y recibe un completion de error
con `words=1`: la entrega correcta es cero; ignorar el enable produce tres.
El terminal retira 31 bytes. Los tres cambios se detectaron en los flancos
previstos, conservando también el oráculo Python en ese caso de error.

| Mutación real | Comparación | Fila |
| --- | --- | ---: |
| swap_fields | NUMERIC_A_MISMATCH | 8 |
| drop_unowned_context | NUMERIC_A_MISMATCH | 11 |
| omit_origin | NUMERIC_A_MISMATCH | 6 |
| wrong_fault | NUMERIC_A_MISMATCH | 5 |
| ignore_enable | NUMERIC_B_MISMATCH | 9 |
| omit_terminal | NUMERIC_B_MISMATCH | 9 |
| omit_cold | NUMERIC_COLD_MISMATCH | 13 |

## Cuatro estados

La prueba focalizada comparó ambos registros completos y su estabilidad
entre flancos: 262.144 combinaciones de conteo/alineación/remaining/enable,
3.672 inyecciones X/Z en descripciones, 792 en palabras, 64 combinaciones
de validez y dos casos dirigidos. Capturó 266.626 orígenes, retiró 266.622
y canceló cuatro en seis flancos de reset. Los siete módulos mutados
fallaron en los pasos 1, 2 o 3. Sus funciones de referencia se copiaron
del modelo y comparten `apply_effect`; complementan la comparación R7
independiente de los bancos crudos, sin reemplazarla.

## Diagnósticos conservados

El banco de contexto inicial no distinguió `ignore_enable`; dos controles
quedaron sin ejecutar. Se conservó ese resultado sin declararlo aprobado.
El primer lanzamiento focalizado falló antes de compilar por sintaxis de
`set_property`; la revisión R2 cambió únicamente a `-dict`. Ambas tandas
diagnósticas tienen respaldo y permanecen en la secuencia de evidencia.

## Respaldo y siguiente paso

- Revisión conjunta: `20261009T160848Z-V4-ffn-supervised-pcie-symbolic-numeric-ab-controls-r1-independent-symbolic-numeric-ab-controls-review.json`, SHA-256 `c6a1ecf3b6ae89a8fc3309544329145c05340c0bd900df83c79af59bb1d7a5d7`.
- Archivo de controles crudos: SHA-256 `bf71bca52695a6430d7e79cad8c84f2e05b2cc8ff6a3813e326e982b1c041ff2`.
- Archivo focalizado: SHA-256 `a6d90e432f27ed285e6ea9adeb1c8be36ca5cb5fe63fe637965d882a35d54edb`.
- Fuentes y cada archivo terminal se verificaron en D: y NFS.

Sigue el [plan de síntesis y replay](PCIE_SYMBOLIC_NUMERIC_AB_SYNTHESIS_PLAN_R1.md).
R, validación de cabeceras con temporales de 15 bits y certificado de vacío
siguen pendientes. Se conservan R12 y todas las dependencias de
[VIVADO_QUEUE.md](VIVADO_QUEUE.md).
