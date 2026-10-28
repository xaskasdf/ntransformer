# Conteo temprano C: banco aislado aprobado, tanda 997

## Cambio y resultado

`20261011T213308Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-admission-count-focus-r1`
terminó en **117,582 s**, salida 0. La revisión independiente aprobó
**51.110 filas de entrada**, con 51.110 comparaciones antes y otras tantas
después del flanco. Se compara el registro C de 3.574 bits, el bitmap
de 256 tags, pending y un comparador adicional de 13 bits para conteos.

Solo cambia la asignación de `allocate_count` del control RQ calificado:
el candidato suma los cuatro bits de admisión temprana en lugar de los
dos indicadores de actualización posteriores. Se conservan el resto del
control, sus entradas públicas y la referencia original. El banco no
fuerza el estado interno.

La cobertura incluye 8.192 casos de control, 256 de tipo, 64 de handshake,
1.024 extensiones de tipo y 1.024 de handshake, además de las 8.850 filas
del banco RQ y las demás combinaciones dirigidas del fixture. El manifiesto
y las fuentes conservan el detalle completo de las 51.110 filas.

## Controles negativos

| Variante defectuosa | Primer fallo, fila/fase | Observado | Referencia |
| --- | --- | --- | --- |
| `drop_lane_three` | 5 / 0 | `0000` | `0801` |
| `collapse_two` | 14 / 0 | `0801` | `1002` |
| `count_blocked` | 5 / 1 | `0802` | `0001` |
| `add_completions` | 19 / 0 | `0002` | `0000` |

Valores hexadecimales del comparador de 13 bits. Las seis fuentes se
verificaron en cada uno de los cinco casos compilados. El revisor rechazó
las 21 alteraciones de evidencia previstas.

Este PASS habilita la regresión completa de 183.058 filas; no califica
todavía su monitor integrado, netlist, timing ni rendimiento. La síntesis
y el replay propios permanecen condicionados a esa regresión.

## Evidencia

Archivo de 2.521.765 bytes y 253 miembros, verificado en D: y NFS a las
21:35:51 UTC: `5b28f062c8031ff983fb8c847066cc3f0e95f33750950915490f8a970b1b1057`.
Revisión: `2943642c80ca3eb600176b11b155de3a5113098a74b6acd6f56f65bed225bbe7`.
Fuentes `source-20261011T213154Z.zip`:
`5540fa1f25757b5348eeeebdb526cbb83759daa6e13c1445edfa23c03b68c495`.
Configuración: `1808497d642056770eca79c862b874110a5033aa8f8510725aa4fc25cbc941db`.

Certificado en `artifacts/ntrpu-board-vivado/`, carpeta de la tanda;
revisión y coordinador R5 en `build/cloud-setup/20261002-board/`.
