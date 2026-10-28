# Semántica pendiente antes de integrar el inventario

Revisión del 2026-10-08 basada en el RTL actual y la evidencia de
`20261008T032247Z-V4-ffn-supervised-pcie-rc-event-endcache-rtl`.
Complementa el [contrato de etapas](PCIE_INVENTORY_PIPELINE_CONTRACT.md).

## Qué prueban los bancos de la etapa RC

La etapa RC R7 conserva las salidas de sus versiones anteriores y pasa
su replay independiente. El inventario monolítico R7 sigue suministrando
pertenencia y metadatos esperados en ese banco. Sus estados públicos se
mantienen iguales porque la nueva etapa lo observa, todavía no lo sustituye.

Los campos `shadow.full_protocol_equivalence=false` y
`shadow.active_pipeline_integrated=false` de la revisión son obligatorios.
No interpretar `all_r7_public_rows_identical=true` como prueba de que los
candidatos RC puedan escribir directamente el estado del inventario.

## Diferencias observadas que deben conservarse como pruebas

Las filas siguientes son filas de entrada, antes de la latencia de la etapa.

| Fila / escenario | Entrada o resultado | Obligación de integración |
| --- | --- | --- |
| 4131 / 11 | La cabecera declara 5 bytes; el estado esperado contiene 4. El candidato descuenta 5 y la referencia 4. | Conservar el fallo y derivar el retiro del contexto esperado, no del byte count recibido. |
| 4140 / 12 | La cabecera declara dirección 257; el estado esperado contiene 256. | La dirección y su offset afectan el cálculo de bytes y las máscaras. Detectar el desacuerdo no convierte la dirección recibida en estado esperado. |
| 4291 / 29 y 4300 / 30 | Los candidatos marcan obligación sin resolver; la referencia produce una finalización que retira 4 bytes. | Definir y comprobar la transición de respuestas no normales, incluidos los contadores después del fallo. |

Hay tres diferencias de eventos, dos de metadatos y dos filas con
obligaciones sin resolver. Los 22.155 ciclos sanos del shadow no presentan
diferencias de eventos; los 184 ciclos con fallo también deben revisarse.

## Consecuencias para el circuito integrado

En `ntrpu_pcie_request_inventory_r7.sv`, `coriginal` y `caddress` proceden
del banco de tags y del forwarding previo del mismo beat. El cálculo de
`delivered` usa esos valores. En `ntrpu_pcie_rc_event_stage_r7.sv`, los
candidatos aritméticos proceden del descriptor recibido; el cotejo con
`expected_metadata` detecta diferencias, pero no corrige sus deltas.

El planificador debe producir sus propios metadatos desde RQ/RC capturados,
conservar el contexto de cada paquete y comparar el estado retirado completo
por secuencia. No basta conectar `event_remaining` y `event_removed`
directamente al banco o descartar todas las filas con fallo. Si se usa una
ruta aritmética rápida basada en la cabecera, demostrar su equivalencia con
el estado esperado y resolver también su alternativa en tráfico inválido,
sin burbujas del observador.

Medir nuevamente el camino de consulta, forwarding y transición integrado:
el WNS positivo de la etapa RC aislada no cubre esa dependencia. Mantener
256 tags, RQ2/RC4, 257 lecturas y 1.052.672 bytes, además de reutilización
de tags, reset frío con etapas ocupadas y wrap explícito.
