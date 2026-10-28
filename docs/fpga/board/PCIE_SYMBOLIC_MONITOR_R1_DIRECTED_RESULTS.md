# Monitor PCIe completo: filtro dirigido R1

Resultado: **1098 filas positivas en cuatro bancos**, con comparación
independiente antes y después de cada flanco y cinco flancos finales de
vaciado. El contenedor real C3/E2/D2/A1/B1/R1 conserva el RTL original.

## Cobertura

| Banco | Filas | Muestras previas | Muestras posteriores |
| --- | ---: | ---: | ---: |
| `monitor` | 252 | 257 | 257 |
| `directed` | 217 | 222 | 222 |
| `controls` | 352 | 357 | 357 |
| `range_return` | 277 | 282 | 282 |

El caso dirigido acredita 31 cambios entre flancos y su restauración:
frío, perfil, reset, link, validez, ready e identidad, incluidos X/Z.
Las entradas de payload desconocidas se ignoran cuando no hay tráfico.
Los testigos exclusivos C/E/D/A/B fueron
1/2/2/2/6; último retiro 4,
fallo tardío aislado 21, temprano aislado 9,
armado demorado 10 y frío ocupado 6.

## Controles reales

Se rechazaron las 16 modificaciones del contenedor, en la comparación y
fila previstas a partir de la secuencia. No son alteraciones del resultado
del simulador. Las 19 fuentes realmente compiladas coinciden con el
manifiesto; se conservan todas las comparaciones R7/modelo/S1/C/E/D/A/B/R.

| Control | Fila | Comparación |
| --- | ---: | --- |
| `omit_c` | 35 | `PENDING` |
| `omit_e` | 35 | `PENDING` |
| `omit_d` | 36 | `PENDING` |
| `omit_a` | 37 | `PENDING` |
| `omit_b` | 38 | `PENDING` |
| `extra_retired` | 39 | `PENDING` |
| `omit_early_fault` | 132 | `EMPTY` |
| `omit_late_fault` | 108 | `EMPTY` |
| `omit_observing` | 10 | `EMPTY` |
| `omit_cold` | 16 | `PROBE` |
| `omit_requester` | 16 | `PROBE` |
| `omit_attributes` | 16 | `PROBE` |
| `omit_rq_ready` | 16 | `PROBE` |
| `omit_rc_ready` | 16 | `PROBE` |
| `omit_raw_rq` | 16 | `PROBE` |
| `omit_raw_rc` | 16 | `PROBE` |

El revisor rechazó 23 corrupciones de resúmenes, muestras y probes.
La contabilidad incluye las cancelaciones por frío; el filtro corto no
sustituye los bancos originales con vaciado.

## Diagnósticos conservados

Tres intentos anteriores están respaldados: corrección de `set_property
-dict` antes de compilar; observación del pendiente C/E/D separada del
pendiente completo; y cambio de la inyección de requester del bit 48 al
51 para aislar el fallo tardío. La tercera tanda comparó las 252 filas,
pero rechazó la ausencia del testigo tardío. No se alteró el RTL ni se
debilitaron condiciones de aceptación. Los revisores conservan las
transformaciones literales y hashes de cada corrección.

## Evidencia y siguiente trabajo

- Tanda: `20261009T192549Z-V4-ffn-supervised-pcie-symbolic-monitor-directed-r4`.
- Archivo D:/NFS: SHA-256 `874b4b28538a42f109da2c24c3732cbd583e2b0b20c9cea51c468e4114b6b68c`.
- Fuentes: SHA-256 `dec90257ce1ae64e426b13b6cc4bd8c4be91295c2271a6f87d7f24643faf2cd8`.
- Revisión: `20261009T192549Z-V4-ffn-supervised-pcie-symbolic-monitor-directed-r4-independent-symbolic-monitor-directed-review.json`, SHA-256 `17ac82c09b5f193ce82bb97eaa6ea8e81abedf32b982958d3604a619df708cd5`.

Ejecutar ahora las **181960 filas restantes**. Reutilizar estas 1098 solo
si coinciden las 19 fuentes compiladas. Después aplicar el
[plan de síntesis conjunta](PCIE_SYMBOLIC_MONITOR_SYNTHESIS_PLAN_R1.md).
No hay aprobación del cierre físico, de colas internas PCIe ni de recepción
posted. Siguen abiertos los WNS aislados negativos y las pruebas de placa.
