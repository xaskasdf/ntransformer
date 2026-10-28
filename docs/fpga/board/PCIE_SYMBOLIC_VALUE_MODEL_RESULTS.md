# Modelo simbólico por valor: resultados

Revisado el 2026-10-09. La tabla y el contexto retenido pueden representarse
con descripciones propias de 51 bits en los bancos ejecutados. **Todavía
no es un pipeline nuevo ni tiene medidas de síntesis**. R12 sigue siendo
la referencia de implementación; el objetivo global continúa abierto.

## Qué cambió

El modelo en `fpga/tb/ntrpu_pcie_inventory_symbolic_model_r1.sv` parte de
R15 y sustituye remaining/dirección almacenados por semilla y resumen de
consumos. TC/atributos permanecen separados. Captura count y condición de
entrega con la cabecera, de modo que EOP/discontinue no recalculen el
delivered retenido. Las vistas numéricas no alimentan sus semillas.

La transformación exacta y sus hashes se conservan en
`pcie-symbolic-value-model-source-preparation.json`. La referencia R7 no
alimenta al modelo. La latencia pública no cambia en esta prueba.

## Evidencia del banco heredado

Vivado/XSIM 2026.1, **356,645 s**. El banco principal pasó 22.339 filas y
68 escenarios, con 5.718.784 comparaciones de palabras de tabla y todos
los estados posteriores a fallo. Pasaron también los bancos de propiedad,
ancestría, RQ, prefijos, máscaras X/Z e integración/suma por lane.

Conserva las capacidades comprobadas: 256 tags, 257 lecturas y 1.052.672
bytes. Se rechazaron los siete mutantes heredados y la inversión real del
orden de composición en el nuevo modelo. El banco no autoriza sintetizar
esta implementación de comportamiento como si ya estuviera segmentada.

## Vida del contexto y observaciones directas

La segunda tanda duró **88,319 s**: 1.047 ciclos, 18 escenarios y 1.048
observaciones contando el flush. Compara consultas de 180 bits, candidatos
de contadores/deltas de 141 bits, contexto RC de 71 bits y RQ de 43 bits.
Además compara 268.032 palabras de tabla y todo el estado público retirado.

Incluye pausas de 257 ciclos, terminal con tag reutilizado y atributos
nuevos, nonterminal retenido, discontinue posterior, reemplazo de contexto
incompleto, cadenas de count cero, suma saturada entre beats, cuatro filas
con X/Z activos y reset frío durante payload pendiente. La copia R7 con
instrumentación exclusivamente observacional se coteja con el R7 original
en estado y 256 filas por observación; ambos conservan su propio estado.

| Mutación real del modelo | Detección |
| --- | --- |
| Recargar la fila actual del tag en EOP | Delta incorrecto, fila 7 |
| Recalcular condición de entrega desde propiedad actual | Contexto retenido incorrecto, fila 7 |
| Sobrescribir TC/atributos durante commit RC | Tabla incorrecta, fila 289 |

Nueve controles rechazan evidencia alterada de consultas, deltas, contextos,
orden, integridad y comparación de referencia. Fuentes y proyectos
compilados se verifican por variante; los mutantes no se simulan mediante
un simple cambio de etiquetas en el log.

## Respaldo y siguiente implementación

| Banco | Tanda | SHA-256 del archivo |
| --- | --- | --- |
| pcie_symbolic_value_inherited_rtl | `20261009T110950Z-V4-ffn-supervised-pcie-inventory-join-symbolic-value-model-rtl` | `6726762b83dab64197a942a1199cf8e00464253d0faba8bb614691938da48c72` |
| pcie_symbolic_value_raw_context | `20261009T112033Z-V4-ffn-supervised-pcie-inventory-join-symbolic-value-context-rtl` | `8094580756780a27a538e8132014fc357f15a970777d00bd1ff25f519dd7658a` |

Ambas tandas tienen fuentes y todos los miembros verificados en D:/NFS,
bajo `artifacts/ntrpu-board-vivado/<tanda>/VERIFIED.json`. R107 enlaza sus
revisiones y el siguiente contrato. Los dos intentos iniciales de preparar
el probe se detuvieron por guardas de reemplazo textual; no ejecutaron
Vivado ni modificaron el modelo probado. Los helpers y diagnósticos se
conservaron y R3 completó la preparación.

Para reservar espacio de trabajo se comprimieron cuatro cachés de evidencia
R15 ya certificadas: **220.901.376 bytes físicos**, sin alterar contenidos,
identidades, fechas ni enlaces. No se eliminó ningún archivo.

Sigue [S1/S2/S3 con registros reales](PCIE_SYMBOLIC_PIPELINE_R1_CONTRACT.md):
primero medir consulta/commit simbólico; después aplicación y retiro
alineados. La regla por valor evita tickets retenidos hacia almacenamiento
reutilizable, pero todavía debe verificarse en el pipeline implementado.
CDC/reset, DDR de 512 GiB, pesos reales y placa conservan sus gates en
[VIVADO_QUEUE.md](VIVADO_QUEUE.md).
