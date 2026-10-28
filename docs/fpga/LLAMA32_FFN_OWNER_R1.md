# Propiedad pública de FFN y protección del residual retenido

Estado 2026-10-15 16:50 UTC: **protocolo aprobado en nativo y Vivado R4; respaldo verificado**.
RTL vigente: `fpga/rtl/ntrpu_llama32_ffn_owner_r3.sv`. Banco:
`fpga/tb/tb_ntrpu_llama32_ffn_owner_r8.sv`. El hijo del banco es un peer
controlado; estas pruebas no ejecutan la aritmética ni el transporte AXI FFN.

## Contrato de integración

El adaptador conserva secuencia y época públicas de 64 bits, junto con
los ocho descriptores de la FFN existente. Exige una fuente válida con
identidad completa coincidente y fija layout 2, D=2048, H=8192 y tile K=32.
El padre deberá obtener buffer 0 del ABI1 confirmado de post-O y mantener
su respuesta retenida hasta terminar la segunda suma residual.

El hijo recibe una etiqueta privada de 64 bits dividida en secuencia y
época de 32 bits. Cada reserva avanza cuatro posiciones alineadas para
las cuatro fases FFN. Se comprueban el cruce de 32 bits, el último grupo
válido y el rechazo por agotamiento o valor inicial desalineado. Un reset
requiere reiniciar también hijo y peer AXI: no autoriza reutilizar etiquetas
frente a respuestas anteriores todavía en vuelo.

Antes de lanzar el hijo, el adaptador verifica alineación de 64 bytes,
límites mediante aritmética de 65 bits y ausencia de solapamiento entre
los destinos 4–7 y **toda la región** del residual retenido. Sus spans son
65.536, 65.536, 16.384 y 8.192 bytes; el residual requiere 8.192 bytes.
El secuenciador FFN conserva la responsabilidad de validar sus ocho regiones.

La respuesta del hijo permanece retenida hasta el ACK público coincidente.
Cancelación, cambio de época completa y respuesta ajena revocan el resultado;
un evento tardío también lo revoca de forma persistente. Los descriptores de
resultado permanecen a cero cuando no son válidos. El éxito exige cuatro
etapas, fase 3, 8.192 bytes finales, 155.648 bytes acumulados y estados cero.

## Evidencia reproducible

- Nativo R8: `20261015T151956Z-V7-llama32-ffn-owner-native-r8`.
  161 trabajos: tres semillas de 52 casos más agotamiento y desalineación.
  Incluye errores, desbordamientos, cancelación, identidad ajena, respuesta
  retenida y reset coordinado en cuatro fases. Revisor: 119 controles rechazados.
- Mutantes R2: `20261015T152043Z-V7-llama32-ffn-owner-mutants-r2`.
  26 variantes RTL compiladas; todas fallan con una aserción explícita,
  sin aceptar errores de compilación o timeout como detección.
- CPU2/3, archivos y miembros verificados tanto en D: como en NFS.

| Evidencia | SHA-256 |
| --- | --- |
| Archivo nativo R8 | `b3b8e9e57cb77ddd886f633accc5413afb42ff6db6b5be4ba068c49722fc691b` |
| Revisión nativa | `df08d67448a466ca248143cbf2b5fb175f8b71d8585bd8259aabac24e731b9ad` |
| Archivo mutantes R2 | `9c9a14613524892b6c91e9764a918a52551794545a6dd8db8de1cf17eef435c9` |
| Revisión mutantes | `637079ee7e0565c8ae71673c848b57fe11d66601cabeb62b803c4d0784c034ab` |
| Fuente sellada Vivado | `2972d06927f1ef3be965adff85834a19be803b3ae75b9044ae02bc28859557e7` |

Los intentos anteriores se conservan: declaraciones ANSI no aceptadas por
Icarus, propagación desconocida desde arrays no empaquetados y dos huecos
del banco revelados por mutantes. R3 usa registros empaquetados con los
mismos puertos; R8 comprueba la época errónea sólo en admisión y la
retención del rechazo con el hijo disponible. Ningún fallo se reclasificó
como aprobación ni se sobrescribieron sus fuentes.

## Cola y trabajo restante

Vivado R4 terminó y la revisión independiente aprobó los cinco bancos:
161 trabajos y 119 controles de trazas, además de siete controles de
identidad del diseño. Síntesis OOC a 5 ns: 1.339 LUT, 2.084 FF, cero
BRAM/DSP, WNS +1,545 ns y WHS +0,068 ns. No hay cajas negras, latches,
lazos ni endpoints internos sin restricción. Es una estimación de síntesis,
no cierre físico; el hijo sigue siendo un peer de protocolo.

- Archivo `20261015T164528Z`: `0d3d15401dc4668b5db3ea95d87278b7a6e9eab04064c9211c9157e42cc13195`.
- Revisión: `706c431880dc170ff2e3a3126a625f02c4831cc2a45142674f71d2911e96de4e`.

La FFN completa inició automáticamente `20261015T164923Z` después
de comprobar esa revisión y el respaldo.

`queue_llama32_ffn_owner_vivado_r4.py` verificó revisión y respaldo de
`llama32-attention-projected-root-physopt-review-r3-coordinator.json` y
abrió la tanda `20261015T164528Z`. Ejecutó síntesis OOC a 5 ns y los cinco bancos
XSim, con dos CPU y límite de 1.800 s. Revisó identidad de diseño,
fuentes realmente compiladas, ausencia de cajas negras/latches/lazos,
resultados y controles antes de respaldar y liberar el slot.

R1 se detuvo antes de lanzar herramientas por el observador de respaldo
de atención/post-O. El sucesor R2 conserva RTL, banco y revisores; fuente
sellada R2: `73feb75f943838f74b139b962271998f78a5ac404c2959b825f4f6dc41e23410`.
R2 tampoco lanzó herramientas por el preflight físico precedente. R3
conserva sus fuentes y espera la corrección de ese preflight; fuente R3:
`88b070bc5c06089a02853db82aa1ad96eb2c43c95260ce6cb25177e048321fa3`.
R3 tampoco lanzó herramientas al fallar la revisión del encabezado
físico. R4 verifica el suplemento corregido y conserva RTL/bancos/revisores;
fuente vigente: `e4ece62637ff382f11010f040fe5af7217263dd986ca7f6592a1fbb8cfd6498f`.

La [conexión con el núcleo FFN real](LLAMA32_OWNED_FFN_R1.md) ya está
preparada y encolada detrás de esta prueba, con ABI1 exacto y escrituras
confirmadas por B. Las [referencias numéricas](LLAMA32_POST_FFN_R1.md)
existen; falta aprobar esa unión, conectar post-O vivo, transportar la
segunda suma y calificar inferencia completa.
