# Timing por etapas: prefijo y bandera

## Resultado de la lectura

La tanda 972, `20261011T151417Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-stage-timing-r1`,
terminó con salida **0** en **382,751 s** y quedó respaldada en D: y NFS.
Se reabrieron los DCP sintetizados 966 y 970 sin optimizarlos ni cambiar
restricciones. Cada variante tiene 27.534 FF, clasificados sin solapamientos
en C, E, D, A, B, R y TOP.

Se consultaron 49 pares dirigidos, cada uno para setup y hold: **98 estados
por variante**. Nueve pares tienen rutas; los otros cuarenta quedan marcados
explícitamente sin ruta. Los informes no vacíos contienen **712 caminos**
entre ambas variantes, hasta veinte por subconjunto.

| Par | Peor setup 966 | Peor setup 970 |
| --- | ---: | ---: |
| C→C | −2,407 ns | −2,407 ns |
| C→E | −1,184 ns | −1,184 ns |
| C→TOP | +3,242 ns | +3,242 ns |
| E→D | +0,742 ns | +0,742 ns |
| D→D | +0,717 ns | +0,717 ns |
| D→A | +1,504 ns | +1,504 ns |
| A→B | +2,283 ns | +2,283 ns |
| B→R | −2,546 ns | −2,768 ns |
| R→R | +2,732 ns | +2,732 ns |

Solo cambia el peor slack de B→R. Los holds de estos subconjuntos son
positivos; su mínimo es +0,066 ns. Esto no cubre las rutas de E/S del
reporte global ni demuestra timing colocado o ruteado. Se conservaron
dos avisos `Timing 38-242` por falta de `HD.CLK_SRC` en el contexto OOC.

## Cuellos de botella y decisión

C→C parte de `allocated_tags_reg[51]` y termina en
`registered_ops_reg[commit_slot][422]`: 29 niveles y 6,389 ns de datos.
C→E va de `out_record_reg[rc_data][32]` al reset síncrono de
`out_e_reg[updates][0][summary][10]`: 19 niveles y 5,067 ns.
Ambos fallos permanecen aunque se resuelva B→R.

El setup positivo estimado E→D no revoca el fallo físico anterior de esa
frontera. La siguiente [comparación de optimización](PCIE_SYMBOLIC_MONITOR_PREDECODE_PAIRED_OPT_PLAN_R1.md)
incluye B→R, E→D, C→C y C→E. No se selecciona otro ruteo a partir de estos
valores sin colocación.

## Corrección del lector y evidencia

El lector R1 permitía que un espacio de su expresión regular cruzara una
línea vacía de primitivas. R2 conserva esa línea vacía en rutas de cero
niveles y comprueba que los conteos de primitivas sumen los niveles.
Corrigió **110 descripciones**, con todos los demás campos y diferencias
numéricas idénticos. Pasaron los ocho controles anteriores y dos nuevos.
La revisión R1 y el paquete original permanecen intactos; no se repitió Vivado.

- Archivo: 48.946.711 bytes, 236 archivos; SHA-256
  `154269ca3b920ca8c51f7c9c1f46fea119a98d2ecd5572f48ba41f6c78d2939f`.
- Revisión R2: SHA-256
  `6a060a53837086bf841703e61c707e168a333aab9248d6c3b1cceee44449a407`.
- Revisión original: SHA-256
  `9ed9f8980c4683578f4d694ea53f679840bb621f67b867d55899080eebd6bc9b`.

Registros y revisores: `build/cloud-setup/20261002-board/`.
