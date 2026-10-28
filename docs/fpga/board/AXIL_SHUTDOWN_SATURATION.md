# Saturación del aislamiento AXI-Lite

## Resultado medido

Tanda `20261005T175658Z-V4-ffn-supervised-shutdown-saturation`: aprobada
en dos perfiles, con reloj host de 7 ns y dispositivo de 5/37 ns.
Vivado 2026.1 ejecutó ambos bancos en 72,214 s. Se conserva la
[configuración AMD del aislamiento](AXIL_SHUTDOWN_MANAGER.md): Shutdown
Manager AXI4-Lite de 32 bits, rechazo SLVERR y convertidor asíncrono real.
El productor y el dispositivo siguen siendo modelos de prueba.

Cada escenario ofrece 128 solicitudes por canal activo. Tras 256 ciclos
host sin nuevas aceptaciones, registra la meseta manteniendo VALID y su
carga hasta la aceptación. Ambos perfiles producen estos resultados:

| Condición de bloqueo | AW aceptadas | W aceptadas | AR aceptadas |
| --- | ---: | ---: | ---: |
| AW sin W | 33 | 0 | 0 |
| W sin AW | 0 | 18 | 0 |
| BREADY bajo | 51 | 51 | 0 |
| RREADY bajo | 0 | 0 | 51 |
| BREADY y RREADY bajos, tráfico simultáneo | 51 | 51 | 51 |

Son límites observados para esta topología, configuración y estímulo.
No equivalen a la profundidad de una única FIFO ni garantizan el límite de
otras versiones o configuraciones. [PG377, cambio de modo](https://docs.amd.com/api/khub/documents/3PNXW7YeYMVucr4rgUV6BA/content)
describe la retención de la decisión de cierre al recibir la transacción;
el banco comprueba esa clasificación con cada AW/AR aceptada.

## Conservación y reapertura

Por perfil se comparan **517 escrituras y 261 lecturas** en U, incluida
una escritura/lectura nueva después de reabrir cada escenario. Llegan a
H y D exactamente 140 escrituras y 107 lecturas; las 377 escrituras y
154 lecturas restantes terminan con el rechazo del manager.

El revisor independiente reconstruye cantidades, orden, direcciones,
datos, máscaras y respuestas de los tres puntos. Distingue el rechazo de
los errores devueltos por el dispositivo y compara también sus datos R.
Verifica los cinco readbacks con máscara después de reabrir. La fixture
no modela DDR ni su política de efectos laterales ante errores.

El inventario U usa ocho bits. Su pico de 51 coincide con la reconstrucción
de la traza; no hay desbordamiento ni respuestas sin obligación previa.
Cada tanda se vacía sin reset operativo, incluidas las mitades AW/W
incompletas. Se comprueba estabilidad de solicitudes U/H y respuestas U
bajo bloqueo, inventario final cero y ausencia de repetición al reabrir.
El reset inicial solo configura el banco.

## Consecuencia para el diseño

La [frontera retenida posterior](AXIL_RETAINED_FENCE.md) ya implementa y
prueba el inventario de ocho bits, un límite de 32 y la detención de su
productor hacia el manager. Los requisitos siguientes motivaron esa tanda;
la detención del endpoint externo y sus colas sigue pendiente.

El siguiente wrapper debe usar inventario U retenido de **ocho bits** y
mantener el fallo por desbordamiento como impedimento del certificado.
Cuatro bits no alcanzan; seis representarían las 51 operaciones observadas,
pero esa medida no establece un límite universal. No se fija un crédito
de admisión a partir de la profundidad nominal de las FIFO.

Todavía falta implementar y probar un límite explícito de admisión y el
contrato real de detención del productor, con finalización de las mitades
pendientes. `in_shutdown` y el certificado H/D continúan siendo insuficientes
para el vaciado completo de XDMA. La siguiente dependencia conserva la
integración de AXI4, colas protegidas y [coordinador de reset](ENDPOINT_RESET_COORDINATION.md).
No se autoriza reutilización de sesiones NTRD ni se cierra V4/V7 integrado.

## Evidencia y reproducción

- Banco: `fpga/tb/tb_ntrpu_axil_shutdown_saturation.sv`.
- Ejecución: `tools/ntrpu_board_axil_shutdown_saturation.tcl`, mediante
  `vivado -mode batch -source SCRIPT -tclargs NUEVA_CARPETA`.
- Revisor: `tools/check_ntrpu_axil_shutdown_saturation.py`; auditoría final
  `build/cloud-setup/20261002-board/ffn-supervised-audit-shutdown-saturation-r2.py`.
- Respaldo local/NFS: 121 archivos, 17.997.070 bytes; SHA-256
  `e532a25842d01405aa878b4c5f5fc46d63127d5bcdb95ac24aa0d5841f0238c6`.
- Snapshot: `source-20261005T175646Z.zip`, SHA-256
  `ffc350ed7ed7fb3a2d1fc6a1c9d448aec294a68f92120889b790cae1a03b70dc`.
- Adenda verificada: SHA-256
  `6051de354b113857b89408085fd0b3695dd000e0dc9cb980e69e2009e20f7921`.

Se verifican las fuentes compiladas, dos wrappers generados, parámetros de
IP y 86 archivos de bibliotecas instaladas. El revisor detecta 74 alteraciones
de evidencia. Se revisan seis avisos: dos de entorno, dos del parámetro
aplicado también a `glbl` y dos del límite de visualización del array de
memoria. El array sí se simula; no hay avisos en los logs de ejecución.

El recolector original esperaba cuatro avisos y conservó un diagnóstico
pendiente. La adenda revisa los dos avisos adicionales sin repetir Vivado
ni modificar la tanda. También conserva el primer auditor, que asumía
incorrectamente el mismo directorio de bibliotecas para IP, primitivas y
`xil_defaultlib`. No se ejecutó síntesis ni implementación física en esta tanda.
