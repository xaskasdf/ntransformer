# Relojes físicos internos de la composición HBM–AXI Bridge

## Contrato del experimento

Esta tanda parte del checkpoint de [HBM_CLOCKED.md](HBM_CLOCKED.md), SHA-256
`4070592049040d4afc8ae4fe70c0625757bf2b7df3c7e72124079221c8e765cf`.
Solo escribe en una copia privada. El RTL, los IP y las validaciones anteriores
permanecen intactos. La tarjeta DDR4 de 512 GiB conserva su diseño separado.

La copia incorpora cuatro primitivas BUFGCE, con habilitación constante:

| Puerto externo | Dominio | Período | Buffer interno |
| --- | --- | --- | --- |
| `clk` | PCIe | 4 ns | `clock_probe_pcie_clk` |
| `reader_clk` | Lector | 5 ns | `clock_probe_reader_clk` |
| `mgmt_clk` | Gestión | 10 ns | `clock_probe_mgmt_clk` |
| `mem_clk` | Memoria | 2,222 ns | `clock_probe_mem_clk` |

`insert_hbm_clocktree.tcl` exige que cada conjunto de cargas antes/después sea
idéntico y que los únicos cinco primitivos añadidos sean los cuatro buffers
y su constante VCC. Usa las operaciones de netlist de
[UG835](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/connect_net) y la
primitiva [BUFGCE](https://docs.amd.com/r/en-US/ug974-vivado-ultrascale-libraries/BUFGCE).

Es una **prueba física interna con fuentes externas ideales**: no selecciona
osciladores, MMCM, pads ni pines. No se debe conectar este buffer PCIe de prueba
como un segundo buffer detrás de `axi_aclk_out`. La composición nativa deberá
usar el árbol que entregue el IP y repetir la implementación. No se emplean
ubicaciones ficticias `HD.CLK_SRC` ni desactivaciones de ruteo dedicado.

## Restricciones y auditoría

Se aplican 31 límites específicos de demora: ocho grupos Gray, cuatro grupos
de captura de RAM y 19 cruces de control, incluidos los vigilantes de memoria
y PCIe y el supervisor del lector. Cubren 1.501 destinos; el presupuesto es
4 ns, salvo la entrada al vigilante de memoria, limitada a 2,222 ns. Los ocho
buses Gray del transporte tienen skew máximo de 2 ns.

Se conservan las restricciones originales de los conversores AMD, incluidos
sus ocho límites de skew. Se añaden únicamente 12 cortes sobre entradas
asíncronas CLR/PRE del guardián ya sincronizadas. El auditor verifica su fuente
exacta, admitiendo exclusivamente la inversión de polaridad LUT1 necesaria
para el control del primitivo. No hay cortes generales entre relojes.

La auditoría comprueba 144 registros privados ASYNC_REG, 88 conexiones entre
etapas, 1.458 capturas de RAM sin consumidores que eviten el registro de
captura, 17 conexiones de reset canónico y el origen de los cuatro árboles.
Los informes CDC, skew, excepciones, timing, DRC y metodología se conservan
sin waivers. Los CDC internos protegidos de AMD conservan su alcance propio.

## Resultado medido: 11 de octubre de 2026

La implementación `route-20261011T111736Z` terminó en 907,28 s y produjo
`routed.dcp`, SHA-256
`d3340acba9cdf133298976465810f2c9e11a40202833496b03a5cbcbd102ebbb`.
La revisión independiente `reopen-20261011T113336Z` tardó 81,74 s; reabrió
ese archivo sin modificarlo ni cambiar sus restricciones.

| Dominio | Setup mínimo | Hold mínimo | Destinos con fallo de setup |
| --- | ---: | ---: | ---: |
| PCIe, 250 MHz | −0,172 ns | +0,015 ns | 1.281 |
| Lector, 200 MHz | +0,759 ns | +0,011 ns | 0 |
| Gestión, 100 MHz | +8,726 ns | +0,025 ns | 0 |
| Memoria, período 2,222 ns | +0,015 ns | +0,019 ns | 0 |

Los cuatro árboles internos quedaron ruteados: 24.939/24.939 redes que
requieren ruteo, cero errores y cero gaps. Los puertos de reloj externos
siguen siendo fronteras ideales OOC; ese resultado no certifica pads ni
fuentes de placa. No hay fallos de hold o ancho de pulso. **El setup global
no cierra**; el margen de memoria es de solo 15 ps.

Las 1.501 rutas CDC privadas tienen margen mínimo +1,758 ns. Los 16 límites
de skew pasan; el máximo Gray privado es 0,620 ns frente a 2 ns permitidos.
Las 88 conexiones entre etapas tienen mínimos +1,765 ns de setup y
+0,050 ns de hold. Se preservan las 1.458 capturas y los 17 consumidores
de reset canónico. El informe CDC contiene 27 CDC-3, 8 CDC-6, 12 CDC-9 y
1.460 CDC-15, sin hallazgos críticos ni waivers.

Vivado añadió un BUFGCE de distribución de reset, alimentado por
`endpoint/guard/cold_a_reg[2]/Q`: CE constante VCC, `CE_TYPE=ASYNC`, sin
inversión ni cargas de reloj. El auditor original, que exigía conexión
directa, rechazó esa topología después de guardar el checkpoint. Su
`RESULT.json` fallido se conserva; la revisión posterior verifica el buffer
y la fuente lógica exacta en lugar de cambiar ese resultado. También se
conservan dos intentos iniciales de preflight con fallos del auditor
(inversión LUT1 del reset y resolución de objetos Tcl).

Permanecen advertencias: 20 PDCN-1569 sobre LUT de los conversores AMD,
un RTSTAT-10 sobre redes sin cargas ruteables y seis LUTAR-1 en resets
internos de IP. El informe de metodología lista 1.000 TIMING-18; `check_timing`
enumera 1.390 entradas y 1.044 salidas sin presupuesto externo. No se han
aprobado ni ocultado esas advertencias.

## Siguiente ajuste de diseño

La peor ruta parte de `writer/src_req_bytes_reg[0]`, atraviesa la suma de
dirección final en `source_join` y termina en el CE de la cola de retorno:
19 niveles, cinco CARRY8 y 3,999 ns de demora de datos. Antes de otro ruteo,
probar una variante privada que simplifique la comprobación del límite de
16 GiB conservando longitud, alineación y prohibición de cruzar 4 KiB.

Una comparación aritmética local de 234.881.024 combinaciones coincide con
la fórmula actual para siete clases de página representativas. Es una
hipótesis de optimización: no sustituye equivalencia HDL, semántica de X ni
regresiones de protocolo. No se modificó el RTL calificado. La variante
necesitará simulación funcional, nueva síntesis y ruteo completo; no basta
con trasladarle los márgenes de esta tanda.

## Reproducción

```bash
python -B run_hbm_clocktree.py NUEVA_TANDA --mode preflight \
  --donor CHECKPOINT_SELECCIONADO --settings VIVADO_SETTINGS
python -B run_hbm_clocktree.py NUEVA_TANDA --mode route \
  --donor CHECKPOINT_SELECCIONADO --settings VIVADO_SETTINGS
python -B review_hbm_clocktree.py TANDA_RESTAURADA
python -B check_hbm_clocktree.py TANDA_RESTAURADA NUEVO_DIRECTORIO_NEGATIVAS
python -B review_hbm_clocktree_bundle.py RESPALDO_RESTAURADO
```

`--mode reopen --donor CHECKPOINT_RUTEADO --donor-sha SHA256` realiza la
auditoría posterior sin aplicar restricciones ni editar el netlist. Los
runners usan dos CPU, prioridad reducida y límites de 1.200 s para ruteo o
600 s para revisión, dentro de `ntrpu-side/axi-bridge-hbm-clocktree/`.

## Criterios para avanzar

El cumplimiento interno requiere ruteo completo de los árboles, ninguna ruta
interna incompleta, cobertura íntegra de restricciones, skew y sincronizadores
sin violaciones, y setup/hold/pulse width sin fallos. Un resultado estructural
correcto o una salida de Vivado con código cero no bastan para afirmar cierre.

Persisten como pasos independientes la conexión de ambos IP nativos, las
restricciones de frontera, las fuentes de reloj de placa, tráfico conjunto,
supervisión de referencia HBM/APB y la procedencia física del ACK. Los márgenes
de esta prueba no se transfieren automáticamente a esa integración.

La evidencia y los intentos no seleccionados se guardan en
`D:/AI/artifacts/ntrpu-axi-bridge-hbm-clocktree/`; la selección y los recibos de
respaldo registran el checkpoint exacto y su revisión restaurada.
El paquete `20261011-hbm-clocktree.zip` incluye las cinco tandas, sus archivos
transferidos, scripts, revisiones, procedencia de la composición y controles
negativos: 11 del preflight y 22 de la revisión física. Las cinco tandas
propias terminaron; no quedan procesos de esta prueba en la VM.
