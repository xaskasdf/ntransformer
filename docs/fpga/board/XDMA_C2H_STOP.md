# C2H nativo: respuesta retenida y recepción del host

## Alcance de la tanda

La tanda `20261006T050718Z-V4-ffn-supervised-xdma-c2h-stop` parte del
ejemplo inmutable `20261004T154440Z-V4-ffn-supervised-pcie-bypass-probe`.
Estado: **observación funcional revisada y respaldada**.
VU5P `xcvu5p-flva2104-2-i`, Vivado 2026.1, XDMA 4.2 y AXI de 512 bits;
se mantiene `comp_timeout=50ms`. Límite del banco: 200 us; proceso: 1800 s,
dos hilos. No modifica las IP protegidas ni introduce `force`.

## Secuencia y criterios

1. Descriptor H2C terminal en `0x100`: copiar 64 bytes de host `0x400`
   a BRAM de aplicación `0x1000`. Patrón de byte: `((i*13)^0xa5)&255`.
   Exigir AW/W/B, contador uno, canal inactivo y lectura de Run=0.
2. Descriptor C2H terminal en `0x200`: leer AXI `0x1000` y escribir host
   `0x600`. La memoria del bypass BAR es independiente y no se usa para
   inicializar la BRAM de DMA.
3. Retener la respuesta pública R mediante VALID/READY, conservando
   ID, datos, RRESP y RLAST. Exigir Busy=1 y contador cero; borrar Run con
   una escritura PCIe real a `0x100c` y repetir lecturas de estado.
4. Liberar R; verificar una escritura PCIe completa y 64 bytes recibidos
   por una memoria de host del banco. Repetir estado con Busy=0 y contador
   uno, observando también que H2C sigue detenido.

El receptor es pasivo: únicamente modifica memoria al aceptar payload
en CQ (`TVALID && TREADY`), según sus byte enables. Conserva intacto el
logger RX de AMD. El verificador Python reconstruye por separado los
descriptores, CC, AXI, escrituras CQ y cada actualización de memoria.
Una escritura posted C2H no se acredita contando completions de lecturas.

La semántica de Run se consulta en [PG195, control C2H](https://docs.amd.com/r/en-US/pg195-pcie-dma/C2H-Channel-Control-0x04);
la interpretación de CQ y sus byte enables, en
[PG213, interfaz CQ de 512 bits](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Completer-Request-Interface).

## Reproducción y revisión

Fuentes: `fpga/tb/ntrpu_xdma_c2h_stop.sv`,
`fpga/tb/ntrpu_xdma_c2h_completion.vh`,
`tools/patch_ntrpu_xdma_c2h_stop.py` y
`tools/ntrpu_board_xdma_c2h_stop.tcl`. Usar un directorio de salida nuevo:

```bash
vivado -mode batch -source tools/ntrpu_board_xdma_c2h_stop.tcl \
  -tclargs /ruta/al/ejemplo/inmutable /ruta/a/salida/nueva
```

`tools/check_ntrpu_xdma_c2h_stop.py` exige coincidencia entre VCD y logs.
El auditor local revisa fuentes compiladas, XCI sin cambios, avisos
heredados y rechazo de evidencia alterada antes de emitir un addendum.
El código de salida cero no basta para aprobar la tanda.

## Resultado y evidencia

Se compararon **64 bytes en cada dirección**. La ventana comprobada
con R retenida fue de **29.220000 µs (7305 ciclos)** impuestos por el
banco. Durante esa retención, Run=0 coexistió con Busy=1 y contador cero.
Después se comprobaron Busy=0, contador uno y todos los bytes en el receptor.

| Evento | Tiempo simulado |
| --- | --- |
| H2C B aceptada | 98.928000 µs |
| C2H AR aceptada | 107.688000 µs |
| Escritura Run=0 emitida | 111.776000 µs |
| Lectura Run=0 terminada | 114.224000 µs |
| C2H R aceptada | 136.920000 µs |
| Primer payload CQ | 137.168000 µs |
| 64 bytes recibidos en host | 137.172000 µs |
| Fin del banco | 165.996000 µs |

La recepción de los 64 bytes concluyó 0.252000 µs
después de aceptar R. Es una latencia de este modelo y fixture; no mide
rendimiento de una placa. Las 31 lecturas de registros
y 10 escrituras de control coinciden con sus TLP nativos.
La escritura C2H observada lleva TC=0/ATTR=0; las lecturas nativas de
descriptores y payload H2C llevan TC=0/ATTR=2. El revisor verifica ambas
políticas por separado.

El auditor rechazó **74 alteraciones de evidencia**.
Los 139 avisos heredados se cotejaron con las fuentes compiladas.
SHA-256 del archivo nativo: `7ba3c6e11713c4e1f2f163647523a117352b0a71f9e288b668c57ff441401105`.
SHA-256 del paquete de revisión: `bc1ec860c84b80b592478fa08941c24605f059a8c470d6b49ebc21162de2dffe`.
Ambos están verificados localmente y en NFS. La memoria host es un receptor
del banco, no una demostración de DMA hacia memoria de un sistema operativo.

## Respaldo y límites

Se usa archivo tar completo, verificado por SHA-256 y por cada miembro,
más el ZIP de fuentes, sin duplicar el árbol extraído. El estado mutable
se escribe mediante archivo temporal y reemplazo atómico.
La compresión transparente NTFS de respaldos anteriores conserva sus
bytes; sus manifiestos se verifican antes y después.

Esta prueba no califica vaciado global de XDMA, cancelación de cadenas,
fence del productor BAR, reset, coherencia con memoria del sistema
operativo, DDR, caudal físico ni una tarjeta de 512 GiB.

Siguiente tanda: [cierre del productor BAR](HOST_BAR_CLOSURE_PLAN.md).
