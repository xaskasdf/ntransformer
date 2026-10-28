# Activaciones desde memoria AXI y GEMV integrado

Estado al 2026-10-03 UTC: `ntrpu_activation_loader.sv` carga activaciones
desde un puerto AXI de 512 bits. `ntrpu_q8_gemv_memory.sv` lo conecta al
[GEMV dual](Q8_GEMV_PARALLEL.md), conservando su RTL y el MAC HLS validado.
La carga directa del banco queda reemplazada por lecturas de memoria.
El puerto se comparte por fases: primero activaciones, después pesos.

## ABI y control

Cada registro de activación ocupa 64 bytes, alineados a 64:

| Bytes | Contenido |
| --- | --- |
| 0–31 | 32 valores INT8, en orden de elemento |
| 32–39 | Escala IEEE754 binary64 little-endian |
| 40–63 | Reservados, obligatoriamente cero |

Se admiten 1–256 registros: hasta 8192 elementos y 16 KiB de transporte.
La escala debe ser no negativa y no superar el límite del MAC
`float_max/127`, bits binary64 `0x4780204070e1c387`. Se preservan cero
negativo y subnormales; NaN, infinito, negativos no nulos y exceso se
rechazan. `tools/ntrpu_activation_layout.py` empaqueta bytes inmutables;
no cuantiza ni asigna memoria del dispositivo.

El job integrado captura dimensiones, layout 2, secuencia, época, rango
de pesos y descriptor de activaciones al aceptar `job_valid/job_ready`.
Este último contiene dirección, base, longitud y época propia, que debe
coincidir con la del job. El banco modifica los puertos inmediatamente
después de aceptar el job para comprobar que se usan los registros.
Ambos buffers deben permanecer inmutables hasta completar la operación.

El cargador valida sumas de 65 bits, rango completo y alineación antes
del primer acceso. Emite un comando por fragmento de página de 4 KiB,
con una ráfaga pendiente como máximo. Conserva los 64 bits de dirección.
La caché recibe entradas privadas conforme llegan; **solo una completion
correcta habilita tiles y accesos a pesos**. Una entrada previa a un error
puede haberse escrito en caché, pero nunca autoriza cómputo o resultados.
El siguiente job invalida el mapa de entradas presentes.

## Errores, cancelación y reset

El primer error del cargador queda retenido. Una cancelación detiene
ráfagas futuras y drena la ya emitida; un dato con `valid` pendiente se
mantiene estable hasta `ready`. El wrapper espera ese drenaje antes de
cancelar el núcleo GEMV: cancelarlo antes cerraría su puerto de activación
y podría impedir terminar la carga fallida.

| Error del cargador | Estado standalone | Estado del job GEMV |
| --- | ---: | ---: |
| Dimensión/rango/alineación | 1 | 2, activación inválida |
| Época | 2 | 2 |
| AXI/RID/RLAST/conteo | 3 | 3 |
| Escala/reservados/formato | 4 | 2 |
| Cancelación | 7 | 7 |
| Watchdog | 8 | 8 |

El watchdog predeterminado es de 65.536 ciclos sin progreso. Marca el
fallo, pero no abandona una transacción AXI: un esclavo o consumidor
permanentemente detenido requiere recuperación coordinada. `rst_n`
debe resetear o aislar también al peer AXI. Todas las interfaces usan
el mismo reloj; este wrapper no introduce un cruce de dominios.

El núcleo mantiene sus errores de pesos/MAC y la publicación por job.
Una vez comprometidas las filas de salida, `cancel_ready` permanece bajo.
La salida de este bloque sigue siendo un stream de resultados. El nuevo
[wrapper de escritura](RESULT_MEMORY.md) lo conecta a memoria AXI con
destino privado y completion tras las respuestas B. El registro de buffers
del runtime y la conexión PCIe–DDR siguen pendientes.

## Evidencia reproducible

`20261003T040812Z-V7-activation-loader-unit`: **96/96 casos**, semillas
1/7/42. Comprueba bytes, épocas, límites, escala, errores tardíos, pausas,
cancelación, timeout y dos resets por semilla. La carga de 256 registros
desde offset 4032 usa cinco ráfagas y 462–480 ciclos. Diez mutaciones de
logs se rechazan. `20261003T041650Z-V7-activation-loader-synth` conserva
checkpoint e informes del cargador.

`20261003T042547Z-V7-gemv-memory-unit`: **60/60 casos**, 21 valores exactos
contra una referencia de producto entero independiente. Incluye cuatro
tiles K, una fila, descriptor externo alterado tras aceptación, errores
de activación sin acceso a pesos, cancelación con AR detenido y durante
MAC, timeout, reset coordinado y job de recuperación. La regresión
`20261003T043551Z-V7-gemv-memory-regression` repite los 60 casos tras
ampliar el banco con soporte de imágenes, conservando sus conteos.

| Síntesis OOC, VU5P, reloj 5 ns | LUT | FF | RAMB36 + RAMB18 | DSP | WNS / WHS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cargador con lector AXI | 788 | 1429 | 0 + 0 | 0 | +2,559 / +0,066 ns |
| Cargador + GEMV dual | 16993 | 12438 | 37 + 2 | 52 | +0,980 / +0,046 ns |

El conjunto corresponde a `20261003T042802Z-V7-gemv-memory-synth`.
No hay LUTRAM ni cajas negras. Frente al dual anterior agrega 735 LUT
y 1871 FF, sin aumentar DSP/BRAM. Los valores son **estimaciones de
síntesis**, no ruteo. El conjunto mantiene 1103 entradas y 281 salidas
sin retardos; faltan reloj padre y restricciones del sistema. Se retienen
avisos de puertos aritméticos sin carga y recomendaciones DSP/RAM.

Nota de auditoría: el `result.md` sellado de la síntesis standalone del
cargador conserva texto genérico sobre hashes MAC y recomendaciones DSP.
Ese bloque no contiene MAC ni DSP; su `review.json` registra correctamente
`copied_mac_rtl_identical=false`, cero DSP y únicamente los tres avisos
de contexto OOC. El archivo original se conserva sin modificar.

Para repetir, cargar el entorno Vivado 2026.1 y usar directorios nuevos:

```bash
vivado -mode batch -source tools/ntrpu_board_activation_loader.tcl \
  -tclargs sim NUEVA_SALIDA
vivado -mode batch -source tools/ntrpu_board_gemv_memory.tcl \
  -tclargs "$NTRPU_MAC_RUN" sim NUEVA_SALIDA_GEMV
python -m unittest discover -s tests -p test_ntrpu_activation_layout.py -v
```

`NTRPU_MAC_RUN` debe identificar la co-simulación VU5P aprobada del MAC
Q8, con manifiesto y RTL generado. Sustituir `sim` por `synth` para
síntesis OOC. Los revisores comprueban valores/conteos y diagnósticos;
el código de salida de Vivado por sí solo no acredita corrección.

## Imágenes del host y pesos reales

`20261003T043551Z-V7-gemv-memory-fixture` terminó en 537,170 s y pasó
**174 jobs / 5376 valores**, sin diferencia numérica frente a Q8. Repite
los 54 jobs de proyecciones Brandon y cuatro formas sintéticas por semilla.
Ambos tipos de datos llegan desde imágenes de bytes inmutables; el banco
ya no escribe activaciones directamente. La revisión CPU decodifica cada
registro físico y recalcula todos los productos/acumulaciones desde los
pesos densos y activaciones originales, sin llamar al empaquetador.

| Medida por semilla 1 / 7 / 42 | Resultado |
| --- | --- |
| Ciclos de 58 jobs, sin pausas entre jobs | 377776 / 377075 / 377175 |
| Ciclos de las 54 proyecciones Brandon | 219840 / 219418 / 219508 |
| Ciclos de 32×8192, pesos sintéticos | 100844 / 100749 / 100689 |
| Bytes AXI de activaciones, incluidos escala y padding | 76288 |
| Bytes AXI de pesos | 1016192 |
| Ráfagas activaciones / pesos | 122 / 496 |

Los 1.092.480 bytes transportados por semilla equivalen a 2,892–2,897
bytes/ciclo de job bajo estas pausas. Es simulación, no caudal DDR físico.
Los patrones de pausas difieren del banco anterior de carga directa;
restar ambas latencias no aísla exclusivamente el coste del cargador.
Las activaciones de gate/up/down siguen siendo entradas sintéticas
independientes: no se encadena SwiGLU ni se ejecuta inferencia Llama.

Fixture: `build/ntrpu-gemv-memory-20261003`, 465 archivos más manifiesto,
SHA-256 `ae519d84dd530d6e457bc44ae25f956e63cd2e62671c4fec808bc008ee2e4f81`.
Conserva revisión y hashes de Brandon del [fixture de pesos](GEMV_WEIGHT_LAYOUT.md).
`prepare_ntrpu_gemv_memory_fixture.py` añade imágenes ABI1 al paquete
aprobado, en un directorio nuevo. El argumento opcional final del Tcl
de GEMV selecciona ese fixture. Diez mutaciones de logs son rechazadas;
los tres tests CPU cubren formato, extremos y entradas inválidas.

El control negativo `20261003T043854Z-V7-gemv-activation-negative`
modifica solo el byte 4032 de la primera imagen de activaciones: 254→255.
Los hashes se actualizan, pero las fuentes densas y referencias permanecen
intactas. La auditoría CPU rechaza la discrepancia y RTL falla al comparar
la primera fila, ciclo 3203. Se conserva su exit 1 esperado. Esto prueba
consumo de la imagen y sensibilidad del comparador; **no implementa una
protección de integridad en hardware** para valores INT8 válidos.

Cada tanda conserva revisión, fuentes, logs y hashes en NFS y bajo
`artifacts/ntrpu-board-vivado/<run_id>/`. El RTL del cargador y wrapper
es idéntico entre síntesis, banco de fallos y pruebas de imágenes.

## Próxima integración

La [escritura de resultados con confirmación AXI](RESULT_MEMORY.md) ya
está integrada y probada. Añadir registro de buffers y secuenciación FFN
(gate/up, SwiGLU, cuantización, down). Preservar la publicación condicionada
al éxito antes de conectar PCIe y DDR. La
calificación del DIMM, reset DDR, CDC y restricciones de placa sigue
dependiendo de [V2–V8](board/VIVADO_STATUS.md).
