# Composición del lector HBM con resets fríos canónicos

## Alcance

Esta tanda privada continúa [HBM_CDC_PHYSICAL.md](HBM_CDC_PHYSICAL.md) para
el nodo **VU47P/HBM**. La tarjeta DDR4 de 512 GiB conserva su diseño separado.
El generador `prepare_hbm_clocked.py` une el cliente AXI Bridge, el transporte
250→200 MHz y el lector seleccionado del perfil `HEAD_NATIVE_PROFILE.json`.
La composición termina en las interfaces AXI del endpoint y de HBM; todavía
no incluye ambos IP completos ni constituye una implementación de tarjeta.

No se modifica el guardián V3, el lector seleccionado ni sus checkpoints.
Se generan variantes nuevas del guardián y del cliente que únicamente
exponen `cold_m[2]` como `management_cold_rst_n`. El revisor exige exactamente
esa transformación y las fuentes originales fijadas por SHA-256.

## Relojes y propiedad del reset

| Dominio | Frecuencia del contrato | Único origen de liberación fría |
| --- | --- | --- |
| PCIe | 250 MHz | `endpoint/guard/cold_a[2]` |
| Gestión | 100 MHz, continuo e independiente | `endpoint/guard/cold_m[2]` |
| Lector | 200 MHz | `reader/startup/source_release[2]` |
| Memoria | ≈450 MHz, período de prueba 2,222 ns | `reader/startup/memory_release[2]` |

Las cuatro cadenas reciben la misma entrada `configuration_rst_n`. El CDC,
los vigilantes y los conversores consumen sus salidas; no generan cadenas
paralelas para el mismo dominio. PERSTn y los resets operativos del endpoint
son señales de disponibilidad, nunca sustitutos del reset frío coordinado.

El endpoint seleccionado entrega `axi_aclk_out`. El perfil HBM seleccionado
recibe relojes externos: sus bancos anteriores usaban estímulos para lector,
memoria, referencia HBM y APB. Esta tanda conserva entradas de reloj en su
frontera; **no selecciona osciladores, MMCM, buffers, pines ni distribución
física**. No resuelve los gaps de reloj del ensayo OOC anterior.

## Vigilancia añadida

`ntrpu_axi_bridge_hbm_mem_watch.sv` vigila el reloj de memoria mediante
solicitud y eco, con tres etapas por cruce y plazo de 64 ciclos de gestión.
La admisión requiere que el vigilante haya arrancado. Después de armarse,
una pérdida de reloj deja un fallo persistente hasta el próximo reset frío.
El fallo llega directamente al dominio PCIe y también al lector; así no
depende de que el lector siga funcionando para cerrar la admisión.

La vigilancia existente del reloj de 200 MHz permanece intacta. Un fallo
operativo conserva las ofertas de las colas; puede haber drenaje durante la
latencia de detección. La gestión debe seguir activa. Las pérdidas del reloj
de referencia HBM, la secuencia APB y la supervisión física de alimentación
continúan pendientes de integración.

## Validación reproducible

El banco nuevo comprueba 13 escenarios con dos desplazamientos temporales:
operación sana; parada alta/baja de los relojes de memoria, lector y PCIe;
pérdida de disponibilidad PCIe; fallo del adaptador; pérdida de calibración;
reloj de memoria ausente al arrancar; reset frío con memoria parada; y parada
simultánea de lector y memoria. Comprueba ofertas retenidas en ambos canales,
recuperación exclusivamente fría y equivalencia del guardián exportado con V3.
Los datos del banco son estímulos explícitos, no tráfico HBM nativo.

```bash
python -B prepare_hbm_clocked.py NUEVA_PREPARACION \
  --selected PERFIL_SELECCIONADO --repo REPOSITORIO
python -B run_hbm_clocked.py NUEVA_TANDA --prepared NUEVA_PREPARACION \
  --engine xsim --settings VIVADO_SETTINGS
python -B run_hbm_clocked.py NUEVA_SINTESIS --prepared NUEVA_PREPARACION \
  --engine synth --settings VIVADO_SETTINGS
python -B review_hbm_clocked.py TANDA_RESTAURADA
```

Para Icarus, usar `--engine icarus --toolchain TOOLCHAIN_PORTABLE`.
La síntesis genera los tres conversores AMD reales del lector, compara sus
configuraciones completas con el perfil seleccionado y audita 17 conexiones
de reset contra cuatro drivers canónicos. Exporta CDC y DRC sin waivers.
La síntesis estructural no valida tráfico conjunto, timing físico ni CDC
integrado. Los límites físicos del transporte deben reaplicarse y comprobarse
en su nueva jerarquía durante la implementación, incluyendo el nuevo vigilante.

## Evidencia y próximos criterios

Icarus y XSIM pasaron **26/26 casos cada uno**. La mayor latencia de fallo
observada fue 695,79 ns, bajo el límite de 900 ns del banco. XSIM tardó 198,9 s
y la síntesis 190,4 s. La síntesis confirmó las 17 conexiones y configuraciones
de IP idénticas al perfil seleccionado: 11.634 LUT y 14.402 FF, sin blackboxes,
latches, bucles ni registros internos sin reloj.

El inventario CDC contiene 29 CDC-3 y 12 CDC-9 informativos, 8 CDC-6 y
1.460 CDC-15 de advertencia, sin críticos. Dos CDC-15 pertenecen a los
conversores AMD; los restantes corresponden al transporte. DRC no reportó
hallazgos en síntesis OOC. Esto no sustituye la revisión de rutas físicas.
Persisten 1.390 entradas y 1.044 salidas sin presupuestos externos. La síntesis
reportó 4.161 warnings, con supresión tras 100 mensajes Synth 8-7129 de puertos
sin carga o no conectados; no es una validación libre de advertencias.

**Timing no cerrado:** WNS estimado −0,530 ns en PCIe y −0,458 ns en memoria.
También existen rutas asíncronas pendientes de sus restricciones específicas.
No hay buffers reales, colocación ni ruteo en esta tanda; no debe interpretarse
la ausencia de endpoints internos sin restricción como corrección del CDC.

Los resultados, intentos no seleccionados, fuentes y revisores se conservan en
`D:/AI/artifacts/ntrpu-axi-bridge-hbm-clocked/`. `SELECTION.json` identifica las
tandas válidas; `BUNDLE_BACKUP.json` y `BUNDLE_REVIEW.json` verifican la copia
local restaurada. Los trabajos usan dos CPU, prioridad reducida y un máximo de
1.200 s para síntesis, dentro del espacio privado de la VM.

Ambos runners de VM conservaron `status=FAILED`: el empaquetador había
clasificado como entradas sus dos logs en crecimiento. `PACKAGING_AUDIT.json`
prueba que únicamente esos logs cambiaron, que las copias ejecutadas coinciden
con sus hashes originales y que todas las fuentes reales permanecieron
intactas. El revisor recupera explícitamente esta evidencia con
`--packaging-audit PACKAGING_AUDIT.json`, sin alterar los resultados originales
ni repetir el cómputo. El runner publicado filtra esos logs en futuras tandas.
Los 12 controles negativos incluyen tres que impiden usar esa recuperación
si cambia RTL real, una copia ejecutada o el conjunto permitido de archivos.

Antes de calificar la composición faltan: integrar relojes físicos y sus
restricciones, revisar los CDC de todo el lector, rutear con ambos IP nativos,
validar tráfico conjunto y enlazar el ACK con su origen físico autorizado.
El hold OOC previo de 4 ps no se hereda como margen de esta composición.
