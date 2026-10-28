# Receptor multibuffer conectado a MIG: arranque en frío

## Resultado y alcance

Tanda lateral del 2026-10-04: el receptor R4 pasa simulación funcional
contra un controlador DDR4 generado por Vivado 2026.1 y sus modelos DIMM.
Se reutiliza sin modificaciones el núcleo de la
[composición física de dos destinos](PEER_RX_MULTIBUFFER_COMPOSITION.md).
**Esta nueva integración todavía no tiene síntesis, ruteo ni validación de
placa.** El MIG funciona en modo de simulación `BFM`; no representa una
medición de calibración, integridad de señal o rendimiento del hardware.

`ntrpu_multibuffer_mig_cold.sv` conecta el destino 0 al MIG; el destino 1
permanece inactivo. Conserva ocho bancos de 4 KiB, cuatro de reserva,
16 descriptores y cuatro créditos por destino. Ambos puertos internos
reciben el mismo reloj UI en esta prueba. No acredita dos controladores
independientes, PCIe/P2P, BAR compartido, HBM ni 512 GiB instalados.

## Configuración y contrato de arranque

El controlador usa `xcvu5p-flva2104-2-i`, dos RDIMM genéricos de 64 GiB,
cuatro ranks, componentes x4 y DQ físico de 72 bits con ECC habilitado.
AXI: datos de 512 bits, dirección de 37 bits e ID de cuatro bits.
DDR4-1600: CL12, CWL9, tRCD/tRP de 15 ns; referencia de 250 MHz y UI de
200 MHz. La entrada también opera a 200 MHz, con distinta fase. Son
geometrías exploratorias, no una cualificación de un DIMM comercial.

`ntrpu_mig_cold_start.sv` añade un adaptador al supervisor existente:

- El reloj de gestión, independiente y continuo, comprueba actividad de
  los relojes de origen y memoria mediante el supervisor.
- Cada dominio confirma al menos ocho ciclos limpio bajo reset; el
  supervisor mantiene el reset al menos 64 ciclos locales y coordina su
  liberación. Estas confirmaciones sólo acreditan el contrato de arranque
  sin tráfico previo, no la recuperación de transacciones arbitrarias.
- La admisión espera calibración, salida de `ui_clk_sync_rst` y liberación
  coordinada. El reset AXI puede liberarse antes de la calibración.
- Tras el primer `RUN`, `rearm` queda bloqueado hasta reconfigurar la FPGA.
  Una pérdida de reloj, calibración o reset invalida la sesión; no se
  promete recuperación en caliente ni reversión de bytes ya escritos.

El reset físico `sys_rst` del MIG permanece separado del supervisor.
AMD describe la secuencia de reset y calibración en
[PG150: Reset Sequence](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/Reset-Sequence).
Esperar calibración para admitir trabajo es una decisión de esta interfaz;
el IP permite presentar comandos antes de completarla según
[PG150: init_calib_complete](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/init_calib_complete).

Una parada ordinaria cierra comandos nuevos y deja terminar datos,
respuestas y completions aceptados. `drained` exige admisión activa para
evitar confundir un reset con un vaciado correcto. La lectura de
verificación es un productor del banco directamente en el dominio UI;
todavía no es un protocolo de lectura para el host.

## Evidencia de la tanda

| Comprobación | Resultado |
| --- | --- |
| Arranque del adaptador | Ocho casos correctos: sano, pérdida de calibración, reset UI, pérdida de cada reloj, reset externo, B pendiente y origen no limpio |
| Comandos / completions | 16 / 16; uno rechazado localmente por rango |
| Escritura real al MIG | 99 beats de 64 bytes: 6.336 bytes |
| Lectura comparada | 279 beats: 17.856 bytes, sobre 98 direcciones distintas |
| Créditos | Cuatro AW aceptados pendientes de B simultáneamente |
| Parada | Con cuatro B retenidos, durante llenado y después de AW; 12 vaciados comprobados |
| Backpressure | Respuesta B, datos W y completion; ninguna respuesta de memoria fabricada |
| Fronteras | 0, más de 4 GiB, 32/64/96 GiB, último bloque y rechazo en 128 GiB; ráfaga de 4 KiB |
| Actividad periódica interna | 12 solicitudes, aceptaciones, CAS y retornos asociados y drenados |
| Verificadores | Seis corrupciones de payload/protocolo y 13 del observador periódico rechazadas |

Se reconstruyen los bytes escritos y leídos desde la traza, el orden
AW/W/B/completion y los comandos físicos ACT/READ/WRITE. Los cuatro
barridos de lectura mantienen centinelas para detectar alias. El rechazo
en 128 GiB no emite AW/W al MIG: la dirección completa se comprueba antes
de reducirla a 37 bits. No hubo interrupciones ECC; esta tanda no inyecta
errores ECC. Las fallas de arranque se prueban en el adaptador aislado,
no durante tráfico pendiente del MIG.

La simulación cubre 93,758498 µs y consumió 391,775 s de VM, con afinidad
a CPU 26–27, prioridad `nice=10` y dos hilos Vivado. La carga incluye
pausas deliberadas: estos totales no son un benchmark de ancho de banda.

## Diagnósticos y revisión independiente

El nuevo verificador periódico admite cero avisos de lectura sin
inicializar **únicamente cuando existe escritura física previa**. Conserva
las demás comprobaciones del original, que permanece intacto. Una
mutación mueve una lectura periódica a un bloque no escrito sin añadir
avisos y debe fallar. También fallan avisos inventados, retornos perdidos,
fugas al canal de usuario y finales sin drenaje.

Se conservan 23 avisos de herramientas: 13 con identificador y diez del
modelo FIFO. Sus limitaciones incluyen latencia de sincronización no
modelada, dos assertions de calibración no soportadas por XSIM y avisos
de índices en código generado. No hubo `CRITICAL WARNING` ni errores.
No se extrapola este resultado funcional a cierre CDC o temporal.

Al abrir la copia, Vivado encuentra inicialmente el CSV antiguo CL11;
el flujo selecciona después el CSV actual y regenera el IP. La auditoría
confirma CL12, CWL9 y tRCD/tRP=12 ciclos en las propiedades y el RTL
generado. El ejemplo original queda intacto. De los 35 archivos del
modelo, sólo cambia `ddr4_sdram_model_wrapper.sv`: `DDR4_16G_X4`,
`DDR4_1250_Timing` y `FIXED_1600`. No se desactivan lecturas periódicas,
ni se precarga memoria ni se alteran los modelos protegidos.

## Reproducción, respaldo y siguiente paso

En Linux con Vivado 2026.1 disponible, ejecutar desde una copia aislada:

```bash
python3 -B tools/run_ntrpu_multibuffer_mig.py /ruta/nueva/tanda \
  --example /ruta/ejemplo/ntrpu_ddr_geometry_ex --timeout 1200
# Para la estructura de archivo descargado: inputs/ y output/
python3 -B tools/check_ntrpu_multibuffer_mig.py /ruta/tanda-descargada
python3 -B tools/check_ntrpu_multibuffer_periodic_controls.py \
  /ruta/tanda-descargada/output/suite/memory.log
```

El runner requiere un directorio nuevo. El verificador crea `REVIEW.json`
sin sobrescribirlo. La auditoría adicional y su comando de recuperación
están en `HANDOFF.md` dentro de
`D:\AI\artifacts\ntrpu-multibuffer-mig`.

La tanda `20261004T214657205023Z-multibuffer-mig` conserva entradas,
salidas, trazas, inventarios y hashes. El paquete
`20261004-mig-handoff.zip` reúne las fuentes, revisores, documentación,
resultados y ejemplo original, con CRC y SHA-256 por miembro. Los 21
archivos preexistentes vigilados, incluido `AGENTS.md`, siguen intactos.

El siguiente avance acotado es conectar un segundo MIG con su propio UI,
definir qué ocurre cuando sólo uno está disponible y verificar tráfico
concurrente, aislamiento de direcciones y uso de reservas. Después debe
cerrarse la implementación física del conjunto y medirse servicio bajo
contención antes de aumentar el pool o atribuir mejoras de ancho de banda.

Ese experimento quedó validado funcionalmente con dos MIG, barrera de
arranque conjunta y reservas compartidas; véase
[receptor con dos controladores DDR4](PEER_RX_DUAL_MIG.md). La
implementación física del nuevo conjunto continúa pendiente.
