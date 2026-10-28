# Consulta independiente de metadatos KV

Estado 2026-10-16 06:34 UTC: **control aprobado en Icarus; síntesis y XSim
encolados**. La jerarquía RoPE/atención sólo tiene compilación nativa en
esta revisión; sus dos sustitutos HLS no se ejecutaron.

## Motivo y contrato

La inspección del cableado R1 encontró que `cache_committed_tokens`
selecciona la capa del comando retenido dentro de captura/despacho. Antes
de aceptar el siguiente comando, puede describir la capa anterior. Por
ello no sirve como consulta anticipada arbitraria del padre que combinará
normalización/Q/K/V con RoPE/atención. Las pruebas previas de cada bloque
no acreditaban ese uso de la interfaz.

La nueva jerarquía agrega `cache_query_layer` de 32 bits,
`cache_query_valid` y `cache_query_committed_tokens`, manteniendo separado
el prefijo usado por la operación interna. El generador verifica los hashes
de los tres padres R1 y produce módulos nuevos:

- [`ntrpu_attention_root_guard_query_r1`](../../fpga/rtl/ntrpu_attention_root_guard_query_r1.sv): segunda lectura de los 16 prefijos confiables.
- [`ntrpu_attention_root_query_r1`](../../fpga/rtl/ntrpu_attention_root_query_r1.sv): expone la consulta del guard.
- [`ntrpu_rope_attention_query_r1`](../../fpga/rtl/ntrpu_rope_attention_query_r1.sv): lleva esos tres puertos hasta el padre.

La consulta es combinacional y no modifica estado. Capa fuera de 0–15 o
autoridad inválida devuelve `valid=0`, prefijo cero. Conserva la publicación
únicamente tras el ACK de una respuesta append propia y satisfactoria;
un error de escritura no publica un token. Época, identidad y fallo de
metadatos conservan las reglas del guard original.

**`cache_query_valid` acredita metadatos, no reserva la caché.** Durante
una operación puede observarse el último prefijo confirmado. El padre
debe comprobar inactividad y prioridad de bind, reservar la caché y fijar
la capa consultada durante toda la propiedad. Antes de iniciar el productor
debe validar las diez asignaciones y su separación de KV: el controlador
de retención existente sólo las verifica después de producir Q/K/V.

## Validación y límites

El [banco](../../fpga/tb/tb_ntrpu_cache_query_r1.sv) conserva los 96 estados
del banco original: bind, append retenido/confirmado, errores, revocación,
rollback de época y cuatro resets. En cada estado cruza 17 capas de
operación con 20 consultas, incluidos `0x10000000`, `0x8000000f` y
`0xffffffff`. Compara los diez puertos de salida anteriores contra una
instancia R1 independiente.

Tres semillas: **97.920 consultas**, 19.584 a capas inválidas, 4.896
comprobaciones originales de prefijo. El revisor rechazó **54 corrupciones
de traza** y las **ocho variantes RTL defectuosas** fallaron por aserciones
semánticas, sin contar timeouts. La jerarquía completa compiló con 25
fuentes RTL y dos sustitutos HLS; no ejecutó aritmética ni AXI.

Revisión: `build/cloud-setup/20261002-board/cache-query-native-r1-review.json`,
SHA-256 `83d2e71e9419e1bc9f7b43fb83039865d00523c6846b74294cb13e0c73bf0dbc`.
Ambas tandas tienen fuentes y resultados restaurados en D:, con copias
verificadas en NFS:

| Tanda | SHA-256 del TAR |
| --- | --- |
| `20261016T062853Z-V7-cache-query-native-r1` | `3a9a5c3c3a5ea5d708e11d15897ca0dc729d040248fb070636ced029cc204d43` |
| `20261016T063045Z-V7-cache-query-mutants-r1` | `d85c03b22fb063936b8b13b70c37d94042c90842dc766695d2fcb8aa1d70e9bb` |

## Turno Vivado y continuación

La [composición desde la entrada](LLAMA32_INPUT_ATTENTION_R1.md) ya usa la
consulta para validar regiones antes de arrancar normalización/Q/K/V.
Su protocolo nativo está aprobado y su síntesis/XSim se añadió después de
esta revisión. La aritmética conjunta sigue pendiente.

Paquete `20261016T063314Z-cache-query-vivado-r1`, 36 archivos y tres
fuentes compiladas: guard nuevo, guard de referencia y banco. ZIP SHA-256
`007199245818297496752a6686dd1a7a871e787e5c7696d191e43eff429d6b0a`.
Coordinador PID 22492: espera el control Q/K/V→RoPE aprobado y restaurado.
Dos CPU, 30 minutos totales, síntesis del guard VU5P a 5 ns OOC y tres
semillas XSim. Sin reintentos automáticos; respaldo y restauración antes
de liberar el slot. La síntesis de los wrappers queda para su composición.

Handoff: `artifacts/ntrpu-board-vivado/cache-query-handoff-20261016-r1/`.
La composición productor real → retención → RoPE/atención ya usa esta
consulta y valida regiones antes de cualquier escritura. Su ejecución
aritmética conjunta, modelo completo y DDR físico siguen pendientes.
DDR R1 conserva su lugar después de BRAM, **sin SAIF ni potencia nueva**.
