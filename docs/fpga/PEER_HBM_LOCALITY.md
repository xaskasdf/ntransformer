# Localidad HBM del origen compartido

## Hallazgo y límites de la evidencia

Revisión documental y de configuración del 2026-10-07, hora de Chile;
**no es una nueva simulación ni una medición de ancho de banda**.

El donante nativo conecta los canales 0/1 a `AXI_00`/`AXI_16`, respectivamente,
en `fixtures/native.sv`. Su `native_hbm_config.tsv` declara `16GB`, dos stacks
y `USER_SWITCH_ENABLE_00/01=TRUE`. SHA-256 de esa configuración:
`d74a3e492748cf8a87100205e04bafe232fcdba09ae7e4a32105cdd02090539e`.
Se conserva en la tanda `20261007T233442277315Z-compile` de
`D:/AI/artifacts/ntrpu-peer-native/20261007T233159389882Z`.

Para 16 GB, AMD asigna el bit 33 al stack, los bits 32:29 al destino dentro
del stack y regiones de 512 MiB por pseudocanal. El destino asociado a AXI 00
comienza en `0x0_0000_0000`; el asociado a AXI 16, en `0x2_0000_0000`.
Con direccionamiento global, la dirección determina el destino aunque el
puerto de ingreso sea otro. Fuentes: [mapa de puertos, PG276](https://docs.amd.com/r/en-US/pg276-axi-hbm/Port-Descriptions)
y [direccionamiento global, PG276](https://docs.amd.com/r/en-US/pg276-axi-hbm/System-Level-Considerations).

**Inferencia para nuestro perfil:** la precarga anterior usa `canal*65536 +
offset`, con offsets pequeños; ambos canales apuntan al primer pseudocanal.
La lease del banco compartido también cubre solo los primeros 64 KiB. Son
pruebas útiles de integridad, propiedad y reutilización, pero no demuestran
reparto físico entre stacks ni ancho de banda agregado. No cambia la validez
funcional de la evidencia anterior.

La revisión de las trazas confirmó 80 AR en la prueba nativa anterior y 118
en el banco compartido comportamental: al aplicar ese mapa, todos corresponden
al destino 0. Las direcciones finales máximas son 107.007 y 29.247 bytes,
respectivamente. `LOCALITY_REVIEW.json`, en la entrega física compartida,
registra hashes y distribución por canal; el segundo banco no contiene HBM
nativa y su clasificación es una proyección del mapa.

## Secuencia de pruebas propuesta

1. **Origen común:** una inicialización y una precarga de la región baja;
   leerla por ambos puertos sin duplicar la precarga. Comprobar publicación,
   cuatro reservas, DMA/ACK, reutilización, revocación y parada final. Esperar
   la finalización de escritura antes de publicar. Este caso acepta contención
   en un pseudocanal; todavía no mide escalado.
2. **Mapa nativo aislado:** escribir patrones diferentes en las bases 0 y 8 GiB;
   leer cada región por ambos ingresos y verificar bytes y respuestas. Usar
   etiquetas distintas por región, no un patrón que ignore los bits altos.
   Comparar luego accesos locales simultáneos con accesos cruzados, registrando
   bytes útiles, ciclos activos y stalls por puerto.
3. **Integración con localidad:** introducir una traducción explícita entre
   offsets lógicos y regiones físicas. Ejemplo de diseño: dos fragmentos
   lógicos contiguos de 4 KiB, ubicados físicamente en las dos bases anteriores.
   Elegir el canal según el fragmento y validar traducción, tamaño y límites
   antes de reservar. Mantener offsets de destino independientes.

## Contrato pendiente de implementar

La lease actual describe una sola región publicada contigua y el lector calcula
`source_base + source_offset`. Aumentar `published_bytes` hasta 8 GiB para
alcanzar el segundo stack también autorizaría el hueco intermedio. La propuesta
de traducción evita ese permiso excesivo: publica solo el rango lógico escrito
y enumera sus fragmentos físicos autorizados.

La tabla y su generación deberán permanecer fijadas hasta retirar todas las
reservas que las usan. La compactación actual depende de una única lease
inmutable; varias leases simultáneas exigirán referencias por contexto y una
nueva validación. Ninguna traducción, tabla de fragmentos ni selección automática
de localidad está implementada en el guard actual.

Resolver este mapa y medir la contención precede a ampliar el pool. Aumentar
reservas, frames o puertos por sí solo no constituye evidencia de mayor ancho
de banda. Los resultados físicos de [la composición compartida](PEER_READ_SHARED_PHYSICAL.md)
siguen siendo otra condición independiente.
