# Composición del propietario real y los certificados de drenaje

## Resultado revisado

El 2026-10-06 se completaron ocho perfiles XSIM con el propietario real,
las secciones AXI-Lite/AXI4 con IP AMD y los dos proveedores R2 de respuestas
locales. La revisión reconstruye las transferencias y los certificados
desde los VCD, comprueba las entradas compiladas contra las fuentes
congeladas y conserva los resultados por perfil.

| Modo | Host | XDMA | Obligación del propietario retenida | `complete_ack` |
| --- | --- | --- | --- | --- |
| 0 | Ausente | Ausente | CMD | Siempre 0 |
| 1 | Fixture | Ausente | DATA | Siempre 0 |
| 2 | Ausente | Fixture | Resultado | Siempre 0 |
| 3 | Fixture | Fixture | CMD | Dos certificados frescos; revocación posterior |

Cada modo se ejecutó con periodos de dispositivo de 5 y 37 ns. La época
real común fue `0x100000001`; no se certificó época cero ni se sustituyó
la época del propietario por una identidad arbitraria de gestión.

## Cobertura y medidas

Se comprobó bloqueo con propietario vivo, pérdida de sesión con reloj
detenido, descarte de una respuesta obsoleta, renovación por tags y
retención de CMD/DATA/resultado y ambas reservas. Tras invalidar la sesión,
restaurar admisión no permite nuevo ARM ni retiro de la reserva.

Por perfil se reenviaron 24 escrituras y 24 lecturas AXI4, con 1.045 beats
por dirección, y cinco escrituras/cinco lecturas AXI-Lite. Las respuestas
locales permanecieron pendientes hasta su aceptación independiente.
En los ocho perfiles, el propietario entregó ocho bloques de 64 bytes;
ninguna reserva se retiró. Los controles positivos añadieron una AW sin W
y observaron revocación en gestión 23 ns después de la falla local, aunque
el reloj host ya estaba detenido. Son medidas de este estímulo simulado.

Dos controles adicionales correctos y tres defectos RTL privados pasaron
la revisión independiente: se detectaron aceptación del propietario sin
prueba, solicitud al host ausente y solicitud al XDMA ausente. Diez
alteraciones de la evidencia también fueron rechazadas.

## Fuentes y respaldos

Los bancos están en `tb_ntrpu_drain_owned_sections.sv`,
`ntrpu_drain_owner_fixture.sv` y las dos fixtures de secciones. Se ejecutan
con `tools/ntrpu_board_drain_owned_{sections,missing,negatives}.tcl`;
`tools/check_ntrpu_drain_owned_sections.py` realiza la revisión de ondas.

Los archivos locales bajo `artifacts/ntrpu-board-vivado/<run>/` y sus
copias NFS tienen estos SHA-256:

| Tanda | SHA-256 del tar.gz |
| --- | --- |
| `20261006T092603Z-V4-ffn-supervised-owned-sections-frozen` | `a9acb99a8507e1bb76574a48e6cb24ce3ed008db06b700327a1e0a6d611711ce` |
| `20261006T093320Z-V4-ffn-supervised-owned-missing-frozen` | `0164eb4014aa1639b7e694805a2d883d98cfc1c47665759f357a923028bae465` |
| `20261006T094040Z-V4-ffn-supervised-owned-negatives-frozen` | `63e280ef823d85a62234ab1a7f49e6e6852fa110f9868506451ea13193d6b214` |

Las revisiones independientes y las mutaciones de evidencia se conservan
como addenda en `build/cloud-setup/20261002-board/`, con el ID de su tanda.

## Alcance y siguiente paso

Los estímulos del propietario y de las secciones siguen siendo caminos
independientes: **no forman todavía un datapath host–PCIe–DDR integrado**.
Host y colas protegidas XDMA continúan sin proveedores reales calificados.
La prueba no dirige reset operativo ni valida el timeout nativo de 50 ms,
CDC-15 del FIFO AMD, timing físico, DIMM exacto o placa de 512 GiB.

El siguiente paso es trasladar la conexión de slot 0 al RTL sintetizable
`ntrpu_owned_drain_coordinator`, conservando seis contratos externos
explícitos. Debe reproducir estos ocho perfiles y sus señales públicas,
y pasar revisión OOC de recursos, timing y CDC antes de integrar los
proveedores host/XDMA y el camino de datos común.
