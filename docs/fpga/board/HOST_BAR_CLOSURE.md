# Cierre del productor BAR: evidencia nativa acotada

Estado: **revisado y respaldado localmente y en NFS**.
Tanda `20261006T052948Z-V4-ffn-supervised-host-bar-closure`, Vivado 2026.1, XDMA 4.2 nativo.
El [plan](HOST_BAR_CLOSURE_PLAN.md) conserva el contrato ensayado.

## Resultado

Un productor del banco admitió dos escrituras de un DWORD en dos generaciones,
rechazó cuatro solicitudes y aceptó dos certificados después de recibir y
comparar los readbacks. Diez intentos de certificación prematuros, atrasados
o con identidad/objetivo incorrectos quedaron rechazados. Ambos canales
SGDMA permanecieron detenidos; sus 18 lecturas de control, estado y contador
se correlacionaron con los TLP reales.

| Evento | Generación 7 | Generación 8 |
| --- | ---: | ---: |
| RQ escritura aceptada | 104.656000 µs | 123.604000 µs |
| Cierre del productor | 104.842000 µs | 123.606000 µs |
| AW aceptada | 104.836000 µs | 123.784000 µs |
| B aceptada | 108.944000 µs | 127.888000 µs |
| RC del readback | 109.164000 µs | 128.108000 µs |
| Certificado del modelo | 110.846000 µs | 129.790000 µs |

La segunda escritura llegó a AXI **después del cierre**: ya había sido
admitida y seguía dentro de PCIe. El cierre conservó su obligación pendiente.
Las respuestas B se retuvieron 4,100 y 4,096 µs (1025 y 1024 ciclos)
por estímulo del banco; una rama independiente las liberó sin esperar una
lectura bloqueada. Esos tiempos no son mediciones de rendimiento de placa.

## Reconstrucción y límites

El revisor reconstruye RQ/RC, AW/W/B/AR/R, bytes habilitados y el registro
de obligaciones. Direcciones: `0x100002000` y `0x100002100`; datos:
`0xd3c2b1a0` y `0x4c3d2e1f`. TC=0, ATTR=0, Requester ID=`0x00af`,
primer BE=`0xf` y último BE=0. Se comprobaron las respuestas AXI y los
20 retornos de lectura nativos, incluidos los dos readbacks.

El auditor rechazó **94 alteraciones de evidencia** y cotejó los
139 avisos heredados con las fuentes compiladas. El XCI conserva el
Completion Timeout nativo de 50 ms; no se fuerza estado interno.
Las llamadas del modelo sin avance temporal se reconstruyen en orden
del registro y se contrastan con el estado agregado del VCD. Esto **no
prueba concurrencia entre hilos de un driver**, un fence universal PCIe,
inventario protegido vacío, drenaje global ni permiso de reset.

## Respaldo y siguiente dependencia

- Archivo nativo SHA-256: `991349885fdd9ebbbcbfb9414a9191f92a383ba45740391b7e5e423b00d41750`.
- Revisión SHA-256: `d79739ff6af2c0b28deff2da97584083e7ab3d1d0bd7a72f8d6a0be98756082a`.
- Revisor: `tools/check_ntrpu_host_bar_closure.py`.
- Auditor: `build/cloud-setup/20261002-board/ffn-supervised-audit-host-bar-closure.py`.

La cola R26 debe integrar identidad, cierre y certificados de sección,
manteniendo bloqueada la propiedad cuando falta el contrato externo de
inventario XDMA. El certificado del modelo no sustituye esa entrada.
El [criterio de drenaje](XDMA_DRAIN_ACCEPTANCE.md) y las dependencias de
reset, DIMM, CDC y diseño físico siguen vigentes.
