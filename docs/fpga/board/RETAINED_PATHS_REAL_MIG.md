# Rutas AXI retenidas con NTRD y dos MIG

## Integración y alcance

El 2026-10-06 se conectaron las secciones retenidas al mismo camino de datos:
AXI-Lite → admisión retenida → Shutdown Manager → CDC → control NTRD;
AXI4 → admisión retenida → CDC → ingreso NTRPUF01. Ambos usan el mismo
propietario coordinado y el receptor multibuffer de dos MIG.

Los nuevos módulos son `ntrpu_retained_axil_path`,
`ntrpu_retained_axi4_path`, `ntrpu_pcie_fragment_retained_host` y
`ntrpu_pcie_fragment_retained_mig`. Los certificados de secciones y respuestas
locales ocupan sus slots reales del coordinador. Los contratos externos de
host y XDMA permanecen ausentes: no se simulan como si fueran pruebas válidas.

## Tanda aprobada a 7/5 ns

`20261006T113614Z-V4-ffn-supervised-retained-paths-mig-fast-frozen` pasó la
simulación y la revisión independiente con Vivado 2026.1, build 6511674,
`xcvu5p-flva2104-2-i`. Host: 7 ns; fuente: 5 ns; gestión: 10 ns;
dos UI de memoria: 5 ns, desfasadas 731 ps.

La revisión reconstruye las transacciones de ambos extremos, los registros
NTRD, las identidades y los datos DDR. Conserva el oráculo anterior de
18 trabajos; su resultado a 5 ns coincide exactamente con la evidencia
nativa previa. El trabajo adicional retiene respuestas B/R del host y B
del MIG durante invalidación, cambio de tag y pausas de reloj.

| Evidencia observada | Resultado |
| --- | --- |
| Control MMIO del host | 288 escrituras, 12 lecturas |
| Frames del host | 38; el último se rechaza localmente tras el cierre |
| Escrituras / lecturas DDR comparadas | 17 / 29; 1.088 / 1.856 bytes |
| ARM / retiros | 19 / 18; el último propietario permanece retenido |
| Certificados aceptados | Propietario y dos secciones: máscara `0x19` |
| Finalización de receptor retenida | Cookie `0x10000000f`; sin aceptación ni publicación |
| Drenaje global | `complete_ack=0` durante toda la prueba |

La memoria conserva el último payload incluso mientras su CPL permanece
pendiente. Se verificaron cuatro ranks por canal y 81/73 solicitudes
periódicas completadas. Los modelos archivados coinciden con la referencia
nativa anterior. Seis alteraciones deliberadas de evidencia fueron
rechazadas; son controles del revisor, no mutantes RTL.

## Segundo perfil y métricas de simulación

La tanda `20261006T115221Z-V4-ffn-supervised-retained-paths-mig-slow-frozen`
también quedó aprobada y respaldada con fuente a **37 ns**, host a 7 ns y
las mismas fuentes. Reprodujo las mismas cantidades de MMIO, frames,
transferencias DDR, retiros y certificados del perfil a 5 ns, conservando
la última CPL. Completó 192/186 solicitudes periódicas y rechazó los mismos
seis controles de evidencia.

| Métrica observada | Fuente 5 ns | Fuente 37 ns |
| --- | --- | --- |
| LINE → B del MIG | 88,498–593,498 ns | 341,498–1.144,498 ns |
| B del MIG → CPL | 55,771–56,502 ns | 350,502–372,502 ns |
| AR → R del MIG | 115–720 ns | 115–740 ns |
| Tiempo simulado | 115,794230 µs | 228,464230 µs |
| Ejecución nativa | 728,309 s | 1.209,554 s |

Los rangos corresponden al prefijo sano de 18 trabajos y sus retenciones
deliberadas; no representan caudal sostenido ni rendimiento físico.
El revisor R2 considera que el reset del host puede liberarse entre dos
flancos de la fuente lenta. Conserva las comprobaciones de profundidad
del sincronizador y de todas las transacciones; reproduce exactamente
los resultados independientes del primer perfil.

## Respaldo y siguiente dependencia

Se verificaron localmente y en NFS los 738 archivos de la tanda.
Archivo de resultados: SHA-256
`950f285758e756587f80acb2b6aa599ab50d5bb060b1ce75702ca7f11cce162c`.
Fuentes congeladas: SHA-256
`4dbc7efc252f1f0d9e502c3c97183e0f147c2fe2b05658c4c7c16cd22e8783eb`.
Revisión independiente: SHA-256
`e73d6b093b8322860f6d661b061122d86c1b4841038452de269527e162ea36b8`.
Los archivos están bajo `artifacts/ntrpu-board-vivado/<run>/` y los informes
bajo `build/cloud-setup/20261002-board/<run>-*`.

La segunda tanda también conserva 738 archivos verificados. Su comprimido
tiene SHA-256
`a21174123cc1fa40368f6d96f64359c6480befb54d8ea75b15396d73453a2094`;
su revisión R2, SHA-256
`2f14ec80bfd01b5161c97f44dbc415b3b2b6b93cb067d1f92dd489e574f3f103`.
La siguiente tanda prueba el [supervisor con retención del receptor](RETAINED_RECEIVER_EPOCH.md)
cuando vence realmente el watchdog.

Las pausas de fuente de esta prueba son inferiores al watchdog existente
de 64 ciclos de gestión, que no se modificó. Una pérdida prolongada todavía
puede resetear las colas del receptor actual: requiere retención coordinada
y pruebas específicas antes de autorizar recuperación. Tampoco se validan
reset operativo, inventario protegido XDMA, cierre del productor host,
lecturas AXI4 del host, DIMM comercial exacto, cuatro canales ni timing
físico. El readback DDR lo conduce el fixture de memoria. Los criterios de
[VIVADO_QUEUE.md](VIVADO_QUEUE.md) permanecen íntegros.
