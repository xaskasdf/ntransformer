# Diagnósticos de integración de certificados de drenaje

Revisión del 2026-10-06. Los resultados originales permanecen inmutables.
Ninguna de las tandas de este documento autoriza un reset de endpoint ni
califica una tarjeta física.

## Captura de las cuatro simulaciones

La tanda `20261006T062300Z-V4-ffn-supervised-drain-sections` terminó con
código 0 y mensajes HDL PASS en sus cuatro perfiles: reloj de dispositivo
de 5/37 ns y contratos externos de prueba ausentes/presentes. Integra las
secciones AXI-Lite y AXI4 reales con IP AMD y dos certificados de respuestas
locales. Los proveedores de propiedad y cierre del host siguen siendo
fixtures; el proveedor XDMA también es un fixture cuando está presente.

La captura independiente está incompleta:

| Perfil `external-period_ps` | Tamaño del VCD |
| --- | ---: |
| `0-5000` | 4.560.201 bytes |
| `1-5000` | 123 bytes, sin señales |
| `0-37000` | 123 bytes, sin señales |
| `1-37000` | 123 bytes, sin señales |

Los PASS no sustituyen las trazas faltantes. La causa de las capturas vacías
no está demostrada. `tools/ntrpu_board_drain_sections_r2.tcl` conserva RTL,
banco y perfiles, ejecuta los scripts nativos con un proceso XSIM nuevo por
perfil y exige señales y contenido en cada VCD. La revisión independiente
de transacciones, identidades, revocación y trazas continúa pendiente.

Archivo original SHA-256:
`c2115eb3f0888dfca0c503d40cb79019ff1e2d21a745305d8027a448efff59bc`.

## Síntesis del proveedor local con fuentes reales

`20261006T062816Z-V4-ffn-supervised-local-reply-synth` terminó con código 0.
El wrapper incluye admisión AXI4, su certificado de sección, Clock Converter
AMD y el nuevo proveedor local. `pair_cdc.rpt` informa **12 CDC-11 Critical**:
seis registros de fallos de admisión alimentan dos cadenas de sincronización
hacia gestión, una en `dut/admission_sources` y otra en
`local_reply/report_fault`.

Las fuentes son `quarantine_b`, `quarantine_r`, `write_fault`, `read_fault`,
`request_fault` y `local_write_fault`. Cada una aparece en dos destinos.
La revocación propia `local_reply/revoked_source_reg` aparece como CDC-3
con tres etapas. Persiste además un CDC-15 Warning. No se aplicó una
excepción para ocultar los CDC-11.

La siguiente revisión debe evitar la duplicación de estas cadenas y
demostrar que ambos consumidores reciben fallos retenidos aun si se detiene
el reloj fuente. Compartir la observación ya sincronizada es una opción de
diseño pendiente de implementar y verificar; no es una corrección probada.

El resumen de síntesis arroja WNS 1,523 ns y WHS 0,013 ns, con **1.439 entradas
y 1.726 salidas sin retardos de interfaz**. No hay endpoints internos sin
restricción, relojes ausentes ni lazos reportados. Estas métricas corresponden
a síntesis OOC, no a cierre de timing físico ni a equivalencia de netlist.

Archivo de síntesis SHA-256:
`aafed24bbeb6d891f2eb5841170efa908a08af7c0cfdc8d3d61f9bce7fc4714f`.

## Conservación y alcance

Ambas tandas conservan comprimido verificado, fuentes y evidencia local/NFS.
Los `VERIFIED.json` acreditan integridad del respaldo; no certifican la
corrección funcional. Los reviews originales con estado pendiente siguen
vigentes y este documento agrega los diagnósticos encontrados.
