# Proveedor local R2: validación de composición

## Resultados revisados el 2026-10-06

`20261006T080527Z-V4-ffn-supervised-drain-sections-r6-frozen` pasó la revisión
independiente de sus cuatro perfiles, con relojes de dispositivo de 5/37 ns
y contrato externo ausente/presente como fixture. Conserva 25 escrituras y
25 lecturas AXI4 completas por perfil, incluidas respuestas locales, además
de seis transacciones AXI-Lite por dirección. Las capturas comprueban los
datos, identidades, relojes y retención de obligaciones; sin contrato XDMA
no aparece `complete`. La petición AW tardía revoca el certificado en 25,5 ns.

`20261006T081319Z-V4-ffn-supervised-drain-shared-fault-r2-frozen` pasó dos
perfiles adicionales. Una respuesta R huérfana atraviesa el Clock Converter
real, queda contenida ante la interfaz protegida y activa el registro de
fallo host. Con gestión detenida se captura el fallo; después se detiene host
y se reanuda gestión. La alarma compartida llega al proveedor local en
24.498/20.498 ps desde el marcador de reanudación, sin otro flanco host.
No hay revocación local propia que pueda ocultar una conexión ausente. La
alarma persiste después de cancelar y reemplazar la identidad.

## Revisión independiente y respaldo

El verificador original detectó 21 alteraciones deliberadas de evidencia.
La revisión R2 del verificador mantiene las comprobaciones durante actividad
y excluye de la medición de fallos los valores X de inicialización anteriores
a `cold=1`. No modifica RTL, estímulos ni resultados nativos. Los controles
negativos adicionales sobre la evidencia compartida se registran en un
addendum separado.

Los cuatro bancos de esta tanda están respaldados localmente y en NFS:
hardware original, síntesis R2, composición R2 y fallo compartido. Los JSON
`*-independent-transaction-vcd-review.json` y `*-independent-cdc-review.json`
en `build/cloud-setup/20261002-board/` fijan logs, VCD y revisores por SHA-256.
El checkpoint `20261006T081502Z-ffn-supervised-checkpoint` conserva 672 tandas
y registra explícitamente que el banco compartido aún estaba en curso al
tomar ese checkpoint; su certificado posterior lleva el total a 673.

## Límites y siguiente ejecución

La síntesis R2 eliminó los 12 CDC-11 duplicados; queda un CDC-15 nativo sin
resolver y faltan restricciones físicas de E/S. Véanse las
[métricas OOC y condiciones de revisión](DRAIN_SECTION_INDEPENDENT_REVIEW.md).

Está en ejecución `20261006T081717Z-V4-ffn-supervised-local-reply-negatives-frozen`:
control positivo y tres copias RTL defectuosas. Cada defecto debe fallar por
su aserción específica. No se da por aprobado antes de revisar y respaldar
su resultado. Continúan abiertos los proveedores externos reales, el reset
completo del endpoint y la calificación DDR/PCIe de la placa de 512 GiB.
