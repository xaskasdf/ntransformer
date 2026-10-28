# Captura DDR R2 con comandos por rank

> **Corte actual:** preparada y con controles sintéticos aprobados, pero el
> coordinador se detuvo antes de lanzar Vivado porque `vocab-memory` requiere
> inspección. El usuario reporta pérdida de acceso a la VM. R2 no se ejecutó y
> no hay nueva estimación de consumo. Véase el [traspaso local](VIVADO_ACCESS_LOSS_HANDOFF_20261016.md).

Fecha: 2026-10-16. Revisión separada de [R1](POWER_ACTIVITY_RESULTS_R1.md).
**Encolada y probada con controles sintéticos; todavía sin ejecución nativa R2.**

Actualización 15:17 UTC: la [asociación SAIF/DCP de R1](POWER_SAIF_MAPPING_R1.md)
se añadió al final, después del controlador de token. Es un diagnóstico
de correspondencia con entradas ya archivadas; no cambia esta captura
R2 ni ejecuta una estimación de consumo.

Comprobación 15:01 UTC: PID 26696 confirmado, estado
`waiting_for_vocab_memory_review_and_backup`. Conserva su posición tras
`vocab-memory-vivado-r1`; el controlador de token se añadió más adelante,
al final de la cola. R1 ya tiene dos SAIF revisados y respaldados.
**La estimación de consumo aún no está encolada**: requiere revisar esta
captura, clasificar X/Z y comprobar correspondencia SAIF/DCP. La actividad
por rank y refresh es evidencia de simulación, no una medida física de vatios.

Registro 13:06 UTC: coordinador `ddr-activity-r2-vivado-r2`, PID 26696,
esperando revisión y respaldo del vocabulario. Memoria Q/K/V conserva
el slot. Paquete de 415 archivos, incluidos los 394 heredados intactos,
restaurado en D:/NFS. Fuente SHA-256
`2207eb44558d51b4940f2a43a53d037b06b722019f2f6e1189deea805ce4fcf1`;
configuración `5a099b0ae60fca5f0acc48016d7309c5bf6a0bc2d6ee78350f1c0954f1a43b0a`.

## Alcance

Conserva el banco, `example_top`, generador PRBS, geometría y parámetros
del paquete R1. El nuevo capturador añade un VCD pasivo con 48 pines:
doce del dispositivo representativo 0 de cada uno de los cuatro ranks.
Abre la traza antes de esperar calibración y la cierra 20 ns después de
las dos ventanas SAIF. No fuerza señales ni cambia los intervalos SAIF.

Las ventanas siguen siendo dos intervalos contiguos de 20,48 µs y 4.096
ciclos UI. Sus tiempos se toman de `windows.tsv`, sin suponer que la
calibración termine siempre en el mismo instante. El ámbito SAIF sigue
siendo el DUT; el VCD observa las interfaces públicas de los modelos RDIMM.

## Cola y límites

Quedó al final de la cola, después de
`vocab-memory-vivado-r1-coordinator.json`, con revisión y restauración
del predecesor obligatorias. Usa el único slot Vivado, CPU 0–1, límite de
1.200 segundos y ninguna repetición automática. El revisor usa CPU 2–3
y 120 segundos. Los resultados parciales también se conservan.

## Criterios de revisión

- Conservar la identidad de todas las fuentes heredadas de R1 y revisar
  la elaboración efectiva, el modelo RDIMM activo y la variante `glbl`.
- Revisar ambos SAIF, contadores y señales de reset, calibración y PRBS.
- Exigir cuatro interfaces completas y coincidencia exacta entre VCD e
  inventario de 48 pines, incluido el scope superior.
- Exigir al menos 10 ns de extensión VCD real antes y después de las
  ventanas; el tiempo solicitado de cierre no basta.
- Contar RD/WR/REF por rank y ventana. Conservar X/Z, relojes ambiguos,
  cambios simultáneos y ausencia de comandos como información pendiente.
- Verificar y restaurar el paquete fuente y la tanda terminal en D:/NFS.

Una codificación no prueba aceptación, cobertura de todos los chips ni
cumplimiento de refresh. R2 tampoco califica potencia: siguen pendientes
la representatividad, X/Z, bibliotecas y correspondencia SAIF/DCP.

## Controles previos

Nueve pruebas del revisor y seis casos de secuenciación Tcl pasaron en
Windows y Linux. La prueba Tcl usa sustitutos de XSim; no elabora RTL.
Se corrigió en una revisión separada el margen inicial de una forma de
onda sintética y se añadió la comprobación del scope VCD. Los borradores
previos quedaron conservados.

Evidencia: `build/cloud-setup/20261002-board/ddr-capture-r2-controls-review.json`.
ZIP restaurado: `cb8a39dcefeee54a10b99e0b70e704be5f5a12d9c2f002b2a64eed09520e4efd`.
Scripts: `tools/ntrpu_capture_ddr_activity_r2.tcl`,
`tools/ntrpu_board_ddr_activity_r2.tcl`, `tools/run_ntrpu_ddr_activity_r2.py`
y `tools/review_ntrpu_ddr_activity_r5.py`.
