# Retención del receptor al vencer el watchdog

## Estado y evidencia

El 2026-10-06 quedaron **aprobados en simulación y respaldados** los perfiles
host/fuente de 7/5 y 7/37 ns, mediante revisión independiente de las trazas,
conteos, identidades, bytes y modelos MIG archivados. Tanda rápida:
`20261006T123351Z-V4-ffn-supervised-retained-long-mig-r2-fast-frozen`.
Tanda de 37 ns:
`20261006T125038Z-V4-ffn-supervised-retained-long-mig-r2-slow-frozen`.
Ambos usan las fuentes congeladas
`c49bb3d9beaff7455d4f18c6058844af8122b14a3fb8244b49b8d102d47b2df1`.
Continúa la integración descrita en [RETAINED_PATHS_REAL_MIG.md](RETAINED_PATHS_REAL_MIG.md).
Las revisiones anteriores permanecen intactas.

| Medida común a ambos perfiles | Resultado por perfil |
| --- | --- |
| Escenarios unitarios del supervisor | 13 aprobados en XSIM |
| Operaciones del banco completo | 19 ARM, 17 CMD/DATA, 16 CPL/DONE, 18 ACK |
| DDR comparada | 17 escrituras / 29 lecturas; 1.088 / 1.856 bytes |
| Host | 288 escrituras / 12 lecturas MMIO; 38 frames |
| Estado final | CPL `0x10000000f` retenida; admission cerrada |
| Certificados aceptados / reset global | Máscara `0x19` / denegado |

| Medida de simulación | Fuente 5 ns | Fuente 37 ns |
| --- | --- | --- |
| Primera muestra MGMT del reloj detenido → cierre observado | 680 ns | 640 ns |
| B de la última escritura → readback | 590 ns | 3.415 ns |
| Solicitudes periódicas completas, MIG 0 / MIG 1 | 82 / 75 | 193 / 188 |
| Duración nativa | 724,709 s | 1.228,327 s |
| Tiempo simulado | 117,474230 µs | 230,129230 µs |

Diez alteraciones deliberadas de evidencia por perfil fueron rechazadas por el revisor;
son controles del análisis, no mutaciones del RTL. Las lecturas periódicas
de ambos MIG permanecieron activas y todas las solicitudes retornaron.
La lectura final usa el fixture de memoria; no califica una lectura BAR.
La retención de una CPL demuestra una obligación pendiente, no vaciado global.

Hashes SHA-256 del perfil rápido:

- Archivo local/NFS: `05cdcca981508933543f7b16a385c0935cd60ebc33224252fea5020c92f6ec7a`.
- Revisión independiente: `e116e856d5cc59c351c42fd42bf43f744904036df648adf37abdcabf80aaa1e8`.
- Controles de evidencia: `7ace7b3b33e1188290c52e455e8a42ece4d05d03b23fe590aeabc2b18adcda2e`.

Hashes SHA-256 del perfil de 37 ns:

- Archivo local/NFS: `11f9f17dd15d83c72bf6126aa11cc404b1f24db525b68d0ab73dc99f3bd51009`.
- Revisión independiente: `d32c521a7a8cadc9d1fabf779a52f12ff40cea0538c9df2fd399a964825b1fab`.
- Controles de evidencia: `0ad290267a59fdc9c31a49677835c2d713f1e7562c8f2eb73ff137831c308f59`.

## Cambio del supervisor

El receptor anterior usa el mismo reset para el arranque y los fallos
durante operación. Al perder un reloj, su supervisor puede borrar las
colas aunque existan transacciones aceptadas. Las pausas breves de la
tanda anterior no califican ese comportamiento.

`ntrpu_dual_mig_retained_start` conserva las barreras del arranque:
relojes presentes, calibración de ambos MIG, intervalos de reset y
ausencia de respuestas antiguas. Al entrar por primera vez en operación,
retiene esa condición durante toda la configuración de la FPGA.

Un fallo posterior cierra admission y deja `restart_required` activo,
pero mantiene liberados los resets del receptor y sus CDC. La vuelta de
relojes o calibración, y `rearm`, no reabren la sesión perdida. El reloj de
gestión debe permanecer activo. El watchdog sigue siendo de 64 ciclos;
no se amplía ni se desactiva.

`ntrpu_dual_multibuffer_mig_retained` conecta este supervisor al receptor
real. Los datos y respuestas de comandos ya aceptados pueden continuar;
se bloquean comandos nuevos. `ntrpu_pcie_fragment_retained_mig_r2` retrasa
la apertura inicial del host hasta que ambas memorias admiten tráfico.

## Banco y diagnóstico conservado

La revisión `tools/ntrpu_board_retained_long_mig_r2.tcl` ejecuta 13 escenarios
del supervisor y después el banco con las secciones AXI, NTRD, receptor
y dos MIG reales. Conserva los 18 trabajos sanos y agrega un trabajo con
B/R del host y B de memoria retenidas.

El reloj fuente se detiene durante dos intervalos de 96 ciclos de gestión,
separados por un único flanco fuente. La invalidación debe proceder del
watchdog real; `invalidate_owner_source` permanece en cero. Se exigen
identidades y respuestas estables, readback completo, CPL retenida y
ausencia de retiro, publicación, reset y reapertura de admission.

La revisión R2 conserva todo el RTL, los tiempos y las comparaciones.
Separa la condición física de memoria de admission y permite exactamente
una lectura del fixture para el trabajo 18, con su B ya recibida, CPL
pendiente, identidad conservada, resets liberados y cierre global denegado.
Las escrituras con admission cerrada siguen prohibidas en este escenario.
No se habilita ninguna nueva lectura desde el host.

El primer intento, `20261006T121519Z-V4-ffn-supervised-retained-long-mig-fast-frozen`,
terminó con código 1 a 117.140.998 ps: `MEMORY_NOT_READY`. Su monitor
exigía admission abierta para la lectura independiente posterior al cierre.
Ese intento sigue clasificado como diagnóstico, sin aprobación, y conserva
fuentes, logs y archivo local/NFS:
SHA-256 `df7e2f483d392a2cfa66880966414e28f47af2ed4525ed68f60c17c5b48eee57`.

## Siguiente validación

Las síntesis del host y supervisor están terminadas, revisadas y respaldadas:
7.899 LUT/14.848 FF y 102 LUT/149 FF, respectivamente. La auditoría de
restricciones del supervisor identificó `TIMING-6/7`; un ensayo con alcance
exacto a 45 CLR los resuelve, manteniendo abierto un aviso CDC en el reset
externo. Los dos perfiles funcionales del netlist pasan sus 13 escenarios.
Véanse [métricas, evidencias y límites](RETAINED_RESET_TIMING_RESULTS.md).

Está en ejecución, **sin resultado aprobado**, el banco
`tb_ntrpu_retained_data_mig_r2.sv` con su Tcl y revisor independientes.
Conserva los 18 trabajos sanos y detiene el reloj tras aceptar el CMD
del trabajo 18, antes de aceptar su DATA. El receptor debe conservar
banco y descriptor durante el watchdog, tomar exactamente una DATA al
recibir un único flanco fuente y escribir después sus 64 bytes en el MIG.
El cierre permanece activo y la CPL final debe quedar retenida.
Las escrituras posteriores al cierre solo son admisibles para ese CMD
ya aceptado, con la misma identidad y todos los bytes comparados.
Su tanda es `20261006T140221Z-V4-ffn-supervised-retained-data-mig-r2-fast-frozen`.
El intento original está archivado como [diagnóstico del banco](RETAINED_DATA_PENDING.md).
Los casos CMD
todavía no aceptado, CPL ya visible y resultado publicado siguen pendientes.

Esta retención no demuestra recuperación después de reset externo del
MIG, autorización de reset global ni vaciado de colas protegidas XDMA.
Esas dependencias, los pendientes de restricciones del supervisor y los criterios
de placa de [VIVADO_QUEUE.md](VIVADO_QUEUE.md) siguen abiertos.
