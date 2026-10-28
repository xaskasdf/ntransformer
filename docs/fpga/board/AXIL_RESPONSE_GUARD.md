# Contención de respuestas AXI-Lite antes del Shutdown Manager

## Cambio y alcance

La sección R3 incorpora `ntrpu_axil_response_guard` entre las respuestas del
Clock Converter AMD y el Shutdown Manager. Conserva admisión R2 y añade
certificado R5. El fallo anterior de contador vacío permanece registrado en
`20261005T194744Z-V4-ffn-supervised-shutdown-section-lifetime`; las revisiones
y trazas selladas no se modifican.

El protector observa continuamente los handshakes AW/W/AR del lado host.
Una B requiere ambas mitades anteriores; una R requiere una AR anterior.
Las respuestas sin crédito quedan con READY=0 debajo del protector y no
producen VALID hacia el manager. Cada canal retiene su cuarentena: solicitudes
posteriores no pueden convertir la respuesta huérfana en una respuesta válida.

Un fallo en B no corta una R válida retenida, ni viceversa. Los contadores
son de ocho bits, saturan y denuncian desbordamiento. El límite de admisión
de 32 sigue vigente. No hay descarte, reapertura ni reset operativo automático.
AXI-Lite no permite distinguir una respuesta incorrecta que ya tenga crédito:
esta protección acredita causalidad de solicitudes, no identidad de datos.

## Alarmas y clocks

`guard_fault_sources={inventory_fault, orphan_r, orphan_b}` expone tres
registros independientes. R5 sincroniza cada uno con tres etapas de gestión
y combina después. `fault_mgmt[3]` representa el protector; los bits 0/1/2
conservan admisión/H/D. Una alarma bloquea nuevas pruebas y revoca los
certificados visibles sin truncar un `src_send` XPM pendiente.

Si D se detiene tras aceptar la respuesta inyectada, su alarma puede alcanzar
gestión mientras la respuesta permanece dentro del CDC. La alarma del
protector aparece cuando esa respuesta llega efectivamente al host. No se
exige que ambas alarmas lleguen simultáneamente.

## Evidencia funcional

Vivado 2026.1, build 6511674, `xcvu5p-flva2104-2-i`. Relojes de gestión/host
de 10/7 ns y dispositivo de 5/37 ns; variantes con origen activo/detenido.

| Tanda del 2026-10-05 UTC | Resultado revisado |
| --- | --- |
| `200930Z-V4-ffn-supervised-shutdown-section-lifetime-r2` | 16 escenarios con IP AMD real: fallo anterior a prueba, mailbox con host detenido, SLVERR local retenido y cierre por fallo D |
| `201350Z-V4-ffn-supervised-shutdown-section-post-r2` | 12 escenarios; conservación de 387 escrituras/131 lecturas upstream por combinación de clocks, más revocación posterior |
| `201810Z-V4-ffn-supervised-response-guard` | Cinco casos aislados: 32 créditos, respuestas concurrentes, cuarentena B/R, conservación del canal sano y overflow a 255 |

El banco de vida útil mide reporte a gestión en **20,8–28,8 ns** y cierre
por fallo D en **41,6–54,6 ns**. La traza distingue la oferta huérfana debajo
del protector y los handshakes protegidos: no hay B adicional hacia el manager.
Las inyecciones directas G/H de otros casos prueban sus observadores y no
acreditan protección frente a una inyección situada después del protector.

La revisión inicial de vida útil esperaba prematuramente el bit del protector
con D detenido. Una adenda revisa las mismas trazas con la secuencia causal
correcta; no se repite simulación ni se cambia RTL. Rechaza 36 corrupciones
de evidencia. La regresión posterior rechaza otras 36 y el banco aislado seis.

## Recursos, restricciones y controles negativos

`20261005T201943Z-V4-ffn-supervised-shutdown-section-synth-r3` termina en
90,362 s. La composición usa **802 LUT y 1.762 FF**, frente a 754/1.725
de R2: incremento de 48 LUT/37 FF. No contiene RAM, DSP, latches ni cajas
negras. El informe de utilización y el conteo de primitivas son métricas
distintas: se conservan también 887 celdas LUT lógicas.

La revisión reconoce **14 CDC-3 de tres etapas**, sin CDC Warning/Critical,
incluidos los tres registros reales del protector. Conserva 14 false paths
del proveedor a primera etapa y tres límites datapath-only de 10/7/7 ns,
con sus waivers XPM acotados. No se añaden excepciones generales.

No hay endpoints internos sin restricción ni registros sin reloj. Persisten
161 entradas y 276 salidas sin retardos de entorno OOC. WNS/WHS/WPWS de
4,716/0,077/3,225 ns son estimaciones de síntesis, no timing físico aprobado.
Los diez avisos revisados corresponden a XPM y clocks OOC. Este top incluye
admisión, protector y certificación; excluye manager/CDC AMD y endpoint.

`20261005T202301Z-V4-ffn-supervised-guard-lifetime-mutants` termina en
96,874 s. Dos copias privadas alteradas fallan ante el banco positivo de vida
útil sin modificarlo: omitir la fuente de admisión 0 activa la aserción de
reporte a los 2.950 ns; cancelar `src_send` por retirada de petición activa
la aserción de respuesta XPM a los 2.885 ns. Se verifican las modificaciones
exactas y los hashes compilados. Estos fallos esperados acreditan sensibilidad
del banco, no funcionamiento de las copias alteradas.

## Respaldo

Archivos en `artifacts/ntrpu-board-vivado/<ID>/`, con prefijo completo
`20261005T`, y réplica en NFS. Cada `VERIFIED.json` verifica archivo y miembros.

| Tanda | SHA-256 del archivo |
| --- | --- |
| `200930Z` | `1344713db1fa224d9da04edf113461554ebc4cb42f2d6c9a728294fb40ed22ce` |
| `201350Z` | `c261849b6e8bc019c0cd4210435ff1304c53786c599740561e6b740ef109bca7` |
| `201810Z` | `512897251ee7194024f8a94a443542eefadd173d49ba87bb3625117f0a4df16f` |
| `201943Z` | `1e2ff41dc3e59a4d7976935af4fd67fbb0e023fcef239c49ac468bbcf42e659f` |
| `202301Z` | `98742a978a18ddb65c45c959573c733da706d0b32a2288ae5451c866faa25440` |

Adenda de vida útil: `a604f0bdbcbc58c59e7da278c2aa281b7889b858dd800ea9fc69aba8a09d641a`.
Adenda de síntesis: `ce98d8ecf050983730cb67dbc5e93606074d66cdb935c81c389a3e209fccbe1d`.

## Continuación y trabajo restante

Las [alarmas exclusivas y controles específicos](AXIL_GUARD_FAULTS.md)
quedaron comprobados en 24 escenarios adicionales y tres mutaciones RTL,
incluidos host detenido y R huérfana por CDC real. La cola R12 cierra los
criterios de fallos de esta sección y pasa al [inventario AXI4](AXI4_DRAIN_PLAN.md),
seguido por drenaje completo XDMA y coordinación de reset.

Estos resultados no cierran el endpoint nativo, reutilización NTRD, timing
físico, DIMM exacto, placa de 512 GiB ni inferencia Llama completa.
