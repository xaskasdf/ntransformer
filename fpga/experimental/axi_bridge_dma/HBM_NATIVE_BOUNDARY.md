# LOCKED y frontera física del nodo VU47P/HBM

## Objetivo de esta tanda

Continuar [HBM_NATIVE_CONSTRAINTS.md](HBM_NATIVE_CONSTRAINTS.md) agregando un
presupuesto de retardo al estado bruto del MMCM e inventariando las interfaces
que impiden cerrar el top de tarjeta. Se conserva el núcleo, los IP, el RTL y
el nodo DDR separado. No se ejecuta placement ni routing.

El donante es `constraints-apply-20261011T222937Z/output/constrained.dcp`, SHA-256
`629bae0ba8bddd48eec2fbdf78224a9f8ac0c149a32299d15bbb82c77f6bf313`.

## Presupuesto de LOCKED

La única restricción nueva es `set_max_delay -datapath_only 10.000` desde
`clock_generator/inst/mmcme4_adv_inst/LOCKED` hasta
`availability/lock_sync_reg[0]/D`. Es una decisión de diseño: un período del
reloj de gestión de 100 MHz para el camino al primer registro. No representa
la latencia del MMCM para detectar pérdida de lock, el tiempo de adquisición
de lock ni la latencia completa del supervisor. No garantiza capturar pulsos
arbitrariamente cortos ni acredita MTBF.

Los tres registros del sincronizador y sus dos conexiones internas mantienen
su análisis síncrono. La auditoría conserva también los 1.503 caminos ya
acotados, los 147 `ASYNC_REG`, las 90 conexiones entre etapas y las restricciones
previas. Exporta el camino antes/después y exige que empiece en `LOCKED`, con
destino en gestión y sin asignarle un reloj de lanzamiento ficticio.

El modelo del MMCM expone arcos `RST→LOCKED` y `PWRDWN→LOCKED`. El primer intento
se detuvo al asumir que no existían arcos de entrada; se conserva como fallido.
La revisión posterior exige esos dos arcos concretos y compara todas sus
propiedades antes, después y tras reabrir. Ambos arcos son `Reg Set/Clr` y ya
tienen `IS_DISABLED=1`, `IS_USER_DISABLED=0` antes de aplicar el límite.

Vivado emite **`CRITICAL WARNING [Constraints 18-515]`** al forzar `LOCKED` como
inicio. Esta tanda conserva y revisa esa advertencia concreta: acepta únicamente
ese pin, sin lógica intermedia y con un único consumidor interno, exige que los
arcos ya estuvieran deshabilitados por el modelo y que sus propiedades no cambien,
y compara los demás caminos y relojes con el donante. No degrada la severidad ni
crea un waiver. La propuesta inicial de exigir cero advertencias críticas se
reemplaza por esta revisión explícita, porque el origen es un estado asíncrono
sin reloj de lanzamiento. Cualquier otra advertencia crítica provoca rechazo.
No se atribuye a este corte una garantía temporal sobre el reset del MMCM o
su salida de estado externa. Esta precaución sigue la descripción de
[segmentación de caminos de AMD](https://docs.amd.com/r/en-US/ug903-vivado-using-constraints/Path-Segmentation).

`LOCKED` no sustituye la secuencia de recuperación: la descripción de
[MMCME4_ADV en UG974](https://docs.amd.com/r/en-US/ug974-vivado-ultrascale-libraries/MMCME4_ADV)
exige reset del MMCM tras perder lock. El banco anterior ejercita estados de
disponibilidad; su relock simulado no demuestra recuperación del silicio.
El supervisor conserva el fallo hasta un nuevo ciclo de configuración.

## Interfaces que deben cerrarse antes del top de tarjeta

El top de integración expone 1.287 bits de puerto en 80 señales/buses. El
inventario registra dirección, referencias de reloj, pines asignados y los
registros alcanzados. Se clasifican 64 bits PCIe serie, seis de referencias,
cuatro de secuencia externa, 1.080 de transacciones y 133 de observación.
Estos puertos lógicos no constituyen un pinout aprobado.

Los 1.080 bits de transacciones sólo alcanzan registros del reloj PCIe de
250 MHz. Los 133 de observación proceden de PCIe, gestión, lector o memoria;
requieren tratamiento por dominio. El informe de timing atribuye 800 entradas
y 279 salidas sin delay a transacciones, dos entradas a secuencia externa y
128 salidas a observación: total 802/407. Quedan además 1.154 destinos internos
sin timing por reloj constante, todos dentro de los bloques XSDB de HBM.

Los 64 puertos serie ya muestran pines de encapsulado derivados de los canales
GT del perfil nativo; los restantes 1.223 puertos no tienen `PACKAGE_PIN`.
Vivado informa `LVCMOS18` en 1.221 puertos y ningún valor en los 66 restantes.
Se registra ese estado del modelo, sin considerarlo una selección eléctrica
aprobada: el XDC preservado no contiene asignaciones explícitas de `IOSTANDARD`.

| Familia | Decisión para la integración | Evidencia pendiente |
|---|---|---|
| PCIe x16 | Mantener los puertos serie del IP | Conector, pares, canales GT, referencia y esquema de reset |
| Referencias de reloj | Mantener separados los contratos PCIe, gestión, fabric y HBM | Osciladores, distribución, jitter, niveles eléctricos y pines |
| Reset/configuración | Definir el secuenciador y su observación de lock | Arranque, pérdida de reloj, reset del MMCM y autorización de nueva época |
| `cfg_*`, `cmd_*`, `ack_*`, `reject_*`, `result_*` | Integrar un bloque de control y colas dentro de la FPGA | Mapa de registros, publicación atómica y backpressure |
| Estados, temperaturas y paridad | Incorporar consumidores internos y lectura coherente | Dominio de cada señal, captura/snapshot y política térmica/paridad |

La clasificación es una propuesta de arquitectura para el top de tarjeta;
esta tanda no implementa el bloque de control ni asigna sus registros a un BAR.
Los vectores de estado que cruzan relojes necesitan un protocolo coherente de
captura; sincronizar cada bit independientemente no demuestra coherencia.

Los presupuestos de entrada/salida dependen del dispositivo externo y del PCB.
Se mantienen pendientes hasta disponer de ese contrato; no se rellenan con
ceros ni se ocultan mediante cortes globales. Para interfaces síncronas físicas
se deben derivar `set_input_delay`/`set_output_delay` mínimos y máximos de los
tiempos del componente externo y del trazado, según
[AMD UG903](https://docs.amd.com/r/en-US/ug903-vivado-using-constraints/Constraining-Input-or-Output-Logic).

## Reproducción y evidencia

Preparar un directorio nuevo con los dos helpers de la tanda anterior,
`audit_hbm_native_boundary.tcl` y `run_hbm_native_join.py`. Copiar
`apply_hbm_native_boundary.tcl` como `inspect_hbm_native_join.tcl` y ejecutar
el runner con `--stage inspect --checkpoint DONOR.dcp`. Repetir con el entrypoint
`reopen_hbm_native_boundary.tcl` sobre el nuevo `output/bounded.dcp`.

```bash
python -B review_hbm_native_boundary.py BASELINE_REOPEN_RESTORED \
  APPLY_RESTORED REOPEN_RESTORED FROZEN_NATIVE_TOP.sv --self-test
```

Cada proceso usa dos CPU, prioridad reducida y límite de 1.200 segundos en el
namespace privado del hilo lateral. Los resultados y fallos se respaldan por
manifiesto en `D:/AI/artifacts/ntrpu-axi-bridge-hbm-native-boundary/`.
`REVIEW.json` contiene el inventario por señal y los límites que siguen abiertos;
`FINAL_BACKUP.json` identifica el paquete restaurado y verificado.

## Resultado de la revisión

`boundary-apply-20261011T225731Z` terminó en 160,8 segundos y
`boundary-reopen-20261011T230251Z` en 152,1 segundos. El nuevo DCP tiene SHA-256
`d1f97e808bf08cf7e8a1e8298621f9def9ab46836c388c0817caa0cc1f3bdf9f`.
El revisor devuelve `NATIVE_LOCK_BOUND_AND_BOUNDARY_REVIEWED` y rechaza
12 controles negativos. Las tablas anteriores, el inventario de puertos,
las propiedades de los arcos MMCM y las restricciones se conservan al reabrir.

El requisito medido del camino `LOCKED→lock_sync[0]/D` es 10 ns y su margen
estimado es +9,835 ns. Es evidencia posterior a síntesis; no acredita retardo
enrutado. La advertencia crítica `Constraints 18-515` permanece visible y está
limitada al caso explicado arriba. No se reclama un informe sin advertencias.

Los recuentos CDC permanecen en 190 `CDC-3`, nueve `CDC-6`, 36 `CDC-9` y
1.746 `CDC-15`, sin `CDC-10`. Los presupuestos externos siguen abiertos y el
pinout sigue sin aprobar. Consultar `IOSTANDARD` en los terminales GT produce
advertencias del instrumento; se preservan, al igual que los avisos de salidas
constantes sin fanin. La comparación XDC prueba que no se añadieron asignaciones
de pines ni estándares eléctricos.

La próxima integración debe sustituir la interfaz de prueba de transacciones
por control interno en el dominio PCIe, y definir snapshots coherentes de
telemetría por dominio. Después se podrá cerrar la interfaz del secuenciador y
seleccionar componentes de reloj y presupuestos del PCB con un contrato concreto.
