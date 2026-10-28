# Candidato fisico del nodo VU47P

Seguimiento: [variante V3 con reset compartido](GUARD_V3.md). Los resultados
de este documento corresponden al V2 congelado y conservan sus hallazgos.

## Decision de arquitectura

El 2026-10-11 el usuario priorizo el nodo **VU47P con HBM**, manteniendo
separado el nodo DDR. Este paquete concreta el PCIe de esa variante y no
modifica el plan DDR ni las simulaciones del hilo principal. La integracion
HBM/PCIe conjunta sigue pendiente: este checkpoint contiene PCIe, BRAM de
prueba, guard V2 y cliente, sin controlador HBM.

## Mapa del dispositivo comprobado

Preflight `physical-preflight-20261011T072930Z`, Vivado 2026.1,
`xcvu47p-fsvh2892-2-e`, desde el DCP auditado de
[GUARD_INTEGRATION.md](GUARD_INTEGRATION.md). Exporta 2.892 bolas, los pines
de cada canal GT y la conectividad real del top. No cambia el DCP ni sus
restricciones activas. El generador cruza funcion, polaridad, banco, pareja
y bonding; no infiere bolas a partir de una fotografia de tarjeta.

Enlace Gen3 x16; bloque `PCIE4CE4_X0Y0`. TX/RX indican la perspectiva del
FPGA. Estas bolas corresponden al circuito generado: faltan conector,
correspondencia en el PCB, SI/PI y revision electrica. No es un pinout liberado.

| Lane RTL | Sitio GTYE4_CHANNEL | Banco | TX P/N | RX P/N |
| ---: | --- | ---: | --- | --- |
| 0 | `GTYE4_CHANNEL_X0Y15` | 127 | AL44/AL45 | AL53/AL54 |
| 1 | `GTYE4_CHANNEL_X0Y14` | 127 | AM46/AM47 | AM51/AM52 |
| 2 | `GTYE4_CHANNEL_X0Y13` | 127 | AN44/AN45 | AN49/AN50 |
| 3 | `GTYE4_CHANNEL_X0Y12` | 127 | AP46/AP47 | AN53/AN54 |
| 4 | `GTYE4_CHANNEL_X0Y11` | 126 | AR44/AR45 | AP51/AP52 |
| 5 | `GTYE4_CHANNEL_X0Y10` | 126 | AR48/AR49 | AR53/AR54 |
| 6 | `GTYE4_CHANNEL_X0Y9` | 126 | AT46/AT47 | AT51/AT52 |
| 7 | `GTYE4_CHANNEL_X0Y8` | 126 | AU48/AU49 | AU53/AU54 |
| 8 | `GTYE4_CHANNEL_X0Y7` | 125 | AU44/AU45 | AV51/AV52 |
| 9 | `GTYE4_CHANNEL_X0Y6` | 125 | AV46/AV47 | AW49/AW50 |
| 10 | `GTYE4_CHANNEL_X0Y5` | 125 | AW44/AW45 | AW53/AW54 |
| 11 | `GTYE4_CHANNEL_X0Y4` | 125 | AY46/AY47 | AY51/AY52 |
| 12 | `GTYE4_CHANNEL_X0Y3` | 124 | BA44/BA45 | BA49/BA50 |
| 13 | `GTYE4_CHANNEL_X0Y2` | 124 | BB46/BB47 | BA53/BA54 |
| 14 | `GTYE4_CHANNEL_X0Y1` | 124 | BC44/BC45 | BB51/BB52 |
| 15 | `GTYE4_CHANNEL_X0Y0` | 124 | BC48/BC49 | BC53/BC54 |

`physical-contract-20261011/pcie_lane_map.csv` conserva los 64 puertos,
funciones de bola, polaridad, pareja diferencial y nets propuestas para ECAD.
El pin del conector queda vacio: no se presupone un adaptador comercial.

## Relojes y reset

| Senal | Candidato / contrato | Pendiente |
| --- | --- | --- |
| PCIe REFCLK 100 MHz | AR40/AR41, MGTREFCLK0 P/N del banco 125 | Fuente host, jitter, acoplamiento y trazado |
| PERSTn | BF41, PERSTN0 banco 65; LVCMOS18 para el ensayo | Adaptacion del nivel del host y tension compartida del banco |
| AXI 250 MHz | BUFG_GT nativo de AMD | Preservar su origen y comportamiento ante perdida del enlace |
| Gestion 100 MHz | BUFGCE, entrada OOC independiente | Oscilador, bola, buffer de entrada y presupuesto de jitter |
| Configuracion fria | `configuration_rst_n`, independiente de PERSTn | Supervisor/origen, pin y anchos minimos |

AR40/AR41 y BF41 coinciden con las funciones consultadas por el ejemplo AMD,
resueltas en este encapsulado. No se importa su XDC completo ni sus waivers
ni excepciones generales. Reservar esos recursos en el futuro top HBM;
la habilitacion de 1,8 V es una hipotesis del candidato, no autorizacion para
conectar directamente una senal host de nivel desconocido.

## CDC del circuito completo

La revision completa del preflight, con `-all_checks_per_endpoint -no_waiver`, encontro
164 CDC-3 informativos, 30 CDC-9 informativos y **288 CDC-15 de advertencia**,
sin CDC criticos reportados. Las 288 rutas son exactamente 18 bits en cada
uno de los 16 contadores de calibracion CPLL del proveedor: `testclk_cnt_reg`
a `freq_cnt_o_reg`. Se conservan las advertencias y sus excepciones originales;
identificarlas no prueba su protocolo ni autoriza nuevas exenciones.

En el preflight, el reporte acotado al cliente conserva tres CDC-3 y ocho CDC-9. Las entradas
externas sin input delay no estan cubiertas por ese conteo. TIMING-9 global
se mantiene visible; la correlacion con los contadores es evidencia de
localizacion, no una equivalencia formal entre reglas.

## Ensayo de implementacion

`implement_guard_candidate.tcl` agrega solo los tres PACKAGE_PIN anteriores
y LVCMOS18 para PERSTn en una copia privada. Ejecuta opt/place/route una vez,
guarda checkpoints por etapa y conserva todos los reportes. No agrega
retardos ficticios de E/S, origen ficticio del reloj, clock groups generales,
waivers ni reducciones de severidad. Limite: dos CPU, nice 10 y 1.200 s.

La gestion y las interfaces de fabric permanecen como fronteras OOC. Por
eso un eventual resultado favorable de rutas internas no cierra placa,
HBM, puertos externos, recovery/removal ni MTBF con actividad real. No se
genera bitstream.

## Resultado y limites de aceptacion

Corrida `physical-candidate-20261011T073351Z`: 824,5 s, sin timeout,
checkpoint de entrada conservado. Las 66.564 nets ruteables quedaron completas,
sin errores de ruteo. El resumen STA bajo las restricciones existentes es:

| Etapa | Setup WNS | Hold WHS | Pulse WPWS | Endpoints que fallan setup / hold |
| --- | ---: | ---: | ---: | ---: |
| Colocacion | +0,513 ns | -0,625 ns | 0,000 ns | 0 / 5.247 |
| Ruteo | +0,331 ns | +0,010 ns | 0,000 ns | 0 / 0 |

**El candidato no esta aceptado para integrar o fabricar.** Los margenes no
resuelven los siguientes hallazgos:

- El CDC ruteado agrega **dos CDC-11 criticos**: `user_reset_reg` alimenta
  `availability[1]` y `[2]`, que sincronizan el mismo reset por separado en
  gestion. El probe confirma el mismo driver en sus seis CLR. Permanecen
  tres CDC-3 y ocho CDC-9 informativos. No se agregaron waivers.
- TIMING-16 conserva cuatro avisos de -1,082/-1,084 ns. Una reapertura
  independiente consulta exactamente CORECLK -> `s_axi_ctl_rdata_reg`
  bits 9/11/21/26: +0,916/+0,918 ns, cuatro rutas cronometradas, ninguna
  ignorada. Todas atraviesan CFGMGMTREADWRITEDONE con `Setup -start 2`
  nativo, presente antes de implementar. La diferencia numerica de 2 ns
  sugiere una diferencia de tratamiento del multicycle; no demuestra que
  el aviso sea descartable. Se guardan ambos reportes.
- Falta `HD.CLK_SRC` en `mgmt_clk_in`: Vivado advierte que su timing puede
  ser inexacto y que no puede comprobar la colocacion aguas arriba.
  Siguen sin presupuesto 1.589 entradas y 511 salidas de fabric.
- DRC conserva RTSTAT-10 por 201 nets sin cargas ruteables; metodologia
  conserva siete LUTAR-1 del proveedor, TIMING-9, TIMING-18 y XDCB-5.

La auditoria `routed-audit-20261011T075047Z` verifica los 42 ASYNC_REG
colocados y 56 chequeos entre etapas: minimo setup +3,524 ns y hold +0,040 ns.
El placer inserto un BUFGCE en `cold_a_reg[2]/Q`, CE conectado a VCC,
CE_TYPE=ASYNC, hacia 1.451 pines de reset y ningun pin de reloj. Sus
propiedades de inversion aparecen sin valor explicito; se conserva ese
dato y el valor por defecto cero documentado en
[UG974](https://docs.amd.com/r/en-US/ug974-vivado-ultrascale-libraries/BUFGCE).
La gestion usa BUFGCE_X0Y0 y AXI el BUFG_GT_X0Y88 nativo.
Esta comprobacion estructural no reemplaza una simulacion posterior al ruteo.

El probe `routed-findings-20261011T075322Z` preserva el DCP y las
restricciones, reproduce el resumen STA y localiza los avisos anteriores.
Los revisores separan evidencia exportada de aceptacion: el modo normal
rechaza este candidato; `--inventory-only` registra expresamente los bloqueos.

Antes de unir HBM, preparar una variante de integracion que sincronice una
sola vez cada fuente fisica de reset y distribuya su resultado. Mantener
la independencia cuando las fuentes sean distintas, y volver a probar
caida de enlace/reloj, liberacion escalonada y obligaciones DMA pendientes.
No modificar silenciosamente el guard V2 congelado. La recomendacion de
evitar sincronizaciones duplicadas esta en
[UG906](https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/Asynchronous-Reset-Synchronizer).

## Siguiente union con HBM

El lector previo usa 200 MHz y la interfaz HBM un periodo de 2.222 ps;
el cliente de este paquete usa AXI nativo a 250 MHz. La union necesita
transportar solicitudes, payload de 512 bits y finalizaciones con identidad
completa a traves de esos dominios, manteniendo las obligaciones pendientes
cuando PCIe pierda su reloj. La aprobacion de la topologia no cambia esos
contratos ni convierte un ACK modelado en confirmacion remota fisica.

El [ensayo nativo previo](../../../docs/fpga/HBM_SOURCE_NATIVE.md) utiliza
XDMA Stream; este candidato es AXI Bridge con escritor y mapper propios.
Su adaptador C2H no es intercambiable con `src_req_*`, `src_*` y `src_cpl_*`
del cliente actual. Reutilizar el lector exige un adaptador especifico y una
prueba nativa conjunta antes de atribuir resultados HBM a este PCIe.

## Reproduccion y controles

```bash
python3 -B run_guard_physical_preflight.py NUEVA_SALIDA \
  --checkpoint AUDITORIA/output/native_audited.dcp \
  --settings VIVADO_SETTINGS --execute-native
python3 -B build_guard_physical_contract.py PREFLIGHT_RESTAURADO NUEVO_CONTRATO
python3 -B check_guard_physical_contract.py PREFLIGHT_RESTAURADO
python3 -B run_guard_physical_candidate.py NUEVA_SALIDA \
  --checkpoint AUDITORIA/output/native_audited.dcp \
  --settings VIVADO_SETTINGS --execute-native
python3 -B run_guard_routed_audit.py NUEVA_AUDITORIA \
  --checkpoint CANDIDATO/output/routed.dcp \
  --settings VIVADO_SETTINGS --execute-native
python3 -B run_guard_routed_probe.py NUEVO_PROBE \
  --checkpoint CANDIDATO/output/routed.dcp \
  --settings VIVADO_SETTINGS --execute-native
python3 -B review_guard_physical_bundle.py BUNDLE_RESTAURADO
```

Once controles negativos rechazan errores de lane, GT, polaridad, banco,
bola no conectada, pareja diferencial y cobertura/clasificacion CDC. El
manifiesto y la igualdad de restricciones comprueban el preflight conservado.
El inventario amplio de propiedades genero avisos de lectura de propiedades
HD no aplicables a OOC; no se aplicaron esas propiedades.

## Respaldo local

Base: `D:/AI/artifacts/ntrpu-axi-bridge-guard-integration/`. Cada tanda
conserva el tar.gz original, SHA-256, recibo `BACKUP.json` y copia restaurada
con manifiesto verificado. El paquete `20261011-guard-physical-bundle.zip`
reune preflight, contrato, checkpoints por etapa, auditoria, probe,
fuentes y revisores; incluye el DCP de entrada para verificar su identidad.
El recibo `20261011-physical-backup-receipt.json` registra hash y restauracion.
No incluye instaladores, credenciales ni modelos.
`PHYSICAL_PROCESS_COMPLETION.json` comprueba que terminaron los ocho PID
propios de estas cuatro tandas. Se conserva el trabajo del hilo principal.

Consulta de recursos: [get_sites](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/get_sites)
y [get_package_pins](https://docs.amd.com/r/2023.1-English/ug835-vivado-tcl-commands/get_package_pins).
Alcance CDC: [reglas y precedencia](https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/Understanding-the-Clock-Domain-Crossings-Report-Rules).
