# Diagnóstico de potencia DDR: ejecución del traspaso R1

Fecha: 2026-10-15. Resultado nativo: **diagnóstico exportado**, confianza
**baja**. No califica potencia de placa ni el DIMM exacto.
Continúa [POWER_PROBE_HANDOFF_R1.md](POWER_PROBE_HANDOFF_R1.md).

**Respaldo completo y restaurado; slot devuelto a las 21:07:39 UTC.**
Se reanudó el mismo coordinador sucesor PID 23708. La auditoría de
sus seis paquetes y 316 archivos vinculados confirmó integridad antes
de liberarlo.

## Ejecución e identidad

Tanda válida: `20261015T205842Z-V8-ddr-power-probe-r2`, Vivado 2026.1
build 6511674; 20:58:46–21:00:24 UTC, 98 segundos, salida 0 y marcador
`NTRPU_POWER_DIAGNOSTIC_EXPORTED`. Dos CPU; límite operativo de 20 minutos.
Abrió el DCP original, sin cambios de actividad, térmicos ni implementación.

Identidad observada: `NAME=checkpoint_route_attempt`, `TOP=example_top`,
`PART=xcvu5p-flva2104-2-i`, 154 puertos; informe en estado `routed`.
Un canal DDR4 genérico con dos slots; sin cuatro canales, PCIe ni cómputo
NTRPU integrado. `board_power_qualified=false`, `exact_dimm_qualified=false`.

El Tcl R1 original permanece intacto. Su comparación de `NAME` con el
módulo rechazó el checkpoint. El [Tcl sucesor R2](../../../tools/ntrpu_board_ddr_power_probe_r2.tcl)
registra NAME/TOP/parte/puertos y comprueba `TOP`, conservando las demás
exigencias. No se repitió síntesis ni ruteo.

## Estimaciones y condiciones heredadas

| Magnitud del informe | Valor |
| --- | ---: |
| Total on-chip | 3,015 W |
| Dinámica | 1,349 W |
| Estática del dispositivo | 1,666 W |
| Ambiente / placa | 25 / 25 °C |
| Temperatura de unión estimada | 26,7 °C |
| TJA efectiva / ThetaSA | 0,6 / 0,7 °C/W |
| Flujo de aire | 250 LFM |
| Máximo ambiente calculado por la herramienta | 98,3 °C |

Son supuestos del informe, no condiciones físicas verificadas: disipador
`medium`, PCB `medium (10"x10")`, 12–15 capas, proceso `typical`, grado
`industrial`. No usar el máximo ambiente calculado como límite de la placa.

Corrientes transcritas del XML; el informe de texto las redondea a tres
decimales. `REVIEW.json` conserva los 16 rails, incluyendo ceros reportados.

| Rail | Tensión (V) | Total (A) | Dinámica (A) | Estática (A) |
| --- | ---: | ---: | ---: | ---: |
| Vccint | 0,850 | 0,978074 | 0,429814 | 0,548260 |
| Vccint_io | 0,850 | 0,483892 | 0,308707 | 0,175186 |
| Vccbram | 0,850 | 0,010643 | 0,000340 | 0,010302 |
| Vccaux | 1,800 | 0,600990 | 0,161123 | 0,439867 |
| Vccaux_io | 1,800 | 0,279933 | 0,156719 | 0,123214 |
| Vcco12 | 1,200 | 0,123650 | 0,123650 | 0,000000 |
| Vccadc | 1,800 | 0,016000 | 0,000000 | 0,016000 |

Arranque, presupuestos, márgenes, cobertura exacta de actividad y duración
de ventana simulada permanecen `null`. No se infieren de los ceros o de
los valores redondeados. Se conservan los totales nativos: el resumen muestra
3,015 W y la tabla de componentes 3,014 W.

## Actividad, advertencias y alcance

Sin SAIF; propagación vectorless con actividad heredada/default. Vivado
informa más del 75 % de entradas sin actividad especificada y menos del
25 % de nodos internos con especificación de usuario. La confianza general
es `Low`; esos umbrales no constituyen cobertura exacta medida.

`Power 33-332` advierte resets de alto fanout inferidos activos demasiado
tiempo; el advisory contiene 320 filas de reset excesivo. No se forzaron
resets ni enables para mejorar la cifra. Se conservan dos avisos de
restricciones DRIVE/SLEW y avisos de modelos de potencia ausentes para
`dci`, `startup`, `xiphy_riu_or` y `genVref.u_hpio_vref`. Cero errores nativos
en la exportación R2; todas las advertencias constan en la revisión.

Falta actividad de una simulación correspondiente al `example_top`
implementado, con calibración y ventanas representativas. El banco de
capacidad existente tiene otra jerarquía. No multiplicar estos vatios por
cuatro, sumarlos a HBM ni dimensionar reguladores/refrigeración con esta
tanda. Consumo total del DIMM, SPD/RCD y placa integrada siguen pendientes.

## Evidencia y continuidad de cola

El escritor residual `20261015T204443Z` terminó, pasó revisión y se respaldó
antes de reservar el slot. Sólo se retuvo su sucesor todavía en espera,
PID 23708; no se interrumpió ninguna validación. Los seis sucesores conservaron
estado, paquetes y dependencias. La reserva pertenece al hilo principal.

Se conservaron también los intentos previos:

| Tanda | Resultado |
| --- | --- |
| `20261015T205133Z-V8-ddr-power-probe-r1` | Lanzador incompatible con Python 3.10; Vivado no inició. Código de salida no capturado, registrado como `null`. |
| `20261015T205454Z-V8-ddr-power-probe-r1-runner-r2` | Tcl R1 intacto; salida 1 por NAME/TOP, sin informes de potencia. |
| `20261015T205842Z-V8-ddr-power-probe-r2` | Exportación correcta con Tcl R2 y lanzador compatible. |

La revisión independiente comprueba hashes antes/después, identidad,
marcador, salida, cuatro informes y XML analizable. Rechaza 13 alteraciones
de evidencia y tres inconsistencias numéricas rehasheadas. Compara 99 filas
XML/texto respetando su precisión de presentación; conserva valores nativos.

Entradas: DCP SHA-256
`161d111ce97d471d461328e615c96d32232b6e6c101673de210cb75b282f205c`;
Tcl R2 `8bbc24239bebee597a84b094333f1960b6e23790ca285e63aa2f2d92c750be57`.
El paquete por tanda contiene DCP, Tcl, lanzador, logs, informes, revisión
y registros del turno. Su certificado está en
`artifacts/ntrpu-board-vivado/<tanda>/VERIFIED.json`.
La liberación exige archivo idéntico en D:/NFS y restauración completa
verificada; el registro operativo es
`build/cloud-setup/20261002-board/power-probe-r1-main-20261015T204728Z/`.

Archivo válido: 27.427.690 bytes, 44 miembros de datos restaurados.
SHA-256 `733dc47a231e5cede530a1b2bc531f85fe55d11e8fffd984733dcdca606cfbbc`.
Revisión: `a056e2fce0054f712840f0caf5e29033f46b536bcd4e8a24466ae6c7c6976f8d`.
Copias idénticas en NFS y
`D:/AI/ntransformer/artifacts/ntrpu-board-vivado/backups/20261015T205842Z-V8-ddr-power-probe-r2/`;
el subdirectorio `restored/` conserva la restauración verificada.

Intento del lanzador: TAR SHA-256
`28c88c62dcbe984f608edabca63bc76e042fbbaed033aef26a61c1332ff975de`;
intento nativo R1: `8de2fbc9b1a935c0894ba5e20e968c1aa8ea226d4542001789c9e11813a2f69f`.
Ambos conservan certificados y restauración independiente.
