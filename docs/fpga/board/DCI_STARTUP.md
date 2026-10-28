# Arranque DCI y reset de memoria

Estado 2026-10-02: secuenciador probado en XSIM, integrado y colocado; timing final y revisión eléctrica pendientes. Elimina los 48 avisos `DCIRST-1` del candidato DDR en bancos 64–67. Es independiente del fallo de recuperación del modelo DDR tras reset.

## Circuito

`fpga/rtl/ntrpu_dci_startup.sv` implementa una calibración DCI global por configuración FPGA. El wrapper `ntrpu_board_startup` conecta una única primitiva `STARTUPE3` y una `DCIRESET`. AMD requiere pulsar RST y esperar LOCKED antes de usar las E/S de configuración compartidas con DCI: [UG570](https://docs.amd.com/r/en-US/ug570-ultrascale-configuration/Start-up-Sequence-Step-8), [UG974](https://docs.amd.com/r/en-US/ug974-vivado-ultrascale-libraries/DCIRESET).

Con referencia de 250 MHz: sincronizar EOS, esperar 64 ciclos, mantener RST alto 64 ciclos (256 ns) y esperar otros 64 ciclos. MIG continúa en reset hasta terminar esa secuencia, disponer de LOCKED y liberar el reset externo. La liberación atraviesa dos registros; la activación por reset externo, pérdida de LOCKED o de EOS es asíncrona.

Los resets posteriores de MIG no reinician ni acortan el pulso DCI. La reconfiguración FPGA inicia una nueva calibración global. La pérdida permanente de LOCKED mantiene MIG en reset; no se intenta una recalibración global mientras otros canales podrían estar utilizando las E/S.

## Reloj e integración

El secuenciador necesita un reloj siempre activo, independiente de `ui_clk` y `sys_rst` de MIG. `tools/ntrpu_board_ddr_dci_synth.tcl` explora un IBUFDS compartido: salida directa hacia MIG configurado `System_Clock=No_Buffer`, y BUFGCE separado hacia el secuenciador. Mantiene el periodo de entrada de 4000 ps y una restricción explícita en el puerto de placa. Verificar clocks, CDC y colocación antes de aceptar esta conexión física.

`20261002T121632Z-V3-dci-integrated-synth` genera un checkpoint con una `DCIRESET`, una `STARTUPE3`, 154 puertos y AM17 libre. Recursos: 30.646 LUT, 34.722 registros, 26,5 BRAM y 3 DSP, incluyendo tráfico/calibración del ejemplo. El DRC conserva cuatro `REQP-1877`; desaparecen los 48 `DCIRST-1`. El proceso termina con código 1 por una consulta final DRC no soportada, después de escribir los resultados; la consulta está corregida en el script actual. El checkpoint se comprueba de nuevo antes de colocarlo.

La síntesis inserta otro BUFGCE en la referencia de MIG. El primer intento de colocación falla porque la entrada GCIO está en banco 64 y el MMCM en banco 66. La siguiente tanda aplica `CLOCK_DEDICATED_ROUTE BACKBONE` exclusivamente a `ntrpu_ref_raw_BUFGCE`, tras comprobar su driver. En UltraScale, BACKBONE equivale a SAME_CMT_COLUMN y conserva rutas globales dedicadas entre regiones de la misma columna: [UG912 2026.1](https://docs.amd.com/r/en-US/ug912-vivado-properties/CLOCK_DEDICATED_ROUTE). Los diagnósticos RIU del XDC generado siguen pendientes; este ajuste de reloj no los resuelve.

`20261002T123715Z-V3-ddr-place-backbone` completa colocación en 491,669 s. La auditoría posterior confirma BUFGCE_X1Y5 en X4Y0 y MMCM_X1Y2 en X4Y2. Las excepciones RIU existentes cubren sus 550 endpoints; no se añadieron excepciones nuevas. Antes del ruteo: WNS +0,553 ns y WHS −0,277 ns, por lo que todavía no cierra timing.

El ruteo posterior `20261002T125230Z-V3-ddr-route` termina en 187,353 s, con WNS +0,334 ns y WHS +0,011 ns. Cumple las restricciones presentes; los contratos de cuatro puertos de placa y la revisión de reset siguen abiertos. Ver [pinout y límites](PIN_PLANNING.md).

El reporte de metodología señala `LUTAR-1` en la combinación de EOS/LOCKED/reset/fin de secuencia que activa los registros de liberación. El banco funcional cubre pulsos breves, pero no glitches analógicos ni metastabilidad. Revisar la estructura de captura de cada fuente asíncrona y sus restricciones antes de cerrar el reset de placa; conservar el aviso, sin degradarlo.

`20261002T131009Z-V4-reset-structure` audita el checkpoint ruteado sin modificarlo. El combinador es una LUT4 y sus únicas cargas son los PRE de dos FDPE con `ASYNC_REG`, juntos en `SLICE_X150Y99`; D del primero es cero y su Q alimenta D del segundo. Los caminos desde `sequence_done` cumplen removal (+0,279 ns) y recovery (+3,139 ns); eso no caracteriza las entradas asíncronas restantes. Se conserva activación sin reloj y liberación sincronizada, coherente con la [topología de reset de UG906](https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/Asynchronous-Reset-Synchronizer).

`report_synchronizer_mtbf` enumera 258 estructuras, con 104 resultados `UNDEFINED`, incluida la cadena de EOS; no entrega MTBF global válido. Su agregado numérico parcial no demuestra fiabilidad de placa. `LUTAR-1` continúa visible: la auditoría delimita sus cargas, pero no sustituye el contrato eléctrico de pulsos ni la evaluación de metastabilidad/glitches.

## Prueba ejecutada

`20261002T121240Z-V4-dci-unit`: **PASS**, 19,546 s de VM, 1,308 µs simulados. Usa `unisims_ver.DCIRESET` instalado y verifica un pulso de 256 ns, espera de LOCKED retrasado y cuatro liberaciones: arranque, reset externo breve, pérdida de lock y pérdida de EOS. Ningún diagnóstico de violación de pulso; conserva el aviso de elaboración sobre `LIBRARY_PATH`.

La prueba no mide resistencias DCI ni valida recuperación de datos DDR. Al integrar el sistema, un reset de memoria debe invalidar calibración, tráfico pendiente y la época de pesos/resultados antes de reanudar accesos.

## Cambios de fuente de reset y ausencia de reloj

`20261002T130647Z-V4-dci-reset-stress`: **PASS**, 19,685 s de VM y 22,258 µs simulados, sin cambiar el RTL de producción. Prueba los siete conjuntos de fallos contra los siete, en cuatro fases y con tres transiciones: solapamiento, simultánea y separación saludable de 5 ps. Total: 588 casos. Añade tres pulsos de fallo con reloj detenido; reset permanece activo hasta que vuelven dos flancos saludables. Cuenta un pulso DCI y 591 liberaciones supervisadas.

`20261002T131532Z-V4-dci-negative-r2` confirma que el banco rechaza dos regresiones en copias del RTL: activación solo síncrona y salida desde una sola etapa. Un intento anterior falló por esperar el texto de otro monitor simultáneo; se conserva, sin falso PASS. Reproducir con `tools/ntrpu_board_dci_sim.tcl NUEVA_CARPETA stress` y `tools/ntrpu_board_dci_negative_sim.tcl NUEVA_CARPETA`, mediante `vivado -mode batch -source ... -tclargs ...`.

Estas pruebas digitales no establecen anchura mínima física del pulso, MTBF ni inmunidad analógica. No se sustituye la activación asíncrona por una dependiente del reloj para eliminar un aviso.
