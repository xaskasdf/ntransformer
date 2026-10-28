# Corte registrado del retorno HBM en la composición peer

## Alcance — 2026-10-07

Este estudio compara el [adaptador peer vigente](PEER_READ_LEDGER.md), con la parada global de `pair5`, frente a la misma composición con la cola C1 revisión 2. Mantiene dos frames y dos contextos por canal, ocho créditos nativos y relojes de 200/450 MHz. La comparación anterior de C1 sobre F1 aislado no determina el resultado de esta composición.

**Se conserva `resumed_cut` como subperfil peer con timing interno nominal cerrado y transacciones del netlist verificadas.** No sustituye la referencia F1 aislada. La [validación funcional posterior con el modelo HBM nativo](PEER_READ_NATIVE.md) pasó sobre el mismo RTL: 20 transferencias y 37.376 bytes. P0 completo, integración física con HBM, PCIe y placa permanecen abiertos; el margen de setup de memoria es de solo 6 ps sobre las restricciones usadas.

`prepare_ntrpu_peer_read.py --return-cut` añade la cola únicamente en una copia privada. Los RTL de referencia y la cola de trabajo principal permanecen intactos. El perfil sin la opción sigue siendo el control.

## Cambio y coste

La cola está entre el convertidor AXI y la FIFO CDC de retorno. Conserva dos palabras de 519 bits por canal: payload, RID, RRESP y RLAST. Añade 256 bytes de capacidad de payload en total; los contextos retenidos hasta ACK y las reservas de frame no aumentan.

La escritura especulativa actualiza exclusivamente el slot libre. VALID avanza la propiedad y los punteros, pero no habilita directamente los registros anchos de payload. READY depende de ocupación registrada; una cola llena puede introducir una burbuja al volver a admitir. `pending` incluye cualquier palabra retenida. Stop/fault no vacían la cola; solo el reset frío coordinado reinicia su estado. Potencia y conmutación no están medidas.

## Validación funcional

Raíz de evidencia: `D:/AI/artifacts/ntrpu-peer-read-cut/20261007T223054509187Z/`.

| Perfil | Escenarios | Bytes públicos comparados | Muestras de cola llena / retenida |
| --- | ---: | ---: | ---: |
| Control sin cola, 200/450 MHz | 21 | 94.720 | No aplica |
| Cola, 200/450 MHz | 21 | 94.720 | 127 / 273 |
| Cola, 50/450 MHz, contrapresión | 21 | 94.720 | 9.429 / 9.638 |

Los tres perfiles pasan 29 rechazos y 18 escenarios de cuarentena. Las 20 transferencias satisfactorias coinciden con el modelo de transporte existente: 37.376 bytes y liberación después de confirmación remota. El perfil lento comprueba retención; no es una medición comparable de rendimiento.

El auditor independiente reconstruye cada entrada/salida de la cola, los punteros, ocupación, payload y metadatos; comprueba estabilidad y `pending`. Rechaza seis alteraciones de su traza, además de las seis del auditor peer. Se observan 1.549/1.548 inserciones/retiradas en el perfil normal y 1.571/1.569 en el lento. Las diferencias son palabras retenidas en la época del escenario 9, ACK prematuro: permanecen contabilizadas en `pending` y en cuarentena hasta el reset frío del banco. No hay remanentes en las épocas satisfactorias.

Dos controles RTL comprueban la sensibilidad de los bancos. Alterar un bit de la cola falla con `PAYLOAD` en el escenario 2. Omitir la cola de `pending` pasa las transacciones externas, pero el auditor independiente lo rechaza por ocupación sin reserva pendiente. Este segundo resultado está archivado como mutante rechazado; el éxito del banco peer por sí solo no lo valida.

## Criterio físico

Se ejecutan control y candidato con la misma versión de Vivado, cuatro CPU por proceso y prioridad reducida. Periodos 5,000/2,222 ns; incertidumbres setup/hold 0,100/0,050 ns. Se conservan checkpoints tras síntesis, optimización y colocación para evitar repetir etapas si se alcanza el límite de ejecución.

Ambas tandas iniciales alcanzaron 1.200 segundos durante ruteo. Sus continuaciones verificaron los checkpoints de colocación y los donantes, sin repetir síntesis ni colocación; terminaron en 336,869 s para el control y 312,467 s para el candidato.

| Ruta nominal inicial | Setup transporte / memoria | Hold transporte / memoria | LUT / registros |
| --- | ---: | ---: | ---: |
| `resumed_control` | +33 / −96 ps | +5 / +12 ps | 16.449 / 18.084 |
| **`resumed_cut`, conservado** | **+21 / +6 ps** | **+15 / +11 ps** | **17.316 / 20.157** |
| `refined_control` | +41 / −43 ps | +12 / +12 ps | 16.451 / 18.084 |
| `refined_cut2` | −15 / +113 ps | +14 / +11 ps | 17.316 / 20.157 |

El candidato pasa las restricciones internas nominales, con margen reducido. Añade 867 LUT y 2.073 registros, sin BRAM36. Ambos mantienen READY nativo registrado, ocho comprobaciones de bus skew, CDC informativo y cero endpoints internos sin restricciones. Los márgenes mínimos de bus skew son +1,546 ns y +1,746 ns, respectivamente. Las advertencias DRC conservan sus criterios de revisión; las E/S externas permanecen sin calificación de placa.

El ajuste posterior de ambos diseños usó incertidumbre de setup de memoria de 0,200 ns durante optimización/ruteo y restauró 0,100 ns antes del checkpoint e informes nominales. Los auditores verifican la restauración, sin cambiar periodos ni hold. `refined_cut2` alcanza +13 ps bajo la exigencia adicional de memoria, pero falla por 15 ps en transporte: diez endpoints, TNS −0,139 ns. Se descarta ese checkpoint. Su primer intento, `refined_cut`, fue rechazado antes de iniciar Vivado porque el lanzador solo aceptaba entradas con timing abierto; el diagnóstico y la corrección están preservados.

El camino limitante del candidato conservado ya no es la escritura de la FIFO CDC: va del selector de palabra del conversor de ancho a la cola del conversor de protocolo, con ocho niveles y 2,013 ns de datos. El ajuste posterior desplaza el límite de transporte desde `frame/tags_reg[0][48]` hasta `frame/data_reg[333]/CE`: doce niveles, 4,578 ns de datos y aproximadamente 73 % de interconexión. Mejorar el margen requiere revisar esa selección/habilitación, manteniendo identidad, orden y estabilidad bajo pausas; aumentar la cantidad de buffers no resuelve por sí solo ese camino.

La selección requiere comparar timing, recursos, CDC, bus skew, cobertura de restricciones y DRC. Un candidato con timing cerrado todavía requiere transacciones del netlist ruteado antes de conservarlo. HBM nativo, endpoint PCIe, rendimiento sostenido y placa mantienen sus criterios propios. El subperfil de identidades independientes no cierra la admisión por origen compartida entre canales.

## Transacciones del netlist conservado

`routed_cut` usa el netlist ruteado como único DUT, en una biblioteca separada, sin editarlo ni sustituirlo por RTL. Pasan 21 escenarios, 29 rechazos, 18 escenarios de cuarentena, 39 resultados y 94.720 bytes comparados. El modelo reproduce las 20 transferencias satisfactorias y sus 37.376 bytes. El auditor rechaza seis alteraciones de evidencia.

Este banco observa exclusivamente puertos externos. La espera usada por el driver es el fin del payload; no afirma observar el comando interno ni CPL local. Su prueba complementa la regresión RTL que sí comprueba esas fronteras. La simulación es funcional, sin SDF ni afirmación de equivalencia por ciclo.

Checkpoint conservado: `281a71072ce22ed2ec52460013c111f2b689fef8fa0d20730463ea2909b85dae`. Netlist: `6c3026cf429f1b1b666edc22e5dbab77a308e8a8070c6c49ee6c25d07eebc35c`. La revisión ruteada liga sus transacciones a estos bytes; no hereda el caudal nativo medido anteriormente en F1.

`drc_selected` auditó ese mismo checkpoint, sin editarlo ni aplicar waivers. Conserva siete advertencias: seis PDCN-1569 por entradas LUT sin uso lógico y una RTSTAT-10. Clasificó 5.933 redes: 5.831 asociadas a puertos externos OOC sin carga interna y 102 sin cargas lógicas hoja, con cero casos pendientes. Es una auditoría completa de esas advertencias, no un DRC limpio ni una calificación de E/S de placa.

## Reproducción

Los fixtures `native_binding.sv` y `native_stub.sv` proceden de la evidencia R0/R1 preservada. Todas las salidas deben ser directorios nuevos.

```bash
python3 -B tools/prepare_ntrpu_peer_read.py /tmp/peer-cut-src \
  --binding /ruta/native_binding.sv --stub /ruta/native_stub.sv --return-cut
python3 -B /tmp/peer-cut-src/tools/run_ntrpu_peer_read.py /tmp/peer-cut-sim --full --pair
python3 -B /tmp/peer-cut-src/tools/check_ntrpu_peer_read.py /tmp/peer-cut-sim > /tmp/peer-review.json
python3 -B /tmp/peer-cut-src/tools/check_ntrpu_peer_read_cut.py /tmp/peer-cut-sim > /tmp/cut-review.json
python3 -B /tmp/peer-cut-src/tools/run_ntrpu_peer_read_physical.py /tmp/peer-cut-route \
  --prior /tmp/peer-cut-sim --review /tmp/peer-review.json \
  --cut-review /tmp/cut-review.json --cpus 4 --timeout 1800
python3 -B /tmp/peer-cut-src/tools/check_ntrpu_peer_read_physical.py /tmp/peer-cut-route > /tmp/physical-review.json
python3 -B /tmp/peer-cut-src/tools/run_ntrpu_peer_read_routed.py /tmp/peer-cut-netlist \
  --prior /tmp/peer-cut-route --review /tmp/physical-review.json
python3 -B /tmp/peer-cut-src/tools/check_ntrpu_peer_read_routed.py /tmp/peer-cut-netlist
```

Si una implementación vence su límite, `run_ntrpu_peer_read_resume.py NUEVA_SALIDA --prior SALIDA_ANTERIOR --checkpoint latest` continúa desde el último checkpoint disponible, verificando fuentes y donante. No acredita una ruta parcial como terminada.

Las revisiones, fuentes y reportes se respaldan por tanda en el almacén SHA-256 compartido, con restauración y repetición de auditores. No se conservan ZIP de transferencia ni instalaciones duplicadas de Vivado.

La entrega consolidada corresponde al manifiesto `peer-read-cut-20261007T223054509187Z-handoff` en `D:/AI/artifacts/ntrpu-content-store`. `HANDOFF/README.md`, `SUMMARY.json` y los manifiestos por tanda permiten recuperar fuentes y resultados; deben conservarse junto con el almacén de objetos. La reconstrucción de SystemVerilog se contrasta por hash con las fuentes del checkpoint seleccionado.
