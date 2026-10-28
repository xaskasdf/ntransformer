# DATA pendiente durante la pérdida de reloj

Estado del 2026-10-06. Este banco amplía la
[retención ante watchdog](RETAINED_RECEIVER_EPOCH.md): se detiene el reloj
después de aceptar CMD y antes de aceptar DATA. La prueba exige conservar
el banco y la identidad, aceptar exactamente una DATA al recibir un flanco,
completar la escritura y lectura DDR y retener la CPL sin publicar el resultado.

## Diagnóstico conservado

`20261006T133708Z-V4-ffn-supervised-retained-data-mig-fast-frozen` se detuvo
como diagnóstico, con código de proceso -15, tras 887,008 s de ejecución
nativa. El banco esperaba `peer_request` después de parar su reloj fuente;
esa señal requiere que el CDC avance en ese mismo dominio.

La evidencia muestra CMD a 113.242.500 ps, parada a 113.245.000 ps y
quarantine observada a 113.905.000 ps. Hasta 155.905.000 ps continuaba
la petición de gestión con tag 1, fuente detenida y `peer_request=0`.
No se alcanzó el primer marcador de cierre ni se aceptó DATA del trabajo 18.
Los 18 trabajos anteriores pasaron la revisión independiente de transporte
y núcleo; **la tanda completa no está aprobada**.

Se envió TERM exclusivamente al supervisor `timeout` de esta tanda,
comprobando PID, argumentos, grupo y directorios de sus procesos. El padre
de orquestación registró el resultado y respaldó todo. El archivo incluye
`EARLY_STOP_DIAGNOSTIC.json`, logs y fuentes originales; SHA-256:
`390c9e39247e1a661bf06319bb11a1bae409dbbec814a15fb45549570ecc88e6`.
Revisor reproducible: `review_retained_data_diagnostic.py` en el checkpoint.

## Revisión R2

`tb_ntrpu_retained_data_mig_r2.sv` espera la petición de gestión y su
identidad, con límite de 16 ciclos de gestión; exige que el reloj fuente
siga detenido y que el peer aún no haya recibido la petición. Registra
ese hecho antes del marcador de cierre. No requiere flancos fuente extra.

El monitor también distingue reserva y despacho: al detenerse, `banks_used`
indica el banco reservado y `pending_counts` vale cero. Este último expone
`outstanding` del receptor, que aumenta al despachar una escritura; no
cuenta un descriptor que todavía está esperando DATA.

El RTL, los dos intervalos de 96 ciclos, el watchdog de 64 ciclos,
la invalidación automática, la conservación de bytes y la denegación
del cierre global permanecen intactos. El revisor añade el testigo de la
petición de gestión. Los 14 controles negativos de evidencia se ejecutaron
sobre la traza rápida completa y rechazaron las 14 alteraciones. Son
controles del verificador, no mutaciones del RTL.

## Resultado rápido y siguiente perfil

`20261006T140221Z-V4-ffn-supervised-retained-data-mig-r2-fast-frozen`
pasó la revisión independiente completa, con duración nativa de 718,745 s.
La compilación usa el RTL real y los modelos MIG archivados coinciden
con la referencia revisada. Fuente:
`003956789aefa24cd932bb5952a9ee3df9679cc9b689244793d0641d0fecf07b`.

CMD se acepta a 113.242.500 ps y el reloj se detiene a 113.245.000 ps.
La muestra de fallo aparece 660 ns después de la parada, o 650 ns desde
la primera muestra de gestión posterior. El campo histórico del revisor
R2 usa ese segundo origen; el addendum `watchdog-interval-review.json`
explicita ambos sin alterar la evidencia original. DATA se acepta
a 114.312.500 ps, después del watchdog; AW/W de memoria llegan a
115.330.998/115.340.998 ps. El readback compara los 64 bytes completos,
570 ns después de B. La CPL `0x10000000f` permanece retenida.

La tanda completa registra 19 ARM, 17 CMD/DATA, 16 CPL/DONE y 18 ACK;
288 escrituras y 12 lecturas MMIO; 17 escrituras y 29 lecturas DDR.
La aceptación de proveedores sigue en `0x19`, con cierre global falso.
Recuperar el reloj no reabre la sesión. Es simulación con DIMM genérico,
sin prueba de caudal físico ni recuperación operativa.

Archivo: `04ce4f31221e890e079eff8aa85912765dc6ec948578ccfaa63585a5f0afc8a7`.
Revisión: `849a7d132b490b1590ef7fe45975752be1213779edd8510887faab53be09520c`.
Controles: `4171c60c59119bda98d7bc86b16cef2a68282b9c781e8b258ebe562ac34a6559`.

El perfil lento de 37 ns,
`20261006T141947Z-V4-ffn-supervised-retained-data-mig-r2-slow-frozen`,
también pasó la revisión completa, la comparación de modelos y los 14
controles de evidencia. Conserva los mismos inventarios y la CPL pendiente.

| Medida de simulación | Fuente 5 ns | Fuente 37 ns |
| --- | ---: | ---: |
| Duración nativa | 718,745 s | 1.209,572 s |
| Parada → primera muestra de fallo | 660 ns | 672 ns |
| Primera muestra de gestión → fallo | 650 ns | 670 ns |
| B final → readback | 570 ns | 3.425 ns |
| Bytes escritos / leídos, tanda completa | 1.088 / 1.856 | 1.088 / 1.856 |

Los intervalos de fallo tienen orígenes explícitos en los addenda
`watchdog-interval-review.json`; no son medidas de metastabilidad ni
tiempos físicos de placa. El perfil lento incluye 193/187 solicitudes
periódicas de los modelos DDR, todas devueltas.

Archivo lento: `40144486cd2efbd71f762cb115605eab4488c0004f46b291686ac6b3e1d1c001`.
Revisión: `a9d442743b2cde43b5fd6487e6cae859ae9571bd4d3c36bd4822360b17cc2157`.
Controles: `5273dbe7a985fafb9b82213610fd50e22c538dc2af36bb9744246d6523f29000`.

La prueba de [CMD no aceptado](RETAINED_CMD_PENDING.md) está preparada.
CPL ya visible antes de la pérdida, resultado publicado, contratos
host/XDMA y reset operativo siguen pendientes.
