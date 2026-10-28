# CPL visible al detenerse el reloj

Actualizado el 2026-10-06; **perfiles de 5 y 37 ns aprobados en simulación,
respaldados y con 14 controles R2 superados por perfil**.
Complementa [DATA pendiente](RETAINED_DATA_PENDING.md) y
[CMD no aceptado](RETAINED_CMD_PENDING.md). Mantiene intactos el RTL,
los modelos DDR4 genéricos y el watchdog de 64 ciclos.

## Escenario

El trabajo 18 completa CMD, DATA y AW/W/B de memoria. Cuando el receptor
expone su CPL, se detiene el reloj fuente antes del flanco que podría
aceptarla. La cookie y el estado se registran en ese instante. El banco
no fuerza READY, VALID ni estado interno de las IP.

Tras dos pausas de 96 ciclos de gestión, separadas por un único flanco
fuente, se exige que el watchdog haya cerrado la admisión y la aceptación
de CPL. La respuesta permanece visible con los mismos datos. No debe
producirse DONE, publicación, ACK del trabajo perdido ni reutilización
del propietario. Una lectura independiente comprueba los 64 bytes escritos.

## Evidencia exigida

`tb_ntrpu_retained_cpl_mig.sv` conserva los 18 trabajos de regresión,
las respuestas host retenidas, las renovaciones de identidad y los dos
MIG. Añade muestras continuas de CPL, cookie, estado e inventario durante
la parada. El inventario de datos puede estar vacío, aunque siga pendiente
la respuesta; esto no permite certificar cierre global.

`check_ntrpu_retained_cpl_mig.py` compara los handshakes completos y exige
que B y la primera CPL visible precedan a la parada. Comprueba que solo
los trabajos anteriores aceptaron CPL y verifica el readback, los modelos
archivados y el aislamiento de la sesión. Los 14 controles que alteran
la evidencia pasaron para el perfil rápido con el guion R2.

Primero se ejecuta a 5 ns y, tras revisión, a 37 ns. La cola exige
que CMD de 37 ns termine y pase revisión completa y 18 controles antes
de lanzar CPL. Las fuentes están fijadas en `source-20261006T145332Z.zip`,
SHA-256 `3e0005ed47a4ed7ab5a860feab3502cc311c9a7bdddcec8f083ddb7fc2cbe4b5`.
La configuración `retained-cpl-mig-frozen-source.json` tiene SHA-256
`91d9f4bd698a8ab1e9bceb2776012011d8844f6e32c87a576bfbfaa99855486b`.
Cada tanda tendrá identificación y respaldo propios. La aprobación se limita
a este escenario de simulación: resultado ya publicado, host/XDMA real,
reset operativo, DIMM exacto, cierre físico e inferencia completa siguen
siendo requisitos distintos.

## Resultado rápido y siguiente perfil

La tanda `20261006T153103Z-V4-ffn-supervised-retained-cpl-mig-fast-frozen`
terminó en **720,958 s**. Se revisaron 19 ARM, 17 LINE/CMD/DATA,
16 CPL/DONE y 18 ACK; 17 escrituras DDR y 29 lecturas de usuario.
Las 82/75 lecturas periódicas de los controladores terminaron.

El fallo se observó **675 ns** después de la parada; desde la primera
muestra de gestión fueron 670 ns. En 410 muestras de gestión la CPL
retuvo cookie `0x10000000f` y estado cero, sin aceptarse. Los 64 bytes
del trabajo 18 se leyeron correctamente; B de escritura → readback
duró **3.955 ns**, incluido el escenario de parada. Es una medida del
banco, no una latencia de lectura aislada. La máscara permaneció en
`0x19`, sin certificado global.

Un control R1 sí rechazó READY=1 en el primer flanco recuperado, pero
esperaba el mensaje de la comprobación posterior a la segunda pausa.
`check_retained_cpl_r2_review_controls.py` corrige únicamente ese mensaje
esperado. No cambia las 14 alteraciones, el verificador positivo, RTL,
banco ni traza. El diagnóstico R1 permanece archivado.

El respaldo local/NFS tiene SHA-256
`95424109a4a053282edef37bd07445ed515169fa5508eae197e5e3a32fb2ddfb`;
la revisión, `8973977ac75758733e24f5dd8acef461a79983e896dd7166eb14493471f2fa23`;
los controles R2, `b9f03f2f7e5e52a87412799630f12162b2084a7a43a8a923eec1dd2bdc3854d5`.
El perfil lento es
`20261006T154747Z-V4-ffn-supervised-retained-cpl-mig-slow-frozen`.
Terminó en **1.233,638 s** y pasó la revisión completa y los 14 controles R2.
Su respaldo local/NFS tiene SHA-256
`fac770cacd673ca37cc68c9b38cc242e157b7c6422c4b3b323550f3fbef50396`;
la revisión, `89f618e25eae4dc3b0551610ab668efc0656361c1deebb5f368c6e6274b8e2a4`;
los controles, `12f4ce89ec80823fd3ab4cf99a417e493e090310f906ab43a472e173e2cce396`.

| Medida observada | Fuente de 5 ns | Fuente de 37 ns |
| --- | ---: | ---: |
| Parada → primera muestra de fallo | 675 ns | 673 ns |
| Primera muestra de gestión → fallo | 670 ns | 670 ns |
| B de escritura → readback final, incluida la parada | 3.955 ns | 8.030 ns |
| Muestras de gestión con CPL pendiente | 410 | 789 |
| Lecturas periódicas devueltas, canales 0/1 | 82/75 | 193/188 |

Ambos perfiles mantienen los mismos conteos de trabajos y transferencias
de usuario, con nueve/ocho escrituras físicas por controlador. CPL, cookie
y estado permanecen retenidos, sin DONE/publicación ni cierre global.
Estos resultados cumplen la dependencia para iniciar el banco de resultado
ya publicado; no califican todavía la retirada ni la recuperación de sesión.
