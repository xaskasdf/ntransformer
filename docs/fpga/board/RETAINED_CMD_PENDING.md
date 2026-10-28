# CMD ofrecido antes de perder el reloj

Actualizado el 2026-10-06; **perfiles de 5 y 37 ns aprobados en simulación,
con revisión R2 y 18 controles de evidencia por perfil**.
Complementa [DATA pendiente](RETAINED_DATA_PENDING.md), manteniendo el RTL
de producción y el watchdog de 64 ciclos sin cambios.

## Obligación que se comprueba

El puente consume una línea del propietario y expone CMD, pero se detiene
el reloj antes del siguiente flanco que podría aceptarlo. Cuando vence el
watchdog, `admit` cierra la entrada del receptor; el puente mantiene VALID,
dirección, cookie, canal y datos. Al recuperar el reloj debe seguir en
COMMAND, con READY bajo, sin enviar DATA ni reutilizar la sesión.

El receptor puede tener cero bancos reservados y cero escrituras
pendientes, aunque el productor conserve un comando irrevocable.
Ese inventario vacío no acredita cierre global. Los proveedores
host/XDMA siguen ausentes y `complete_ack` debe permanecer en cero.

## Banco y criterios

`tb_ntrpu_retained_cmd_mig.sv` mantiene los 18 trabajos iniciales,
los caminos AXI/AXI-Lite, ambos MIG genéricos y sus modelos originales.
La pérdida ocurre durante el trabajo 18, después de LINE y antes de CMD.
Se conservan las dos pausas de 96 ciclos de gestión y el único flanco
fuente intermedio, con respuestas host retenidas y renovación de identidad.

El trabajo 16 escribió antes la misma dirección con otro patrón. Una
lectura independiente de esa dirección después de la pérdida debe
recuperar sus 64 bytes originales. No debe existir ningún AW/W/B del
trabajo 18, CMD/DATA aceptado, CPL, publicación ni ACK de ese trabajo.
La lectura pertenece al fixture de memoria, no al BAR cerrado.

`check_ntrpu_retained_cmd_mig_r2.py` compara todas las muestras y transacciones,
incluido el descriptor y payload pendiente en cada muestra de gestión
y flanco fuente recuperado. Se prepararon 14 alteraciones de evidencia
para comprobar el verificador. Las 14 fueron rechazadas con el diagnóstico
esperado; otras cuatro alteraciones comprobaron escrituras físicas y
contadores de lecturas periódicas.

Ejecutar primero 5 ns y revisar antes de repetir con 37 ns. Cada tanda
usa fuentes fijadas y respaldo de todos los resultados, incluidos fallos.
La primera versión preparada está fijada en
`source-20261006T143525Z.zip`, SHA-256
`cec89a095c2816e4fa42d37b37f1e56c6c6e0125fa77495474eedc5df485d1ad`;
configuración `retained-cmd-mig-frozen-source.json`. El runner y los
revisores están incluidos en el respaldo de orquestación.
La tanda rápida revisada es
`20261006T144418Z-V4-ffn-supervised-retained-cmd-mig-fast-frozen`;
duró **719,588 s** en la VM. El perfil de 37 ns es
`20261006T150801Z-V4-ffn-supervised-retained-cmd-mig-slow-frozen` y
usa las mismas fuentes nativas inmutables; terminó en **1.227,784 s**.
Esto no califica recuperación operativa, host/XDMA nativo, DIMM exacto,
implementación física ni inferencia completa.

## Resultado rápido y corrección del revisor

La traza completa conserva 19 ARM, 17 LINE, 16 CMD/DATA/CPL/DONE y
18 ACK. Se verificaron 16 escrituras DDR y 29 lecturas de usuario.
Por separado, hubo 82/75 solicitudes periódicas por controlador, todas
devueltas. La lectura final de 64 bytes recuperó
el patrón del trabajo 16, diferente del trabajo 18, en la misma dirección.
Su latencia AR→R fue **135 ns**. La parada→fallo fue **665 ns**;
desde la primera muestra de gestión fueron **660 ns**. La máscara de
certificados permaneció en `0x19`, sin autorización de cierre global.

El primer revisor rechazó la traza porque heredaba del caso DATA una
novena escritura física en el canal 0. Este caso exige ocho escrituras
por controlador: el trabajo 18 no se acepta. R2 conserva las comprobaciones
de pines, CAS, avisos, retornos y drenaje, cambiando esa expectativa a
`[8,8]`; las lecturas de usuario siguen siendo `[15,14]`. Dos controles
adicionales inventan una novena escritura y son rechazados. No se cambió
el RTL, el banco, la traza ni el revisor original para obtener este resultado.

El wrapper original también conservaba un título/campo de estado del
caso DATA. Un addendum distingue ese error de metadatos; el runner R2
registra CMD y restaura el puntero DATA desde el checkpoint verificado.

El archivo local/NFS del perfil rápido tiene SHA-256
`c27a396c394b76194ebe497ab553d2effdb0b04de851ddbb99297ddbde8fedf5`;
la revisión independiente R2,
`a5949ad42d4b2b9b4c987390abcf3acb85b4b28a365100507c7644a7d19b5a73`.
Revisiones, controles y diagnóstico original se conservan por ID en
`build/cloud-setup/20261002-board/` y en el checkpoint de la cola R45.

## Resultado con reloj fuente de 37 ns

La revisión lenta confirma los mismos conteos e inventarios: ocho
escrituras físicas por controlador, 15/14 lecturas de usuario y ningún
CMD/DATA ni escritura aceptados para el trabajo 18. Las 193/187 lecturas
periódicas terminaron. Los 64 bytes anteriores permanecieron intactos.

| Medida observada | Fuente de 5 ns | Fuente de 37 ns |
| --- | ---: | ---: |
| Parada de fuente → primera muestra de fallo | 665 ns | 599 ns |
| Primera muestra de gestión → fallo | 660 ns | 590 ns |
| Lectura final AR → R | 135 ns | 560 ns |
| Muestras de gestión con comando pendiente | 418 | 831 |

Son latencias del escenario simulado; no límites de respuesta de placa.
En ambos perfiles la máscara queda en `0x19` y `complete_ack=0`.
El respaldo lento local/NFS tiene SHA-256
`30d217271c2c0f7de7a1c1c1eeff3c54b2fa1bd5df619e39e3daf76c542130e6`;
la revisión independiente R2,
`dc2c14a5ef8745543ab230138e848c991717b38782b0044b8bf47c7856311706`.
Sus 14 controles de cierre y cuatro controles físicos pasaron; esta
evidencia permite avanzar a CPL visible según la dependencia de la cola.
