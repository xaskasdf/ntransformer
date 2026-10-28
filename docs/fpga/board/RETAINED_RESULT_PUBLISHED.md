# Resultado publicado antes de perder el reloj

Actualizado el 2026-10-06; **ambos perfiles aprobados en su alcance,
revisados y respaldados**. Tanda rápida:
`20261006T161102Z-V4-ffn-supervised-retained-result-mig-fast-frozen`.
El perfil de 37 ns también terminó y pasó revisión y 16 controles como
`20261006T163229Z-V4-ffn-supervised-retained-result-mig-slow-frozen`;
se detalla abajo.
Este caso sigue a [CPL visible](RETAINED_CPL_VISIBLE.md). El RTL de
producción, watchdog de 64 ciclos, IP y modelos DDR4 permanecen intactos.

## Contrato de diseño

`ntrpu_pcie_fragment_ingress` conserva un resultado ya publicado hasta su
handshake de retirada. La pérdida posterior de sesión no debe borrar
VALID ni cambiar época, tag o estado. Un estado de resultado exitoso
describe el trabajo terminado; por sí solo no acredita que la sesión
siga disponible ni autoriza reutilizar el destino.

`ntrpu_pcie_fragment_pool` exige identidad, barrera del productor,
transporte quieto y `retire_ready`, además de una sesión válida, para
aceptar ACK. La pérdida debe cerrar esa aceptación incluso cuando el
resultado y la identidad correcta del ACK ya estén presentes. El host
debe tratar disponibilidad de sesión y resultado como condiciones distintas.

## Banco preparado

`tb_ntrpu_retained_result_mig.sv` conserva los 18 trabajos de regresión
y añade un trabajo 18 con dos MIG reales y secciones AXI/AXI-Lite.
Carga época/tag de ACK mediante cuatro escrituras MMIO, completa
CMD/DATA/escritura DDR/CPL/DONE y publica el resultado. Comprueba un
flanco fuente con resultado visible y ACK listo antes de parar el reloj;
todavía no emite el comando ACK.

La barrera previa a publicación procede del fixture de productor, después
de DONE y del drenaje de su interfaz. **No es un certificado del driver
host ni del XDMA nativo.** Tras dos pausas de 96 ciclos de gestión y un
flanco fuente intermedio se exige retener resultado, propietario y rails
de reset, con retirada bloqueada. Los intentos tardíos de ACK y ARM
deben rechazarse en la entrada ya cerrada. Una lectura independiente del
fixture comprueba los 64 bytes de DDR; no es una lectura BAR reabierta.

## Evidencia y dependencias

El revisor reconstruye todos los handshakes y exige 19 ARM,
17 LINE/CMD/DATA/CPL/DONE y 18 ACK: el último resultado queda sin retirar.
Compara 293 escrituras y 12 lecturas MMIO, 38 frames, 17 escrituras y
29 lecturas DDR de usuario, además del tráfico periódico y todos los
modelos archivados. Dieciséis controles de evidencia cubren retirada
inventada, pérdida o cambio del resultado, identidad incorrecta, reloj,
sesión y readback; los dieciséis rechazan las alteraciones en cada perfil.

En ese perfil, B/CPL/DONE preceden a la publicación. El watchdog cierra
la retirada a **690 ns desde STOP** (680 ns desde la primera muestra de
gestión). Se retienen época `0x100000013`, tag `0x800000010000abcd`, VALID
y estado exitoso; el trabajo terminado no se retira ni se reutiliza.
La lectura independiente recupera sus 64 bytes. B→readback ocupa 4.015 ns,
incluidas las pausas: no es latencia aislada de DDR. Hay 9/8 escrituras
físicas y 15/14 lecturas de usuario por controlador; las 84/76 lecturas
periódicas regresan completas. `accepted_mask=0x19` y `complete_ack=0`.

Duración nativa: 747,346 s. Archivo: 115.964.944 bytes, 739 miembros,
SHA-256 `c4fcf855afb8b3b989426ee801c56c9e3206edf6ecfffc991ec6b8282151ca32`.
Revisión independiente:
`943c11882bc71ef41167be020f2e0987776a8d499d1fa325f866c3a3922446f0`.
Dieciséis controles:
`b5c07c89ff08d59a905e6f4a06cebcbe1a0e913922b155b010cee150bdfe3b0a`.

El perfil de 37 ns conserva los mismos inventarios y datos: cierre a
**607 ns desde STOP**, 600 ns desde la primera muestra de gestión,
resultado e identidad retenidos, sin ACK completo. B→readback ocupa
8.215 ns incluidas las pausas; las 195/190 lecturas periódicas regresan.
Duración nativa: 1.230,450 s. Archivo: 114.627.774 bytes, 739 miembros,
SHA-256 `8845ab8fa2ae777c3d8ff5e19fa70b2049f8afdd1b78b3d056871850b77a0534`.
Revisión: `4355ea7847b4203471fb6687247aba5fcfd41a69765eaaf0b2c122ce581accb0`.
Controles: `a06fe192a5c3e6a2a8a6729a304c646f83e9600bb1abd3bde2be88cda9d56ce0`.

Orden: CMD lento revisado → CPL rápido/lento revisados → resultado
publicado a 5 ns → revisión completa y controles → perfil de 37 ns.
CMD, CPL y resultado publicado cumplen sus revisiones y controles. Esta
ejecución usa `source-20261006T152226Z.zip`, SHA-256
`63ec12791a01f0d25e99559c4370f11e40f012bc09b5b4616b5d347f1289e4fd`.
Los scripts `run_retained_result_mig.py`, `review_retained_result_mig.py`
y `check_retained_result_review_controls.py` están en la orquestación local.
La siguiente preparación es la [gestión local XDMA](XDMA_LOCAL_MANAGEMENT.md).
Inventario host/XDMA, recuperación/reset operativo, DIMM exacto, cuatro
canales, implementación física e inferencia completa mantienen sus propios
criterios de cierre; este banco no los sustituye.
