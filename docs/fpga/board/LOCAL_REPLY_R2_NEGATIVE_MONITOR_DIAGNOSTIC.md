# Control negativo: ventana temporal del monitor

La tanda `20261006T081717Z-V4-ffn-supervised-local-reply-negatives-frozen`
terminó con código 1 en su control positivo, antes de ejecutar mutantes.
Se conserva como diagnóstico, nunca como rechazo válido de un defecto RTL.
Respaldo verificado: SHA-256
`f682a2a2f99787c115fe08aa0bb6ef6de20004266da81af12f92b88618d40a11`.

El monitor añadido comprobaba obligaciones locales en **todos** los flancos
de gestión con `complete=1`. El banco introduce deliberadamente una AW tardía
después de obtener una nueva prueba. A 27.013,002 ns ya se había registrado
la revocación host y se había detenido ese reloj; a 27.015 ns el monitor
falló porque gestión aún no había recibido la alarma. Las capturas positivas
revisadas muestran el cruce de tres etapas en 25,5 ns. El monitor exigía una
propagación instantánea que no forma parte de ese contrato CDC.

`ntrpu_board_local_reply_negatives_r2.tcl` comprueba las obligaciones **al
flanco de subida de `complete`**, después de 1 ps de asentamiento de señales.
Mantiene la revisión de identidad de ACK y todas las comprobaciones originales
de petición tardía, reloj detenido, revocación y retención. El verificador
independiente sigue midiendo la latencia y exige como máximo 30 ns.
No se cambian RTL productivo, relojes, periodos, estímulos ni timeout.

La cola original está detenida y respaldada. La revisión R2 usa otra carpeta
y fuentes congeladas; debe completar primero el control positivo y luego
rechazar cada uno de los tres defectos por su aserción específica. Su resultado
sigue pendiente hasta revisar evidencia nativa.

Este diagnóstico actualiza el estado «en ejecución» registrado en el handoff
R28 y en `LOCAL_REPLY_R2_VALIDATION.md`. No invalida las cuatro revisiones
positivas ya respaldadas ni convierte los fixtures externos en contratos
reales de host/XDMA. El reset del endpoint y la placa siguen sin calificarse.
