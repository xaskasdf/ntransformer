# Control e identidad para la proyección O

Estado 2026-10-13, 20:27 UTC: **RTL nativo y revisión independiente
aprobados; XSim y síntesis OOC encolados**. El banco usa un interlocutor
de protocolo, sin ejecutar el GEMV aritmético ni tráfico AXI real.

## Función e interfaces

`fpga/rtl/ntrpu_attention_projection_control_r1.sv` adapta la identidad
raíz de 64 bits a los puertos de 32 bits de `ntrpu_q8_matrix_memory`.
Captura secuencia, época y los nueve campos de dirección/región completos;
modificar los pines del comando después de aceptarlo no cambia la operación.
El GEMV posterior debe instanciarse con `FP32_CAPABLE=1`.

La forma queda fijada para O de Llama 3.2 1B: 2.048 filas, 64 bloques K,
tiles de 32 bloques y salida FP32 de 8.192 bytes. Los controles de rango
y solapamiento siguen a cargo del motor matriz existente; este adaptador
no declara válidas direcciones antes de recibir su finalización correcta.

## Mapeo sin truncamiento

Un contador interno de 64 bits reserva 64 secuencias consecutivas por
proyección, una por grupo de 32 filas. La parte baja es la secuencia del
primer grupo; la parte alta es la época interna del hijo. Esta identidad
es independiente de la identidad raíz, que se conserva y comprueba completa.
Las épocas internas de activación y salida coinciden con la del GEMV.

Las reservas están alineadas a 64: el último grupo no desborda su secuencia.
Al agotar el espacio interno se rechazan comandos nuevos con estado 9;
no se reutilizan etiquetas silenciosamente. `INITIAL_CHILD_TAG` permite
ensayar los límites sin miles de millones de ciclos; producción comienza
en cero. Resetear el contador requiere reiniciar o detener también el hijo
y sus pares AXI, evitando respuestas antiguas después del reset.

## Propiedad y fallas

Una vez ofrecido el comando al hijo, VALID y descriptor permanecen estables
aunque llegue cancelación con READY bajo. Después de aceptarlo, el adaptador
solicita cancelación y espera una finalización con identidad coincidente.
Una respuesta ajena nunca demuestra que el hijo terminó; queda sin reconocer
y no permite publicar ni liberar la operación. Si persiste, se necesita
reset coordinado.

Timeout, cambio de época o cancelación invalidan la salida y conservan la
espera de finalización. Una salida exitosa exige exactamente 2.048 filas y
8.192 bytes. La autorización del rango de salida se revoca inmediatamente
ante cancelación/época tardía y permanece revocada aunque vuelva la época
anterior. El status de completion retenido permanece estable.

`child_activity` es la entrada de progreso del watchdog: su conexión y los
intervalos legítimos sin tráfico deben validarse al integrar el GEMV real.
El banco aislado no acredita ese límite de inactividad en la composición.

## Evidencia obtenida

Tanda `20261013T201953Z-V7-attention-projection-control-native-r1`:
tres semillas de 47 escenarios más un banco de agotamiento de cuatro
comandos. Total **145 escenarios**, incluidos rechazos y fallas esperadas,
137 aceptaciones del hijo, 131 respuestas reconocidas, 133 eventos de
completion y 20 resets coordinados.

La revisión independiente reconstruye etiquetas, orden y publicación de
rangos; rechazó 22 alteraciones por semilla y 17 en agotamiento. Ocho
mutaciones RTL reales compilaron y fueron detectadas: comparación truncada,
identidad devuelta truncada, avance incorrecto de etiquetas, aceptación de
respuesta ajena, revocación reversible, agotamiento ignorado, descriptor
dependiente de pines vivos y tamaño final incompleto.

- Archivo base SHA-256: `ad34612c6b2db1210355ba856ecac6a04d107df73906aa7a8682d4cc805a94e5`.
- Revisión: `3b7c4e77e9f52a00e936cddda604596dee419d35d7f4dde8734332af6867a684`.
- Archivos locales: `D:/AI/ntransformer/artifacts/ntrpu-attention-projection-control-native/`.

Fuentes, logs y certificados de cada mutante se conservaron en D:/NFS.

## Cola y trabajo pendiente

Paquete `20261013T202655Z-llama32-attention-projection-control-vivado-r1`,
16 miembros verificados en D:/NFS. SHA-256 del paquete:
`62d8b61f5e82585cbe7271cc7b66208fb67f5d445a77a5517821e93269b400a8`;
fuente ZIP: `de5a23cb8beac109fe0ef2569e660d2fe3f3f47a476dcf39130f6db3900fc6e8`.

XSim repetirá cuatro bancos y sintetizará los parámetros de producción
en VU5P, reloj de 5 ns, dos CPU y límite de 1.200 s. El ruteo anterior fija
el orden del slot: un diagnóstico físico respaldado no impide esta prueba
independiente de protocolo, ni se transforma en aprobación de timing.

Falta conectar cuantizador, escritura de los 64 registros ABI1, este
adaptador y el GEMV O real, conservando la propiedad de las regiones hasta
su finalización. La comparación aritmética de extremo a extremo,
residual/entrada FFN, DDR física y timing integrado siguen pendientes.
