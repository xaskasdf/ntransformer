# Normalización final y cabeza de vocabulario

Actualización 2026-10-16 14:21 UTC: el
[banco aritmético conjunto](LLAMA32_FINAL_VOCAB_ARITHMETIC_R1.md) pasó
compilación y veinte controles sintéticos del revisor. Está encolado
después de esta tanda; ambas conservan sus fuentes y criterios propios.
Todavía no hay resultados aritméticos de la conexión conjunta.

Actualización 2026-10-16 14:03 UTC: **protocolo nativo aprobado, jerarquía
compilada y síntesis/protocolo XSim encolados**. La ejecución aritmética
conjunta sigue pendiente. Coordinador PID 8740, después del vocabulario
aritmético completo; las 83 entradas anteriores permanecen intactas.
Paquete de 94 archivos restaurado en D:/NFS, SHA-256
`40ca1354536be7b1caac4de16986c8b4350b60ea32a5f5b334052001168b2bdf`.

## Contrato de memoria

`ntrpu_final_vocab_real_r1.sv` conecta el HLS RMSNorm/Q8 original con la
cabeza completa de dos MAC. Un único maestro AXI cambia de propietario
cuando la cabeza acepta su comando, después de terminar las escrituras ABI1.
El propietario permanece retenido hasta el drenaje de ambos hijos.

El controlador captura cinco regiones de 64 bits: residual final de 8.192
bytes, pesos de norma de 8.192, ABI1 de 4.096, pesos compactos de 279.085.056
y logits de 513.024. Comprueba alineación, extensión, desbordamiento y
solapamiento de las reservas completas de escritura antes de iniciar la norma.
Las regiones de sólo lectura pueden compartir memoria.

El padre conserva todas las asignaciones hasta el ACK público y el drenaje.
La identidad de entrada es un descriptor capturado; no confirma por sí sola
la retención de la capa productora. La norma mantiene su respuesta durante
los dieciséis segmentos. Sólo se publican logits con fase 15, dieciséis
matrices y todos los bytes completos. Fallos tardíos revocan la validez
inmediatamente y quedan retenidos; una identidad ajena no recibe ACK.

El watchdog del padre es 268.435.456 ciclos: el límite de 67.108.864 del
recorrido Q/K/V sería insuficiente para los aproximadamente 100 millones
estimados de la cabeza. El límite por segmento sigue en 16.777.216 ciclos.
Estos límites y el reloj de banco de 5 ns no certifican frecuencia física.

## Pruebas y límites

Tres semillas pasaron **540 trabajos de protocolo**: regiones inválidas,
identidades de 64 bits, cancelación, timeout, reset coordinado, retención,
revocación y recuperación. Se rechazaron **33 mutaciones RTL** por aserciones
semánticas y **174 alteraciones de transcripción** en Windows/Linux.
La compilación de 25 fuentes comprueba interfaces; sus dos firmas HLS
auxiliares nunca se ejecutaron.

La tanda R1 conserva un diagnóstico de cobertura: retirar `VALID` también
borraba descriptores y otro control ocultaba la mutación. El banco R2 conserva
descriptores correctos durante esa inyección. El controlador y los criterios
de aprobación no cambiaron. Ambos intentos están restaurados en D:/NFS.
La tanda aprobada tiene SHA-256
`dc58f350db9bb36bcbc7a0ebd3f6a2ba71451989da020a32fc3e2e8f8e051667`.

## Integración Vivado

La siguiente tanda sintetiza la jerarquía real de 58 fuentes, con los HLS
originales de norma y MAC, y ejecuta el banco de protocolo en un proyecto
separado. Comprueba fuentes, jerarquía, ausencia de cajas negras y latches,
recursos y timing OOC. Límite de una hora, CPU 0–1, sin reintento automático.
Debe esperar la [aritmética completa del vocabulario](LLAMA32_VOCAB_ARITHMETIC_R1.md)
revisada y respaldada, además de la norma real previamente aprobada.

Todavía falta ejecutar el recorrido aritmético conjunto sobre el residual
final, conectar las dieciséis capas y cerrar el arbitraje DDR y el diseño
físico. El protocolo con pares sintéticos no demuestra inferencia ni consumo.
