# R14: registrar las sumas antes de acumular bytes

Propuesta del 2026-10-09 UTC, derivada de R12. Pendiente de simulación,
síntesis y replay propio; no sustituye aún a la candidata preferida.

## Motivo medido y cambio

La lectura del DCP R12 conserva el WNS global de −7,068 ns. Las rutas hacia
bytes/rango dominan también entre registros (−7,039 ns); las de tabla
alcanzan −5,795 ns. Separar la suma global puede retirar parte de la cadena
crítica, pero no resuelve por sí solo tabla, admisión, validación o contexto.

Conservar íntegros el selector R3, la reducción de dieciséis carriles RC de
R12 y el cálculo de las dos contribuciones RQ. El acumulador nuevo captura
la suma RC **sin signo de 15 bits**, la suma RQ **con signo de 16 bits** y
una base de **21 bits**. Produce combinacionalmente un candidato de 23 bits
exclusivamente desde esos registros. No estrechar ni reinterpretar entradas.

El ensayo R3 anterior registraba cuatro deltas RC de 13 bits y empeoró otra
versión del productor. R14 conserva la semántica por carriles de R12 y
registra su suma final; es otra combinación, que necesita medición completa.
No trasladar el resultado antiguo ni prometer una mejora física.

## Secuencia y estado

En el flanco N habilitado, guardar los 21 bits bajos del candidato anterior
y las sumas del beat N. Después del flanco, reconstruir:
`candidate = zero_extend(previous_low) + sign_extend(saved_rq) - zero_extend(saved_rc)`.
Sus bits bajos reemplazan `remaining_bytes`; los 23 bits alimentan la misma
guarda de rango. El wrapper sigue capturando estado y fallos en N+1.

El armado conserva cero; `observing` habilita el avance. La pausa retiene
los tres registros. Cold reset domina el habilitador y los limpia. Conservar
el truncamiento entre beats, incluso después de negativos, overflow o fallo;
reset del enlace no cancela obligaciones. No añadir dependencia de la
referencia, burbujas, cambios de capacidad ni nuevas excepciones de timing.

## Puertas de aceptación

La unidad secuencial compara **20.265 muestras**, incluidas 62 inyecciones
X/Z activas, pausa con entradas desconocidas, estabilidad antes del flanco,
reset, negativos y ambos límites: 1.052.672 y 2.097.151 bytes. La referencia
reconstruye la recurrencia con aritmética de 32 bits y truncamiento explícito.
Un acumulador mutante que suma RC debe fallar en la primera resta dirigida.

Conservar todos los bancos R12: 22.339 filas/68 escenarios, tablas completas,
estados después de fallo, selector/ownership/ancestry, RQ, prefijos, máscaras,
carriles y siete mutantes de integración. Añadir las dos pruebas de acumulador.
Solo tras revisión RTL archivada: síntesis comparable a 4 ns, auditoría del
DCP y cinco bancos binarios RTL/netlist propios, 78.712 filas y 30 controles.
Comparar todos los destinos críticos, no solo el contador transformado.

Si empeora, mantener R12. La segmentación completa, endpoint nativo,
reset/CDC, memoria exacta/512 GiB, cómputo e implementación de placa conservan
los criterios de [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
