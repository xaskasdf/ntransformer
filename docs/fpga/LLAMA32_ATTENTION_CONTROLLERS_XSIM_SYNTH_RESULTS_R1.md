# Controladores de atención: resultados XSim y síntesis

Estado 20261013T201041Z: **ambos controladores aprobados y respaldados en D:/NFS**.
Los bancos usan interlocutores de protocolo; no sustituyen la prueba del
wrapper completo con la aritmética generada de atención, actualmente en curso.

## Cobertura funcional

El secuenciador R1 pasó cuatro bancos: semillas 1, 7 y 42 con watchdog corto,
y semilla 1 con watchdog de producción 65.536. Total: **140 trabajos,
124 completions, 73.728 palabras públicas, 1.488 operaciones hijas y 20 resets**.
El revisor independiente rechazó diez corrupciones de traza por banco.
Se verificaron orden de cabezas, captura privada, backpressure, cancelación,
revocación persistente y recuperación.

El guardián de la raíz R1 pasó tres semillas: **288 pasos, 4.896 comprobaciones
por capa y 12 resets**, con ocho corrupciones de traza rechazadas por banco.
Conserva la prioridad y exclusión entre vinculación, append y raíz; publica
prefijos solo tras completar su mantenimiento y revoca metadatos por época.

Las fuentes compiladas, top sintetizado, parte y reloj se contrastaron contra
los paquetes congelados. Los mutantes RTL y pruebas nativas anteriores se
conservan; no se repitieron para generar estas cifras de XSim.

## Recursos estimados

Síntesis OOC sobre `xcvu5p-flva2104-2-i`, reloj de 5 ns:

| Controlador | LUT | FF | Tiles BRAM | DSP | WNS ns | WHS ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Secuenciador de 32 cabezas | 446 | 2357 | 9.5 | 0 | +2.221 | +0.065 |
| Guardián de la raíz | 936 | 679 | 0 | 0 | +2.319 | +0.095 |

Sin black boxes, latches, bucles combinacionales ni endpoints internos sin
reloj/restricción. Los retardos externos y el timing ruteado están pendientes.
No sumar estas cifras a las de un wrapper que ya incluya los controladores.

## Evidencia por tanda

Los archivos completos están bajo `artifacts/ntrpu-board-vivado/<tanda>/`,
con almacenamiento local en D: y copia persistente NFS.

### Secuenciador de 32 cabezas

Tanda `20261013T200002Z-V7-llama32-attention-head-sequencer-vivado-r1`.

- Archivo SHA-256: `1668c198235af1e2ad85362421daa69a0ce8085f99dbcdeb0ba50791f0ee3bf0`.
- Revisión: `999fd340762a6b6ad98344dfd2ad57f6a14fdbf62a22bcb14bb07d1577162ead`.

### Guardián de la raíz

Tanda `20261013T200403Z-V7-llama32-attention-root-guard-vivado-r1`.

- Archivo SHA-256: `02a45e2916531c150cd2b313a4e3c6fa50b76f6a50fe88f8da07f21acadc0841`.
- Revisión: `da057fd5357b94876b9761a613e7f7880947dbeebbd77004abd3dacb723e1d6c`.

## Continuación

Tras ambos respaldos, la cola inició
`20261013T200719Z-V7-llama32-attention-root-vivado-r1`: wrapper completo R2,
23 escenarios, 426 consultas hijas y 24.576 palabras públicas previstas.
Estos son objetivos del banco en curso, no resultados ya aprobados.
Siguen cuantizador HLS, controlador de cuantización y estudio de ruteo del
motor, cada uno tras la revisión y respaldo de sus dependencias.
