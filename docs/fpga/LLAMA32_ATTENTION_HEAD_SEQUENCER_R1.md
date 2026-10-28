# Secuenciador de cabezas Llama 3.2 1B: contrato propuesto R1

Estado: **controlador RTL implementado y probado con un peer de protocolo**.
`ntrpu_attention_head_sequencer_r1` carga Q y publica el vector concatenado.
El wrapper que conecta y bloquea bind/append sobre `ntrpu_attention_engine_r1`
sigue pendiente. El motor real todavía se valida en la cola Vivado; las
pruebas actuales del secuenciador no usan su aritmética ni interfaces AXI.

## Interfaz y almacenamiento

Un comando raíz identifica capa, posición, secuencia, epoch y base de leases.
Se acepta únicamente con caché válido, identidad coincidente, posición dentro
del prefijo comprometido y sin overflow al calcular `lease_base + 31`.
Todos los campos se retienen hasta consumir la finalización.
La interfaz acepta comandos inválidos cuando el caché está inactivo para
devolver un error explícito, sin cargar Q ni emitir consultas hijas.
`cache_committed_tokens` es metadata confiable del wrapper, no un valor
que deba poder proporcionar libremente el host. Integrar esa autoridad
y arbitrar simultaneidad de comandos es parte del wrapper pendiente.

El vector Q de 2.048 valores FP32 llega como 128 beats de 512 bits: índice
monótono 0–127, señal `last` solamente en 127 e identidad del comando raíz
en cada beat. Cada beat contiene 16 valores consecutivos en orden de lane.
Un error de índice, identidad, longitud o valor no finito cancela la carga;
no se emite ninguna consulta hija hasta validar los 128 beats.

Se proponen dos memorias privadas, pendientes de inferencia y recursos reales:

| Memoria | Organización lógica | Tamaño |
| --- | --- | ---: |
| Q completo | 128 × 512 bits | 8 KiB |
| Resultado concatenado | 2.048 × 32 bits | 8 KiB |

Se leen cuatro beats Q por cabeza para formar el puerto de 2.048 bits del
motor existente. El resultado se lee mediante `result_ce`, `result_address`
y `result_q_valid`; solamente las respuestas válidas incrementan el contador.
No se presupone lectura combinacional ni aceptación implícita del dato.

## Secuencia y publicación

`IDLE → LOAD_Q → PREPARE_HEAD → ISSUE_HEAD → WAIT_HEAD → COPY_HEAD`.
Se recorren las cabezas 0–31; el motor aplica la asociación GQA
`kv_head = query_head / 4`. La lease hija es `lease_base + query_head`.
Durante `COPY_HEAD` se retiene `completion_ready=0`, se copian exactamente
64 palabras a `head*64 + lane`, y luego se consume la finalización hija.

El resultado raíz permanece privado hasta completar las 32 cabezas sin
error. Entonces se publica una única finalización exitosa y se habilita la
lectura de las 2.048 palabras. Consumir esa finalización libera el buffer;
el consumidor debe haber terminado de leerlo. No se solapan comandos raíz
en esta primera revisión. La cuantización ABI1 y el lanzamiento de O son
una integración posterior, con referencia ya disponible.

## Propiedad, cancelación y reset

El wrapper bloquea nuevos binds **y appends** desde la aceptación del comando
hasta consumir su finalización. También exige que no haya una operación de
escritura previa en vuelo. Así se evita cambiar de caché o modificar el
prefijo entre leases hijas; una revisión futura podrá permitir appends
concurrentes mediante una prueba explícita de no interferencia.

Un cambio de `expected_epoch` revoca inmediatamente la lectura publicada y
queda retenido aunque el valor vuelva al anterior. Ante cancelación o fallo
de una cabeza, no se lanzan las restantes: se cancela y drena la operación
activa conforme al contrato del motor, se invalida el buffer y se publica
una única finalización de error. No se libera la propiedad antes del drenaje.
Una finalización ya publicada se mantiene estable bajo backpressure: una
cancelación o cambio de epoch tardío revoca la autorización de lectura de
forma inmediata y retenida, sin reescribir el estado de esa finalización.
El reset es compartido con el motor y requiere el mismo quiesce del peer AXI;
no se añade un reset parcial capaz de abandonar una transferencia emitida.

## Criterios para implementar y encolar

1. Revisar el PASS y respaldo del motor generado, incluyendo sus 155 consultas.
2. Implementar el wrapper y probar primero control/orden con un peer declarado
   como tal; esa prueba no califica aritmética de atención.
3. Conectar el motor real y comparar las cuatro muestras de 32 cabezas con
   `attention-output-r1`, incluyendo las 8.192 salidas FP32 y su orden.
4. Inyectar carga Q incompleta, duplicada y no finita; overflow de lease;
   stalls; fallo, cancelación y cambio de epoch durante carga, consulta,
   copia y finalización retenida; reset en esas mismas fases.
5. Verificar ausencia de publicación parcial, bind/append bloqueados,
   contadores exactos, recuperación posterior, BRAM inferida, latches,
   loops y timing OOC. Registrar ciclos medidos; no estimar tokens/s desde
   el banco nativo ni extrapolar latencia de una cabeza al sistema completo.

## Evidencia del controlador independiente

Tanda `20261013T180753Z-V7-attention-head-sequencer-native-r1`: cuatro
suites Icarus, semillas 1/7/42 con watchdog 32 y semilla 1 con watchdog
65.536. Cada suite pasó 35 escenarios, 31 finalizaciones, cinco resets,
372 comandos de cabeza y 18.432 palabras publicadas. Ciclos totales:
169.404, 169.347, 169.044 y 365.919; son ciclos de este banco con peer,
no latencias de atención. Incluye errores de carga, metadatos, leases,
timeouts, fallo de cabeza, cancelación/revocación durante cómputo y copia,
revocación tardía y recuperación tras cuatro fases de reset.

El revisor independiente rechazó diez corrupciones por suite. Cinco
mutantes RTL compilados también fallaron como se esperaba: direcciones
intercambiadas, publicación en la cabeza 30, revocación ignorada, lease
constante y liberación de propiedad durante carga Q.

- Archivo positivo D:/NFS: `5edc693fe19e29a46adce5cbea14b3d71cc8bc20ea5974a1bbcc46824a03c803`.
- Revisión: `fa00ba7e156b92a432d305c17e63814d50370abbfdfbae9131a0fbbcb91a4a1b`.
- Revisor: `tools/check_ntrpu_attention_head_sequencer_r1.py`.

El primer colector falló al escapar un salto de línea. Se recuperó y
verificó la tanda ya terminada sin volver a ejecutarla. La revisión R2
del colector usa cadenas literales para sus scripts remotos.
