# Recuperación de la integración HLS: lecturas especulativas

Los bloques aislados de memoria pasaron, pero la primera integración del
RTL HLS real detectó dos diferencias entre el peer de protocolo y el
calendario de lectura generado por Vitis. Se conservaron ambas tandas.

## Hallazgos y cambios

1. `20261013T182922Z-V7-llama32-attention-engine-vivado-r1` falló en la primera
   consulta: Vitis precarga `state_input[0]` aun cuando `tile_start == 0`.
   El contexto considera inválida cualquier lectura de continuación inicial.
   Archivo del diagnóstico: `11681197a49e3b67ddbb3572d5648e0489720eaae3a1df8b0876a0ca96cc2bd1`.
2. La revisión siguiente aisló ese acceso y avanzó hasta K/V. La tanda
   `20261013T183916Z-V7-llama32-attention-engine-vivado-r1` detectó la lectura
   especulativa en `tile_count*64` al salir del bucle de copia. Su comparación
   combinacional invalidaba la respuesta del último elemento válido.
   Archivo: `7dd3695ece41e98b9d82197558a5345e73b0f104b3b0c92e577a309ffd6cd561`.

El adaptador R2 usa dos protecciones explícitas, sin modificar los 33
archivos generados, los pesos, los bloques de memoria calificados ni la
referencia numérica:

- `ntrpu_attention_state_read_guard_r1` devuelve cero en el primer tile y
  no expone bancos anteriores. Direcciones fuera de los 104 elementos siguen
  llegando al control de rango del contexto.
- `ntrpu_attention_kv_read_guard_r1` descarta exclusivamente la dirección
  exacta posterior al último elemento de tiles cortos. Las demás direcciones
  incorrectas siguen llegando a la RAM y provocando error. Retiene el descarte
  durante el ciclo de respuesta; no anula combinacionalmente la última palabra
  válida cuando avanza la dirección del núcleo. Un tile de 64 tokens llena
  las 4.096 direcciones; el índice truncado adicional vuelve a cero y sigue
  leyendo datos del mismo tile confirmado.

## Validación y cola

La protección de estado pasó 2.304 combinaciones nativas. La de K/V pasó
811.206 comprobaciones: las 4.096 direcciones para tamaños 0–65, CE activo e
inactivo en los límites, conservación de la última respuesta y reset.
Archivo K/V: `d45a6126dc6568e0cb35a7af4ed4231cc938a66ebe7a1ccbe3c4c770b673396f`.
Esto califica las protecciones aisladas; la integración sigue pendiente.

Motor R6 → secuenciador R3 → guard raíz R2 es la cola vigente. Motor R6 usa
`run_ntrpu_attention_engine_vivado_r5.py`, banco R3 y revisor R3. Conserva
las 155 consultas, 9.600 resultados y auditoría Decimal90; añade trazas de
las lecturas aisladas y diez controles de corrupción a los 27 anteriores.
Cada padre debe pasar y quedar respaldado antes de ejecutar al siguiente.

Fuentes de recuperación D:/NFS:
`fa5e1dcabca630f3bbbec4d33f7d274453141d64b32c445f0a2a1f4c26d76d93`.
Cápsula de propietarios/paquetes:
`a307bfbdf3fd032a44cab9e25c08a58737dd221c2533e5989551c4339ccedadf`.
Los resultados físicos, cierre de timing e inferencia completa siguen abiertos.

## Continuación: recuperación R3, 19:20 UTC

El motor R6 avanzó hasta la consulta 135 y falló al continuar en el token 64:
el bucle generado `Pipeline_VITIS_LOOP_103_3` solicita la dirección de estado
104 al salir, aunque solo consume los índices 40–103. Su CE no está condicionado
por la comparación de salida; la escritura del acumulador sí lo está.
La tanda `20261013T184728Z-V7-llama32-attention-engine-vivado-r1` terminó
con diagnóstico y se respaldó: SHA-256
`9f895dab53cdeb410e4995f0a901a2b454b08d3ac0646a4e0fb6d827e320f692`.
No cuenta como PASS del motor.

`ntrpu_attention_state_read_guard_r2` descarta exclusivamente el índice 104
durante continuación y registra el descarte hasta el ciclo de respuesta.
Conserva la palabra 103 cuando avanza la dirección; mantiene aislamiento del
primer tile y reenvía las demás direcciones inválidas al control de rango.
El adaptador R3 incorpora este guard; los 33 archivos generados siguen intactos.

Prueba nativa: `20261013T191455Z-V7-attention-state-read-guard-native-r2`.
Tres semillas, 2.304 comprobaciones de todas las direcciones 0–127, CE activo
e inactivo y tres posiciones de tile; además transición 103→104, reset y
recuperación. Un revisor independiente reproduce la respuesta registrada.
Archivo D:/NFS: `6604cba005e5c9af5e4e39f156bb01411aeea5d8841e5ad9fac05da60cf28807`.

El **motor R7** ejecuta banco R4 y revisor R4 en
`20261013T191941Z-V7-llama32-attention-engine-vivado-r1`. Conserva las 155
consultas y la referencia Decimal90; exige aislar cuatro lecturas de
continuación y aumenta los controles negativos de 37 a 42. Sus dependientes
son secuenciador R4, guard raíz R3 y wrapper completo R2. Cada etapa exige
PASS y respaldo de su padre; no se repiten los bloques de memoria ya calificados.

Fuentes R3: `0ff884611cca2f544c9132a5f88ff0168ec3690941b8c4182d9500e89a3dfb4f`.
Cápsula de cola: `20131e650cd1be32652c8a7fd44773045a28304a719042add0fd2d466f47122d`.
Ambos archivos están verificados en D: y NFS. La prueba nativa no sustituye
la integración XSim y la síntesis pendientes.
