# Motor de atención: estudio de implementación física R1

Estado 2026-10-13, 20:08 UTC: **encolado, todavía sin mediciones de ruteo**.
El motor pasó simulación y síntesis, pero su WNS estimado es −0,456 ns a 5 ns.
La ruta crítica de control K/V tiene 76,7 % de retardo de interconexión
estimado; el estudio comprobará su comportamiento después de colocar y rutear.

## Diseño y contexto

Se reutiliza el checkpoint de la tanda
`20261013T191941Z-V7-llama32-attention-engine-vivado-r1`, extraído de su
archivo verificado y contrastado con la evidencia de síntesis. SHA-256:
`ec95b43b16544e94d39f76e69db107bede87836dc030a4d534b4dfc684bd99a6`.

El experimento añade un BUFGCE habilitado permanentemente en `BUFGCE_X0Y60`,
verificando que su inserción preserve exactamente los consumidores del reloj.
La ubicación es provisional. Se conserva el periodo de 5 ns; no se añaden
excepciones de timing, waivers ni rebajas de severidad. Los puertos externos
siguen sin presupuesto de integración: el estudio califica únicamente rutas
internas, sin afirmar timing de placa o del padre completo.

## Ejecución y revisión

`tools/ntrpu_board_attention_engine_route_r1.tcl` ejecuta optimización,
colocación, optimización física y ruteo, con dos CPU y límite de 1.800 s.
Conserva checkpoint y reportes también ante error. Reabre el checkpoint
resultante antes de consultar reloj, rutas, DRC, recursos y timing.

`tools/check_ntrpu_attention_engine_route_r1.py` comprobará procedencia,
top/parte, reloj físico, presupuestos invariantes, contadores de ruteo,
huecos explícitos, cobertura y slacks de setup/hold/pulso. Un resultado
medido puede incumplir timing: se registran separadamente validez de la
evidencia y aprobación de cada criterio físico. Catorce controles de revisión
están implementados y se ejecutarán sobre la evidencia real disponible.

Se exporta una netlist funcional para una validación posterior; este estudio
no ejecuta aún simulación funcional posterior al ruteo. La evidencia aritmética
anterior pertenece al checkpoint de entrada.

## Dependencias y respaldo

El coordinador `queue_llama32_attention_engine_route_vivado_r1.py` espera al
controlador de cuantización y preserva la cola funcional anterior. Antes de
ejecutar vuelve a comprobar el padre motor y los hashes del paquete. La
transferencia de reportes usa compresión; un timeout de observación no
vuelve a lanzar la implementación.

Paquete `20261013T200803Z-llama32-attention-engine-route-vivado-r1`, 15 miembros,
verificado en D:/NFS:

- Paquete SHA-256: `ab6cbb6ebacda7b19220bdd8e9c170c7b6e5e35a0665488f76f78411e92a19c4`.
- Fuente ZIP: `6ca767b23f88ca63616cbff0e9f3a79eb73251e67112a2e2a466c8be5bf0c346`.

No se ha acreditado todavía que esta implementación cumpla 200 MHz.
