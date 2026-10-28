# Fronteras AXI elásticas del receptor multibuffer

## Alcance

Experimento lateral sobre [fillstage](PEER_RX_MULTIBUFFER_FILLSTAGE.md).
`ntrpu_peer_multibuffer_elastic` envuelve el núcleo preservado con colas
independientes de dos entradas para AW, W y B por destino. Conserva los
puertos de comandos, datos de entrada, completions y telemetría. No se ha
adoptado en el receptor de placa ni en la cola principal de Vivado.

La [composición posterior con CDC y filtro de capacidad](PEER_RX_MULTIBUFFER_MEMORY.md)
ya tiene validación funcional en tres perfiles de reloj. Mantiene este núcleo
sin cambios y registra la serialización del filtro como siguiente límite de caudal;
su integración física con PCIe y DDR/HBM sigue pendiente.

La separación de canales permite que W llegue antes que AW, y que se
acepte otra dirección antes de completar los datos anteriores. La respuesta
B debe esperar ambos handshakes: AW y WLAST. El banco nuevo comprueba
ese contrato sin imponer simultaneidad entre canales
([Arm IHI 0022H, A3](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/IHI0022H_amba_axi_protocol_spec.pdf)).

## Capacidad y coste lógico

| Cola por destino | Ancho | Entradas | Datos almacenados |
| --- | ---: | ---: | ---: |
| AW | 85 bits | 2 | 170 bits |
| W | 577 bits | 2 | 1154 bits |
| B | 6 bits | 2 | 12 bits |

Son 1336 bits de payload y seis bits de ocupación por destino antes de
optimización; para dos destinos, 2684 bits. Vivado puede eliminar atributos
constantes. W añade sólo 128 bytes de datos por destino: estas colas cortan
rutas temporales y absorben pausas breves; no sustituyen el pool de 32 KiB.

Cada cola introduce un ciclo mínimo, mantiene el dato mientras VALID espera
READY y admite una transferencia por ciclo en régimen continuo. Al llenarse,
READY vuelve a habilitarse después de liberar una entrada; no depende
combinacionalmente de READY al otro lado. La estructura sigue el principio
de dos entradas descrito por
[AMD PG373](https://docs.amd.com/r/en-US/pg373-axi-register-slice/Fully-Registered);
el RTL propio no instancia ni acredita equivalencia con esa IP.

## Propiedad, fallo y vaciado

- El banco vuelve al pool al aceptar el último **W interno**. A partir de
  ese flanco, la cola posee una copia completa de los datos pendientes;
  reutilizar BRAM no puede cambiar los beats ya encolados.
- `banks_used` describe BRAM, no la ocupación de las colas. `pending_counts`
  conserva el significado interno: trabajos iniciados que esperan B válido.
- `drained` exige núcleo vacío y las seis colas vacías. Haber liberado todos
  los bancos no basta para declarar finalizado el transporte.
- `stop` cierra admisión y deja vaciar lo aceptado. `fault` se publica cuando
  el núcleo procesa B; una respuesta errónea puede permanecer en la cola
  antes de que cierre admisión. No es una señal combinacional del B externo.
- B con ID ajeno conserva la semántica del núcleo: provoca fallo y no retira
  el trabajo esperado. El destino debe entregar después su B válido. Una
  respuesta espontánea sin trabajo no tiene recuperación automática.
- Reset síncrono activo en bajo descarta las colas. Requiere coordinar la
  época de reset con el destino; no implementa aborto/rearme ni cruce CDC.

## Validación

`tools/run_ntrpu_multibuffer_elastic.py` conserva snapshots y archivos ZIP
por tanda. Prueba nueve geometrías, dos semillas y colas de 6, 85 y 577 bits.
El scoreboard externo comprueba contenido, longitud, orden por destino,
respuesta y completion. `tools/check_ntrpu_elastic.py` reproduce esas
transacciones independientemente. El oráculo anterior sigue comprobando
leases y transacciones en los pines internos del núcleo, sin modificaciones.

En el perfil de un beat, las salidas externas trabajan simultáneamente,
pero las entradas internas a las colas no coincidieron. Se registra esa
excepción de cobertura; siguen vigentes todas las comprobaciones de
corrección del oráculo anterior. No se exige igualdad ciclo a ciclo con
fillstage: la latencia añadida es intencional.

Las pruebas incluyen AW pausado con W adelantado, W pausado con bancos
reutilizados, destinos independientes, stop, créditos de completion,
B erróneo/ajeno y descriptores inválidos. Reset con datos encolados se
verifica en la cola unitaria; reset coordinado del transporte completo
queda para integración.

La tanda final `20261004-rtl-r6` pasó en 12,161 s: 18 simulaciones del
receptor y seis unitarias, 1426 trabajos y 606.080 bytes comprobados. Se
detectaron ocho mutaciones RTL y cuatro corrupciones de trazas. El monitor
observó 281 reutilizaciones de bancos antes del último W externo. En las
pruebas unitarias se descartan deliberadamente dos entradas mediante reset.
Un ensayo adicional del banco de pines públicos pasó 86 trabajos y 81.152
bytes. Las tandas anteriores y sus fallos de infraestructura/cobertura se
conservan junto a la última, sin sobrescribir evidencia.

## Medición física y siguiente integración

La tanda R4 usa VU5P, dos destinos, 64 beats por banco, 16 descriptores y
cuatro pendientes por destino: ocho bancos totales, cuatro de reserva.
Se compara a 5 ns con el mismo arnés de registros y BUFG del experimento
anterior. Las tandas usan dos CPU, prioridad reducida y directorios propios.

La síntesis tardó 267,276 s, limitada a dos CPU: 5034 LUT, 11.975 registros,
15 BRAM y 206 CARRY8, sin DSP, URAM ni LUTRAM. Frente a fillstage añade
681 LUT y 2388 registros, sin aumentar BRAM. Su setup estimado es +1,616 ns;
ese resultado no sustituye el timing después del ruteo. No hubo errores,
avisos críticos, latches, cajas negras ni endpoints internos sin restringir.

La implementación fresca tardó 396,115 s, con colocación `Default` y ruteo
`RuntimeOptimized`, sin reintentos ni cambios de frecuencia:

| Métrica después de P&R | Fillstage | Elastic |
| --- | ---: | ---: |
| LUT del bloque | 4323 | 4982 |
| Registros del bloque | 9587 | 11.975 |
| BRAM | 15 | 15 |
| Setup a 200 MHz | +0,032 ns | **+0,316 ns** |
| Hold | +0,010 ns | **+0,012 ns** |

El arnés añade los mismos 2159 registros y un BUFG: 14.134 registros
totales en esta variante. Cumple setup/hold en 36.292 endpoints y ancho
de pulso en 14.167; no hay violaciones. Las 15.601 nets enrutables están
completas, sin GAP ni errores de ruteo, DRC, latches o cajas negras.
Setup de las fronteras de medición de entrada/salida: +0,518/+1,395 ns;
hold: +0,018/+0,022 ns. No hay endpoints internos sin restringir.

La mejora observada de setup es 284 ps; es una comparación de una
implementación por variante, no una búsqueda de Fmax ni una garantía
para otro floorplan. La peor ruta sigue siendo interna: generación activa
hacia el enable de dirección de un descriptor, 11 niveles, 4,362 ns de
datos y 74,65 % de cableado. Las colas no eliminan ese camino de admisión.

Se revisaron dos avisos críticos `Project 1-486` anteriores a la importación
EDIF; ninguno posterior. Los recursos se conservan durante la importación
y no quedan cajas negras. Se registra un aviso informativo de congestión
`Route 35-443`; el ruteo terminó completo. Persisten avisos por puertos de
prueba sin delays externos, LOC del BUFG y RAM sin registro adicional.
No se aplicaron waivers. Al cumplir directamente 5 ns no se gastó otra
tanda en probar frecuencias menores.

La netlist ruteada pasó en 72,690 s: 86 trabajos, 81.152 bytes, cuatro
pendientes por destino y 261 ciclos con W simultáneo. Comprobó W antes
de AW y B erróneo/ajeno. La reproducción Python coincide con el scoreboard
externo. Es simulación funcional a 10 ns sin SDF, no una medición de
rendimiento a 200 MHz; no observa leases internos. Los avisos son los
conocidos de `glbl` y `LIBRARY_PATH`.

Antes de adopción quedan el puente PCIe/BAR-DMA, controlador DDR/HBM,
arbitraje, reset compartido, CDC si procede y STA del conjunto con su
floorplan. El bus AXI de este módulo no crea una memoria coherente entre
tarjetas ni incrementa por sí solo el ancho de banda físico.

## Evidencia y recuperación

- [RTL final](../../artifacts/ntrpu-multibuffer-elastic/20261004-rtl-r6/REVIEW.json)
  y [resumen](../../artifacts/ntrpu-multibuffer-elastic/RTL_SUMMARY.json).
- [Síntesis](../../artifacts/ntrpu-multibuffer-elastic/20261004T160934939265Z-elastic-synth-r4/REVIEW.json)
  y [coste comparado](../../artifacts/ntrpu-multibuffer-elastic/SYNTH_COMPARISON.json).
- [Implementación](../../artifacts/ntrpu-multibuffer-elastic/20261004T161515328068Z-elastic-route-r4/REVIEW.json),
  [comparación](../../artifacts/ntrpu-multibuffer-elastic/ROUTE_COMPARISON.json)
  y [revisión de avisos](../../artifacts/ntrpu-multibuffer-elastic/IMPORT_WARNINGS_REVIEW.json).
- [Netlist](../../artifacts/ntrpu-multibuffer-elastic/20261004T162247200502Z-elastic-netlist-r4/REVIEW.json)
  y [descarga verificada](../../artifacts/ntrpu-multibuffer-elastic/20261004T162247200502Z-elastic-netlist-r4/DOWNLOAD_VERIFIED.json).
- [Revisión final](../../artifacts/ntrpu-multibuffer-elastic/FINAL_REVIEW.json),
  [fuentes protegidas](../../artifacts/ntrpu-multibuffer-elastic/SOURCE_PRESERVATION.json)
  e [instrucciones de recuperación](../../artifacts/ntrpu-multibuffer-elastic/HANDOFF.md).

La unidad C: se quedó sin espacio durante la extracción de la última
tanda. El ZIP original está completo y verificado. La extracción completa
y el respaldo consolidado se guardan en
`D:\AI\artifacts\ntrpu-multibuffer-elastic\`; `OUTPUT_STORAGE.json` registra
la ubicación. Se conservaron las copias anteriores. El respaldo contiene
fuentes, pruebas, informes, checkpoints y evidencia comparativa de fillstage.
