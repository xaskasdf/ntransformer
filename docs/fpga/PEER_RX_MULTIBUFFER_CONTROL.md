# Registro anticipado de la devolución de bancos

Experimento aislado del 2026-10-04 sobre el
[multibuffer compacto medido](PEER_RX_MULTIBUFFER_PHYSICAL.md). Se busca
acortar el control de R4 sin cambiar su protocolo, capacidad, asignador
ni política de reutilización. El RTL anterior permanece disponible.

**Resultado: R4 con registro de devolución cumple STA a 166,7 MHz en el
arnés; la referencia compacta falla a esa frecuencia.** Ambas pasan la
regresión funcional. La nueva variante todavía incumple 200 MHz. Es una
candidata para continuar la integración, no un cierre temporal de placa.

## Cambio y contrato

La variante [control](../../fpga/rtl/ntrpu_peer_multibuffer_control.sv)
captura `bank` y `generation` del descriptor en dos registros por canal,
`active_bank` y `active_generation`, cuando se activa `start_write`.
La devolución usa esos registros en el handshake del último beat W.

```mermaid
flowchart LR
    D[Descriptor seleccionado] -->|start_write| C[Registro de banco y generación]
    C --> V[Validación de devolución]
    W[Último W aceptado] --> V
    V --> A[Asignador y posible reutilización]
```

La selección entre descriptores ocurre antes del ciclo que devuelve el
banco. El identificador y la generación se mantienen mientras la
escritura está activa. El primer W sólo puede aceptarse después de
`start_write`, por lo que no se añade espera al handshake. La devolución
sigue ocurriendo en el mismo ciclo que antes; el asignador puede
reasignar inmediatamente ese banco. B y CPL mantienen su independencia.

Reset limpia los registros. Stop y fault conservan el drenaje de trabajos
aceptados. La generación mantiene sus 64 bits y las reglas anteriores de
propiedad, agotamiento y rechazo de devoluciones. Los registros se
actualizan al comenzar la siguiente escritura, no al aceptar otro comando.
El incremento lógico previo a optimización es de 134 FF para R4 y 136
para R8; la síntesis determina el coste físico efectivo.

## Regresión local

El [ejecutor](../../tools/run_ntrpu_multibuffer_control.py) conserva copias
inmutables del RTL, banco y oráculo existentes. Sólo especializa el nombre
del DUT en una copia del banco y añade un monitor: mientras un canal está
activo, su banco y generación registrados deben coincidir con los del
descriptor que se está enviando.

La tanda inicial pasó 36 simulaciones: nueve perfiles y dos semillas,
tanto para la referencia compacta como para la variante. Las líneas
TRACE/CASE/PASS coinciden ciclo por ciclo para todas las comparaciones.
La variante verificó 1336 trabajos, 600.320 bytes y 16.182 comprobaciones
del registro de devolución. Dos mutaciones, banco incorrecto y generación
obsoleta, dispararon el monitor. Tiempo local total: 12,22 s.

Se cubren R4/R8, un solo canal, reserva cero, un beat, pendientes limitados,
cantidades no binarias de descriptores/bancos y stride de tres beats,
además de las pausas, errores, propiedad y drenaje del banco existente.
Es una regresión dinámica, no equivalencia formal exhaustiva. El asignador
no cambia; conserva su evidencia previa de generaciones y cuotas.

En Linux/WSL con Icarus disponible:

```bash
python3 -B tools/run_ntrpu_multibuffer_control.py /ruta/nueva/rtl
```

El ejecutor acepta `--iverilog`, `--vvp` y `--ivl-base` para la instalación
local extraída. Nunca reutilizar una carpeta de salida.

## Síntesis y medición física

La síntesis R4 terminó en 263,979 s:

| Medición de síntesis | Compacta R4 | Registro de devolución R4 |
| --- | ---: | ---: |
| LUT | 4284 | 4304 |
| FF | 9386 | 9528 |
| Tiles BRAM | 15 | 15 |
| CARRY8 | 164 | 164 |
| WNS estimado a 5 ns | +0,742 ns | +1,326 ns |
| Niveles de la peor ruta | 16 | 14 |

Son 20 LUT y 142 FF adicionales después de optimización; el coste físico
no tiene por qué coincidir con los 134 bits nuevos del RTL. Las dos
réplicas RAM siguen usando 14 RAMB36 y dos RAMB18. No hay latches, cajas
negras, lazos, endpoints internos sin restricciones ni hallazgos DRC.
El nuevo aviso `Netlist 29-101` señala muchos elementos primitivos en una
celda y recomienda conservar jerarquía para un futuro floorplan.

Colocación/enrutado R4 terminó en 750,018 s, sin reintentos. Se conservan
el dispositivo VU5P
`xcvu5p-flva2104-2-i`, reloj objetivo de 5 ns y el arnés registrado anterior:
BUFGCE real, 673 bits de entrada, 1486 de salida e incertidumbres adicionales
setup/hold de 0,100/0,050 ns. La tanda usa dos CPU y `nice 10`, separada
de la simulación principal. No se modifican excepciones temporales.

Se usaron placement `Default` y route `RuntimeOptimized`, dos CPU y
`nice 10`. La referencia anterior reanudó su checkpoint colocado; esta
variante implementó desde la netlist sintetizada. No es una evaluación
estadística de varias colocaciones ni una búsqueda de Fmax.

| Medición posterior a P&R | Compacta R4 | Registro de devolución R4 |
| --- | ---: | ---: |
| LUT / FF del núcleo | 4275 / 9386 | 4313 / 9528 |
| Tiles BRAM / CARRY8 | 15 / 164 | 15 / 164 |
| WNS a 200 MHz, 5 ns | −1,214 ns | −0,781 ns |
| WNS a 166,7 MHz, 6 ns | −0,214 ns | +0,219 ns |
| WNS a 150 MHz, 6,667 ns | +0,453 ns | +0,886 ns |
| WHS en los periodos auditados | +0,012 ns | +0,005 ns |

El coste adicional medido es de 38 LUT y 142 FF; payload y BRAM no
cambian. El arnés añade 2159 FF y un BUFGCE, separados del núcleo.
Las 13.319 nets enrutables están completas, sin errores de ruteo, GAP,
latches, cajas negras ni hallazgos DRC. No quedan endpoints internos sin
reloj o restricciones. A 200 MHz quedan 3683 violaciones setup, TNS
−1348,326 ns; a 6 ns no hay violaciones setup, hold ni ancho de pulso.
El margen de hold es pequeño y deberá medirse de nuevo en el padre real.

Las auditorías de ambas implementaciones a 6 y 6,667 ns verifican que
ningún LOC, BEL o ROUTE cambia y que los DCP originales conservan sus
hashes. Mantienen las mismas incertidumbres y no añaden excepciones.
La frecuencia de 166,7 MHz describe STA del arnés, no rendimiento medido
en hardware. R8 sólo se validó funcionalmente en esta variante.

La netlist ruteada pasó 81 trabajos, 80.832 bytes y 3869 ciclos, con cuatro
pendientes por puerto, 253 ciclos de W simultáneo y los dos casos B
anómalos. Las 2881 líneas TRACE/CASE/PASS coinciden con la netlist compacta
anterior. Tiempo XSIM: 62,241 s; simulación funcional a 10 ns, sin SDF.
La identidad observada en RTL y netlist no prueba equivalencia exhaustiva.

Se conservan dos avisos críticos `Project 1-486` al sintetizar/reabrir el
arnés con su `core` todavía vacío. Ambos preceden a la importación EDIF;
después no hay avisos críticos, las cantidades de recursos coinciden
antes/después de importar y el recuento final de cajas negras es cero.
No se aplicaron waivers. Persisten los avisos OOC de pines sin partpins,
delays externos ausentes, LOC del BUFG, RAM sin registro de salida
adicional y el incumplimiento de timing a 5 ns. XSIM conserva los avisos
conocidos de `glbl` y `LIBRARY_PATH`.

Los pines de prueba externos del arnés siguen sin contrato temporal;
todavía faltan integración PCIe/HBM, fronteras elásticas, CDC,
ventanas/épocas y recuperación del sistema.

## Siguiente optimización acotada

La [variante de validación por banco y concesiones one-hot](PEER_RX_MULTIBUFFER_DECODED.md)
ya fue medida: R4 cumple STA a 175 MHz nominales en el mismo arnés,
conserva las trazas y todavía incumple 200 MHz. Sus resultados y costes
se registran por separado; las mediciones de esta página no cambian.

La peor ruta nueva va de `allocator/bank_state[6].generation_reg[6][18]`
al enable de `allocator/bank_state[2].generation_reg[2][16]`. Tiene 12
niveles y 5,539 ns de datos: 1,045 ns de lógica y 4,494 ns de cableado
(81,14 %). Se acortó la selección del descriptor; el camino crítico
permanece en la validación de devolución y reasignación.

La siguiente variante debe comparar validación por banco con índices
constantes y selección one-hot para los enables. Medir el coste de
replicar comparadores frente a reducir selección y fanout. Conservar
generación de 64 bits, cuotas, prioridad, round-robin, rechazo de retornos
obsoletos, stop/fault y reutilización en el mismo ciclo. Primero repetir
la regresión; sólo medir físicamente las variantes que la superen.

## Evidencia

- [Regresión y mutaciones](../../artifacts/ntrpu-multibuffer-control/20261004-rtl-r1/REVIEW.json).
- [Resumen y compatibilidad del arnés](../../artifacts/ntrpu-multibuffer-control/RTL_SUMMARY.json).
- [Síntesis R4](../../artifacts/ntrpu-multibuffer-control/20261004T135909167168Z-control-synth-r4/REVIEW.json).
- [Implementación R4](../../artifacts/ntrpu-multibuffer-control/20261004T140442521580Z-control-route-r4/REVIEW.json).
- [Netlist posterior a P&R](../../artifacts/ntrpu-multibuffer-control/20261004T140442521580Z-control-route-r4-netlist/REVIEW.json).
- [STA de la variante a 166,7/150 MHz](../../artifacts/ntrpu-multibuffer-control/20261004T140442521580Z-control-route-r4-period/REVIEW.json)
  y [comparación con R4 anterior](../../artifacts/ntrpu-multibuffer-control/20261004T140442521580Z-control-route-r4-baseline-period/REVIEW.json).
- [Revisión de los avisos de importación](../../artifacts/ntrpu-multibuffer-control/IMPORT_WARNINGS_REVIEW.json).
- [Hashes iniciales de referencia](../../artifacts/ntrpu-multibuffer-control/SOURCE_BASELINE.json).
- [Fuentes de referencia preservadas](../../artifacts/ntrpu-multibuffer-control/SOURCE_PRESERVATION.json).
- [Revisión consolidada](../../artifacts/ntrpu-multibuffer-control/FINAL_REVIEW.json).
- [Respaldo local](../../artifacts/ntrpu-multibuffer-control/20261004-control-handoff.zip),
  [verificación](../../artifacts/ntrpu-multibuffer-control/20261004-control-handoff.VERIFIED.json)
  e [instrucciones](../../artifacts/ntrpu-multibuffer-control/HANDOFF.md).

La variante no sustituye automáticamente al receptor integrado ni a la
referencia R4. Su adopción depende del resultado físico y de la integración
posterior con un transporte real.
