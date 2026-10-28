# Implementación física del multibuffer compacto

Evaluación aislada del 2026-10-04: Vivado 2026.1, VU5P
`xcvu5p-flva2104-2-i`. Se conserva el
[RTL compacto validado](PEER_RX_MULTIBUFFER_COMPACT.md), sin integrar cambios
en el receptor ni en la cola principal. Dos destinos, bancos de 4 KiB,
16 descriptores y cuatro operaciones pendientes por destino.

## Resultado disponible

**R4 y R8 terminaron el enrutado y pasaron la simulación funcional posterior.**
Ambas implementaciones incumplen 200 MHz. El análisis estático sobre sus
colocaciones y cableados fijos cumple a 150 MHz para R4 y a 125 MHz para
R8 dentro del arnés descrito abajo. Esto no es cierre temporal de una
placa ni una búsqueda de Fmax. Mantener R4 como referencia de integración
y optimizar el control antes de ampliar la reserva.

| Medición | R4 | R8 |
| --- | ---: | ---: |
| Bancos de reserva + propios | 4 + 4 | 8 + 4 |
| Payload lógico | 32 KiB | 48 KiB |
| LUT / FF del núcleo tras enrutado | 4275 / 9386 | 4869 / 9715 |
| Tiles BRAM; RAMB36 / RAMB18 | 15; 14 / 2 | 29; 28 / 2 |
| CARRY8; LUTRAM / URAM / DSP | 164; 0 / 0 / 0 | 240; 0 / 0 / 0 |
| FF adicionales del arnés; BUFGCE | 2159; 1 | 2163; 1 |
| Nets enrutables / completamente enrutadas | 13.087 / 13.087 | 14.278 / 14.278 |
| Errores de ruteo / nets con GAP / hallazgos DRC | 0 / 0 / 0 | 0 / 0 / 0 |
| Endpoints internos sin reloj o restricciones | 0 | 0 |

| Frecuencia nominal | Periodo | WNS R4 | WNS R8 | WHS ambos | WPWS ambos |
| --- | ---: | ---: | ---: | ---: | ---: |
| 200 MHz | 5,000 ns | −1,214 ns | −2,234 ns | +0,012 ns | +1,958 ns |
| 150 MHz | 6,667 ns | +0,453 ns | −0,567 ns | +0,012 ns | +2,791 ns |
| 125 MHz | 8,000 ns | +1,786 ns | +0,766 ns | +0,012 ns | +3,458 ns |

A 200 MHz hay 9316/9728 endpoints con violaciones setup y TNS
−4403,609/−13.775,655 ns para R4/R8. R8 aún tiene 1933 violaciones y
TNS −393,612 ns a 150 MHz. Los perfiles que cumplen no tienen violaciones
setup, hold ni ancho de pulso. Se comprueban 30.899/32.212 endpoints
setup/hold. La auditoría conserva todas las ubicaciones LOC/BEL y rutas
físicas y verifica el hash del DCP antes/después; sólo cambia el periodo
analizado, conservando incertidumbres.

R8 aporta 50 % más payload lógico con 93,3 % más BRAM en estos perfiles.
Las estrategias de placement y los presupuestos CPU difieren: esta prueba
no aísla cuánto de la diferencia temporal causa exclusivamente la reserva.
No demuestra un límite arquitectónico de frecuencia.

## Contexto físico y límites

El [arnés](../../tools/prepare_ntrpu_multibuffer_physical_fixture.py)
instancia un BUFGCE real en `BUFGCE_X0Y60` y registros preservados en
frontera: 673 bits de entrada y 1486/1490 de salida para R4/R8. Todos
comparten reloj; incertidumbre adicional setup/hold: 0,100/0,050 ns.

Vivado abre el DCP sintetizado, exporta EDIF y lo incorpora como `core`
mediante `update_design -cells core -from_file`. Los recursos coinciden
antes/después de importar. Sólo se sintetiza el arnés; no se modifica el
RTL del núcleo. Esta importación está documentada en
[AMD UG909](https://docs.amd.com/r/en-US/ug909-vivado-partial-reconfiguration/Method-4-Open-Checkpoint/Update-Design).

**El arnés sirve para medir timing; no implementa un adaptador AXI.**
Registrar ready/valid independientemente no crea una frontera elástica.
Los pines de prueba externos quedan sin delays y sin posiciones partpin:
sus trayectos hasta los registros de frontera no están calificados. Sí
se analizan los caminos registrados del núcleo, ingreso y egreso. No
hay false paths, multicycle, waivers ni excepciones para ocultar rutas.

Para R4 a 200 MHz, el peor ingreso registrado tiene +0,010 ns y el egreso
+0,304 ns; la violación dominante es interna. El reloj real está enrutado.
Los arneses completos usan 11.545/11.878 FF. Separar ese coste del núcleo;
R8 termina con 20 FF más que su netlist sintetizada tras optimización física.

## Ruta crítica y siguiente diseño

Actualización posterior: el [registro anticipado de devolución](PEER_RX_MULTIBUFFER_CONTROL.md)
ya se midió como variante separada. R4 alcanza STA a 166,7 MHz en su
arnés, conserva el comportamiento probado y mantiene 15 tiles BRAM.
Los valores de referencia de este documento permanecen sin cambios.

La peor ruta R4 va de `core/bank_reg[10][0]` al enable de
`core/allocator/bank_state[4].generation_reg[4][44]`. Atraviesa 15 niveles
lógicos: 5,972 ns de datos, con 1,478 ns de lógica y 4,494 ns de cableado
(75,25 %). La devolución validada afecta disponibilidad, selección y
reasignación del banco en el mismo ciclo.

En R8, `core/send_head_reg[0][1]_replica` llega a
`core/generation_reg[5][17]/D`: 16 niveles, 7,120 ns de datos y 5,576 ns
de cableado (78,31 %). Ingreso/egreso también fallan a 200 MHz:
−0,233/−0,396 ns. Ambas mediciones señalan el control de devolución,
selección y actualización de descriptores como trabajo prioritario.

El siguiente experimento debe reducir ese recorrido y su fanout:

1. Revisar colocación relativa de descriptores, validación de devoluciones
   y asignador; usar una región de reloj acorde con el futuro padre.
2. Comparar una etapa explícita de devolución/asignación si la colocación
   no basta. Contabilizar cualquier concesión en tránsito en propietario,
   generación, créditos, stop, fault y drenaje. Conservar la reutilización
   en el mismo ciclo o versionar deliberadamente ese contrato y sus pruebas.
3. Repetir regresión, medición física y replay de netlist; comparar caudal
   y pausas antes de seleccionar R4/R8. Más reserva no añade ancho de banda
   sostenido al ingreso.

La integración necesitará fronteras AXI elásticas, reloj y floorplan del
padre, framing PCIe, ventanas/épocas, CDC y recuperación. Esta evaluación
no prueba coherencia BAR, conectividad HBM ni una tarjeta de 512 GiB.

## Validación funcional y avisos

El [verificador de netlists](../../tools/check_ntrpu_multibuffer_routed_netlist.py)
repite el oráculo de transacciones/bytes y compara todas las trazas con
la síntesis compacta anterior:

| Perfil | Trabajos / bytes | Ciclos | W simultáneo | Líneas idénticas |
| --- | ---: | ---: | ---: | ---: |
| R4 | 81 / 80.832 | 3869 | 253 | 2881 |
| R8 | 85 / 81.088 | 4046 | 252 | 2905 |

Ambos alcanzaron cuatro pendientes por puerto e incluyen una respuesta
B con error y otra con ID ajeno. Es simulación funcional a 10 ns, sin
SDF, mediante pines públicos; las comprobaciones internas de préstamos
siguen en la regresión RTL previa. Tardaron 61,205/63,306 s; las auditorías
de frecuencia, 56,506/58,059 s.

Se conservan avisos de partpins y delays externos ausentes por los pines
de prueba; `TIMING-18` muestra 1000 entradas por límite del informe, aunque
`check_timing` enumera 673 entradas y 1486/1490 salidas. `SYNTH-6` informa
16/30 RAM sin registro adicional de salida integrado; `TIMING-16`,
157/1000 hallazgos de violaciones grandes (R8 alcanza el límite del informe).
`CLKC-3`/`Constraints 18-4434` recomiendan
`CLOCK_REGION` frente al LOC fijo del BUFG. No hay errores ni avisos
críticos en las tandas físicas completadas. XSIM conserva avisos de `glbl`
duplicado y `LIBRARY_PATH`.

## Intentos conservados y ejecución

La [lista de tandas](../../artifacts/ntrpu-multibuffer-physical/ATTEMPTS.json)
incluye los fallos. Dos comprobaciones iniciales leían la propiedad
equivocada del top del DCP; dos importaciones `read_checkpoint -cell`
fallaron con `Project 1-9`, antes de usar EDIF. Un contexto OOC con
`HD.CLK_SRC`, delays entrada 1/0,2 ns y salida 1/−0,1 ns agotó 1200 s:
placement daba WNS −2,131 y WHS −3,133 ns, dominados por fronteras con
reloj externo supuesto. `HD.CLK_SRC` no instancia un buffer; véase
[AMD: reloj OOC](https://docs.amd.com/r/2021.2-English/ug905-vivado-hierarchical-design/I/O-and-Clock-Buffers).

El primer arnés registrado agotó 900 s. Reanudar su checkpoint colocado
con `RuntimeOptimized` completó R4 en 411,810 s usando dos CPU; R8 agotó
1200 s. La última tanda R8 completó en 893,925 s al reubicar con
`AltSpreadLogic_high`, ocho CPU y límite de 1200 s; esta directiva es una
alternativa documentada para
[congestión](https://docs.amd.com/r/en-US/ug949-vivado-design-methodology/Use-Alternate-Placer-and-Router-Directives).
Todos usan `nice 10`, afinidad acotada y procesos independientes de la
cola principal. Cada timeout detiene sólo sus propios procesos.

Desde una copia independiente, con Vivado cargado:

```bash
python3 -B tools/run_ntrpu_multibuffer_route.py /tanda/output \
  --synth-input /tanda/inputs/checkpoints --context registered \
  --reserves 4 --threads 2 --timeout-seconds 1200 \
  --route-directive RuntimeOptimized
python3 -B tools/check_ntrpu_multibuffer_route.py /tanda --reserves 4
```

La carpeta de salida debe ser nueva; mantener fuentes en `/tanda/inputs`
y los DCP bajo `checkpoints/reserve_4` o `reserve_8`. Para reproducir la
tanda concreta, usar su `inputs.zip` y comando registrado: R4 reanuda el
placement previo mediante `--resume-placed-from`; un placement nuevo no
garantiza sus mismos resultados.

## Evidencia local

- [R4: recursos, rutas, timing y avisos](../../artifacts/ntrpu-multibuffer-physical/20261004T081903830609Z-registered-runtime/REVIEW.json).
- [R4: replay funcional tras P&R](../../artifacts/ntrpu-multibuffer-physical/20261004T081903830609Z-registered-runtime-netlist-r4/REVIEW.json).
- [R4: auditoría a 150/125 MHz](../../artifacts/ntrpu-multibuffer-physical/20261004T081903830609Z-registered-runtime-period-r4/REVIEW.json).
- [R8: recursos, rutas, timing y avisos](../../artifacts/ntrpu-multibuffer-physical/20261004T084926214888Z-registered-spread-r8/REVIEW.json).
- [R8: replay funcional tras P&R](../../artifacts/ntrpu-multibuffer-physical/20261004T084926214888Z-registered-spread-r8-netlist/REVIEW.json).
- [R8: auditoría a 150/125 MHz](../../artifacts/ntrpu-multibuffer-physical/20261004T084926214888Z-registered-spread-r8-period/REVIEW.json).
- [Preservación de RTL](../../artifacts/ntrpu-multibuffer-physical/RTL_PRESERVED.json).
- [Revisión consolidada](../../artifacts/ntrpu-multibuffer-physical/FINAL_REVIEW.json).
- [Respaldo local](../../artifacts/ntrpu-multibuffer-physical/20261004-physical-handoff.zip),
  [verificación](../../artifacts/ntrpu-multibuffer-physical/20261004-physical-handoff.VERIFIED.json)
  e [instrucciones de restauración](../../artifacts/ntrpu-multibuffer-physical/HANDOFF.md).

Cada tanda descargada conserva entradas, comandos, logs, checkpoints,
informes y manifiestos con SHA-256 y CRC verificados. El estado global
fallido de la tanda mixta se conserva: su revisión selecciona únicamente
R4 y excluye explícitamente el intento R8 incompleto.
