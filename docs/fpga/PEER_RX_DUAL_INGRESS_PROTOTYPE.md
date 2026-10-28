# Prototipo de recepción con dos entradas independientes

Fecha: 2026-10-07. Estado: **RTL experimental, simulación funcional aprobada
y temporización interna OOC positiva tras ruteo**. Implementa el núcleo de recepción de la
[propuesta](PEER_RX_MULTI_INGRESS_PROPOSAL.md); no sustituye el transporte
calificado ni está conectado todavía a PCIe o al controlador HBM nativo.

## Contrato implementado

[ntrpu_dual_ingress.sv](../../fpga/experimental/ntrpu_dual_ingress.sv) ofrece
comandos y payloads independientes por destino. Aceptar un comando reserva
un banco, tag, longitud y generación para esa entrada; `data_ready[lane]`
solo permite llenar ese marco. Cada entrada mantiene su contador y puede
avanzar mientras la otra se detiene.

El árbitro concede como máximo un banco por ciclo. Dos marcos ya reservados
pueden aceptar un beat de 512 bits cada uno en el mismo ciclo. Se mantienen
ocho bancos de 4 KiB: dos propios por destino y cuatro reservas compartidas,
dieciséis descriptores por destino y cuatro escrituras pendientes por destino.

Cada banco físico tiene un escritor seleccionado y un lector seleccionado.
La exclusividad de las reservas impide que ambas entradas escriban en el
mismo banco. El asignador es una copia exacta del corte de devolución
calificado, conservada en
[ntrpu_dual_allocator.sv](../../fpga/experimental/ntrpu_dual_allocator.sv).
El banco vuelve al pool tras su último W aceptado; B y completion conservan
su estado independiente hasta consumirse.

`stop` queda retenido hasta reset frío y cierra comandos nuevos mientras
permite drenar. El fence global cierra ambas admisiones, espera las
completions externas y retiene su respuesta hasta aceptación. `drained`
excluye una aceptación de comando en ese mismo flanco. Un comando inválido
o una respuesta de error cierra la admisión.

La interfaz es un núcleo **command/payload**, no un reemplazo pin a pin de
`fetch` ni de las colas del transporte. El productor debe enviar exactamente
`LEN+1` beats completos por comando aceptado. La autorización de regiones,
el framing de red/PCIe, las épocas externas y la recuperación coordinada
siguen perteneciendo a la integración superior.

## Validación funcional

[El banco](../../fpga/tb/tb_ntrpu_dual_ingress.sv) cubre ocho escenarios:

1. Llenado simultáneo: ocho marcos de 4 KiB por entrada.
2. Saturación de los ocho bancos con marcos de 4 KiB dos veces sin reset,
   apertura separada de las salidas y reutilización de todos los bancos.
3. Payload de un canal detenido mientras el otro completa ocho marcos.
4. Fence esperando completions bloqueadas y publicación posterior al fence.
5. Stop durante el llenado, con drenaje de los marcos aceptados.
6. Pausas asimétricas de entrada, AW, W y completion; marcos de 1, 9 y 64 beats.
7. Respuesta BRESP de error propagada a completion y cierre de admisión.
8. Comando desalineado rechazado antes de reservar un banco.

Se ejecuta el mismo estímulo con `SERIAL_INPUT=0` y `=1`. El segundo modo
permite como máximo un beat agregado por ciclo y conserva la organización
de bancos: es un control de serialización, no la antigua implementación
compacta. Cada modo verifica **89 descriptores y 253.568 bytes**, con replay
independiente de datos, propiedad, generaciones, orden y completions.

Cada replay rechaza dieciocho corrupciones o supresiones de eventos. Dos
mutaciones reales de RTL, ejecutadas en copias aisladas, también son
detectadas: usar datos del canal equivocado y devolver una generación
incorrecta. El primer intento del prototipo descubrió y corrigió un error
en `drained` al aceptar un comando; ese resultado queda rechazado y guardado.

## Medición de la carga de 64 KiB

Reloj simulado de 200 MHz, dieciséis marcos de 4 KiB, salidas sin pausas
artificiales y respuestas con la misma latencia en ambos modos:

| Medida | Control serial | Doble entrada |
|---|---:|---:|
| Ciclos hasta terminar de entregar payload | 1.033 | 536 |
| Ciclos hasta drenaje completo | 1.105 | 608 |
| Caudal en la ventana de aceptación de payload | 12,713 GB/s | 24,545 GB/s |
| Caudal desde inicio del estímulo hasta drenaje | 11,862 GB/s | 21,558 GB/s |
| Ciclos con dos beats aceptados simultáneamente | 0 | 504 |

La mejora hasta completar toda esta carga es **1,817×**. La ventana de
payload se calcula entre el primer y último beat aceptados, incluyendo
un ciclo final; la ventana hasta drenaje incluye reserva y vaciado. Son
medidas de este banco RTL, sin latencia PCIe/HBM ni distribución física.

## Coste físico inicial

Síntesis OOC para `xcvu47p-fsvh2892-2-e`, Vivado 2026.1:

| Recurso | Control serial | Doble entrada |
|---|---:|---:|
| LUT | 12.144 | 10.068 |
| Registros | 9.964 | 9.913 |
| RAMB36 / RAMB18 | 56 / 8 | 56 / 8 |
| BRAM equivalentes de 36 Kib | 60 | 60 |
| Latches / blackboxes / URAM | 0 / 0 / 0 | 0 / 0 / 0 |

La referencia compacta del transporte usaba quince BRAM equivalentes.
Los tops son diferentes, pero el resultado muestra el coste de la nueva
organización de memoria: **cuatro veces esa cantidad de BRAM**, manteniendo
32 KiB lógicos. No es un coste inevitable de cualquier doble entrada; es
el resultado de ocho arrays anchos y poco profundos.

Con reloj de 5 ns aplicado para el análisis posterior a síntesis, el margen
interno estimado de doble entrada es +1,401 ns en setup y +0,075 ns en hold;
el control serial da +1,585 ns y +0,075 ns. No hay endpoints
internos sin reloj ni latches. Quedan 1.388 entradas y 1.402 salidas sin
presupuesto externo. Estos números no califican 200 MHz tras colocación,
ruteo e integración.

Una tanda posterior de colocación, optimización y ruteo del mismo núcleo de
doble entrada terminó en 423,892 s: **+0,477 ns de setup y +0,042 ns de hold**
con reloj de 5 ns. Las 21.308 redes informadas están completamente ruteadas,
sin errores de ruteo, DRC, latches ni blackboxes. El resultado ocupa
10.006 LUT, 9.908 registros y las mismas 60 BRAM equivalentes.

Es cierre de caminos internos del núcleo aislado. Permanecen las 1.388
entradas y 1.402 salidas sin presupuestos externos; no hay floorplan conjunto
con PCIe/HBM ni prueba de hardware. Esta tanda no incluye simulación de la
netlist ruteada. El [runner](../../tools/run_ntrpu_dual_ingress_route.py) usa
una copia privada del DCP de síntesis; el
[revisor](../../tools/check_ntrpu_dual_ingress_route.py) conserva esos límites
explícitamente en su resultado.

## Reproducción y evidencia

```bash
python3 -B tools/run_ntrpu_dual_ingress.py /doble/output --stage sim
python3 -B tools/run_ntrpu_dual_ingress.py /serial/output --stage sim --serial 1
python3 -B tools/run_ntrpu_dual_ingress.py /sintesis/output --stage synth
python3 -B tools/check_ntrpu_dual_ingress.py /doble/output > /doble/REVIEW.json
```

Cada salida debe ser un directorio nuevo. Las tandas usan dos CPU distintas
de las reservadas para HBM, prioridad baja y límite de 900 s. Fuentes,
configuraciones, informes y trazas se preservan por hash. Evidencia local:
`C:\Users\xasko\artifacts\ntrpu-reserve-dual\20261007T035149419987Z`;
respaldo incremental en `D:\AI\artifacts\ntrpu-content-store`.

## Trabajo siguiente

La [comparación de bancos agrupados](PEER_RX_PACKED_DUAL_INGRESS.md) ya
dispone de un prototipo separado, con pruebas de préstamos y conflictos de
puertos. Sus resultados y límites se registran allí; este RTL permanece
como referencia de ocho arrays independientes.

Antes de integrar el candidato compacto, ampliar la comparación con pausas
representativas del sistema y una asignación que prefiera reservas del
grupo propio. Debe conservar préstamos entre canales y medir conflictos
de puertos RAM, equidad, ocupación, caudal y coste tras ruteo.

Después se necesitan colas/fetch por entrada, integración de CDC y
conversores, réplica de los escenarios con HBM nativo, retorno de lectura
y conexión PCIe/P2P. La presión nativa del receptor anterior se registra por
separado: aprobarla no califica automáticamente este prototipo nuevo.
