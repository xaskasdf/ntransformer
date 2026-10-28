# Control NTRD para el recolector y dos canales

Perfil experimental separado del control NTRF y del control FFN. Estado
2026-10-05: validado en RTL y netlist sintetizada ante pares de protocolo.
No modifica las ABI anteriores.
Fuentes: `ntrpu_pcie_fragment_dual_control.sv` y composición
`ntrpu_pcie_fragment_dual_managed.sv`. Una reserva exclusiva de 64 bytes;
el canal procede del descriptor privilegiado, nunca de un bit de dirección.

## Descubrimiento y registros

El controlador host debe verificar la tupla completa: ID `0x4452544e` (NTRD),
versión `0x00010000`, capacidades `0x000000ff`, magia F01 y dimensiones.
No aceptar por ID aislado ni tratar NTRF/FFN como compatibles. Capacidades
0–4: barrera externa, F01, identidad de 64 bits, snapshot diagnóstico y
contadores de rechazo saturados; 5: selector de dos canales; 6: sesión
retenida/invalida; 7: ACK con retiro conjunto.

| Dirección | Acceso | Contenido |
| --- | --- | --- |
| `00/04/08` | R | ID, versión, capacidades |
| `10` | W | 1: encolar ARM; 2: STOP; 4: ACK; 8: snapshot; 0: ninguna operación |
| `14` | RW | Habilitación IRQ, bit 0 |
| `18/1c` | R | ARM aceptados / ACK consumidos, contadores módulo 2³² |
| `20–3c` | RW | Época, tag, ventana BAR y destino local; pares de 32 bits, bajo primero |
| `40–4c` | RW | Época y tag del ACK, completos |
| `60–70` | R | Época, tag y estado del resultado retenido |
| `78/7c/80` | R | Pruebas externas de quiescencia y época; no son mandos del host |
| `90/94/98/ac` | R | Snapshot: rechazos, identidades, razón y validez |
| `9c/a0/a4/a8` | R | Magia F01, 64 bytes por trama/línea, 32 bytes de payload |
| `b0` | RW | Canal shadow de 32 bits; únicamente 0 o 1 |
| `b4` | R | Canal capturado al encolar ARM |
| `b8` | R | Bits 0–6: sesión inválida, reserva activa del puente, ACK listo, admisión del pool, cookies agotados, fallo del puente, fallo del pool |
| `bc/c0` | R | Dos canales / 37 bits de dirección local |

El resto de diagnósticos `00–ac` conserva el mapa de fragmentos. En `0c`,
bits 10/11 describen puente libre/reserva activa. No reinterpretar `ac`,
que sigue siendo `snapshot_valid`. Direcciones desalineadas o fuera de
`00–c0` devuelven DECERR; escribir registros de solo lectura devuelve SLVERR.

## Propiedad, errores y recuperación

Aplicar WSTRB por byte antes de validar el nuevo canal; valores mayores a 1
se rechazan sin modificarlo. Descriptor y selector quedan bloqueados con
ARM pendiente, reserva en ejecución, resultado retenido o sesión no utilizable.
ARM exige época creciente y ventana/destino alineados a 64 bytes; el destino
de 64 bits debe ser menor a 2³⁷ antes de reducirlo para cada canal. Puede
esperar calibración y barreras; su aceptación y payload son conjuntos.

ACK comprueba identidad completa, resultado, sesión y `ack_ready` de la
composición. Una escritura rechazada no emite pulso de retiro. El ACK válido
consume resultado y reserva en un solo flanco. STOP conserva el trabajo
expuesto; no es un reset. Un fallo anterior a LINE permite retiro y nueva
época mediante la cancelación limitada a la reserva ya validada.

IRQ habilitada indica resultado, fallo de control, pérdida de sesión,
agotamiento de cookies o fallo del puente/pool. Los diagnósticos y su snapshot
siguen legibles durante un fallo. Restablecer admisión no reabre la sesión;
recolector, puente y control comparten la vida fría de configuración. Los
resets operativos del pool no deben borrar estos registros. El módulo `ntrpu_pcie_fragment_dual_managed` no incluye CDC, supervisor,
MIG ni PCIe nativo; su composición posterior se documenta al final.

## Validación y recuperación del banco

`20261005T043738Z-V7-ffn-supervised-fragment-dual-startup` pasa seis
simulaciones: semillas 1/7/42 en RTL y netlist sintetizada. Por ejecución:
257 escrituras y 69 lecturas AXI-Lite; 51 escrituras rechazadas, 11 reservas,
23 tramas y ocho retiros conjuntos. Doce escenarios cubren barreras,
identidades completas, WSTRB, canal inmutable, diagnóstico retenido, IRQ,
conflicto local → ACK → nueva época y sesión perdida tanto activa como libre.
El verificador reconstruye registros y transacciones desde puertos, incluida
la estabilidad de R durante un snapshot concurrente.

El RTL de la tanda `042455Z` detectó nueve mutantes; su revisor rechazó
236 alteraciones de evidencia. La repetición conserva ese RTL y reutiliza
su netlist sin sintetizar otra vez: SHA-256
`6f58afe782590456bd068fbaa70be0fea4e3415cfc4135d6965277896000a325`.
Recursos: 2.182 LUT, 3.405 registros y cero BRAM/URAM/DSP. Timing interno
OOC de 5 ns sin colocar: WNS 1,988 ns / WHS 0,066 ns; faltan presupuestos
de 974 entradas y 2.442 salidas, además de inserción/skew y ruteo.

Se conservan dos fallos previos, sin calificarlos como éxitos: `042341Z`
no compiló por una declaración tardía del banco; `042455Z` pasó RTL pero
su primera lectura de netlist llegó a 27,5 ns, dentro del GSR inicial,
y terminó por timeout. La repetición usa otra copia del banco: reset
inicial de 200 ns y observación de `glbl.GSR`, sin forzarlo ni modificar
el modelo público AMD. GSR se observa alto hasta 100 ns; primer AR a
207,5 ns. Ocho corrupciones de esa observación fueron rechazadas.

Las trazas RTL/netlist coinciden desde el ciclo 2; el primer muestreo
precede al reset síncrono y difiere por INIT de primitivas. Esto valida el
banco corregido, no el arranque físico ni un reset caliente DDR. El top
con supervisor/pool/MIG deberá repetir su propia prueba de vida fría.

Respaldo local/NFS verificado de la repetición: 104 archivos, 8.687.524 bytes;
SHA-256 `8d9002623cf8f6c9ae13b6aad89585652c31987df8436854d57f80aed3bdc571`.
El intento de compilación se conserva con SHA-256
`d98a02e661c49540ca213f6b52ff741e86a008e07f0962b6ecb93b43cba45662`;
RTL/síntesis y replay fallido, con
`1561e1f0d960c4b4474beaa685a1f67f6afde018a6306bd0f671a475df1f6529`.
Vivado empleó 75,911 s en la repetición. Línea → ACK ocupa 68–268 ciclos
de 5 ns, incluidos accesos MMIO, rechazos y pausas del banco; no mide
caudal sostenido ni latencia física de DDR.


## Vida retenida con supervisor y pool reales (2026-10-05)

`20261005T045125Z-V7-ffn-supervised-managed-dual-lifetime` queda aprobado.
El nuevo `ntrpu_pcie_fragment_dual_mig.sv` une el control NTRD/recolector/puente
con el supervisor, pool y dos CDC AMD. Conserva una sola reserva; no duplica
el puente. Los tres primeros bloques comparten un reset de vida fría,
separado de los resets operativos. STOP llega al pool solamente cuando la
reserva del puente está libre; la marca retenida del recolector no sirve
para decidirlo.

Ocho escenarios: cuatro trabajos sanos con STOP parcial y conflicto local;
pérdida de calibración con media línea; reset tras AW/W de memoria;
reloj fuente detenido con resultado; reloj de memoria detenido con AW MMIO
sin W; reset UI con R MMIO retenida; reset con AW F01 sin W; y pérdida de
calibración con B MMIO retenida. Los siete fallos conservan reserva, época,
cookie, diagnóstico y obligaciones aceptadas. Tras restaurar entradas,
ACK, nuevo ARM y cambio de canal siguen rechazados; no hay recuperación
caliente implícita. El reinicio sincronizado llega 3–5 muestras después
de la señal del supervisor.

El revisor reconstruye MMIO y F01 por ciclo, compara los 64 bytes de cada
escritura y exige causalidad B → completion → resultado → ACK. Detecta los
tres mutantes RTL (reset que borra sesión, canal invertido y dato alterado)
y 259 corrupciones de evidencia. Los 325 avisos corresponden a 320 modelos
FIFO, cuatro avisos LIBRARY_PATH y una regeneración de constraints; los FIFO
conductuales no modelan retardo de sincronización. Las señales de calibración,
reloj y destinos AXI son drivers del banco: esta tanda todavía no usa MIG.
El reset externo de dos casos consta en la fuente congelada y sus efectos;
no tiene un campo independiente en la traza de esta revisión.

Simulación positiva hasta 7,990 us; Vivado empleó 191,869 s para las cuatro
variantes. Respaldo local/NFS: 131 archivos, 29.563.331 bytes, SHA-256
`d1df17a5c33a86224de5c7410d6c6a30e2a899f5ac640d07b8c113fdcc77cc9c`.
La siguiente sección registra la tanda posterior con dos MIG/modelos
genéricos DDR4. Mantener separadas sus pruebas funcionales y el timing físico.


## NTRD sobre dos MIG: readback completo (2026-10-05)

`20261005T050945Z-V7-ffn-supervised-managed-dual-mig` aprobado y respaldado. El banco
`tb_ntrpu_pcie_managed_dual_mig.sv` envía 35 tramas F01 mediante el control
NTRD: 18 reservas/resultados/retiros, 16 líneas completas y 16 escrituras
DDR, cada una de 64 bytes. Las 28 lecturas comparan 1.792 bytes con una
memoria de referencia reconstruida desde las escrituras confirmadas.
Se verifican direcciones sobre 4 GiB, los cuatro ranks de ambos canales,
el extremo `0x1fffffffc0`, sobrescrituras y conservación tras STOP parcial
o conflicto local. STOP con W o B retenidas no publica éxito anticipado;
una nueva época funciona después del retiro correcto.

El verificador reconstruye 274 escrituras y
9 lecturas AXI-Lite, con identificación NTRD completa,
canal explícito e identidades de 64 bits. La traza emparejada comienza en
el primer ciclo fuente y contiene 11,445 muestras.
Ambas calibraciones son necesarias; durante 1.200 ciclos solo el primer
canal está listo y no se admite trabajo. Los UI tienen período de 5 ns
con diferencia de fase de 731 ps.

Se auditaron 35 archivos del modelo público y 143 entradas compiladas;
solo cambian las tres definiciones de geometría/timing ya documentadas.
Los 25 avisos de herramienta coinciden con la referencia revisada y los
diez avisos FIFO declaran su limitación conductual. Los avisos del modelo
DDR (864/144 por canal) se asignan a lecturas periódicas no escritas,
con control de multiplicidad por chip y retorno. No se suprimen diagnósticos.
El revisor rechaza 209 alteraciones de evidencia.

Vivado empleó 434.815 s; el banco termina a
57.225 us simulados. Línea → respuesta B ocupa
88.498–678.498 ns;
lectura AR → R, 115.000–735.000 ns.
Son latencias funcionales con pausas deliberadas y modelos DDR; no son
caudal sostenido, timing de placa ni resultados de inferencia.

Respaldo local/NFS verificado: 650 archivos, 78,790,889 bytes;
SHA-256 `d555e2efcb6c27c2f7da0b87c529ca60021b306cc3a6b709fbd3ed10334c5bc5`. Esta tanda contiene dos MIG genéricos y un host
AXI de prueba. No homologa DIMM comercial, reset caliente, PCIe nativo,
cuatro canales ni implementación física. Continúa el
[plan de integración PCIe nativa](PCIE_DUAL_NATIVE_PLAN.md).

## Transporte host CDC con dos MIG (2026-10-05)

`20261005T055601Z-V7-ffn-supervised-managed-host-cold-r2` aprobado y
respaldado: dos conversores AMD conectan el host de 7,3 ns al núcleo de
5 ns. Coinciden 35 transferencias AW/W/B de datos, 276 escrituras y diez
lecturas de control en ambos extremos; 18 reservas se retiran después
del drenaje. Las 28 lecturas DDR comparan 1.792 bytes. El revisor rechaza
80 alteraciones y el intento anterior que terminaba antes del último
retorno de confirmación. No cambia el RTL entre ambos intentos.

El [contrato y las métricas](PCIE_DUAL_HOST_CDC.md) detallan la diferencia
entre presentación y aceptación AXI-Lite, modelos, avisos y límites.
Respaldo de 716 archivos: SHA-256
`73793b715a45fdc5be7adc3163196cb0ceab967fc5369ae736be5d9b4a78cbb9`.
Este resultado cubre arranque frío y host AXI de prueba; la integración
PCIe nativa, recuperación del enlace y cruce de IRQ siguen pendientes.
