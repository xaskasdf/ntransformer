# Créditos de escritura para el receptor multibuffer

## Alcance y candidato seleccionado

Experimento lateral del 2026-10-04 sobre la
[frontera de memoria con CDC](PEER_RX_MULTIBUFFER_MEMORY.md).
Se añade un filtro de escritura con uno, dos o cuatro descriptores; el
filtro original se conserva y se ejecuta como referencia. El pool sigue
siendo R4: dos destinos, ocho bancos de 4 KiB, cuatro de reserva.
La variante no está adoptada en el diseño de placa ni en la cola principal.

Con B lento, cuatro créditos alcanzan **2,31 veces** el caudal funcional
de la referencia. Dos alcanzan 1,86 veces. Con respuestas rápidas empatan.
Cuatro es el candidato para la siguiente composición física: frente a dos
cuesta 47 LUT y 33 registros adicionales por filtro, y también cumple el
timing del arnés aislado. Queda pendiente el contrato del controlador y
el cierre de la composición completa.

## Contrato y organización

`ntrpu_axi_write_capacity_queue` es un bloque de escritura especializado
en el ID AXI cero usado por el receptor. Cada entrada retiene dirección,
longitud, atributos y estado hasta retirar su respuesta B aguas arriba.
El ancho de dirección se valida en 64 bits antes de reducirlo a 37 bits;
se permiten transferencias INCR alineadas de 64 bytes sin cruzar 4 KiB,
dentro de la ventana de 128 GiB. Un ID distinto de cero también se rechaza
localmente, devolviendo ese ID con DECERR. No sustituye un interconnect
general con varios IDs o reordenamiento.

Los cursores de dirección y datos avanzan independientemente del cursor
de respuestas. La cola W de dos entradas transporta también el número
de descriptor: `w_done` se establece al aceptar el último W externo.
Un rechazo consume sus beats de entrada, no emite AW/W y espera su turno
para devolver DECERR, sin consumir la B de una escritura válida posterior.
Un descriptor con respuesta retenida sigue ocupando crédito.

Todas las escrituras externas usan el mismo ID. El destino debe mantener
el orden de respuestas y el origen el orden de datos asociado a AW, como
exige AXI4. W puede ser aceptado antes de AW; la respuesta espera ambos.
([Arm IHI 0022H](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/IHI0022H_amba_axi_protocol_spec.pdf)).
El bloque presupone beat counts, WLAST y respuestas conformes; no añade
recuperación frente a BID ajeno o tráfico malformado.

`admit` sólo controla AW nuevas. El wrapper de memoria lo mantiene activo
para drenar trabajos aceptados por el receptor. `stop` actúa en la admisión
del receptor. Reset sigue requiriendo coordinación con ambos extremos y
los dominios de reloj; no revierte bytes escritos ni implementa aborto AXI.

## Comparación funcional

La entrada compartida sigue a 200 MHz y 512 bits: 12,8 GB/s brutos máximos
entre ambos destinos. La IP CDC conserva la configuración anterior.
Se compara `GUARD_SLOTS=0` (original), 1, 2 y 4 contra el mismo destino.

Se corrigió el modelo de destino para asignar un vencimiento B independiente
a cada pareja AW/W completa. El modelo anterior comenzaba la demora del
siguiente trabajo después de retirar B anterior. Esa serialización artificial
ocultaría parte del beneficio. La referencia original y la variante de un
crédito reproducen exactamente los tres caudales anteriores.

| Variante | B rápido | Relojes asimétricos | B lento |
| --- | ---: | ---: | ---: |
| Original | 11,609566 | 6,213416 | 4,559026 |
| Un crédito | 11,609566 | 6,213416 | 4,559026 |
| Dos créditos | 11,609566 | 8,943842 | 8,494621 |
| Cuatro créditos | 11,609566 | 8,943842 | 10,536334 |

Unidades: GB/s decimales del banco funcional, incluyendo admisión y drenaje
de 32 transferencias alternadas de 4 KiB. Perfiles: periodos de destino
4/7,3 ns con demora B de 3 ciclos, 11,5/4,3 ns con 32 ciclos y 4/7,3 ns
con 160 ciclos, respectivamente. El modelo añade dos ciclos al parámetro
de demora. Cuatro frente a dos mejora un 24,04 % en el caso B lento.

**No son estimaciones de rendimiento de placa.** Los modelos FIFO AMD
omiten retardos físicos de sincronización y los destinos son conductuales.
Aquí no hay PCIe, MIG/HBM ni arbitraje de memoria compartido.

## Validaciones

La matriz de cuatro variantes y tres perfiles pasó 14 escenarios cada una:
840 comandos, 828 completions, 12 descartes deliberados por reset y
1.774.080 bytes comprobados en interfaces de destino. Las completions
incluyen 48 DECERR y 24 SLVERR inyectadas; los bytes incluyen los trabajos
posteriormente descartados por reset.

Se alcanzó la ocupación de uno, dos y cuatro pendientes externos por
destino. Además de los casos anteriores, se comprobó saturación con B
retenida y stop, un rechazo intercalado entre escrituras válidas y un
SLVERR intercalado, drenando el trabajo ya aceptado. Se detectaron tres
mutaciones RTL y 72 corrupciones de trazas mediante reproducción Python.

El banco directo del filtro cubre las profundidades 1, 2 y 4, con 36
comandos cada una: 26 escrituras externas y diez rechazos por caso.
Comprueba tamaño/burst no admitidos, exclusivas, ID no cero, desalineación,
cruce de página, desbordamiento, extremos de capacidad, strobes parciales,
respuesta SLVERR y estabilidad bajo backpressure. También ejercita cierre
de `admit`, W antes de AW, saturación y reutilización circular de entradas.
Son 108 comandos y 19.968 bytes de bus comprobados, con 30 rechazos.
El banco detecta tres mutaciones RTL; el verificador independiente detecta
otras 18 corrupciones de trazas, incluidas alteraciones de los strobes.

## Medición física

La tanda del filtro aislado usa VU5P, un arnés común de registros y BUFG
a 200 MHz. Mide las cuatro variantes con colocación `Default` y ruteo
`RuntimeOptimized`. Recursos del filtro, excluyendo registros del arnés;
slack del conjunto filtro + arnés, después de P&R:

| Variante | LUT | LUT usadas como RAM | Registros | Setup | Hold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 447 | 338 | 86 | +2,355 ns | +0,019 ns |
| Un crédito | 406 | 0 | 1234 | +2,571 ns | +0,032 ns |
| Dos créditos | 485 | 36 | 1199 | +2,142 ns | +0,027 ns |
| Cuatro créditos | 532 | 36 | 1232 | +2,260 ns | +0,028 ns |

Las cuatro usan 16 CARRY8, cero BRAM y cero DSP; las LUT RAM están incluidas
en la columna LUT. El mapeo cambia respecto del original: su cola usa
memoria distribuida, mientras la nueva cola W elástica usa registros.
El incremento de cuatro frente al original es 85 LUT y 1146 registros por
filtro, distinto del coste incremental de pasar de dos a cuatro créditos.

No hay endpoints internos sin restringir, latches ni fallos de setup,
hold o ancho de pulso. Se completaron 887/2041/2045/2101 nets enrutables,
respectivamente, sin errores de ruteo ni errores DRC en los chequeos
ejecutados. Las constantes eliminan algunos registros del arnés; no se
supone que su número sea idéntico entre variantes.

Esta es una medición **OOC** con reloj de 5 ns, sin incertidumbre adicional
explícita. Los puertos externos no tienen delays ni ubicaciones físicas,
y el reloj de entrada no tiene `HD.CLK_SRC`; esos avisos se conservan.
También se registra que OOC excluye algunos chequeos de conectividad DRC.
Los márgenes califican las rutas internas del arnés, no el reloj de placa,
sus I/O o el sistema completo. No se aplicaron waivers ni se buscó Fmax.
La revisión de avisos y conteos está en el `REVIEW.json` de la tanda física.

## Siguiente integración

Mantener cuatro créditos como candidato y dos como alternativa de menor
coste. La [composición lateral completa](PEER_RX_MULTIBUFFER_COMPOSITION.md)
ya pasó colocación, ruteo, auditoría CDC y regresión del netlist: receptor,
colas, CDC por destino y filtro de cuatro créditos. Ese documento registra
los márgenes del conjunto y sus límites OOC. Sigue pendiente la integración
con el controlador real y no se reemplaza el filtro de placa.
La entrada actual y el receptor limitan a cuatro pendientes por destino;
ampliar el filtro por encima de cuatro requiere revisar también ese límite.

## Reproducción y evidencia

Con Vivado 2026.1 cargado y salidas nuevas:

```bash
python3 -B tools/run_ntrpu_multibuffer_credits.py /ruta/tanda/output --timeout 900
python3 -B tools/check_ntrpu_multibuffer_credits.py /ruta/tanda
python3 -B tools/run_ntrpu_credit_guard.py /ruta/otra-tanda/output --timeout 900
python3 -B tools/check_ntrpu_credit_guard.py /ruta/otra-tanda
```

Cada tanda debe conservar `inputs/` con las fuentes exactas. Los runners
generan manifiesto SHA-256 y ZIP de resultados; limitan la ejecución a dos
CPU, `nice=10` y dos hilos Vivado. El verificador de trazas se completó
durante la primera simulación: su hash final queda en `REVIEW.json`; no
cambiaron los archivos RTL/TB/Tcl de esa tanda.

Evidencia local: `D:\AI\artifacts\ntrpu-multibuffer-credits\`.
La tanda `20261004T195735873251Z-credits-sim` tardó 316,044 s; su ZIP tiene
SHA-256 `4d76654811e642eaa6313003c5f14425675c203857b45445f5646a6f05c6884c`.
El índice final, la preservación de fuentes y el respaldo consolidado se
guardan en esa carpeta. La tanda `20261004T200627167528Z-credits-guard`
tardó 802,888 s e incluye pruebas directas y las cuatro implementaciones;
su ZIP tiene SHA-256
`fd1fc9b331dc083db70b7ddf533ccecfb8fe259ae8de09bf331d9ab177ddc4c8`.
Ambas tandas finalizaron dentro del límite de 900 s, secuencialmente.

`FINAL_REVIEW.json` reúne los resultados; `SOURCE_PRESERVATION.json`
comprueba los archivos protegidos y `HANDOFF.md` explica la recuperación.
`20261004-credits-handoff.zip` contiene fuentes, testbenches, informes,
checkpoints y ambos archivos de resultados; su `.VERIFIED.json` registra
CRC y hashes SHA-256. El diseño original y `AGENTS.md` se preservan.
