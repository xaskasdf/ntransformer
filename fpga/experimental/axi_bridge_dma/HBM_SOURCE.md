# Frontera del lector HBM hacia AXI Bridge

## Variante de tarjeta y alcance

Se prioriza **VU47P / FSVH2892 con HBM**. La tarjeta DDR4 de 512 GiB es
otro diseño; puede compartir protocolo y software, pero necesita su propia
memoria, alimentación, relojes, encapsulado y PCB. Este ensayo no cambia
el nodo DDR ni la cola de simulación principal.

`ntrpu_axi_bridge_hbm_source.sv` une un contexto del escritor con las dos
interfaces del lector, destinadas a PC0/PC16. `ntrpu_axi_bridge_hbm_client.sv`
compone esa unión con V3, cliente, mapper y maestro de control existentes.
Los módulos anteriores permanecen intactos. La unión no usa descriptores
XDMA ni hereda la calificación de aquel camino.

## Identidad, direcciones y retiro

La unión conserva los 128 bits de identidad; entrega al lector el tag local
de 64 bits y restituye la identidad completa al devolver datos y CPL.
Solo admite un contexto global pendiente. No generalizar esta localización
de tags a varios contextos concurrentes sin una tabla de correspondencias.

El perfil usa direcciones HBM de 34 bits, conserva la dirección completa y
elige el canal por el bit 33. Corresponde al mapa de dos stacks 8H del
[PG276, descripción de puertos](https://docs.amd.com/r/en-US/pg276-axi-hbm/Port-Descriptions)
y al donante congelado; no es el mapa DDR de 512 GiB. El escritor rechaza
solicitudes fuera de rango antes de reducir anchos. Se mantienen longitud
64–4096 bytes, alineación de 64 bytes y límite de página de 4 KiB.

Un registro elástico conserva cada oferta de datos. La CPL debe llegar
después del último handshake del lector y solo se publica después de
entregar el último dato al escritor. La comprobación usa contadores
registrados, evitando realimentar `fault -> stop -> ready -> fault`.

Liberar un frame HBM no libera el contexto. La unión y el escritor se
retiran al consumir el resultado, después de B correcto y ACK atribuido.
Un fallo conserva ambos contextos; ofertas existentes de comando, AW/W o
resultado permanecen estables. Recuperar disponibilidad no reabre la época.
Solo el reset frío coordinado borra estado. Esto conserva contexto lógico;
la autorización y protección de regiones HBM siguen siendo externas.

## Banco y procedencia

El banco compila V3, el escritor real y `ntrpu_hbm_read_frames.sv` congelado
desde `ntrpu-tracker-head-native/.../PREPARED`, SHA-256
`28863cc75ac91edf265540cdb8a6c02dc110620aad4a88e11b86f97506c14b2f`.
La memoria AXI, el mapping, B y ACK son fixtures explícitos. El mapper del
wrapper completo se comprueba por síntesis; no participa en este banco.

Dos patrones de pausas ejercitan cuatro lecturas correctas de 64, 128, 256
y 4096 bytes, direcciones superiores a 4 GiB y fallos de tag, longitud,
status, RRESP, CPL prematura/duplicada, canal incorrecto y disponibilidad.
También se prueba retiro con identidad alta incorrecta y fallo mientras un
resultado espera aceptación. El revisor reconstruye los bytes y exige
retención sin reset; ocho controles alteran la evidencia deliberadamente.

La primera versión pasó simulación pero produjo dos bucles combinacionales
en síntesis. Se conserva como intento no seleccionado. La corrección exige
recibo registrado de todos los datos antes de CPL, siguiendo el contrato
del lector. La selección y los resultados finales se registran en el respaldo.

## Resultado seleccionado

| Comprobación | Resultado |
| --- | --- |
| Icarus `local-r5` | 40/40 escenarios, 3,4 s |
| XSIM `xsim-20261011T091048Z` | 40/40 escenarios, 335,2 s |
| Retención tras fallos | 36 escenarios por simulador |
| Bytes comprobados / bytes de recorridos correctos | 10.880 / 9.088 por simulador |
| Controles de evidencia alterada | 8/8 rechazados |
| Síntesis `synth-20261011T091052Z` | 173,3 s, salida cero |
| Bucles / latches / black boxes | 0 / 0 / 0 |
| Endpoints internos sin reloj o sin restricción | 0 / 0 |
| Wrapper completo, sin los IP nativos | 2.291 LUT, 2.265 FF |
| Unión nueva incluida en ese total | 582 LUT, 740 FF |

La síntesis mantiene 2.174 bits de entrada y 1.225 de salida sin presupuestos
externos. No se ejecutó place/route ni se afirma cierre temporal. Los fallos
de compilación iniciales del banco y la variante con bucles quedan conservados;
el revisor del paquete rechaza expresamente aquella síntesis aunque la
herramienta haya terminado con código cero.

## Relojes y siguiente condición de integración

Todos los puertos de esta frontera pertenecen al reloj AXI PCIe de 250 MHz.
El banco del lector comparte ese reloj únicamente para validar el protocolo.
El lector nativo calificado usa 200 MHz y su dominio HBM aproximadamente
450 MHz; esta prueba no califica cambiarlo a 250 MHz. Las reglas físicas
del controlador siguen [PG276, Clocking](https://docs.amd.com/r/en-US/pg276-axi-hbm/Clocking).

Antes de conectar los IP nativos se requiere:

| Cruce | Condición |
| --- | --- |
| Comandos 250 → 200 MHz | Captura estable, una reserva y dirección/tag completos |
| Datos 200 → 250 MHz | FIFO con espacio reservado y ofertas retenidas al parar |
| CPL 200 → 250 MHz | Ordenar detrás del último dato; FIFOs independientes no garantizan ese orden |
| Estado y relojes | Detectar pérdida del dominio HBM con gestión independiente; conservar obligaciones |
| Reset frío | Liberación coordinada por dominio; ningún reset PCIe borra colas HBM |

Siguen pendientes esa frontera CDC, tráfico conjunto con ambos IP nativos,
la ruta física del ACK, autorización de regiones y cierre de timing/pines.
Esta tanda no contiene HBM nativa, endpoint PCIe nativo ni calificación PCB.

## Reproducción y evidencia

La raíz local es `D:/AI/artifacts/ntrpu-axi-bridge-hbm-source/`.
Los runners requieren salida nueva, guardan fuentes, logs, trazas y hashes;
XSIM/síntesis usan dos CPU, prioridad reducida y límites por proceso.

```bash
python -B run_hbm_source.py NUEVA_SALIDA --engine xsim \
  --reader-source LECTOR_CONGELADO --master-source MAESTRO_CONGELADO \
  --settings VIVADO_SETTINGS
python -B review_hbm_source.py TANDA_RESTAURADA
python -B check_hbm_source.py TANDA_RESTAURADA
```

`--engine icarus --toolchain TOOLCHAIN` permite repetir localmente el banco.
`run_hbm_source_synth.py` sintetiza el wrapper completo en VU47P; exige cero
black boxes, latches, bucles y endpoints internos sin reloj/restricción.
Su resultado estructural no constituye cierre físico ni hereda el ruteo V3.

`20261011-hbm-source-frontier.zip` conserva la selección y los intentos
anteriores; se restaura en `20261011-hbm-source-restored/` y se audita con
`source/review_hbm_source_bundle.py`. `BUNDLE_BACKUP.json` registra tamaño y
SHA-256, `BUNDLE_REVIEW.json` contiene la revisión y `SELECTION.json` vincula
los donantes y las tandas. `PROCESS_COMPLETION.json` confirma el fin de los
cinco trabajos privados de VM; no consulta ni modifica procesos del principal.
