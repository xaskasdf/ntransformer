# Transporte asíncrono conectado al HBM nativo

## Alcance

Composición aislada del [transporte aprobado](PEER_RX_HBM_NETLIST.md) y el
[enlace nativo de lectura/escritura](PEER_RX_HBM_NATIVE_TRAFFIC.md).
Conserva los trece archivos RTL de referencia y las nueve unidades generadas
del transporte. No cambia la cola ni los trabajos del hilo principal.

```text
descriptores + payload (200 MHz)
  → transporte / pool / completaciones
  → dos cruces AXI de reloj
  → adaptadores AXI4 512 bits → AXI3 256 bits (~450 MHz)
  → HBM nativo: AXI 00 / AXI 16
  ← lectura de comprobación desde un maestro del banco
```

El lector de comprobación reside en el dominio HBM. Todavía no implementa
el retorno de datos por CDC hasta PCIe ni un motor de inferencia.
Se configura VU47P, dos stacks, 16 GB y switches habilitados. XSIM usa el
responder interno documentado por [AMD PG276](https://docs.amd.com/r/en-US/pg276-axi-hbm/Simulation).

## Arranque y estímulos

Un único arranque frío libera APB y después el supervisor de transporte.
Los dos `apb_complete` reales alimentan la calificación de arranque.
Se ofrecen descriptores antes de calibrar; no deben aceptarse hasta que
el supervisor habilite publicación. Los casos posteriores comparten ese
arranque. No se fuerza la inicialización ni se reinicia AXI con tráfico,
respetando el [contrato de reset del controlador](https://docs.amd.com/r/en-US/pg276-axi-hbm/Port-Descriptions).

| Caso | Descriptores | Comprobación |
| --- | ---: | --- |
| 1 | 8 | Dos canales; ráfagas de 1, 8, 9 y 64 palabras de 512 bits; arranque bloqueado |
| 2 | 2 | Completaciones externas retenidas, fence y resultado del fence bloqueado |
| 3 | 12 | Colas bajo bloqueo de completaciones y posterior drenaje |
| 4 | 2 | Parada con descriptores aceptados y fetch retenido; drenaje sin reabrir publicación |

Tras cada caso se leen los destinos de ambos stacks y se comparan con
patrones calculados a partir del tag y posición originales. Los mismos
offsets contienen patrones distintos en cada stack. La lectura introduce
pausas y exige estabilidad de datos, ID, respuesta y RLAST.

## Revisión independiente

El revisor reconstruye cada descriptor, payload, escritura AXI4, fragmento
AXI3, respuesta, completion y lectura. Exige cobertura exacta y causalidad.
Las escrituras y lecturas se agrupan por dirección, sin exigir iguales
límites de ráfaga: en la prueba corta, una escritura de 18 beats se divide
en 2+16 y su lectura en 16+2. Ambas deben cubrir los mismos 576 bytes.

También verifica fences, contadores y máscaras del pool. `banks_used` es
una máscara de ocho bits; su ocupación es el número de bits activos.
Los 24 controles negativos eliminan eventos o corrompen datos, direcciones,
descriptores y alarma térmica en copias de la traza. No son inyecciones de
fallo en el RTL o en el controlador nativo.

## Estado de la validación

La prueba corta `20261007T023116354729Z-preflight` pasa en 54,902 s con un
destino AXI conductual: 24 descriptores, 41 ráfagas de escritura, 41 de
lectura y 18.752 bytes verificados. Esta etapa valida el banco con los IP
reales de CDC/conversión, pero no incluye el IP HBM nativo.

La primera prueba corta se rechazó porque el monitor R comprobaba señales
desconocidas durante reset. Se corrigió exclusivamente el monitor; el
diagnóstico quedó respaldado. Una segunda corrida confirmó los datos y
permitió corregir la presentación de la máscara de ocupación. El banco
final observa un pico de dos bancos (`used_mask=0x05`): no acredita uso de
las cuatro reservas ni saturación del pool.

La compilación nativa `20261007T023237386909Z-compile` alcanzó el límite de
900 s después de que XELAB informara que había construido el snapshot.
La tanda original permanece rechazada. Una copia independiente,
`20261007T025010081530Z-snapshot-check`, cargó el ejecutable en tiempo cero
en 77,018 s y pasó la revisión de sus quince entradas, jerarquía nativa,
configuraciones y hashes. No ejecutó calibración ni tráfico.

La ejecución nativa `20261007T025204228576Z-traffic` pasa en **861,459 s**
con estado `NATIVE_HBM_TRANSPORT_READBACK_PASS`. La revisión independiente
reproduce los datos desde descriptores hasta lectura y rechaza los
24 controles negativos de traza.

| Evidencia nativa | Resultado |
| --- | ---: |
| Casos / descriptores completados | 4/4 / 24 |
| Ráfagas AXI3 de escritura / lectura | 41 / 41 |
| Beats de 256 bits escritos / leídos | 586 / 586 |
| Bytes distintos comprobados | 18.752 |
| Muestras R bloqueadas / completaciones bloqueadas | 301 / 539 |
| Muestras de arranque con publicación cerrada | 8.734 |
| Pico de bancos ocupados / máscara utilizada | 2 / `0x05` |
| Ciclos de memoria con estado térmico desconocido | 3.641 |

El primer grupo se admite al calificar ambos stacks; el marcador de
arranque se registra a 48,3952 µs y el banco termina a 56,4852 µs.
Los indicadores públicos e internos de inicialización están en uno al
final; `fault`, créditos en vuelo, bancos, colas y pendientes están en cero,
con `drained=1`. Ambos `CATTRIP` permanecen en X.

La ocupación nativa confirma el límite de cobertura del preflight: hubo
reutilización de bancos, pero no consumo de las cuatro reservas. La lectura
de comprobación sigue siendo local al banco, no un retorno PCIe/CDC.

## Ejecución y respaldo

Vivado 2026.1, tandas aisladas, dos CPU, prioridad baja y límite de 900 s
por tanda. Compilación y ejecución nativa se separan. La ejecución copia
el snapshot y comprueba sus hashes y los del donante antes y después.
El preflight requiere `fixture/native_hbm.veo`, copiado del enlace aprobado,
y el directorio exportado de la referencia asíncrona.

```bash
python3 -B tools/run_ntrpu_hbm_native_transport.py /preflight/output \
  --stage preflight --reference /referencia-async/output
python3 -B tools/run_ntrpu_hbm_native_transport.py /compile/output \
  --stage compile --reference /referencia-async/output
python3 -B tools/check_ntrpu_hbm_native_transport.py /compile/output > /compile/REVIEW.json
python3 -B tools/run_ntrpu_hbm_native_transport.py /traffic/output \
  --stage traffic --prior /compile/output --review /compile/REVIEW.json
python3 -B tools/check_ntrpu_hbm_native_transport.py /traffic/output > /traffic/REVIEW.json
```

Si el proceso padre agota su tiempo después de construir el snapshot,
`verify_ntrpu_hbm_native_transport_snapshot.py /verificacion/output
/compile/output` permite una comprobación separada de carga. Solo acepta
ese diagnóstico, conserva el rechazo original y requiere revisión antes
de utilizar la copia para tráfico. No marca la compilación original como
exitosa ni evita el arranque frío de la ejecución posterior.

Evidencia local:
`C:\Users\xasko\artifacts\ntrpu-hbm-native-transport\20261007T022322904971Z`.
Respaldo incremental por SHA-256/gzip en
`D:\AI\artifacts\ntrpu-content-store`, con manifiestos por tanda; no se
conservan ZIP de transferencia. La caché ejecutable de XSIM queda en la VM
y se puede regenerar a partir de las fuentes y configuraciones exportadas.

## Límites y siguiente paso

La protección térmica y la paridad siguen sin calificar; un `CATTRIP`
afirmado falla, mientras X/Z se registra explícitamente como desconocido.
Este banco no acredita rendimiento, cierre temporal, PCIe, P2P entre
tarjetas, coherencia ni capacidad de 512 GB.

La composición expone una única entrada compartida `payload_data[511:0]`
a 200 MHz: su máximo teórico es `64 bytes × 200 MHz = 12,8 GB/s`, antes
de cualquier pausa. Los dos destinos HBM no duplican esa entrada. Es un
límite del contrato ensayado, no una medición de rendimiento. Para aumentar
el caudal agregado hará falta estudiar varias entradas de payload o cambiar
ancho/frecuencia, con su correspondiente validación física.

Con la composición nativa aprobada, falta someter las reservas a pausas
controladas del destino, comprobar su devolución al drenar y llevar la
composición completa a síntesis y análisis físico. También debe estudiarse
la entrada compartida antes de ampliar el número de destinos. El retorno
de lectura hacia el transporte requiere su propia implementación y validación.
