# Regresión del transporte HBM ruteado y revisión DRC

## Alcance

Continuación aislada de [la prueba física y el arranque HBM](PEER_RX_HBM_PHYSICAL.md).
Se conserva el checkpoint VU47P ruteado, sus restricciones y el RTL aprobado.
Esta tanda no modifica la cola de placa ni las validaciones del hilo principal.

Entradas congeladas:

- DCP: `e04bc6fc8b5aedfee9e87a94fdd962b6e1b3b0f6d1b2ebd3065bf48ed9b71ef5`.
- Netlist funcional: `20816af702e2d71a05cfc9272d21ad849aea8f5dc234df7cb30b16797caca5d5`.
- Banco de referencia: los 21 casos CDC de `20261006T215159479702Z-async`.

## Revisión de las 13 advertencias

La tanda `20261006T235010092487Z-drc` abre una copia del DCP, ejecuta
`report_drc -no_waivers` y exporta todos los objetos, conexiones e INIT.
Termina en 88,589 s; el DCP original y la copia conservan su SHA-256.

| Hallazgo | Evidencia | Disposición |
| --- | --- | --- |
| 12 `PDCN-1569` | Doce LUT4 de FIFO AMD; el pin señalado no altera INIT para ninguna combinación de entradas | Revisados; conservar aviso y regenerar/revisar IP en la siguiente implementación |
| 1 `RTSTAT-10`: 1.032 conexiones | Salidas OOC, estado `HIERPORT`, sin cargas lógicas internas | Requieren el contexto de integración externo |
| Mismo `RTSTAT-10`: 186 conexiones | Estado `NOLOADS`, sólo conductor, sin cargas ni puertos | Salidas internas sin consumidor; inventario conservado |

El aviso comprende **1.218 redes individuales**. La frase «first 15 of 368
listed» del informe agrupa nombres de buses; no significa 368 conexiones.
El inventario completo contiene 1.218 redes y 1.218 conductores distintos.
No se encontró una carga lógica activa desconectada entre estos objetos.
Las 30.716 redes enrutables siguen completas y el informe registra cero
errores de ruteo. **Persisten las 13 advertencias**: no hay waivers ni una
declaración de DRC limpio, y esta inspección no demuestra equivalencia.

La clasificación usa los objetos de violación, no sólo el texto abreviado:
[UG835: get_nets](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/get_nets).
La distinción entre ausencia de cargas y rutas pendientes está documentada
en [UG912: ROUTE_STATUS](https://docs.amd.com/r/2022.1-English/ug912-vivado-properties/ROUTE_STATUS).

## Banco de comparación funcional

La referencia RTL y el circuito ruteado reciben las mismas entradas.
Se comparan controles y datos válidos en el flanco ascendente de cada
dominio, antes de actualizar registros. Se conservan los scoreboards de
propiedad, payload, fragmentación AXI3, respuestas, créditos y drenaje.
Los datos sin VALID no tienen significado contractual y se enmascaran.
Se exportan además eventos observados directamente en las salidas ruteadas.

El reset frío coordinado se valida por separado tras 200 ps de asentamiento:
los dos resets de cada implementación deben estar afirmados. El modelo AMD
`FDCE.v` declara unidades de 1 ps y `assign #100 Q = Q_out`; por ello, el
comparador no exige igualdad instantánea durante la afirmación de reset.
La tanda `20261007T000028100894Z-sim` pasó los casos 1–14 y se rechazó al
muestrear 88 ps después del reset entre casos. Se conserva esa traza y el
hash del modelo en `RESET_MODEL_EVIDENCE.json`. La corrección sólo cambia
el monitor; conserva estímulos, RTL, netlist y comprobaciones del tráfico.

El banco conserva el arranque global de 150 ns, los relojes independientes,
la pausa de memoria, la pérdida de estado de inicialización y el escenario
de memoria más lenta. Dos inyecciones en la vista del comparador alteran
payload y `submit_ready`; deben terminar en el fallo específico esperado.
Son controles del comparador, no mutaciones del circuito aprobado.

Los intentos `20261006T235235807829Z-sim` y
`20261006T235719796906Z-sim` se rechazan durante elaboración, antes del
tráfico. El primero renombraba módulos, pero los bloques AMD cifrados
conservaban referencias XPM. El segundo separaba bibliotecas sin fijar su
prioridad. El flujo corregido conserva el netlist byte por byte y usa un
archivo de inclusión con `uselib lib=hbm_routed`, cerrado con `uselib` al
terminar. [UG900 explica la prioridad de búsqueda](https://docs.amd.com/r/2023.1-English/ug900-vivado-logic-simulation/Verilog-Search-Order).

La tanda corregida `20261007T000554934830Z-sim` termina en **188,609 s**,
con estado revisado `ROUTED_FUNCTIONAL_REGRESSION_PASS`:

- **21/21 escenarios**, sin diferencias en las señales comparadas.
- **9.705 muestras de transporte y 19.683 de memoria**, más 21 controles
  de reset asentado. La elaboración confirma las dos bibliotecas distintas.
- **122 descriptores publicados, 117 completados, 411 fragmentos AXI3 y
  186.688 bytes verificados**. Cinco descriptores quedan retenidos por los
  escenarios de fallo previstos; no se contabilizan como completados.
- Replay independiente de la referencia y de las salidas ruteadas, más
  31 controles negativos sobre las trazas y dos inyecciones de error detectadas.
- Doce mutaciones adicionales de INIT confirman que el analizador de LUT
  detecta cuando una entrada antes irrelevante pasa a afectar la salida.

El DCP, netlist y fuentes calificadas permanecen intactos. Es simulación
funcional sin SDF; no sustituye equivalencia formal, timing de placa ni
tráfico del controlador HBM nativo. Los avisos DRC quedan **clasificados**,
no eliminados.

## Reproducción y respaldo

```bash
python3 -B tools/run_ntrpu_hbm_netlist.py /tanda/output --stage drc \
  --prior /ruteo/output --review /ruteo/REVIEW.json
python3 -B tools/run_ntrpu_hbm_netlist.py /tanda/output --stage sim \
  --prior /ruteo/output --review /ruteo/REVIEW.json --reference /cdc/output
python3 -B tools/check_ntrpu_hbm_netlist.py /tanda/output --review /tanda/REVIEW.json
```

Dos CPU, prioridad baja y límites de 300 s para DRC y 900 s para simulación.
Fuentes, netlist, entradas generadas y checkpoint se verifican por hash.
Evidencia local: `C:\Users\xasko\artifacts\ntrpu-hbm-netlist\20261006T234629677381Z`.
Cada tanda, incluidos los rechazos, se respalda incrementalmente en
`D:\AI\artifacts\ntrpu-content-store`; no se retienen ZIP de transferencia.

## Pendientes de integración

La [prueba posterior de HBM nativo](PEER_RX_HBM_NATIVE_TRAFFIC.md) pasa
cinco casos de escritura y lectura en los pseudo-canales 00 y 16 con el
modelo de XSIM; la supervisión térmica queda sin validar. Sigue conectar
el transporte asíncrono a ese controlador y comprobar desde descriptores
hasta lectura final. Después, integrar el contexto real de reloj/puertos,
repetir timing y DRC, y comprobar PCIe/DMA con tráfico.
Los márgenes internos anteriores de +0,025 ns setup y +0,006 ns hold siguen
siendo estrechos. Esta prueba no acredita 512 GB ni ancho de banda entre NTRPUs.
