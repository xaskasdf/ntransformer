# Arranque y transporte HBM entre relojes

## Alcance

Continuación aislada del [transporte con adaptación HBM](PEER_RX_HBM_TRANSPORT.md).
`ntrpu_hbm_transport_async.sv` conecta el transporte de 512 bits a dos
adaptadores de escritura mediante instancias reales de
`axi_clock_converter:2.1`. Conserva las fuentes acreditadas y la distribución
de ocho bancos: dos por canal y cuatro de reserva.

La simulación funcional pasa **21 escenarios** sobre una VU47P exploratoria.
Sus destinos AXI3 son conductuales: este resultado no acredita tráfico en
el controlador HBM nativo, PCIe/BAR, recuperación en caliente ni timing.
La cola de placa y `AGENTS.md` permanecen fuera de esta tanda.

## Contrato de relojes y arranque

- `clk`: transporte, publicación, completions y contador de operaciones.
  El banco lo estimula a 200 MHz.
- `mem_clk`: adaptadores y puertos AXI3. Se prueban períodos de 2,222 ns y
  8,026 ns, con fase distinta de `clk`; también se detiene este reloj.
- Cada convertidor es AXI4 de escritura, datos de 512 bits, direcciones
  de 64 e ID de 4; `ACLK_ASYNC=1` y tres etapas de sincronización.
- `cold_reset_n` se afirma asíncronamente y se libera mediante tres
  registros por dominio. Ambos relojes deben funcionar durante el arranque.
  No es un mecanismo para abortar escrituras ni reiniciar una sola mitad.

`ntrpu_hbm_startup.sv` sincroniza las dos señales `init_complete_async`,
el reset liberado de memoria y los fallos de adaptador. Exige ocho ciclos
consecutivos de disponibilidad sincronizada antes de publicar trabajo.
AMD requiere esperar ambos `apb_complete` antes del tráfico AXI:
[puertos no AXI](https://docs.amd.com/r/en-US/pg276-axi-hbm/Non-AXI-Ports).
En esta tanda, dichas señales se estimulan desde el banco; el supervisor
todavía no está conectado al controlador nativo.

Una pérdida posterior de disponibilidad enclava fallo al observarse en
el dominio de origen. Restaurar las señales no reabre la publicación.
No se reinicia automáticamente HBM. El caso de prueba permite terminar
una escritura ya enviada cuando el destino vuelve a responder; no promete
recuperación de datos tras un fallo físico. La supervisión térmica queda
pendiente.

## Inventario y drenaje

El convertidor puede conservar operaciones entre dominios. Por eso,
`inflight` cuenta AW aceptadas menos B consumidas en el reloj de origen,
con seis bits por canal. No se reutiliza un `adapter_idle` sincronizado
como prueba suficiente de vaciado.

`drained` requiere arranque habilitado, transporte drenado e inventario
cero. El transporte sigue esperando las completions externas y las cercas.
`stop` cierra nuevas publicaciones; los adaptadores permanecen disponibles
para drenar trabajo comprometido. Cada adaptador conserva sus cuatro
créditos y la cuarentena de respuestas inválidas.

Los puertos de diagnóstico `adapter_idle` y `adapter_fault` pertenecen a
`mem_clk`. El fallo usado para cerrar el transporte atraviesa el supervisor.
La simulación observó como máximo cuatro padres pendientes por canal;
no determina todavía la capacidad efectiva de todas las colas CDC.

## Evidencia funcional

Tanda `20261006T215159479702Z-async`, **194,072 s**:

| Medida | Resultado |
| --- | ---: |
| Solicitudes publicadas / completadas | 122 / 117 |
| Fragmentos AXI3 / bytes verificados | 411 / 186.688 |
| Muestras del inventario entre dominios | 9.726 |
| Alteraciones de trazas rechazadas | 21 |

Las cinco solicitudes retenidas son intencionales: tres descriptores por
fallos y dos respuestas inválidas en cuarentena. La reproducción independiente
comprueba datos, strobes, direcciones, orden y causalidad con marcas en
picosegundos, comunes a ambos relojes.

Se mantienen los 17 escenarios anteriores. Los cuatro nuevos comprueban
inicialización escalonada de stacks, pausa de memoria con trabajo pendiente
y parada, fallo persistente tras pérdida de disponibilidad, y memoria más
lenta que el transporte. Durante la pausa no se anuncian completions ni
drenaje prematuro. La configuración exportada de la IP y los hashes de
fuentes y resultados forman parte de la revisión.

También se rechazaron dos mutaciones RTL aisladas: omitir la espera de
arranque dispara `CDC_BEFORE_START` en el caso 1; ignorar la pérdida de
disponibilidad dispara `LOSS_LATCHES_FAULT_WITHOUT_RESET` en el caso 20.
Las copias alteradas y sus diagnósticos se conservan separadamente; las
fuentes del diseño aprobado permanecen intactas.

Se conservan los avisos de simulación, sin avisos críticos. Entre ellos
está `VRFC 10-3705` sobre un índice del reset en el modelo AMD
`fifo_generator`; también aparece en la referencia anterior. El pase
funcional no sustituye la revisión de esos modelos ni de los resets físicos.

## Límites y próximo trabajo

La [auditoría posterior de CDC y arranque](PEER_RX_HBM_CDC_AUDIT.md)
registra la síntesis, las restricciones de estado y el avance del modelo
nativo. Los límites siguientes corresponden al cierre de esta primera tanda.

La simulación digital no acredita ausencia de metastabilidad. Faltan
síntesis, `report_cdc`, revisión de resets, colocación, ruteo y timing de
esta composición. Deben declararse ambos relojes y conservar las restricciones
de la IP; no ocultar los cruces con excepciones globales. Véase
[conversión de reloj AMD](https://docs.amd.com/r/en-US/pg059-axi-interconnect/Clock-Conversion).
Sigue abierto el incumplimiento temporal de la síntesis anterior.

El ensayo nativo de arranque se registra separadamente. XSIM utiliza un
**modelo interno de respuestas** para HBM, según la
[documentación de simulación](https://docs.amd.com/r/en-US/pg276-axi-hbm/Simulation).
Incluso un arranque exitoso en ese modelo requeriría después escrituras,
lecturas de retorno y validación en hardware.

La tanda nativa `20261006T214406679385Z-boot` elaboró el modelo en 850 s,
pero alcanzó el límite global de 900 s al iniciar XSIM. La continuación
`20261006T220051395880Z-boot-resume` reutilizó una copia verificada del
ejecutable y sus tablas, sin recompilar ni modificar el original. Ejecutó
la liberación de reset a **4,702 µs simulados**, pero no obtuvo `BOOT PASS`
antes del límite adicional de **300,944 s reales**. Por tanto, el arranque
HBM nativo sigue **sin validar**; el límite de tiempo no demuestra un fallo
del controlador. Se respaldan los parámetros, archivos `.mem`, hashes de
la instantánea y diagnósticos. El ejecutable de simulación permanece en
la VM y no forma parte del respaldo local de resultados.

Antes de ampliar ese ensayo, conviene añadir hitos de tiempo simulado y
observar el avance del inicializador APB con la instantánea existente.
Esto permitirá presupuestar la prueba sin repetir la elaboración costosa.

Para avanzar hacia ancho de banda entre NTRPUs: conectar esta frontera al
HBM nativo, validar el mapa por pseudo-canal y resolver el productor compartido
de payload. Su techo aritmético de 512 bits a 200 MHz sigue siendo 12,8 GB/s;
añadir bancos no aumenta por sí solo ese caudal.

## Reproducción y respaldo

```bash
source /ruta/Vivado/settings64.sh
python3 -B tools/run_ntrpu_hbm_cdc.py /tanda/output --stage async \
  --reference /return-cut-rtl/output
python3 -B tools/check_ntrpu_hbm_cdc.py /tanda/output --review /tanda/REVIEW.json
python3 -B tools/run_ntrpu_hbm_cdc.py /arranque/output --stage boot
```

Los runners limitan cada tanda a dos CPU, prioridad baja y 900 s.
Evidencia local:
`C:\Users\xasko\artifacts\ntrpu-hbm-cdc\20261006T214228214037Z`.
Respaldo por contenido en `D:\AI\artifacts\ntrpu-content-store`, con hashes
SHA-256 y objetos gzip compartidos; no se conservan ZIP de transferencia.
