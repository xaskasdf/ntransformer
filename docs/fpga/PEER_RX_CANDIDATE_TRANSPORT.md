# Frontera elástica del candidato R4

Estado: **validación funcional, timing interno ruteado a 200 MHz y equivalencia
de netlist; integración PCIe/HBM pendiente**. Experimento aislado de la cola principal.

El [estudio posterior de caudal y evaluación física](PEER_RX_TRANSPORT_STUDY.md)
conserva esta envoltura y documenta los resultados de implementación.

`fpga/rtl/ntrpu_candidate_transport.sv` envuelve el
[candidato con admisión registrada](PEER_RX_CANDIDATE_DISPATCH.md).
El núcleo generado y sus fuentes de referencia permanecen intactos.
Esta envoltura sí almacena transferencias y respeta handshakes; sustituye
la función de integración que el montaje anterior de muestreo no ofrecía.

## Interfaces y capacidad

Un reloj y reset coordinado, dos destinos AXI de escritura, payload de
512 bits y descriptores de 232 bits previamente autorizados. Los bloques
siguen limitados a 1–64 beats completos: 64 B–4 KiB, direcciones alineadas
a 64 B y sin cruzar una página de 4 KiB.

| Interfaz | Almacenamiento elástico |
| --- | --- |
| Publicación de descriptores | Dos entradas por destino |
| Solicitud de payload (`fetch`) | Dos entradas de 233 bits, destino incluido |
| Entrada de payload | Dos entradas de 512 bits |
| Finalizaciones externas | Dos entradas de 66 bits por destino |
| AW / W / B | Dos entradas por canal, ya presentes en el receptor |

Las colas nuevas suman **2.682 bits de datos**, además de sus contadores y
control. No amplían el pool de ocho bancos de 4 KiB: dos dedicados por
destino y cuatro compartidos. `queued` incluye las dos posiciones nuevas
y las ocho del núcleo: máximo diez descriptores aún sin reservar por
destino. Las dos finalizaciones externas pueden liberar crédito interno
antes de que el consumidor las recoja; `drained` sigue esperando a ese
consumidor. `pending_counts` conserva el significado interno del núcleo.

El receptor mantiene una sola operación de llenado activa. Las dos entradas
de fetch no habilitan dos DMA concurrentes. Tampoco se presenta esta prueba
como una mejora medida de ancho de banda. La cola elástica puede introducir
una burbuja al salir de llena; no anticipa crédito por una extracción simultánea.

## Stop, fallos y barreras

- **Stop queda enclavado hasta reset.** Cierra publicaciones nuevas y permite
  que los descriptores ya aceptados en la frontera entren al núcleo y drenen.
  También siguen avanzando fetch, payload, AW/W/B y finalizaciones comprometidas.
- **Un fallo conserva las reservas comprometidas** y pone en cuarentena los
  trabajos sin reserva, incluidos los que solo ocupan la frontera de entrada.
  No los descarta ni declara drenaje falso. El reset requiere un dominio/época
  sin transacciones vivas; no implementa recuperación frente a pérdida del enlace.
- **Payload necesita un fetch aceptado externamente.** Su presupuesto es
  `AWLEN+1`; datos previos o sobrantes no obtienen READY. El productor mantiene
  VALID y datos estables mientras espera, incluso después de stop/fallo para
  transferencias ya comprometidas.
- **Una barrera cierra admisión**, incorpora lo aceptado previamente y espera
  las finalizaciones aceptadas por el consumidor externo. Un resultado ya
  ofrecido mantiene VALID/id ante un fallo posterior. Un fallo previo deja
  la barrera pendiente, sin éxito artificial.

La barrera acredita terminación del transporte según respuestas AXI B y
consumo de finalizaciones. No acredita persistencia de memoria, coherencia
de cachés ni ordenamiento global PCIe entre dispositivos.

## Evidencia

Banco `fpga/tb/tb_ntrpu_candidate_transport.sv`, Vivado/XSIM 2026.1,
`xcvu5p-flva2104-2-i`. Trece escenarios cubren tráfico equilibrado,
contrapresión independiente, stop con entradas pendientes y cola llena,
errores externos/SLVERR, saturación de bancos y descriptores, progreso del
otro destino, W anterior a AW, payload no solicitado y barreras retenidas.

La versión ampliada publica 91 trabajos: 88 terminan, tres permanecen en
cuarentena por fallos inyectados y se verifican **127.040 bytes**.
Una barrera queda intencionalmente pendiente en el caso de fallo anterior
al resultado. El revisor Python reconstruye propiedad, orden, presupuestos,
bytes y finalizaciones desde eventos; rechaza trece alteraciones de traza.
El banco también comprueba estabilidad de las salidas bajo contrapresión.

Síntesis: 6.755 LUT, 16.024 FF, 15 bloques BRAM, cero latches y cero bloques
sin resolver. DRC de síntesis sin hallazgos, con cobertura limitada por el
modo fuera de contexto. A 5 ns, la estimación previa a placement arroja
setup +1,024 ns y hold −0,008 ns; **no es cierre físico**. Falta asignar el
reloj físico y restricciones externas. Se conservan los avisos de entorno
XSIM, `HD.CLK_SRC` y alcance limitado de DRC.

La tanda de síntesis `20261005T232927891307Z-transport` duró 305,234 s.
Las ampliaciones posteriores cambian únicamente el banco; reutilizan la
síntesis tras comparar hashes del RTL y generadores. Cada tanda usa dos CPU,
prioridad reducida y un límite de ejecución; no deja una validación en bucle.

## Reproducción y respaldo incremental

```bash
source /ruta/Vivado/settings64.sh
python3 -B tools/run_ntrpu_candidate_transport.py /tanda/output --timeout 900
python3 -B tools/check_ntrpu_candidate_transport.py /tanda/output \
  --review /tanda/REVIEW.json
# Si solo cambió el banco:
python3 -B tools/run_ntrpu_candidate_transport.py /otra/output --simulation-only
python3 -B tools/check_ntrpu_candidate_transport.py /otra/output \
  --synthesis /tanda/output --review /otra/REVIEW.json
```

Fuentes, trazas, checkpoint y revisiones locales:
`C:\Users\xasko\artifacts\ntrpu-candidate-transport\20261005T232106845192Z`.
`LATEST.json` identifica la simulación vigente. Las versiones previas del banco
se conservan como evidencia de desarrollo, no como validación de la última versión.

El respaldo usa `D:\AI\artifacts\ntrpu-content-store`, con objetos gzip
identificados por SHA-256 y manifiestos por tanda. Archivos idénticos comparten
objeto; no se conservan ZIP de transferencia ni cachés de compilación de Vivado.
Los respaldos históricos consolidados permanecen separados.

```powershell
python -B tools/ntrpu_artifact_store.py verify D:\AI\artifacts\ntrpu-content-store candidate-transport-20261005
python -B tools/ntrpu_artifact_store.py restore D:\AI\artifacts\ntrpu-content-store candidate-transport-20261005 --destination C:\ruta\nueva
```

## Próximas decisiones de integración

1. Medir caudal y latencia con bloques de 64 B, 512 B y 4 KiB y pausas
   controladas; separar coste de admisión, memoria y consumo de finalizaciones.
2. Repetir placement/ruteo y STA al incorporar las conexiones e IP reales.
   El montaje de esta envoltura con un solo reloj ya fue ruteado y comparado
   funcionalmente; su cierre interno no acredita los tiempos de la placa.
3. Conectar adaptadores PCIe/HBM con créditos, validación de ventanas,
   fragmentación, IDs y ordenamiento definidos. Añadir CDC/reset coordinado
   donde corresponda. Este dispositivo de ensayo no valida una plataforma HBM.

La unión lógica de memoria entre NTRPU requiere además tablas de regiones,
propiedad/época y traducción de direcciones: los buffers no fusionan por sí
solos los BAR ni multiplican el ancho de banda físico.
