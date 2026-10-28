# Receptor RTL aislado y drenaje de escrituras NTRPU

Bloque experimental de un reloj, separado de los diseños y la cola Vivado del
principal. Recibe **un span ya traducido y autorizado**, almacena todos sus
datos y lo escribe mediante AXI de 512 bits. No es un endpoint PCIe ni una
implementación completa de PUT/FENCE. El
[modelo PCIe/AXI](PCIE_AXI_TRANSPORT_MODEL.md) sigue siendo la referencia de
protocolo, memoria y recuperación.

Validación local del **2026-10-04 UTC: 74 casos RTL PASS**, 62 trazas
reproducidas en el modelo y 12 rechazos locales comprobados sin AXI.
Los tres controles negativos fallaron con el diagnóstico esperado.

La [síntesis aislada y simulación de netlists](PEER_RX_SYNTHESIS.md) también
se completaron: 74 casos funcionales sobre la lógica sintetizada, sin cambiar
este RTL. La [adaptación a PCIe y al supervisor](PEER_RX_INTEGRATION.md)
está definida como contrato; su implementación conjunta sigue pendiente.

## Archivos y ejecución local

- [RTL](../../fpga/rtl/ntrpu_peer_rx_guard.sv).
- [Banco de señales](../../fpga/tb/tb_ntrpu_peer_rx_guard.sv).
- [Ejecutor, contraste de trazas y respaldo](../../tools/run_ntrpu_peer_rx_guard.py).

En Linux/WSL con Python 3.10+ e Icarus Verilog 12:

```bash
python3 -B tools/run_ntrpu_peer_rx_guard.py
```

Para el paquete Ubuntu extraído localmente, sin instalación del sistema:

```bash
python3 -B tools/run_ntrpu_peer_rx_guard.py \
  --iverilog artifacts/ntrpu-peer-rx-toolchain/extracted/usr/bin/iverilog \
  --vvp artifacts/ntrpu-peer-rx-toolchain/extracted/usr/bin/vvp \
  --ivl-base artifacts/ntrpu-peer-rx-toolchain/extracted/usr/lib/x86_64-linux-gnu/ivl
```

El ejecutor crea una carpeta única en `artifacts/ntrpu-peer-rx-guard/`, conserva
fuentes de entrada, versiones, comandos, logs, trazas y hashes; después genera
un ZIP y verifica CRC y contenido. Un fallo devuelve código distinto de cero.
Los mutantes de control negativo existen exclusivamente en esa carpeta.

## Evidencia y respaldo

La [tanda local](../../artifacts/ntrpu-peer-rx-guard/20261004T024810533658Z/result.json)
conserva las dos configuraciones: 620 ciclos con 4 beats y 2531 con 64.
Son duraciones del banco, no mediciones de frecuencia o rendimiento físico.
El [ZIP verificado](../../artifacts/ntrpu-peer-rx-guard/20261004T024810533658Z.zip)
contiene 42 miembros, incluidas las fuentes y documentación de entrada.
El [registro de verificación](../../artifacts/ntrpu-peer-rx-guard/20261004T024810533658Z/VERIFIED.json)
confirma CRC y hashes, con SHA-256 del ZIP
`f83409e2570c13b2f6fa7a9c7d741ba9ceb309bc113fb113013c12685a509a0c`.
Este párrafo y los enlaces desde otros documentos se añadieron después
del respaldo; no se modificaron las fuentes probadas.

Se usó Ubuntu 24.04 en WSL y el paquete `iverilog_12.0-2build2_amd64.deb`
extraído en `artifacts/ntrpu-peer-rx-toolchain/`. Se comprobó su SHA-256
contra los metadatos APT locales:
`f19394cee05ad526e0eb3873bc6c17a5828f56df17fbe2db40cb66513a2d4667`.
No se utilizó la VM ni se instaló software globalmente.

## Contrato del receptor

`cmd_valid/ready` captura dirección AXI de 64 bits, longitud menos uno,
época y tag opaco. `MAX_BEATS` admite 1–64; por defecto 4 beats/256 bytes.
El receptor rechaza época incorrecta, desalineación de 64 bytes, exceso de
buffer y cruce de 4 KiB. El productor debe mantener cada descriptor/dato
estable hasta aceptarlo o recibir una revocación coordinada del ingreso.

`data_valid/ready` llena el buffer; `data_last` debe coincidir exactamente
con la longitud. Un framing incorrecto termina sin exponer AXI. Una vez
completo hay un estado intermedio de un ciclo: todavía puede descartarse
antes de lanzar AW/W. Tras el lanzamiento, **ningún stop cancela VALID**.
AW y W avanzan independientemente, incluso con todos los W anteriores a AW.

El perfil usa INCR, AWSIZE=6, ID=0, WSTRB completo, AWPROT=000 y
AWCACHE=0000. Solo hay una ráfaga activa y un slot de finalización. Este
perfil prioriza orden y drenaje; no demuestra todavía máximo ancho de banda.

| Estado de finalización | Valor | Consecuencia |
| --- | --- | --- |
| OK | 0 | Solo después de consumir B OKAY del ID 0 |
| STOP | 1 | Puede publicarse y retirarse mientras AXI sigue pendiente |
| Descriptor inválido | 2 | No se lanza AXI; cuarentena |
| Framing inválido | 3 | No se lanza AXI; cuarentena |
| Error AXI | 4 | BRESP distinto de OKAY o BID ajeno; cuarentena |

Una respuesta con BID ajeno se consume como anomalía, pero **no descarga
la transacción del ID 0**. Si esta no responde, permanece retenida. BREADY
depende de estado registrado; no existe un camino combinacional BID→BREADY.

La finalización conserva tag/época/status bajo backpressure. STOP gana si
coincide con el B que formaría una finalización nueva; no modifica una
finalización ya publicada. `lease_held` es una solicitud de retención para
el gestor externo de ventanas: el módulo no administra un BAR real.

## Parada y rearme

El reset síncrono `rst_n` es exclusivamente de arranque o de reset coordinado
de todos los participantes. No es un mecanismo de timeout ni de aborto.
Después del reset, época 0 y cuarentena impiden admitir trabajo.

Para `rearm_valid/ready` deben cumplirse simultáneamente:

1. Cuarentena activa y `stop=0`.
2. Ningún AW/W/B pendiente ni finalización sin consumir.
3. `source_stopped` e `ingress_isolated` certificados por el entorno, con
   `quiesce_epoch` igual a la época actual.
4. `rearm_epoch` estrictamente mayor; no se admite wraparound.

El rearme libera la retención y abre la nueva época. La retirada de un error
por sí sola nunca libera la ventana. Los ACK deben cubrir todos los emisores,
buffers y puentes pertinentes: son obligaciones del supervisor externo,
no pruebas físicas producidas por este bloque. Tras un reset de alimentación,
evitar colisiones con épocas anteriores requiere negociación externa.

## Validación y límites

El banco contiene 37 casos por configuración y se ejecuta con buffers de
4 y 64 beats. Incluye 20 semillas de pausas independientes AW/W, direcciones
sobre 4 GiB y al final de 512 GiB, estabilidad de VALID/datos/finalización,
rechazos de entrada, errores B, escrituras posteriores al retiro de error,
prioridades simultáneas y certificados de rearme obsoletos/incompletos.
Un watchdog acota las simulaciones. Tres mutaciones deben fallar: cancelar
AW al parar, liberar la retención al retirar un error y rearmar con AXI activo.

El ejecutor reproduce eventos LAUNCH/AW/W/B/STOP/CPL/RETIRE/REARM en el
modelo Python y compara los bytes transmitidos. Usa una instancia de modelo
por comando, con sesión normalizada; no es equivalencia formal entre RTL y
modelo. Los seis rechazos locales por descriptor/framing se comprueban sin
AXI, fuera de ese modelo. La época se valida además en el banco de señales.

La memoria del replay es un **oráculo asumido**: B OKAY implica visibilidad
final; SLVERR deja un beat y DECERR ninguno en los ejemplos. El RTL observa
solamente B, no el estado físico de DDR/HBM. El perfil no bufferable requiere
preservar sus garantías en toda la ruta real.
[Arm IHI0022H, A4.4.1](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/IHI0022H_amba_axi_protocol_spec.pdf).
El B del puente PCIe de origen no prueba esa visibilidad remota.
[AMD PG194](https://docs.amd.com/r/en-US/pg194-axi-bridge-pcie-gen3/Transaction-Ordering-for-PCIe).

Quedan pendientes colocación/ruteo y timing integrado, CDC, endpoint/TLP, fragmentación MPS,
permisos y generaciones de ventanas, deduplicación de tags, timeouts del
supervisor, integración DDR/HBM, transportes de confirmación remota y ensayos
P2P físicos. No hay cambios de ABI2, MSI-X ni integración con la FFN. Para
rendimiento harán falta varias colas/bancos y más concurrencia, manteniendo
retención por transacción y aislamiento verificable.
