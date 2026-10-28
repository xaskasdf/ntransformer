# Captura privada Q/K/V antes de caché y atención

Actualización 2026-10-16 01:37 UTC. **Protocolo aprobado en Icarus;
síntesis OOC y XSim encolados después de RoPE real**. No conecta todavía
la caché KV ni ejecuta la aritmética RoPE o una inferencia completa.

## Función y memoria

[`ntrpu_qkv_snapshot_r1.sv`](../../fpga/rtl/ntrpu_qkv_snapshot_r1.sv)
retiene una respuesta RoPE y carga V desde su asignación DDR mediante
el lector AXI real de 512 bits. Sólo publica después de recibir los
32 beats V y copiar las 2.560 palabras FP32 Q/K sin errores.
La memoria privada contiene 12 KiB de datos; su coste físico en BRAM,
LUT y registros queda pendiente de Vivado.

| Tramas de 512 bits | Contenido | Tamaño |
| --- | --- | ---: |
| 0–127 | Q rotado, 2.048 FP32 | 8 KiB |
| 128–159 | K rotado, 512 FP32 | 2 KiB |
| 160–191 | V, 512 FP32 | 2 KiB |

Cada trama contiene dieciséis palabras, con la primera en los bits bajos.
El lector limita cada ráfaga a una página de 4 KiB y mantiene una sola
pendiente. Valida extensión con sumas de 65 bits, alineación de 64 bytes,
RLAST/RRESP/RID y ausencia de NaN/infinito. No escribe memoria externa.

## Propiedad y publicación

La identidad es secuencia/época de 64 bits y posición de 32 bits.
La capa es una asociación suministrada por el planificador: RoPE todavía
no proporciona una etiqueta de capa independiente. El llamador conserva
inmutable la asignación V hasta el ACK público y el drenaje final.

La respuesta RoPE permanece retenida durante la captura y la lectura del
resultado. Cambios de época, cancelación o alteraciones del productor
revocan inmediatamente la lectura pública y mantienen esa revocación.
El ACK al productor ocurre después del ACK público, con identidad completa;
una respuesta ajena se retiene hasta restaurar su identidad o realizar un
reset coordinado. El watchdog marca error, pero una transacción AXI ya
emitida debe drenar antes de publicar la respuesta. Reset requiere
quiescencia conjunta del lector, memoria y productor.

## Evidencia ejecutada

Banco [R3](../../fpga/tb/tb_ntrpu_qkv_snapshot_r3.sv), semillas 1/7/42:

- 321 escenarios, 312 respuestas, 12 resets y 12 revocaciones retenidas.
- 35.712 tramas públicas: **2.285.568 bytes**, reconstruidos por el
  [revisor independiente](../../tools/check_ntrpu_qkv_snapshot_r1.py)
  desde las palabras Q/K y los beats V realmente recibidos.
- 177 corrupciones de traza rechazadas y 23 mutantes RTL detectados
  por aserciones directas; ningún timeout se contabiliza como éxito.
- Capturas válidas: 5.165–5.191 ciclos desde admisión hasta respuesta,
  con pares sintéticos. No es latencia medida en DDR física.

El primer banco inyectaba fallos antes de admitir el trabajo nuevo porque
leía un contador retenido del anterior. R2 exige el handshake del trabajo
actual. R3 separa las esperas de AR de los huecos del productor R: se
observaron cero esperas del consumidor R. Los tres intentos y la tanda de
mutantes quedan conservados y restaurados; el RTL R1 no cambió.

## Cola y respaldo

Paquete `20261016T013549Z-qkv-snapshot-vivado-r1`: 39 archivos, tres fuentes
compiladas, dos CPU, reloj OOC de 5 ns y límite de 30 minutos sin reintento.
Coordinador PID 28052; espera aprobación y restauración de RoPE real R2.
No desplaza la captura DDR posterior a BRAM. Resultados, fallos y timeouts
se respaldan y restauran antes de liberar el slot.

| Evidencia | SHA-256 |
| --- | --- |
| Nativo R3 | `119d17f299848c6c26579343fd8b9733df798ecb993f616dad41756bde64c69c` |
| Mutantes R1 | `eb21a55131133f8091373eeff512283cbbc0a97c1c477363ea256e5934262953` |
| Revisión nativa | `a1e67e30217212ddc46df80368a11545efe7936265402423239b5c46688638f5` |
| ZIP fuente Vivado | `cdc0403852511c11b25ccdca2d779245b0074ebd76d16ffa5c89c2b226c58232` |
| Configuración Vivado | `3860da871f21d0d28ff15228ec8aceb686ea55a01adb8c9a2992f3f7276990b5` |

Respaldo nativo en `D:/AI/ntransformer/artifacts/ntrpu-qkv-snapshot-native/`
y NFS. Traspaso documental y de cola:
`artifacts/ntrpu-board-vivado/qkv-snapshot-queue-20261016-r1/`.

## Siguiente integración

Implementar el planificador que envía primero las tramas K/V al append,
confirma su commit y acepta su completion. Sólo entonces puede admitir
el comando de atención y transmitir las 128 tramas Q. El guard de
atención actual actualiza el prefijo disponible al aceptar la completion
KV; recibir el último B por sí solo no habilita el comando siguiente.
Faltan esa conexión, sus cancelaciones/drenajes y la prueba con RoPE real.
