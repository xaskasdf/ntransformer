# Lector AXI de pesos: contrato y validación RTL

Fecha: 2026-10-02. Implementación: `fpga/rtl/ntrpu_axi_read_master.sv`. Este bloque sintetizable lee memoria a través de AXI4 y entrega un flujo de 512 bits. Todavía no está conectado a la FFN HLS, a una IP DDR ni a PCIe.

## Interfaces

Un solo reloj; reset síncrono activo bajo. Cada comando contiene una dirección de 64 bits y una longitud de 32 bits en **bytes**. Dirección y longitud deben ser múltiplos de 64; longitud cero y desbordamiento de dirección se rechazan antes de emitir AR. La comprobación del rango físico registrado corresponde al futuro controlador de comandos: una dirección representable no implica memoria instalada ni permisos.

Solo hay un comando activo y una ráfaga pendiente. AR usa ID 0, INCR, 64 bytes por transferencia y hasta 64 transferencias por ráfaga. Los comandos pueden cruzar páginas; cada ráfaga termina antes del límite de 4 KiB. Esta restricción y la obligación de completar las transferencias emitidas siguen [Arm IHI 0022H, A3.4](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/IHI0022H_amba_axi_protocol_spec.pdf).

`ARCACHE=0010`, `ARPROT=000`, `ARLOCK=0`: memoria normal no cacheable/no bufferable, acceso de datos seguro no privilegiado y no exclusivo. El integrador debe adaptar protección, ancho de ID/direcciones y señales opcionales al interconnect elegido; no se ha validado una configuración Zynq de seguridad. El puerto solo tiene canales AR/R.

`out_valid/out_ready` transportan datos completos, con `out_last` en el último beat del **comando**, no de cada ráfaga. Un registro elástico retiene datos durante pausas. El consumidor puede aceptar un beat por ciclo cuando llegan respuestas consecutivas; las separaciones entre ráfagas limitan el promedio.

La finalización usa `done_valid/done_ready`. Estado y contador permanecen estables hasta aceptación; no se admite otro comando antes. `done_bytes` cuenta bytes entregados al consumidor, no bytes descartados del bus.

| Estado | Significado |
| --- | --- |
| 0 | Comando completo correcto |
| 1 | Dirección/longitud inválida |
| 2 | RRESP distinto de OKAY, incluido EXOKAY inesperado |
| 3 | RID incorrecto o RLAST adelantado/retrasado |

Ante un error se conserva el primer estado, se suprimen los datos del beat defectuoso y posteriores, y se drena hasta RLAST sin emitir otra ráfaga. Las respuestas anteriores ya pueden haber llegado al consumidor. Por eso el destino debe ser un buffer privado: **publicar su contenido solo después de `done_status=0`**. El flujo por sí solo no ofrece atomicidad.

Un RLAST incorrecto es una violación de protocolo; detectar estos casos no garantiza recuperación frente a cualquier esclavo averiado. No hay timeout ni cancelación: un esclavo que nunca responde o nunca termina puede bloquear el comando. El reset durante tráfico requiere reset coordinado del dominio/interconnect, no únicamente de este módulo.

## Pruebas reproducibles

Con Vivado disponible en PATH:

```bash
source /ruta/AMDDesignTools/2026.1/Vivado/settings64.sh
python3 scripts/run_ntrpu_axi_reader.py --synth
```

El ejecutor crea una carpeta nueva en `build/ntrpu-axi/`, copia las fuentes, registra SHA-256, comandos y logs, y exige la marca de éxito de los 52 casos por semilla. No modifica ejecuciones HLS existentes. `--seed` permite elegir semillas; por defecto ejecuta 1, 7 y 42. `--synth` añade síntesis fuera de contexto para `xczu7ev-ffvc1156-2-e`, modificable con `--part`.

El banco comprueba datos/direcciones, límites de 4 KiB, direcciones mayores de 4 GiB y 512 GiB, el extremo del espacio de 64 bits, longitudes inválidas, pausas prolongadas y aleatorias, estabilidad de señales y finalización, SLVERR/DECERR/RID y RLAST incorrectos. Incluye errores al principio/final de una ráfaga y después de ráfagas correctas, comandos consecutivos y reset en reposo. Los datos son patrones sintéticos; no pesos de Llama.

## Evidencia del 2 de octubre

- Simulación final `20261002T071645Z-ixyscin7`: **156 casos correctos** (52 × 3 semillas); 599 ráfagas y 1.965.312 bytes entregados. Se observaron pausas en AR, R, salida y finalización.
- Síntesis `20261002T071405Z-nb18_e1r`, mismo RTL: **343 LUT, 655 FF, 0 BRAM, 0 DSP**. Su banco anterior también pasó, pero la simulación final amplía la cobertura de fallos tardíos.
- Reloj solicitado de 5 ns; estimación de síntesis WNS +3,186 ns y WHS +0,084 ns. **No hay colocación/ruteo ni cierre temporal de la placa**: 621 entradas y 618 salidas carecen de retrasos externos, y no se fijó HD.CLK_SRC. Los avisos de puertos constantes corresponden a atributos AXI fijos.

Los archivos completos y su índice verificado se guardan en `artifacts/ntrpu-axi-20261002/` y en `/lambda/nfs/vivado/ntrpu/validation/`. La integración posterior con [dos bancos de pesos](WEIGHT_TILE_BUFFER.md) ya prueba publicación/invalidación del tile y solapamiento de lectura/llenado. Su regresión del lector `20261002T074644Z-2jjm01fa` repitió los 156 casos correctamente. Sigue pendiente el adaptador de bloques Q8_0 hacia el cómputo.
