# Normalización real con transporte AXI — R1

Estado 2026-10-15 22:58 UTC: **preparada, comprobada en interfaces y encolada**.
Todavía no se ha ejecutado esta integración en XSim. Los recuentos del banco
son cobertura prevista, no mediciones de aritmética integrada.

## Núcleo y conexión

[`ntrpu_input_norm_real_r1.sv`](../../fpga/rtl/ntrpu_input_norm_real_r1.sv)
conecta el [recorrido AXI aprobado en protocolo](LLAMA32_INPUT_NORM_PATH_R1.md)
al núcleo original `ntrpu_norm_q8_abi1_r1`. Conserva los 27 archivos Verilog
y siete scripts IP de la [tanda HLS aprobada](LLAMA32_NORM_Q8_R1.md), sin
modificar su aritmética. La vinculación inspecciona los 19 puertos reales,
el reset síncrono activo alto y las dos RAM de entrada; no añade latencia.
El contrato de puertos rechazó 58 alteraciones.

El llamador mantiene las asignaciones de entrada y pesos hasta el ACK final.
El bloque lee dos regiones de 8.192 bytes, calcula RMSNorm y Q8, conserva
512 palabras ABI1 privadas y escribe 4.096 bytes por AXI. La publicación
requiere finalización aritmética correcta y respuestas B recibidas.

## Banco y revisión

El banco prepara 20 trabajos sobre cuatro embeddings reales de Llama 3.2 1B,
con recuperación después de cada fallo: desbordamiento con entradas finitas,
cancelación durante HLS y escritura, revocación tardía por época/cancelación,
reset durante cómputo y después de AW, y última respuesta B posterior al
watchdog. Se espera drenar esa respuesta sin publicar el resultado.

Cobertura prevista: 8.704 palabras HLS exactas, 14 ofertas correctas y 12
resultados aceptados, equivalentes a 49.152 bytes. El revisor calcula una
referencia independiente mediante redondeos racionales y raíz Decimal;
reconcilia escrituras HLS, R/AW/W/B, identidad y aceptación pública. Exportará
latencias por trabajo, intervalos sin actividad y recursos/timing de síntesis.
No convierte el reloj solicitado de 5 ns en rendimiento físico demostrado.

La comprobación Icarus de diez fuentes pasó con una firma exacta del núcleo,
sin ejecutar esa firma. Respaldo completo restaurado en D: y NFS:
`20261015T225142Z-V7-input-norm-real-compile-r1`, SHA-256
`a435339bed170efa304f04a5c726e59b822ef824d7f30dfb32a7156e0a407ddf`.
El revisor rechazó 51 corrupciones sobre 17.464 eventos sintéticos; esto
comprueba el revisor y no acredita ejecución RTL.

## Cola y recuperación

Paquete `20261015T225723Z-input-norm-real-vivado-r1`: 93 archivos,
36 fuentes reales y siete IP, sin firmas sustitutas en la compilación Vivado.
SHA-256: `833e638acbc143122f2e221bd035ef7a7e93c996cc2c472568aeec3cd5276273`.
Copias D:/NFS y restauración verificadas. Coordinador PID 9808: espera
aprobación y respaldo del recorrido de protocolo, además del slot vacío.
Usa dos CPU, límite de dos horas y ningún reintento automático. Resultados
o diagnósticos se archivan y restauran antes de liberar el slot.

Traspaso: `artifacts/ntrpu-board-vivado/input-norm-real-queue-20261015-r1/`.
La captura DDR conserva su posición anterior. Siguen pendientes la ejecución
de esta tanda, el consumidor Q/K/V vivo y la validación física de la placa.

El [control de traspaso normalización→Q/K/V](LLAMA32_INPUT_QKV_HANDOFF_R1.md)
ya pasó su banco nativo con hijos sintéticos y está encolado después de esta
tanda. Desde las 23:43 UTC, la [conexión real con Q/K/V](LLAMA32_INPUT_QKV_REAL_R1.md)
también está implementada y encolada detrás del traspaso. Compiló sus
interfaces; la validación aritmética conjunta todavía no se ha ejecutado.
