# Propiedad de RAM para normalización de entrada — R1

Estado 2026-10-15 22:28 UTC: **control nativo aprobado y encolado para Vivado**.
Este bloque prepara la conexión desde el vector FP32 y los pesos de norma
hasta el ABI1 que consume [Q/K/V](LLAMA32_QKV_PRODUCER_R1.md). Las pruebas
actuales usan peers HLS controlados; todavía no prueban esa conexión aritmética.

El [recorrido AXI R1](LLAMA32_INPUT_NORM_PATH_R1.md) ya superó su banco nativo
de transporte y quedó encolado después de este propietario. Reutiliza el RTL
R1 sin cambios; también mantiene pendiente la aritmética HLS integrada.

## Contrato y estructura

`fpga/rtl/ntrpu_input_norm_owner_r1.sv` retiene dos RAM de 2.048 × 32 bits:
entrada y pesos de RMSNorm. Presenta puertos síncronos independientes al
núcleo `ntrpu_norm_q8_abi1_r1`, con 2.048 elementos y epsilon `0x3727c5ac`.
La respuesta de una lectura pertenece al CE del ciclo anterior, incluso
cuando el núcleo cambia de dirección o introduce una pausa en CE.

Una RAM privada de 512 × 64 bits recibe ABI1 en orden. Se comprueban cantidad
y dirección de escrituras, INT8 distinto de −128, escala FP64 no negativa
y acotada, y palabras reservadas cero. Sólo una finalización completa,
correcta y no revocada permite leerla. La lectura interna usa el estado
registrado; la publicación sigue condicionada por validez y propiedad.

El bloque captura secuencia y época de 64 bits. Fallo de memoria, cancelación,
cambio de época o watchdog invalidan el resultado y esperan `ap_done` antes
de liberar la fuente. Una respuesta con identidad ajena no recibe ACK ni
permite reutilizar el bloque; debe volver la identidad correcta o aplicarse
reset coordinado. Una revocación retenida no desaparece al restaurar la época.
La RAM ABI1 permanece ocupada hasta el ACK público de su consumidor.

## Validación ejecutada

Banco R4: tres semillas, **243 trabajos y 66.048 palabras ABI1** comprobadas.
Cubre errores de ambas RAM, datos no finitos, escritura corta/excesiva,
desorden, formato, errores HLS, cancelación, watchdog, retención, revocación,
identidades ajenas persistentes y reset/recuperación. Cada semilla consume
222.170–222.228 ciclos incluyendo lectura de resultados y fallos; esto no
mide la latencia de RMSNorm ni rendimiento físico.

El revisor rechazó 63 alteraciones de traza. Los 21 mutantes compilados
fallaron con aserciones específicas, sin aceptar timeout como detección.
La primera tanda de mutantes dejó dos controles insuficientes: identidad
reducida terminaba por timeout y el caso de desorden también violaba formato.
R4 comprueba identidad al iniciar HLS y usa una dirección equivocada de la
misma clase ABI1. El RTL R1 permanece idéntico; las seis tandas, incluida la
fallida, están archivadas y restauradas en D: con copia en NFS.

## Cola y evidencia

Paquete `20261015T222702Z-input-norm-owner-vivado-r1`: 40 archivos,
copias D:/NFS verificadas y restauración local completa. Coordinador PID 26776.
Espera el cierre y respaldo/restauración del banco Q/K/V aritmético; esa
dependencia sólo ordena el slot. Un diagnóstico archivado de Q/K/V no aprueba
su aritmética. La FFN y la reserva de actividad DDR permanecen intactas.

Vivado ejecutará síntesis OOC a 5 ns y tres suites XSim, dos CPU, límite de
30 minutos y sin reintento automático. Se revisan fuentes compiladas, diseño,
recursos, timing y ausencia de cajas negras/lazos; resultado y diagnóstico
se respaldan y restauran antes de liberar el slot. No hay métricas nuevas
de Vivado para este propietario todavía.

- Revisión: `build/cloud-setup/20261002-board/input-norm-owner-native-r1-review.json`.
- Base R4 `20261015T222400Z`, TAR SHA-256
  `1fcffdea7fd16287f81deb6e5a6190a6f6aa2b93cb0b93f2e63aee8bbb9529da`.
- Mutantes R2 `20261015T222440Z`, TAR SHA-256
  `b8737954140f0b967841a20b76ca29a400afa44d426479fee2c575f0c0c1c55f`.
- ZIP Vivado SHA-256
  `3fe3710539684e34e277d2368b441ad2e3c3263d7b1de8a136617ac780920dd7`.

## Siguiente integración

Reutilizar el cargador de dos regiones FP32 `ntrpu_residual_loader_r1`:
su primer plano será la entrada y el segundo los pesos, sin sumar operandos.
Conectar el núcleo RMSNorm/Q8 generado ya calificado, escribir sus 4.096
bytes mediante AXI y retener esa finalización mientras Q/K/V consume la
misma memoria escrita. Exigir igualdad byte a byte con los cuatro fixtures
de embeddings reales. Después siguen RoPE, append K/V y atención conectada;
capas completas, logits, tokens y DDR/PCIe físico permanecen pendientes.
