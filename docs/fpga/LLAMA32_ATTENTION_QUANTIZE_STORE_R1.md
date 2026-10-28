# Integración del cuantizador con publicación AXI (R1)

Estado 20261013T213049Z: **protocolos nativos aprobados; composición con el RTL
HLS generado encolada**, pendiente de ejecutar y revisar.

## Arquitectura y propiedad

`fpga/rtl/ntrpu_attention_quantize_store_r1.sv` conecta el cuantizador generado
con `ntrpu_attention_quantize_owner_r2`, `ntrpu_attention_activation_store_r1`
y el escritor real `ntrpu_activation_writer`.

1. Captura identidad padre de 64 bits y región de destino del comando.
2. Lee la fuente síncrona de 2.048 FP32, manteniéndola retenida hasta terminar HLS.
3. Libera esa fuente; el propietario conserva los 512 enteros de 64 bits ABI1.
4. El escritor materializa 64 registros AXI de 64 bytes, con AW y W independientes.
5. Tras drenar B y reconocer el resultado cuantizado, publica los 4.096 bytes.

Un error o cancelación impide publicar. Cancelación durante HLS exige esperar
su finalización; durante AXI exige drenar las transferencias ofrecidas.
La revocación tardía permanece aunque se restaure la época. Después del ACK
de la fuente, su reutilización para otro trabajo no revoca la copia privada.
Reset exige detener también la fuente, el núcleo y la memoria AXI.

## Evidencia nativa

Tres semillas, **72 escenarios**, con cuatro pares FP32/ABI1 reales.
El sustituto HLS lee memoria síncrona y emite los resultados de referencia;
**no calcula la cuantización**. Esta prueba acredita protocolos, no aritmética.

- 63 completions; 66 ACK de fuente y 63 de cuantización; 12 resets.
- 3.099 AW, 3.102 W y 3.099 B: tres resets dejan un W sin AW/B.
- 45 regiones válidas al cierre: **184.320 bytes** comprobados.
- Cancelación/época durante cálculo, no finitud tardía, error B, cancelación
  con AW pendiente de W, revocación tardía, reset durante HLS y W sin AW,
  identidad ajena persistente y reutilización de fuente tras ACK.
- Verificador independiente: **90 corrupciones rechazadas**. Seis mutantes
  RTL compilados y detectados: identidad truncada, destino desplazado,
  error HLS descartado, cancelación suprimida, completion prematura y
  dependencia indebida de una fuente ya liberada.

Baseline `20261013T212236Z-V7-attention-quantize-store-native-r2`; archivo `bf64e85110a75e090db81db9cf04974522a9b8206ac4b5e13806e1d9da8e75e0`.
Revisión `e78f7aade01ab1ee7f48be7ae628441b60df502deb9932db9b4d112a158ba819`.
Las fuentes, pruebas positivas, seis fallos deliberados y un fallo inicial
de sintaxis del sustituto están respaldados con hashes en D: y NFS.
El sustituto R2 corrigió solo inicializadores ANSI incompatibles con Icarus.

## Tanda Vivado encolada

Paquete `20261013T212833Z-llama32-attention-quantize-store-vivado-r1`, SHA-256 `859c6154ff2d540682396b62fbdd0d978ea25d6eee1d29b7b6b6ea7b687ffc50`;
fuente `fe3ca565bffd720ee8a25c384a93fa916697ae953d43cad5f1625680311ec08b`; 57 miembros congelados.
Se excluye el sustituto de protocolo. Se compilan los 18 Verilog y cuatro
scripts de IP del [HLS aprobado](LLAMA32_ATTENTION_QUANTIZE_RESULTS_R1.md).

Depende del escritor AXI aprobado y respaldado. Ejecutará los 24 trabajos
con semilla 1 y aritmética generada, seguida de síntesis OOC VU5P a 5 ns:
dos CPU, máximo 5.400 s, revisión y respaldo antes de liberar el turno.
La cola contiene 22 etapas; auditoría vigente R11.

La fuente de este banco reproduce los vectores de la raíz desde memoria.
Falta la conexión simultánea con la raíz RTL y el GEMV O, seguida de residual
y FFN, además del controlador DDR real y cierre de timing de la tarjeta.
No se declara inferencia completa ni funcionamiento físico a 200 MHz.
