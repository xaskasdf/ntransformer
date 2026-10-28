# Cuantización/AXI aprobada y composición de salida (R1)

Estado 20261013T222735Z: cuantización real y publicación AXI **aprobadas en XSim
y síntesis, revisadas y respaldadas**. La nueva composición con proyección O
tiene control validado con sustitutos; su aritmética integrada sigue pendiente.

## Resultado con HLS generado

Tanda `20261013T214728Z-V7-llama32-attention-quantize-store-vivado-r2`: cuatro fuentes FP32 de atención reales, cuantizador
HLS, propietario y escritor AXI reales. La fuente se reproduce desde memoria
síncrona; todavía no se conecta aquí el wrapper de atención completo.

- 24 trabajos, 21 completions, 22 ACK de fuente, 21 ACK de cuantización y cuatro resets.
- 42.497 lecturas de fuente y 11.264 palabras de salida de 64 bits.
- 1.033 AW, 1.034 W, 1.033 B; un reset interrumpe W antes de AW/B.
- **15 regiones válidas: 61.440 bytes exactos**. Una escritura completa adicional
  queda revocada por época tardía y no se cuenta como región válida final.
- 30 corrupciones rechazadas; errores HLS/B, cancelación, identidad ajena,
  reset y reutilización de fuente tras ACK conservan los contratos previstos.

Los cuatro primeros trabajos completos consumieron **103.173, 103.167,
103.173 y 103.166 ciclos** entre BEGIN y COMPLETE: 515,830–515,865 µs
a 5 ns. Incluyen control, escritura y retención del banco; no son latencia de
inferencia completa. Banco total: 2.224.822 ciclos.

Síntesis OOC VU5P a 5 ns: **7.221 LUT, 6.566 FF, 3,5 tiles BRAM, 3 DSP**.
WNS +0,807 ns; WHS +0,039 ns. Cero cajas negras, latches o extremos internos
sin restricciones; 564 entradas y 804 salidas sin presupuesto de E/S.
Faltan colocación/ruteo y tiempos del diseño padre.

Archivo SHA-256 `55d4b4fd3a823213ca7464188da6adacc0edd256b3e298f272dd3492876e9340`;
revisión `7a26d1a175e70ea55c11b30b9ce674df9eac005e114ea970f3fa3736292e51a8`. Archivo completo verificado en D: y NFS.
La [prueba nativa anterior](LLAMA32_ATTENTION_QUANTIZE_STORE_R1.md) conserva
su alcance de protocolo; esta tanda añade los cálculos HLS reales.

## Enlace ABI1 → proyección O

`fpga/rtl/ntrpu_attention_output_path_r1.sv` instancia el cuantizador/escritor
y la proyección matricial. Captura nueve campos de memoria y la identidad
padre completa. Tras publicar ABI1 cambia el dueño del puerto AXI al GEMV;
**retiene la completion del cuantizador durante todas las lecturas O**.
Libera ambos hijos juntos al consumir la completion pública.

La cancelación y la época incorrecta quedan retenidas. Un error cuantizador
impide iniciar O; un comando O ya ofrecido se mantiene y se drena aunque llegue
una cancelación. Las identidades ajenas no se reconocen. El caller conserva
pesos inmutables y regiones de trabajo disjuntas; reset coordina fuente,
cómputo y receptor AXI externo.

El banco nativo ejecutó **105 escenarios**, tres semillas: 99 completions,
99 liberaciones cuantizadas, 84 liberaciones O y nueve resets. Contempla
fallos de tamaño/resultado, cancelación antes y después del cambio de dueño,
época tardía restaurada, identidad de 64 bits ajena, oferta bloqueada,
reutilización de fuente y resets con W o R pendientes.

Son **sustitutos de protocolo que transfieren un beat sintético por etapa**.
Los tamaños públicos de 4/8 KiB son modelados: este banco no prueba la
cuantización, los tensores completos ni la multiplicación matricial.
No se interpretan sus ciclos como rendimiento del diseño aritmético.

Nueve mutantes RTL compilaron y fallaron donde correspondía: liberación
prematura, dueño AW incorrecto, dirección ABI1 desplazada, identidad truncada,
error cuantizador ignorado, identidad ajena aceptada, cancelación ignorada,
revocación olvidada y región cuantizada nunca liberada. El verificador de
trazas rechazó otras 57 corrupciones.

Baseline `20261013T222036Z-V7-attention-output-path-native-r1`, archivo
`63b77c6e9b9f75cc0a0d9afb6faf682701a87e1ff9bf7a398a82a626868f62a0`; revisión
`e03c70c7e198363220d88585266a643daf9a4a0ff98079b2a40c6ff7c4a562e1`.
Cada intento y mutante conserva fuente, invocación y logs en D:/NFS.

## Ejecución O y reparación del observador

La [proyección O real](LLAMA32_ATTENTION_PROJECTION_MEMORY_R1.md) corre como
`20261013T221343Z-V7-llama32-attention-projection-memory-vivado-r1`. La instancia de XSim sigue usando las fuentes selladas;
no se reinició. Su coordinador R1 asignó `state=load(parent)` al validar
dependencias y reutilizó accidentalmente metadatos de la FFN anterior.
Los campos de identidad de ese registro R1 no identifican al proceso actual.

El observador R2 reconstruye la procedencia desde el registro de lanzamiento,
hashes del driver/paquete y la carpeta de la ejecución. Conserva el registro
original, espera el veredicto y verifica archivo/revisión antes de adoptar
el resultado y respaldar una corrección adicional. No lanza herramientas
ni escribe el turno compartido. Coordinador vigente para sucesores:
`llama32-attention-projection-memory-vivado-r2-coordinator.json`.
Auditoría vigente: `audit_attention_composition_active_r15.py`.

Siguiente validación: banco con **cuantizador generado + memoria ABI1 escrita
por él + GEMV O generado**, seguida de raíz, residual/norma y FFN conectadas.
El enlace de salida aún no tiene tanda aritmética Vivado encolada. La tarjeta
DDR y la inferencia completa permanecen pendientes.
