# Propietario de la RAM de salida residual

Estado 2026-10-15 16:46 UTC: **RTL nativo aprobado y respaldado; síntesis/XSim en cola**.
El bloque conserva los dos operandos hasta que termina el cálculo y publica
la RAM residual sólo después de recibir sus 2.048 palabras y una terminación correcta.

## Contrato

`fpga/rtl/ntrpu_residual_owner_r1.sv` captura secuencia y época de 64 bits.
Exige identidad y estado válidos del cargador, respuestas RAM síncronas
y datos FP32 finitos. Comprueba orden, cantidad y contenido finito de las
escrituras; cuenta también la última escritura cuando coincide con `ap_done`.
La lectura pública es síncrona, controlada por CE y devuelve cero fuera
de una respuesta válida.

Cancelación o cambio de época revocan la salida de forma persistente,
incluida una respuesta retenida. Un fallo durante HLS conserva la propiedad
de los operandos hasta el drenaje. El watchdog no permite abandonar un
núcleo colgado y reutilizar sus RAM: ese caso requiere reset coordinado.

## Validación nativa

Banco `fpga/tb/tb_ntrpu_residual_owner_r2.sv`, semillas 1, 7 y 42:

- 219 trabajos; 245.760 palabras residuales verificadas por dirección y valor.
- 63 corrupciones de trazas rechazadas por el revisor independiente.
- 17 mutantes RTL compilados y detectados mediante aserciones explícitas.
- Cada semilla: 73 trabajos, 68 respuestas, 66 inicios, 64 terminaciones,
  67 ACK y seis resets; entre 368.953 y 369.039 ciclos del banco.

Incluye identidad truncada, respuestas ausentes, NaN/Inf, escritura corta,
duplicada o extra, última escritura con `ap_done`, cancelación, revocación
tardía, reset y recuperación. El banco R1 detectaba el mutante de identidad
por timeout; R2 añade comprobaciones en el flanco de inicio. Se conserva
el intento fallido y se repitieron las tres semillas y los 17 mutantes.
El RTL no cambió.

| Evidencia D:/NFS | SHA-256 |
| --- | --- |
| Nativo `20261015T164012Z`, R2 | `ed835aa5d40f7b85e7e723d98624d9651396026af6ff7d98bf70fd949e567404` |
| Mutantes `20261015T164012Z`, R2 | `4f92f5810c5721a5367c672a477b7a4f0e7545520654bdb07d81503386ce5899` |
| Revisión independiente | `ae566174b07212af98767bbfbb3dbf09a453d59cb10779d3cf246cd6281ab10a` |
| Fuente sellada Vivado | `53c747951bec75992284f28afc300a3420340a41b05448d740ddbc366ec763ef` |

## Cola y alcance

`queue_llama32_residual_owner_vivado_r1.py` espera revisión y respaldo
del [cargador residual Vivado R2](LLAMA32_RESIDUAL_LOADER_R1.md).
Sintetizará en VU5P a 5 ns antes de ejecutar las tres semillas XSim;
CPU0/1, dos hilos, límite de 1.800 segundos.

Los peers del banco son sintéticos: estos ciclos no miden aritmética ni
inferencia. El [escritor AXI confirmado por B](LLAMA32_RESIDUAL_STORE_R1.md)
ya pasó su banco nativo y está encolado detrás del propietario.
Sigue pendiente conectar el [HLS residual real](LLAMA32_RESIDUAL_R1.md),
validar toda la cadena de transporte y unirla a la salida FFN viva.
