# Contexto privado y publicación de atención R1

## Alcance

`fpga/rtl/ntrpu_attention_context_r1.sv` administra Q, estado intermedio,
salida y control del contrato de atención por tiles. Su par HLS es externo:
este módulo no calcula atención ni contiene el RTL generado por Vitis.
El banco usa patrones identificables, **no resultados de softmax**.

La integración prevista conecta `query_*`, `compute_*`, `finish_*` y
`query_done_*` con [el camino KV R1](LLAMA32_ATTENTION_KV_PATH_R1.md).
El par aritmético leerá K/V directamente de sus dos puertos RAM. El
adaptador conjunto y la comprobación de los puertos HLS generados siguen
pendientes; probar los componentes separados no los acredita.

## Memorias y control

Un comando captura 64 palabras Q FP32 mediante un puerto interno de
2048 bits, posición, capa, cabezal, secuencia, época y concesión. No es
un bus de pines ni una interfaz de carga DMA. Se rechazan Q no finito,
identificadores nulos y dimensiones fuera del contrato Llama 1B.

Q ocupa 256 bytes; dos bancos de estado distintos contienen 104 palabras
de 64 bits cada uno; la salida privada ocupa otros 256 bytes: 2176 bytes
lógicos en total. Las lecturas HLS son síncronas, de un ciclo. Cada tile
exige 104 escrituras únicas de estado; sólo el último permite y exige
64 escrituras FP32 finitas de salida. Se incluyen las escrituras que
coinciden con `hls_done`. El banco de estado cambia sólo al aceptar
`finish_valid && finish_ready` exitoso.

`hls_start` permanece alto hasta `hls_ready`; un ready anticipado no
autoriza terminar antes de `hls_done`. Esto sigue el contrato
[AMD de control de bloque](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Block-Level-Control-Protocols).
Una cancelación propaga `query_cancel_valid`, pero drena el par HLS
hasta ready y done antes de confirmar finish. No existe watchdog que
permita abandonar arbitrariamente un HLS todavía activo.

El estado externo de atención sigue el formato del [bloque aritmético](LLAMA32_ATTENTION_TILE_R1.md).
El controlador verifica cobertura y propiedad de RAM; la interpretación
numérica del estado corresponde al HLS. Estado 10 representa retorno HLS
no cero; 11, certificado global incompleto. Se conservan 1 para comando,
2 para identidad, 4 para memoria/formato y 7 para cancelación.

## Publicación y revocación

El resultado permanece privado hasta recibir el certificado del camino
KV: estado cero, misma identidad, prefijo completo, número correcto de
tiles y todos los finishes aceptados. El `ap_done` de un tile aislado no
basta para publicar el resultado de la consulta completa.

`completion_valid/status` son un registro terminal estable mientras se
retiene la respuesta. **La autorización vigente es `result_valid`**, no
`completion_status == 0` por sí solo. Se debe comprobar al leer o confirmar
el resultado. Una época distinta bloquea inmediatamente las lecturas;
si se observa en un flanco, queda revocada aunque vuelva el valor anterior.
`result_q_valid` y los datos también quedan bloqueados al revocar o retirar
la respuesta. El consumidor debe mantener privada cualquier copia leída
hasta su propia confirmación de época; aún falta ese publicador exterior.

Un reset coordinado limpia propiedad y validez, no el contenido físico
de RAM. También deben resetearse/drenarse KV, HLS y el consumidor externo.
El contrato usa un reloj; CDC y reset independiente no están calificados.

## Evidencia y reproducción

Tanda nativa `20261013T145956Z-V7-attention-context-native-r1`, archivo
SHA-256 `989cc4d8f929598c7172891e6e571deaf7b401499482f0b9b2de4027acc232cc`.
Tres semillas aprobaron **207 casos, 231 tiles y 12 resets**. Por suite:
4672 palabras Q, 1976 lecturas de estado, 2240 lecturas de resultado;
7591 escrituras de estado y 3424 de salida observadas. Hay 66 completados
y tres abortados por reset por suite. Incluye GQA completo, 1/64/65/129
posiciones, backpressure, identidad/forma inválidas, cobertura incompleta,
escrituras duplicadas, NaN, cancelación y revocación tardía.

El revisor independiente reconstruye las palabras, el orden de tiles y
el plan de casos; rechaza 15 corrupciones por suite. Seis variantes RTL
compiladas fueron rechazadas: pulso start corto, banco equivocado,
publicación anticipada, época ignorada, cobertura de estado omitida y
retorno HLS ignorado. Todos los intentos están respaldados en D: y NFS.
El primer intento falló por muestrear finish en el mismo delta que el
banco retiraba señales del par; se corrigió sólo el muestreo del banco.

```powershell
python build/cloud-setup/20261002-board/run_llama32_attention_context_native_dev_r1.py
```

La tanda Vivado prevista repite las tres suites y sintetiza este módulo
aislado en VU5P a objetivo 5 ns, dos CPU y 1200 s. No acredita todavía
integración KV/HLS, cálculo de atención, inferencia completa, memoria
física ni cierre de timing ruteado.
