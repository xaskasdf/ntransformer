# Despacho Q/K/V hacia KV y atención

Estado 2026-10-16 03:04 UTC: **banco de fallos aprobado en Icarus;
síntesis y XSim encolados**. El [controlador R1](../../fpga/rtl/ntrpu_qkv_dispatch_r1.sv)
consume una captura Q/K/V completa. El
[banco R4](../../fpga/tb/tb_ntrpu_qkv_dispatch_r4.sv) conecta append KV,
guardia de caché y secuenciador de cabezas reales; la RAM fuente, memoria
AXI y aritmética por cabeza son modelos del banco.

## Secuencia y propiedad

1. Validar identidad, época, capa, posición, capacidad y reserva de leases.
   El llamador debe excluir otros usuarios de caché mientras
   `cache_exclusive` esté activo, incluido el drenaje final.
2. Enviar las 64 tramas K/V al append. Esperar commit, respuesta y ACK;
   comprobar la actualización del prefijo antes de admitir atención.
3. Ofrecer el comando al root y enviar las 128 tramas Q. Conservar tanto
   la captura fuente como el resultado del root hasta el ACK público.
4. Drenar al consumidor antes de liberar la fuente. La lectura pública
   utiliza la RAM FP32 del root; no duplica el vector completo.

Una cancelación posterior al commit KV **no revierte ese commit**.
`completion_kv_committed` conserva ese hecho; el llamador no debe repetir
el append en la misma posición y lease. Una respuesta de identidad ajena
no se reconoce. El reset debe coordinar controlador, pares y destino AXI.

`completion_kv_bytes` conserva `done_bytes`: cuenta handshakes W, incluidos
los de ráfagas cuya respuesta B falla. No representa bytes confirmados.
Estado correcto, todas las respuestas B y prefijo publicado son necesarios
para certificar el commit; el banco comprueba también el fallo de la última B
después de haber enviado los 4.096 bytes.

## Banco de fallos y revisión independiente

La tanda `20261016T025834Z-V7-qkv-dispatch-native-r4` pasó tres semillas,
48 escenarios por semilla: 144 trabajos, 132 respuestas, 105 errores esperados,
12 resets y 15 revocaciones de resultados retenidos. Contrastó 10.338 tramas,
978 cabezas y 55.296 palabras públicas. Incluye cancelaciones durante captura,
AW pendiente, Q, cálculo y después del ACK KV; RAM ausente, NaN, error B,
fuente ajena y cambios de época. Estos ciclos pertenecen a pares sintéticos;
no son una medida de rendimiento de inferencia.

El [revisor R2](../../tools/check_ntrpu_qkv_dispatch_r2.py) reconstruye bytes,
direcciones físicas, límites de página, respuestas B, prefijos y orden de
liberación. Rechazó 267 corrupciones de traza. La tanda
`20261016T025947Z-V7-qkv-dispatch-mutants-r2` detectó 25 mutantes compilados
por aserciones explícitas, sin aceptar timeouts como detección.

Se conserva el intento anterior: un mutante que omitía el filtro NaN del
despachador escapaba del banco R3 porque append/secuenciador lo rechazaban
después. El revisor de traza sí lo rechazó. R4 añade dos aserciones de finitud
en las transferencias aceptadas; el RTL de producción no cambió. También
se conservó el revisor R1, que confundía bytes W con bytes confirmados y
rechazaba respuestas de admisión observadas después del NBA del mismo flanco.

Las cuatro tandas y sus fuentes tienen restauración completa verificada.
Revisión: `build/cloud-setup/20261002-board/qkv-dispatch-native-r2-review.json`,
SHA-256 `38dd21c5cdf5f2e7f6be0c7f065caeafd999002987c6625431562e36f9ae6cdc`.
TAR del banco R4: `8113e905155e8cb22fcd0521f39f52f1e7e30359f929f4e3fff1346083d0cb26`.
TAR de mutantes R2: `7ea5fe3869c0aaadea0e9060d5d767e0ff363d5b6a445c5cb2cc69f22fea7c1e`.

## Turno Vivado

Coordinador `qkv-dispatch-vivado-r2-coordinator.json`, PID 3996: espera captura
Q/K/V R2 revisada y restaurada, y el slot libre. Paquete de 54 archivos,
cinco fuentes compiladas, 5 ns OOC, dos CPU y 30 minutos sin reintento automático.
ZIP fuente: `a8af82d6b2c2a00643a7320e2337b8113efd26750bf9dc6161f337aed12e3bea`.
La síntesis mide sólo el despachador; la simulación incluye append, guardia
y secuenciador. No acredita todavía recursos ni timing de la integración.

## Evidencia inicial

La tanda `20261016T015242Z-V7-qkv-dispatch-native-r2` terminó con salida
cero en Icarus: tres semillas y cuatro casos por semilla, incluidos dos
tokens consecutivos, capa 15 y reserva máxima de leases. El banco informa
2.304 tramas, 384 cabezas y 24.576 palabras públicas verificadas.
Comprueba K/V confirmado por B antes del root y lectura desordenada de
los 2.048 resultados por trabajo. Esto no mide aritmética de atención
ni rendimiento DDR.

El intento R1 se conserva: su aserción de propiedad observaba el estado
anterior al NBA en el flanco de admisión. R2 observa ese mismo flanco
después de asentarse, manteniendo la aserción; el RTL no cambió.
Ambas tandas y sus fuentes se restauraron y verificaron en
`D:/AI/ntransformer/artifacts/ntrpu-qkv-dispatch-native/`.
SHA-256 del TAR R2:
`537b0c42a9c49c8f18860b5be9e0eff605d0a5761b3b80cbaacd3ba0fa40c01c`.

## Criterios pendientes

Ampliar el banco con ofertas de comando/datos retenidas mediante adaptadores
de interfaz y respuestas ajenas o cambiantes de los hijos; no modificar
el paquete ya encolado. La [conexión con captura Q/K/V real](LLAMA32_QKV_CAPTURE_DISPATCH_R1.md)
pasó 93 escenarios nativos y quedó encolada después de este bloque.
RoPE y aritmética HLS reales, arbitraje global de caché y memoria física
siguen pendientes en esa composición.
