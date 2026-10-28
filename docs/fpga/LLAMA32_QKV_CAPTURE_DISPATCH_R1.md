# Captura Q/K/V y despacho con propiedad compartida

La [prueba de reutilización entre capas](LLAMA32_QKV_LAYER_REUSE_R1.md)
añade evidencia sobre una misma asignación KV, cambios de capa, reintentos
y vinculación de una nueva época, con el mismo RTL de producción.

Estado 2026-10-16 03:33 UTC: **protocolo integrado aprobado en Icarus;
síntesis y XSim encolados**. El
[módulo R1](../../fpga/rtl/ntrpu_qkv_capture_dispatch_r1.sv) conecta la
[captura Q/K/V](LLAMA32_QKV_SNAPSHOT_R1.md) con el
[despachador](LLAMA32_QKV_DISPATCH_R1.md). El banco utiliza además el append
KV, la guardia de caché y el secuenciador reales. La fuente RoPE, la memoria
AXI y la aritmética por cabeza siguen siendo modelos del banco.

## Secuencia y contrato

Una admisión retiene secuencia, época, capa, posición y lease. La captura
lee V por AXI y 2.560 palabras Q/K de la RAM RoPE, formando las 192 tramas
privadas. Sólo después ofrece el comando al despachador. Éste envía K/V,
espera respuestas B y publicación del prefijo, y entrega Q a atención.
No se añadió otra copia de la RAM de captura.

`cache_exclusive` cubre captura, despacho, respuesta pública y drenaje.
El planificador debe respetar esa señal para excluir todos los otros
usuarios de la caché. La validación de capacidad/lease ocurre en el
despachador después de capturar; un comando inválido puede haber realizado
lecturas privadas, pero no autoriza un append inválido.

La cancelación queda retenida desde su aceptación. Esto incluye el flanco
entre terminar la captura y admitir el despacho: aunque la captura conserve
estado cero y revoque sus datos, el resultado compuesto debe indicar
cancelación. Un commit KV exitoso permanece registrado ante fallos posteriores.

La liberación sigue este orden: ACK público, drenaje del root, ACK de captura,
ACK de la fuente RoPE y retorno a reposo. Una identidad RoPE ajena mantiene
el módulo ocupado y la exclusión de caché hasta restaurarla o efectuar un
reset coordinado. V y la RAM fuente permanecen inmutables hasta su liberación.
El reset debe abarcar ambos bloques, los consumidores y los destinos AXI.

## Evidencia nativa

El [banco R2](../../fpga/tb/tb_ntrpu_qkv_capture_dispatch_r2.sv) pasó tres
semillas y 31 escenarios por semilla: 93 trabajos, 81 respuestas, 57 errores
esperados, 12 resets y 12 revocaciones retenidas. Contrastó 6.873 tramas,
867 cabezas y 49.152 palabras públicas. Registró 168.963 solicitudes de
palabras RoPE y 2.520 beats V aceptados. Estos contadores no representan
rendimiento de inferencia con memoria física.

El [revisor](../../tools/check_ntrpu_qkv_capture_dispatch_r1.py) reconstruye
las tramas desde las respuestas reales de RAM y V del banco; contrasta
escrituras KV, límites de página, publicación, consultas por cabeza y
resultados. Rechazó 258 corrupciones de traza. Diez mutantes compilados
fallaron por aserciones explícitas, incluidos pérdida del pulso de
cancelación, liberación prematura y omisión del drenaje de la fuente.

Tandas restauradas en D: y NFS:

- `20261016T031546Z-V7-qkv-capture-dispatch-native-r1`: prueba inicial de 12 trabajos.
- `20261016T031927Z-V7-qkv-capture-dispatch-native-r2`: banco de 93 trabajos.
- `20261016T032234Z-V7-qkv-capture-dispatch-mutants-r1`: diez mutantes.

Revisión `build/cloud-setup/20261002-board/qkv-capture-dispatch-native-r1-review.json`,
SHA-256 `55f58637af007e5c65459baab9ae85bab7fff884b418349085b9100e034cb104`.
TAR del banco R2: `03cf907a52fd78b4d75bafdadf0d7386c37c1d0727a7b70838ec1c4627e2b715`.
TAR de mutantes: `9f169f481ed26e07abd582e468cb85343489f6d4fe52efb89bd68986405f9ca1`.

## Turno Vivado y trabajo restante

Coordinador `qkv-capture-dispatch-vivado-r1-coordinator.json`, PID 9824,
espera despacho R2 revisado/restaurado y el slot libre. Paquete de 44 archivos,
ocho fuentes compiladas, dos CPU, 30 minutos y reloj OOC de 5 ns, sin reintento
automático. ZIP: `c7d21494215532d63e069055b799dfe4a8cec33dfca2746e3ac36c763d56c84c`.

La síntesis medirá captura, lector AXI y despacho; el revisor exige RAMB
para la captura. La simulación incluye los consumidores reales descritos
arriba. Recursos y timing aún no están medidos para esta composición.
Falta conectar RoPE y atención HLS reales, integrar el planificador/árbitro
global, ampliar pruebas de ofertas retenidas y respuestas ajenas de hijos,
y validar memoria física y timing con colocación/ruteo.
